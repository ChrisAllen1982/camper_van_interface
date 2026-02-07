#!/usr/bin/env python3
import asyncio
import json
import logging
import os
import re
import shlex
import subprocess
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import urllib.error
import urllib.request
import urllib.parse

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# -------------------- CONFIG --------------------

WIFI_IFACE = "wlan0"  # STA/client interface
AP_IFACE = "wlan1"    # AP interface (RaspAP hotspot)

# CarConnectivity CLI (prefer absolute/venv path, fall back)
_CLI_CANDIDATES = [
    str((Path(__file__).resolve().parent / ".venv/bin/carconnectivity-cli")),
    "/usr/local/bin/carconnectivity-cli",
    "/usr/bin/carconnectivity-cli",
    "carconnectivity-cli",
]
CARCONNECTIVITY_CLI = next(
    (c for c in _CLI_CANDIDATES if ("/" not in c) or os.path.exists(c)),
    "carconnectivity-cli"
)
CARCONNECTIVITY_CONFIG = "carconnectivity.json"  # passed as positional argument to the CLI

# Data files
BASE_DIR = Path(__file__).resolve().parent
SETTINGS_PATH = BASE_DIR / "settings.json"
CACHE_PATH = BASE_DIR / "vehicle_cache.json"

# Victron defaults can come from env too
VICTRON_API_BASE = os.getenv("VICTRON_API_BASE", "").rstrip("/")
VICTRON_API_KEY = os.getenv("VICTRON_API_KEY", "")
VICTRON_DEVICE_NAME = os.getenv("VICTRON_DEVICE_NAME", "")

# Background vehicle updater
VEHICLE_UPDATE_ENABLED = os.getenv("VEHICLE_UPDATE_ENABLED", "1") == "1"
VEHICLE_UPDATE_INTERVAL_SEC = int(os.getenv("VEHICLE_UPDATE_INTERVAL_SEC", "300"))
_vehicle_update_lock = asyncio.Lock()

logging.basicConfig(level=logging.INFO)

# -------------------- HELPERS --------------------

def run(cmd: str) -> str:
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if p.returncode != 0:
        msg = (p.stderr or p.stdout or f"Command failed: {cmd}").strip()
        raise RuntimeError(msg)
    return p.stdout


def wpa(cmd: str) -> str:
    return run(f"sudo wpa_cli -i {shlex.quote(WIFI_IFACE)} {cmd}").strip()


def parse_kv(text: str) -> dict:
    kv = {}
    for ln in text.splitlines():
        if "=" in ln:
            k, v = ln.split("=", 1)
            kv[k.strip()] = v.strip()
    return kv


def load_settings() -> dict:
    if SETTINGS_PATH.exists():
        return json.loads(SETTINGS_PATH.read_text())
    return {}


def save_settings(data: dict) -> None:
    SETTINGS_PATH.write_text(json.dumps(data, indent=2))


def load_cache() -> dict:
    if CACHE_PATH.exists():
        return json.loads(CACHE_PATH.read_text())
    return {}


def save_cache(data: dict) -> None:
    CACHE_PATH.write_text(json.dumps(data, indent=2))


def serve_file_bytes(name: str) -> Response:
    p = BASE_DIR / name
    if not p.exists():
        raise HTTPException(status_code=404, detail=f"Missing file: {name}")
    return Response(p.read_bytes(), media_type="text/html; charset=utf-8")


# -------------------- MODELS --------------------

class WifiStatus(BaseModel):
    connected: bool
    ssid: Optional[str] = None
    ip: Optional[str] = None
    wpa_state: Optional[str] = None


class SettingsModel(BaseModel):
    vin: str = Field(min_length=8)
    victron_base_url: Optional[str] = None
    victron_api_key: Optional[str] = None
    victron_device_name: Optional[str] = None


# -------------------- APP --------------------

# Use FastAPI lifespan event handler instead of deprecated on_event
@asynccontextmanager
async def lifespan(app: FastAPI):
    if VEHICLE_UPDATE_ENABLED:
        asyncio.create_task(_vehicle_update_loop())
    yield

app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")


# -------------------- PAGES --------------------

@app.get("/", response_class=HTMLResponse)
def page_index():
    return serve_file_bytes("index.html")


@app.get("/wifi", response_class=HTMLResponse)
def page_wifi():
    return serve_file_bytes("wifi.html")


@app.get("/vehicle", response_class=HTMLResponse)
def page_vehicle():
    return serve_file_bytes("vehicle.html")


@app.get("/power", response_class=HTMLResponse)
def page_power():
    return serve_file_bytes("power.html")


@app.get("/settings", response_class=HTMLResponse)
def page_settings():
    return serve_file_bytes("settings.html")


# -------------------- SETTINGS API --------------------

@app.get("/api/settings")
def api_get_settings():
    s = load_settings()
    return {
        "vin": s.get("vin"),
        "victron_base_url": s.get("victron_base_url"),
        "victron_api_key": s.get("victron_api_key"),
        "victron_device_name": s.get("victron_device_name"),
    }


@app.post("/api/settings")
def api_set_settings(req: SettingsModel):
    data = load_settings()
    data["vin"] = req.vin

    if req.victron_base_url is not None:
        data["victron_base_url"] = req.victron_base_url
    if req.victron_api_key is not None:
        data["victron_api_key"] = req.victron_api_key
    if req.victron_device_name is not None:
        data["victron_device_name"] = req.victron_device_name

    save_settings(data)
    return {"ok": True}


@app.get("/api/vehicle_cache")
def api_vehicle_cache():
    c = load_cache()
    if not c:
        return {"ok": False}
    return c


# -------------------- CARCONNECTIVITY / VW --------------------

def cc_cli(args: List[str]) -> str:
    """
    carconnectivity-cli usage is: carconnectivity-cli <config> <command> ...
    Example: carconnectivity-cli carconnectivity.json get /garage/<VIN>/drives/total_range
    """
    cmd = [CARCONNECTIVITY_CLI, CARCONNECTIVITY_CONFIG] + args
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        msg = (p.stderr or p.stdout or "carconnectivity-cli failed").strip()
        raise HTTPException(status_code=500, detail=msg)
    return p.stdout.strip()


@app.get("/api/vw/get")
def vw_get(path: str = Query(..., min_length=3)):
    val = cc_cli(["get", path])
    try:
        return json.loads(val)
    except Exception:
        return {"raw": val}


def _get_value(path: str) -> Any:
    """
    Helper to unwrap common {"value": ...} or {"raw": ...} responses.
    """
    try:
        v = vw_get(path=path)
        if isinstance(v, dict):
            if "value" in v:
                return v.get("value")
            if "raw" in v:
                return v.get("raw")
        return v
    except Exception:
        return None


def _first(*vals: Any) -> Any:
    for x in vals:
        if x is None:
            continue
        s = str(x).strip()
        if not s or s.lower() in ("none", "null", "nan"):
            continue
        return x
    return None


@app.get("/api/vw/summary")
def vw_summary(vin: str = Query(..., min_length=8)):
    """
    IMPORTANT: This returns the nested structure that vehicle.html expects:
      { ok, vin, data:{...}, coords:{latitude,longitude}, address:{...} }
    """
    base = f"/garage/{vin}"

    # Minimal "data" block expected by vehicle.html
    data: Dict[str, Any] = {
        "name": _first(_get_value(f"{base}/name"), _get_value(f"{base}/nickname")),
        "model": _first(_get_value(f"{base}/model")),
        "odometer": _first(_get_value(f"{base}/odometer"), _get_value(f"{base}/odometer_km")),
        # Address-derivative placeholders used by vehicle.html (safe if empty)
        "addr_display_name": None,
        "addr_name": None,
        "addr_road": None,
        "addr_postcode": None,
        "addr_county": None,
        "addr_state": None,
        "addr_country": None,
        "addr_lat": None,
        "addr_lon": None,
        "pos_lat": None,
        "pos_lon": None,
    }

    # Best-effort coordinate extraction (paths vary by config/provider)
    lat = _first(
        _get_value(f"{base}/position/latitude"),
        _get_value(f"{base}/position/lat"),
        _get_value(f"{base}/location/latitude"),
        _get_value(f"{base}/location/lat"),
        _get_value(f"{base}/coordinates/latitude"),
        _get_value(f"{base}/coordinates/lat"),
    )
    lon = _first(
        _get_value(f"{base}/position/longitude"),
        _get_value(f"{base}/position/lon"),
        _get_value(f"{base}/location/longitude"),
        _get_value(f"{base}/location/lon"),
        _get_value(f"{base}/coordinates/longitude"),
        _get_value(f"{base}/coordinates/lon"),
    )

    if lat is not None:
        data["pos_lat"] = lat
    if lon is not None:
        data["pos_lon"] = lon

    coords = {"latitude": lat, "longitude": lon}

    # Get address from carconnectivity position_location
    address = {
        "display_name": _get_value(f"{base}/position/position_location/display_name"),
        "name": _get_value(f"{base}/position/position_location/name"),
        "road": _get_value(f"{base}/position/position_location/road"),
        "postcode": _get_value(f"{base}/position/position_location/postcode"),
        "county": _get_value(f"{base}/position/position_location/county"),
        "state": _get_value(f"{base}/position/position_location/state"),
        "country": _get_value(f"{base}/position/position_location/country"),
    }

    return {"ok": True, "vin": vin, "data": data, "coords": coords, "address": address}


@app.post("/api/vw/preload")
def vw_preload():
    s = load_settings()
    vin = s.get("vin")
    if not vin:
        raise HTTPException(status_code=400, detail="No VIN set in settings")

    summary = vw_summary(vin=vin)

    drive_level_path = f"/garage/{vin}/drives/primary/level"
    total_range_path = f"/garage/{vin}/drives/total_range"

    drive_level = None
    total_range = None
    try:
        dl = vw_get(path=drive_level_path)
        if isinstance(dl, dict):
            drive_level = dl.get("value") or dl.get("raw")
    except Exception:
        pass
    try:
        tr = vw_get(path=total_range_path)
        if isinstance(tr, dict):
            total_range = tr.get("value") or tr.get("raw")
    except Exception:
        pass

    payload = {
        "ok": True,
        "vin": vin,
        "fetched_at": int(time.time()),
        "summary": summary,
        "extras": {"drive_level": drive_level, "total_range": total_range},
    }
    save_cache(payload)
    return payload


# -------------------- BACKGROUND VEHICLE UPDATER --------------------

async def _run_vw_preload_background() -> None:
    if _vehicle_update_lock.locked():
        logging.info("Vehicle update skipped (already running)")
        return
    async with _vehicle_update_lock:
        await asyncio.to_thread(vw_preload)


async def _vehicle_update_loop() -> None:
    await asyncio.sleep(5)
    while True:
        try:
            logging.info("Background vehicle update")
            await _run_vw_preload_background()
        except Exception:
            logging.exception("Vehicle update failed")
        await asyncio.sleep(VEHICLE_UPDATE_INTERVAL_SEC)


# -------------------- VICTRON / POWER --------------------

def _victron_config() -> Dict[str, str]:
    s = load_settings()
    base = (s.get("victron_base_url") or VICTRON_API_BASE or "").rstrip("/")
    key = s.get("victron_api_key") or VICTRON_API_KEY or ""
    name = s.get("victron_device_name") or VICTRON_DEVICE_NAME or ""
    return {"base": base, "key": key, "name": name}


def _victron_http_get_json(url: str, api_key: str) -> Dict[str, Any]:
    req = urllib.request.Request(url)
    if api_key:
        req.add_header("x-api-key", api_key)  # adjust if your Victron API expects another header
    req.add_header("accept", "application/json")

    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            raw = r.read().decode("utf-8", errors="replace")
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        raise HTTPException(status_code=e.code, detail=f"Victron API error: {detail or e.reason}")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Victron API unreachable: {e}")


def victron_get_device_by_name(name: str) -> Dict[str, Any]:
    cfg = _victron_config()
    if not cfg["base"]:
        raise HTTPException(status_code=400, detail="Victron base URL not configured")
    if not name:
        raise HTTPException(status_code=400, detail="Victron device name not configured")

    url = f"{cfg['base']}/devices/by-name/{urllib.parse.quote(name)}"
    return _victron_http_get_json(url, cfg["key"])


# Victron SmartSolar charge state mapping (common values)
_VICTRON_CHARGE_STATE_MAP = {
    0: "Not charging",
    2: "Fault",
    3: "Bulk",
    4: "Absorption",
    5: "Float",
}


def _get_nested(d: Dict[str, Any], path: List[str]) -> Any:
    cur: Any = d
    for k in path:
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            return None
    return cur


def summarize_power_state(device: Dict[str, Any]) -> Dict[str, Any]:
    """
    Your observed payload:
      device.data._data.charge_state (int)
      device.data._data.battery_voltage (float)
      device.data._data.battery_charging_current (float)
      device.data._data.solar_power (int, W)
      device.data._data.external_device_load (int, W)
    """
    name = device.get("name")
    dd = _get_nested(device, ["data", "_data"]) or {}

    charge_state = dd.get("charge_state")
    try:
        charge_state_int = int(charge_state) if charge_state is not None else None
    except Exception:
        charge_state_int = None

    state = _VICTRON_CHARGE_STATE_MAP.get(charge_state_int, "Unknown")

    voltage = dd.get("battery_voltage")
    current = dd.get("battery_charging_current")
    solar_w = dd.get("solar_power")
    load_w = dd.get("external_device_load")

    power_w = solar_w
    if power_w is None and voltage is not None and current is not None:
        try:
            power_w = float(voltage) * float(current)
        except Exception:
            power_w = None

    return {
        "ok": True,
        "name": name,
        "state": state,
        "charge_state": charge_state_int,
        "soc": None,  # SmartSolar typically doesn't provide SOC unless you have a BMV/SmartShunt
        "voltage": voltage,
        "current": current,
        "power_w": power_w,
        "solar_w": solar_w,
        "load_w": load_w,
        "raw": device,
    }


@app.get("/api/power")
def api_power():
    cfg = _victron_config()
    device = victron_get_device_by_name(cfg["name"])
    return summarize_power_state(device)


# -------------------- WIFI API --------------------

@app.get("/api/status", response_model=WifiStatus)
def status():
    try:
        out = wpa("status")
        kv = parse_kv(out)

        wpa_state = kv.get("wpa_state", "")
        connected = (wpa_state == "COMPLETED")
        ssid = kv.get("ssid") if connected else None

        ip = None
        try:
            ip_out = run(f"ip -4 addr show {shlex.quote(WIFI_IFACE)}")
            m = re.search(r"inet (\d+\.\d+\.\d+\.\d+)", ip_out)
            if m:
                ip = m.group(1)
        except Exception:
            pass

        return WifiStatus(connected=connected, ssid=ssid, ip=ip, wpa_state=wpa_state)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/saved_networks")
def saved_networks():
    out = wpa("list_networks")
    lines = [ln for ln in out.splitlines() if ln.strip()]
    if not lines:
        return {"networks": []}
    nets = []
    for ln in lines[1:]:
        parts = ln.split("\t")
        if len(parts) >= 2:
            nets.append({"id": parts[0], "ssid": parts[1]})
    return {"networks": nets}


@app.post("/api/scan")
def scan():
    wpa("scan")
    time.sleep(2)
    out = wpa("scan_results")
    lines = [ln for ln in out.splitlines() if ln.strip()]
    aps = []
    for ln in lines[1:]:
        parts = ln.split("\t")
        if len(parts) >= 5:
            bssid, freq, sig, flags, ssid = parts[:5]
            aps.append({"bssid": bssid, "freq": freq, "sig": sig, "flags": flags, "ssid": ssid})
    return {"aps": aps}


class ConnectRequest(BaseModel):
    ssid: str = Field(min_length=1)
    password: Optional[str] = None


@app.post("/api/connect")
def connect_network(req: ConnectRequest):
    """
    Connect to a WiFi network. If the network is already saved, select it.
    Otherwise, add it as a new network.
    """
    try:
        # Check if network already exists
        saved = saved_networks()
        existing_id = None
        for net in saved.get("networks", []):
            if net.get("ssid") == req.ssid:
                existing_id = net.get("id")
                break
        
        if existing_id is not None:
            # Network exists, select it
            wpa(f"select_network {existing_id}")
            wpa("reassociate")
            return {"ok": True, "message": f"Connecting to existing network {req.ssid}"}
        else:
            # Add new network
            add_result = wpa("add_network")
            network_id = add_result.strip()
            
            # Set SSID
            wpa(f'set_network {network_id} ssid \"{req.ssid}\"')
            
            # Set password or configure as open network
            if req.password:
                wpa(f'set_network {network_id} psk \"{req.password}\"')
            else:
                wpa(f"set_network {network_id} key_mgmt NONE")
            
            # Enable and select the network
            wpa(f"enable_network {network_id}")
            wpa(f"select_network {network_id}")
            wpa("save_config")
            wpa("reassociate")
            
            return {"ok": True, "message": f"Added and connecting to {req.ssid}", "network_id": network_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/captive_check")
def captive_check():
    url = "http://connectivitycheck.gstatic.com/generate_204"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "curl"})
        with urllib.request.urlopen(req, timeout=5) as r:
            return {"ok": True, "status": r.status}
    except Exception as e:
        return {"ok": False, "error": str(e)}
