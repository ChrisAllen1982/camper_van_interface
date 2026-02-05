#!/usr/bin/env python3
import json
import re
import shlex
import subprocess
import time
from pathlib import Path
from typing import List, Optional, Dict, Any

import urllib.error
import urllib.request

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# ---- CONFIG ----
WIFI_IFACE = "wlan0"   # STA/client interface
AP_IFACE = "wlan1"     # AP interface (RaspAP hotspot)

# CarConnectivity / VW Connect (UK/EU)
CARCONNECTIVITY_CLI = "carconnectivity-cli"
CARCONNECTIVITY_CONFIG = "carconnectivity.json"  # in root

# Settings storage (persisted on Pi)
SETTINGS_PATH = Path("./settings.json")  # stored next to server.py
CACHE_PATH = Path("./vehicle_cache.json")

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")


def run(cmd: str) -> str:
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if p.returncode != 0:
        msg = (p.stderr or p.stdout or f"Command failed: {cmd}").strip()
        raise RuntimeError(msg)
    return p.stdout


def wpa(cmd: str) -> str:
    return run(f"sudo wpa_cli -i {shlex.quote(WIFI_IFACE)} {cmd}").strip()


def cc_cli(args: list[str]) -> str:
    cmd = [CARCONNECTIVITY_CLI, CARCONNECTIVITY_CONFIG] + args
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        msg = (p.stderr or p.stdout or "carconnectivity-cli failed").strip()
        raise HTTPException(status_code=500, detail=msg)
    return p.stdout.strip()


def parse_kv(text: str) -> dict:
    kv = {}
    for ln in text.splitlines():
        if "=" in ln:
            k, v = ln.split("=", 1)
            kv[k.strip()] = v.strip()
    return kv


def read_first_existing(paths: list[str]) -> Optional[str]:
    for p in paths:
        try:
            path = Path(p)
            if path.exists():
                return path.read_text(errors="ignore")
        except Exception:
            pass
    return None


def get_ap_ssid() -> Optional[str]:
    candidates = [
        "/etc/hostapd/hostapd.conf",
        "/etc/raspap/hostapd.conf",
        "/etc/raspap/hostapd.ini",
        "/etc/hostapd/hostapd.conf.orig",
    ]
    txt = read_first_existing(candidates)
    if not txt:
        return None
    for line in txt.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("ssid="):
            return line.split("=", 1)[1].strip()
    return None


def get_iface_mac(iface: str) -> Optional[str]:
    try:
        p = Path(f"/sys/class/net/{iface}/address")
        if p.exists():
            return p.read_text().strip().lower()
    except Exception:
        pass
    return None


def recent_supplicant_logs(lines: int = 140) -> str:
    unit_candidates = [
        f"wpa_supplicant@{WIFI_IFACE}.service",
        "wpa_supplicant.service",
    ]
    for unit in unit_candidates:
        try:
            out = run(f"sudo journalctl -u {shlex.quote(unit)} -n {int(lines)} --no-pager")
            if out.strip():
                return out
        except Exception:
            pass
    return ""


def serve_file_bytes(filename: str) -> Response:
    try:
        with open(filename, "rb") as f:
            return Response(content=f.read(), media_type="text/html; charset=utf-8")
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail=f"{filename} not found next to server.py")


# -------------------- Settings Helpers --------------------

def load_settings() -> Dict[str, Any]:
    if not SETTINGS_PATH.exists():
        return {}
    try:
        return json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_settings(data: Dict[str, Any]) -> None:
    SETTINGS_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")


def save_cache(data: Dict[str, Any]) -> None:
    try:
        CACHE_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except Exception:
        pass


def load_cache() -> Dict[str, Any]:
    if not CACHE_PATH.exists():
        return {}
    try:
        return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


# -------------------- Models --------------------

class Network(BaseModel):
    bssid: str
    frequency: int
    signal: int
    flags: str
    ssid: str


class ConnectRequest(BaseModel):
    ssid: str = Field(min_length=1, max_length=32)
    psk: Optional[str] = Field(default=None, max_length=63)


class WifiStatus(BaseModel):
    connected: bool
    ssid: Optional[str] = None
    ip: Optional[str] = None
    bssid: Optional[str] = None
    freq: Optional[int] = None
    signal: Optional[int] = None
    key_mgmt: Optional[str] = None
    wpa_state: Optional[str] = None
    network_id: Optional[int] = None


class CaptiveCheck(BaseModel):
    captive: bool
    online: bool
    portal_url: Optional[str] = None
    detail: Optional[str] = None


class ConnectResult(BaseModel):
    ok: bool
    ssid_requested: str
    network_id: Optional[int] = None
    wpa_state: Optional[str] = None
    ssid: Optional[str] = None
    ip: Optional[str] = None
    bssid: Optional[str] = None
    key_mgmt: Optional[str] = None
    detail: Optional[str] = None
    wpa_cli_status_raw: Optional[str] = None
    wpa_logs_tail: Optional[str] = None


class SimpleResult(BaseModel):
    ok: bool
    detail: Optional[str] = None


class SavedNetwork(BaseModel):
    network_id: int
    ssid: str
    bssid: str
    flags: str
    is_current: bool = False
    is_disabled: bool = False


class ForgetRequest(BaseModel):
    network_id: Optional[int] = None
    ssid: Optional[str] = None


class ForgetResult(BaseModel):
    ok: bool
    removed_network_id: Optional[int] = None
    removed_ssid: Optional[str] = None
    detail: Optional[str] = None


class SettingsModel(BaseModel):
    vin: str = Field(min_length=8)


# -------------------- Pages --------------------

@app.get("/", response_class=HTMLResponse)
def page_index():
    return serve_file_bytes("index.html")


@app.get("/wifi", response_class=HTMLResponse)
def page_wifi():
    return serve_file_bytes("wifi.html")


@app.get("/vehicle", response_class=HTMLResponse)
def page_vehicle():
    return serve_file_bytes("vehicle.html")


@app.get("/settings", response_class=HTMLResponse)
def page_settings():
    return serve_file_bytes("settings.html")


# -------------------- Settings API --------------------

@app.get("/api/settings")
def api_get_settings():
    s = load_settings()
    return {"vin": s.get("vin")}


@app.post("/api/settings")
def api_set_settings(req: SettingsModel):
    data = load_settings()
    data["vin"] = req.vin
    save_settings(data)
    return {"ok": True, "vin": req.vin}


@app.get("/api/vehicle_cache")
def api_vehicle_cache():
    c = load_cache()
    return c if c else {"ok": False, "detail": "No cache yet"}


# -------------------- WiFi API --------------------

@app.get("/api/status", response_model=WifiStatus)
def status():
    try:
        out = wpa("status")
        kv = parse_kv(out)

        wpa_state = kv.get("wpa_state", "")
        connected = (wpa_state == "COMPLETED") and bool(kv.get("ssid"))

        signal = None
        if "rssi" in kv:
            try:
                signal = int(kv["rssi"])
            except ValueError:
                signal = None
        if signal is None:
            try:
                sp = wpa("signal_poll")
                for ln in sp.splitlines():
                    if ln.startswith("RSSI="):
                        signal = int(ln.split("=", 1)[1].strip())
                        break
            except Exception:
                pass

        freq = None
        if "freq" in kv:
            try:
                freq = int(kv["freq"])
            except ValueError:
                freq = None

        nid = None
        if "id" in kv and kv["id"].isdigit():
            nid = int(kv["id"])

        return WifiStatus(
            connected=connected,
            ssid=kv.get("ssid") if connected else None,
            ip=kv.get("ip_address") if connected else None,
            bssid=kv.get("bssid") if connected else None,
            freq=freq if connected else None,
            signal=signal if connected else None,
            key_mgmt=kv.get("key_mgmt") if connected else None,
            wpa_state=wpa_state or None,
            network_id=nid if connected else None,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/scan", response_model=List[Network])
def scan():
    try:
        wpa("scan")
        out = wpa("scan_results")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
    if len(lines) <= 1:
        return []

    results: List[Network] = []
    for ln in lines[1:]:
        parts = ln.split("\t")
        if len(parts) < 5:
            continue
        bssid, freq, signal, flags = parts[0], parts[1], parts[2], parts[3]
        ssid = "\t".join(parts[4:])

        if not re.match(r"^[0-9a-fA-F:]{17}$", bssid):
            continue

        try:
            results.append(
                Network(
                    bssid=bssid,
                    frequency=int(freq),
                    signal=int(signal),
                    flags=flags,
                    ssid=ssid,
                )
            )
        except ValueError:
            continue

    ap_ssid = get_ap_ssid()
    ap_mac = get_iface_mac(AP_IFACE)

    if ap_ssid or ap_mac:
        filtered: List[Network] = []
        for n in results:
            if ap_ssid and n.ssid == ap_ssid:
                continue
            if ap_mac and n.bssid.lower() == ap_mac:
                continue
            filtered.append(n)
        results = filtered

    results.sort(key=lambda n: n.signal, reverse=True)
    return results


@app.post("/api/connect", response_model=ConnectResult)
def connect(req: ConnectRequest):
    try:
        nid = wpa("add_network")
        if not nid.isdigit():
            return ConnectResult(ok=False, ssid_requested=req.ssid, detail=f"add_network returned: {nid}")

        out = wpa(f"set_network {nid} ssid '\"{req.ssid}\"'")
        if "FAIL" in out:
            return ConnectResult(ok=False, ssid_requested=req.ssid, network_id=int(nid), detail=f"set_network ssid failed: {out}")

        if req.psk and req.psk.strip():
            out = wpa(f"set_network {nid} psk '\"{req.psk}\"'")
            if "FAIL" in out:
                return ConnectResult(ok=False, ssid_requested=req.ssid, network_id=int(nid), detail=f"set_network psk failed: {out}")
        else:
            out = wpa(f"set_network {nid} key_mgmt NONE")
            if "FAIL" in out:
                return ConnectResult(ok=False, ssid_requested=req.ssid, network_id=int(nid), detail=f"set_network key_mgmt failed: {out}")

        out = wpa(f"enable_network {nid}")
        if "FAIL" in out:
            return ConnectResult(ok=False, ssid_requested=req.ssid, network_id=int(nid), detail=f"enable_network failed: {out}")

        out = wpa(f"select_network {nid}")
        select_detail = f"select_network: {out}"

        try:
            wpa("save_config")
        except Exception:
            pass

        deadline = time.time() + 18.0
        last_status_raw = ""
        last_kv = {}
        while time.time() < deadline:
            time.sleep(1.0)
            last_status_raw = wpa("status")
            last_kv = parse_kv(last_status_raw)
            if last_kv.get("wpa_state") == "COMPLETED":
                for _ in range(3):
                    if last_kv.get("ip_address"):
                        break
                    time.sleep(1.0)
                    last_status_raw = wpa("status")
                    last_kv = parse_kv(last_status_raw)
                break

        logs = recent_supplicant_logs(lines=140) or None
        state = last_kv.get("wpa_state")
        connected_ok = (state == "COMPLETED") and (last_kv.get("ssid") == req.ssid)

        if connected_ok:
            return ConnectResult(
                ok=True,
                ssid_requested=req.ssid,
                network_id=int(nid),
                wpa_state=state,
                ssid=last_kv.get("ssid"),
                ip=last_kv.get("ip_address"),
                bssid=last_kv.get("bssid"),
                key_mgmt=last_kv.get("key_mgmt"),
                detail=select_detail,
                wpa_cli_status_raw=last_status_raw or None,
                wpa_logs_tail=logs,
            )

        return ConnectResult(
            ok=False,
            ssid_requested=req.ssid,
            network_id=int(nid),
            wpa_state=state,
            ssid=last_kv.get("ssid"),
            ip=last_kv.get("ip_address"),
            bssid=last_kv.get("bssid"),
            key_mgmt=last_kv.get("key_mgmt"),
            detail=f"Did not reach COMPLETED for requested SSID within timeout. {select_detail}",
            wpa_cli_status_raw=last_status_raw or None,
            wpa_logs_tail=logs,
        )

    except Exception as e:
        logs = recent_supplicant_logs(lines=140) or None
        return ConnectResult(ok=False, ssid_requested=req.ssid, detail=str(e), wpa_logs_tail=logs)


@app.post("/api/disconnect", response_model=SimpleResult)
def api_disconnect():
    try:
        out = wpa("disconnect")
        return SimpleResult(ok=True, detail=out or "disconnected")
    except Exception as e:
        return SimpleResult(ok=False, detail=str(e))


@app.get("/api/saved_networks", response_model=List[SavedNetwork])
def api_saved_networks():
    try:
        out = wpa("list_networks")
        lines = [ln.rstrip("\n") for ln in out.splitlines() if ln.strip()]
        if not lines:
            return []

        rows = lines[1:] if len(lines) > 1 else []
        saved: List[SavedNetwork] = []

        for row in rows:
            parts = row.split("\t")
            if len(parts) < 4:
                continue
            nid, ssid, bssid, flags = parts[0], parts[1], parts[2], parts[3]
            if not nid.isdigit():
                continue
            saved.append(
                SavedNetwork(
                    network_id=int(nid),
                    ssid=ssid,
                    bssid=bssid,
                    flags=flags,
                    is_current=("CURRENT" in flags),
                    is_disabled=("DISABLED" in flags),
                )
            )

        saved.sort(key=lambda n: (not n.is_current, n.is_disabled, n.ssid.lower()))
        return saved
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/forget", response_model=ForgetResult)
def api_forget(req: ForgetRequest):
    try:
        nid: Optional[int] = req.network_id
        target_ssid: Optional[str] = req.ssid

        if nid is None:
            if not target_ssid:
                return ForgetResult(ok=False, detail="Provide network_id or ssid.")
            out = wpa("list_networks")
            for row in out.splitlines()[1:]:
                parts = row.split("\t")
                if len(parts) >= 2 and parts[1] == target_ssid and parts[0].isdigit():
                    nid = int(parts[0])
                    break

        if nid is None:
            return ForgetResult(ok=False, removed_ssid=target_ssid, detail="No matching saved network found.")

        try:
            wpa(f"disable_network {nid}")
        except Exception:
            pass

        out_rm = wpa(f"remove_network {nid}")

        try:
            wpa("save_config")
        except Exception:
            pass

        return ForgetResult(ok=True, removed_network_id=nid, removed_ssid=target_ssid, detail=out_rm or "removed")
    except Exception as e:
        return ForgetResult(ok=False, detail=str(e))


@app.get("/api/captive_check", response_model=CaptiveCheck)
def captive_check():
    test_urls = [
        "http://connectivitycheck.gstatic.com/generate_204",
        "http://clients3.google.com/generate_204",
        "http://neverssl.com/",
    ]

    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            return None

    opener = urllib.request.build_opener(NoRedirect())
    last_detail = None

    for url in test_urls:
        try:
            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "pi-captive-check",
                    "Cache-Control": "no-cache",
                    "Pragma": "no-cache",
                },
                method="GET",
            )
            with opener.open(req, timeout=5) as resp:
                code = resp.getcode()

                if url.endswith("generate_204") and code == 204:
                    return CaptiveCheck(captive=False, online=True, portal_url=None, detail="204 OK")

                body = resp.read(512) or b""
                low = body.lower()

                if b"<html" in low or b"<form" in low:
                    return CaptiveCheck(captive=True, online=False, portal_url=None, detail=f"{url} returned HTML (status {code})")

                if "neverssl.com" in url and code == 200:
                    return CaptiveCheck(captive=False, online=True, portal_url=None, detail="HTTP reachable")

                last_detail = f"{url} status {code}"

        except urllib.error.HTTPError as e:
            loc = e.headers.get("Location")
            if e.code in (301, 302, 303, 307, 308):
                return CaptiveCheck(captive=True, online=False, portal_url=loc, detail=f"Redirect {e.code}")

            try:
                body = e.read(512) or b""
                low = body.lower()
                if b"<html" in low or b"<form" in low:
                    return CaptiveCheck(captive=True, online=False, portal_url=loc, detail=f"HTTPError {e.code} with HTML")
            except Exception:
                pass

            last_detail = f"HTTPError {e.code}"

        except Exception as e:
            last_detail = str(e)

    return CaptiveCheck(captive=True, online=False, portal_url=None, detail=last_detail or "No response")


# -------------------- VW Connect API (via CarConnectivity CLI) --------------------

@app.get("/api/vw/vehicles")
def vw_vehicles():
    out = cc_cli(["list"])
    vins = set()
    for ln in out.splitlines():
        ln = ln.strip()
        if ln.startswith("/garage/") and ln.count("/") >= 2:
            parts = ln.split("/")
            if len(parts) >= 3 and parts[2]:
                vins.add(parts[2])
    return {"vins": sorted(vins)}


@app.get("/api/vw/get")
def vw_get(path: str = Query(..., min_length=3)):
    val = cc_cli(["get", path])
    return {"path": path, "value": val}


@app.get("/api/vw/summary")
def vw_summary(vin: str = Query(..., min_length=8)):
    base = f"/garage/{vin}"

    def norm(v: str | None) -> str | None:
        if v is None:
            return None
        s = str(v).strip()
        if not s:
            return None
        low = s.lower()
        if low in ("none", "null", "nan"):
            return None
        return s

    fields = {
        "model": f"{base}/model",
        "name": f"{base}/name",
        "odometer": f"{base}/odometer",
        "pos_lat": f"{base}/position/latitude",
        "pos_lon": f"{base}/position/longitude",
        "position_type": f"{base}/position/position_type",
        "addr_country": f"{base}/position/position_location/country",
        "addr_county": f"{base}/position/position_location/county",
        "addr_display_name": f"{base}/position/position_location/display_name",
        "addr_lat": f"{base}/position/position_location/latitude",
        "addr_lon": f"{base}/position/position_location/longitude",
        "addr_name": f"{base}/position/position_location/name",
        "addr_postcode": f"{base}/position/position_location/postcode",
        "addr_raw": f"{base}/position/position_location/raw",
        "addr_road": f"{base}/position/position_location/road",
        "addr_source": f"{base}/position/position_location/source",
        "addr_state": f"{base}/position/position_location/state",
        "addr_uid": f"{base}/position/position_location/uid",
    }

    out = {"vin": vin, "data": {}}

    for key, path in fields.items():
        try:
            out["data"][key] = cc_cli(["get", path])
        except Exception:
            out["data"][key] = None

    for k in list(out["data"].keys()):
        out["data"][k] = norm(out["data"][k])

    d = out["data"]

    address = {}
    mapping = {
        "display_name": "addr_display_name",
        "name": "addr_name",
        "road": "addr_road",
        "postcode": "addr_postcode",
        "county": "addr_county",
        "state": "addr_state",
        "country": "addr_country",
        "source": "addr_source",
        "uid": "addr_uid",
        "raw": "addr_raw",
        "latitude": "addr_lat",
        "longitude": "addr_lon",
    }
    for out_key, data_key in mapping.items():
        if d.get(data_key):
            address[out_key] = d[data_key]
    if address:
        out["address"] = address

    current_address = d.get("addr_display_name") or d.get("addr_name")
    if current_address:
        out["current_address"] = current_address

    lat = d.get("addr_lat") or d.get("pos_lat")
    lon = d.get("addr_lon") or d.get("pos_lon")
    if lat and lon:
        out["coords"] = {"latitude": lat, "longitude": lon}

    return out


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
        drive_level = vw_get(path=drive_level_path).get("value")
    except Exception:
        pass
    try:
        total_range = vw_get(path=total_range_path).get("value")
    except Exception:
        pass

    payload = {
        "ok": True,
        "vin": vin,
        "fetched_at": int(time.time()),
        "summary": summary,
        "extras": {
            "drive_level": drive_level,
            "total_range": total_range,
        },
    }
    save_cache(payload)
    return payload


def main():
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8090, reload=False)


if __name__ == "__main__":
    main()
