"""Airport data loading and lookup."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple


def load_airports_map(json_path: Path) -> Dict[str, str]:
    """Load IATA code to city name mapping from airports.json.
    
    File format: array of {IATA, City/Name, Latitude, Longitude}
    """
    if not json_path.exists():
        return {}
    try:
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
        except UnicodeDecodeError:
            data = json.loads(json_path.read_text(encoding="utf-8-sig"))
    except Exception:
        return {}

    if not isinstance(data, list):
        return {}

    mapping: Dict[str, str] = {}
    for item in data:
        if not isinstance(item, dict):
            continue
        code = str(item.get("IATA", "")).upper().strip()
        if len(code) != 3:
            continue
        city = (item.get("City") or item.get("Name") or "").strip()
        if city:
            mapping[code] = city
    return mapping


def load_airport_coords(json_path: Path) -> Dict[str, Tuple[float, float]]:
    """Load IATA code to (latitude, longitude) mapping.
    
    Ignores entries without valid coordinates.
    """
    if not json_path.exists():
        return {}
    try:
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
        except UnicodeDecodeError:
            data = json.loads(json_path.read_text(encoding="utf-8-sig"))
    except Exception:
        return {}

    coords: Dict[str, Tuple[float, float]] = {}
    if not isinstance(data, list):
        return coords
    for item in data:
        if not isinstance(item, dict):
            continue
        code = str(item.get("IATA", "")).upper().strip()
        if len(code) != 3:
            continue
        lat = item.get("Latitude") or item.get("latitude")
        lon = item.get("Longitude") or item.get("longitude")
        try:
            if lat is None or lon is None:
                continue
            lat_f = float(str(lat))
            lon_f = float(str(lon))
        except Exception:
            continue
        coords[code] = (lat_f, lon_f)
    return coords
