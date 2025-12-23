"""CSV parsing utilities for flight scheduling."""

from __future__ import annotations

import csv
import io
import re
import unicodedata
from datetime import datetime, date
from typing import List

from datatypes import FlightRow, FlightParts


def _normalize(s: str) -> str:
    """Normalize string for case-insensitive column matching."""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower().strip()
    s = " ".join(s.split())
    return s


def _pick_field(fieldnames: List[str], canonical: str) -> str:
    """Find CSV column name matching canonical name (ignoring case/accents)."""
    canon = _normalize(canonical)
    for f in fieldnames:
        if _normalize(f) == canon:
            return f
    raise KeyError(f"No se encontró la columna requerida: {canonical}")


def _parse_date(d: str) -> date:
    """Parse date string in DD/MM/YYYY format."""
    return datetime.strptime(d.strip(), "%d/%m/%Y").date()


def _parse_time(t: str) -> str:
    """Parse time in HHMM or HH:MM format, return HH:MM."""
    t = t.strip()
    if len(t) == 4 and t.isdigit():
        return f"{t[:2]}:{t[2:]}"
    return t


def extract_route_from_subject(subject: str) -> FlightParts | None:
    """Extract origin/dest codes from flight subject (e.g., 'MAD123-BCN456')."""
    s = subject.upper()
    m = re.search(r"\b([A-Z]{3})\d{3,4}-([A-Z]{3})\d{3,4}\b", s)
    if not m:
        return None
    return FlightParts(origin=m.group(1), dest=m.group(2))


def read_ib_flights_from_csv_bytes(data: bytes) -> List[FlightRow]:
    """Parse airline scheduling CSV (Outlook export format) from bytes.
    
    Supports multiple encodings: utf-8-sig, cp1252, latin-1.
    Extracts flights with subject matching pattern IATA###-IATA###.
    """
    last_err: Exception | None = None
    for enc in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            text = data.decode(enc)
            reader = csv.DictReader(io.StringIO(text))
            f_asunto = _pick_field(reader.fieldnames or [], "Asunto")
            f_fecha_comienzo = _pick_field(reader.fieldnames or [], "Fecha de comienzo")
            f_comienzo = _pick_field(reader.fieldnames or [], "Comienzo")
            f_fecha_fin = _pick_field(reader.fieldnames or [], "Fecha de finalización")
            f_fin = _pick_field(reader.fieldnames or [], "Finalización")

            rows: List[FlightRow] = []
            for rec in reader:
                subject = (rec.get(f_asunto) or "").strip()
                if not extract_route_from_subject(subject):
                    continue
                try:
                    sd = _parse_date(rec[f_fecha_comienzo])
                    ed = _parse_date(rec[f_fecha_fin])
                    stime = _parse_time(rec[f_comienzo])
                    etime = _parse_time(rec[f_fin])
                except Exception:
                    continue
                rows.append(FlightRow(subject, sd, stime, ed, etime))
            return rows
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"No se pudo leer el CSV: {last_err}")
