"""Calendar building and flight grouping logic."""

from __future__ import annotations

from typing import Dict, List, Tuple

from datatypes import FlightRow, FlightParts


def hhmm_to_tuple(hhmm: str) -> Tuple[int, int]:
    """Convert HH:MM time string to (hour, minute) tuple."""
    try:
        h, m = hhmm.split(":")
        return int(h), int(m)
    except Exception:
        return (0, 0)


def build_calendar_lines(rows: List[FlightRow]) -> Tuple[int, int, Dict[int, List[str]], set[str], set[int]]:
    """Build month calendar data from flight rows.
    
    Returns:
        (year, month, by_day, dest_codes, plane_days)
        - by_day: {day: [flight_lines]}
        - dest_codes: unique destination airports
        - plane_days: days where plane remains away (ends next day)
    """
    if not rows:
        raise ValueError("No se encontraron vuelos en el CSV.")

    year = rows[0].start_date.year
    month = rows[0].start_date.month

    by_day: Dict[int, List[str]] = {}
    dest_codes: set[str] = set()
    plane_days: set[int] = set()
    last_end_by_day: Dict[int, Tuple[Tuple[int, int], str]] = {}
    last_start_by_day: Dict[int, Tuple[Tuple[int, int], str]] = {}

    for r in rows:
        parts = FlightParts(
            origin=r.subject.split("-")[0][-3:].upper() if "-" in r.subject else "",
            dest=r.subject.split("-")[1][:3].upper() if "-" in r.subject else ""
        )
        # Better extraction via regex
        import re
        m = re.search(r"\b([A-Z]{3})\d{3,4}-([A-Z]{3})\d{3,4}\b", r.subject.upper())
        if not m:
            continue
        parts = FlightParts(origin=m.group(1), dest=m.group(2))

        same_day = r.start_date == r.end_date
        if same_day:
            if r.start_date.year == year and r.start_date.month == month:
                line = f"{parts.origin} - {parts.dest} ({r.start_time} - {r.end_time})"
                by_day.setdefault(r.start_date.day, []).append(line)
                dest_codes.add(parts.dest)
                day = r.start_date.day
                tkey = hhmm_to_tuple(r.end_time)
                prev = last_end_by_day.get(day)
                if (prev is None) or (tkey > prev[0]):
                    last_end_by_day[day] = (tkey, parts.dest)
        else:
            if r.start_date.year == year and r.start_date.month == month:
                by_day.setdefault(r.start_date.day, []).append(
                    f"{parts.origin} ({r.start_time}-"
                )
                sday = r.start_date.day
                tkey_s = hhmm_to_tuple(r.start_time)
                prev_s = last_start_by_day.get(sday)
                if (prev_s is None) or (tkey_s > prev_s[0]):
                    last_start_by_day[sday] = (tkey_s, parts.dest)
            if r.end_date.year == year and r.end_date.month == month:
                by_day.setdefault(r.end_date.day, []).append(
                    f"{parts.dest} - {r.end_time})"
                )
                dest_codes.add(parts.dest)
                day_end = r.end_date.day
                tkey = hhmm_to_tuple(r.end_time)
                prev = last_end_by_day.get(day_end)
                if (prev is None) or (tkey > prev[0]):
                    last_end_by_day[day_end] = (tkey, parts.dest)

    all_days = set(last_end_by_day.keys()) | set(last_start_by_day.keys())
    for day in sorted(all_days):
        end_rec = last_end_by_day.get(day)
        start_rec = last_start_by_day.get(day)
        if start_rec and (not end_rec or start_rec[0] > end_rec[0]):
            dest = start_rec[1]
            if dest != "MAD":
                plane_days.add(day)
        elif end_rec:
            dest = end_rec[1]
            if dest != "MAD":
                plane_days.add(day)

    return year, month, by_day, dest_codes, plane_days
