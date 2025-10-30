from __future__ import annotations

import io
import csv
import json
import unicodedata
from dataclasses import dataclass
from datetime import datetime, date
from pathlib import Path
from typing import Dict, List, Tuple, Union
from collections import Counter

import streamlit as st
import pydeck as pdk
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# Paths
APP_DIR = Path(__file__).parent
AIRPORTS_JSON = APP_DIR / "airports.json"

# UI labels
WEEKDAYS_ES = ["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"]
MONTHS_ES = {
    1: "Enero",
    2: "Febrero",
    3: "Marzo",
    4: "Abril",
    5: "Mayo",
    6: "Junio",
    7: "Julio",
    8: "Agosto",
    9: "Septiembre",
    10: "Octubre",
    11: "Noviembre",
    12: "Diciembre",
}


@dataclass
class FlightRow:
    subject: str
    start_date: date
    start_time: str  # HH:MM
    end_date: date
    end_time: str  # HH:MM


@dataclass
class FlightParts:
    origin: str
    dest: str


# -------- CSV helpers ---------

def _normalize(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower().strip()
    s = " ".join(s.split())
    return s


def _pick_field(fieldnames: List[str], canonical: str) -> str:
    canon = _normalize(canonical)
    for f in fieldnames:
        if _normalize(f) == canon:
            return f
    raise KeyError(f"No se encontró la columna requerida: {canonical}")


def _parse_date(d: str) -> date:
    return datetime.strptime(d.strip(), "%d/%m/%Y").date()


def _parse_time(t: str) -> str:
    t = t.strip()
    if len(t) == 4 and t.isdigit():
        return f"{t[:2]}:{t[2:]}"
    return t


def _extract_route_from_subject(subject: str) -> FlightParts | None:
    import re

    s = subject.upper()
    m = re.search(r"\b([A-Z]{3})\d{3,4}-([A-Z]{3})\d{3,4}\b", s)
    if not m:
        return None
    return FlightParts(origin=m.group(1), dest=m.group(2))


def read_ib_flights_from_csv_bytes(data: bytes) -> List[FlightRow]:
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
                if "IB" not in subject.upper():
                    continue
                if not _extract_route_from_subject(subject):
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


# -------- Calendar logic ---------

def _hhmm_to_tuple(hhmm: str) -> Tuple[int, int]:
    try:
        h, m = hhmm.split(":")
        return int(h), int(m)
    except Exception:
        return (0, 0)


def build_calendar_lines(rows: List[FlightRow]) -> Tuple[int, int, Dict[int, List[str]], set[str], set[int]]:
    if not rows:
        raise ValueError("No se encontraron vuelos IB en el CSV.")

    year = rows[0].start_date.year
    month = rows[0].start_date.month

    by_day: Dict[int, List[str]] = {}
    dest_codes: set[str] = set()
    plane_days: set[int] = set()
    # Último fin y última salida del día (para vuelos que terminan al día siguiente)
    # Por día del mes seleccionado: (hh, mm), dest
    last_end_by_day: Dict[int, Tuple[Tuple[int, int], str]] = {}
    last_start_by_day: Dict[int, Tuple[Tuple[int, int], str]] = {}

    for r in rows:
        parts = _extract_route_from_subject(r.subject)
        if not parts:
            continue
        same_day = r.start_date == r.end_date
        if same_day:
            if r.start_date.year == year and r.start_date.month == month:
                line = f"{parts.origin} - {parts.dest} ({r.start_time} - {r.end_time})"
                by_day.setdefault(r.start_date.day, []).append(line)
                dest_codes.add(parts.dest)
                day = r.start_date.day
                tkey = _hhmm_to_tuple(r.end_time)
                prev = last_end_by_day.get(day)
                if (prev is None) or (tkey > prev[0]):
                    last_end_by_day[day] = (tkey, parts.dest)
        else:
            if r.start_date.year == year and r.start_date.month == month:
                by_day.setdefault(r.start_date.day, []).append(
                    f"{parts.origin} ({r.start_time}-"
                )
                # Registrar salida del día; el destino es conocido aunque aterrice al día siguiente
                sday = r.start_date.day
                tkey_s = _hhmm_to_tuple(r.start_time)
                prev_s = last_start_by_day.get(sday)
                if (prev_s is None) or (tkey_s > prev_s[0]):
                    last_start_by_day[sday] = (tkey_s, parts.dest)
            if r.end_date.year == year and r.end_date.month == month:
                by_day.setdefault(r.end_date.day, []).append(
                    f"{parts.dest} - {r.end_time})"
                )
                dest_codes.add(parts.dest)
                # Considerar el fin real del día por la hora de fin de este vuelo
                day_end = r.end_date.day
                tkey = _hhmm_to_tuple(r.end_time)
                prev = last_end_by_day.get(day_end)
                if (prev is None) or (tkey > prev[0]):
                    last_end_by_day[day_end] = (tkey, parts.dest)

    # Determinar último evento del día: comparar última llegada (end) vs última salida (start-only)
    all_days = set(last_end_by_day.keys()) | set(last_start_by_day.keys())
    for day in sorted(all_days):
        end_rec = last_end_by_day.get(day)
        start_rec = last_start_by_day.get(day)
        if start_rec and (not end_rec or start_rec[0] > end_rec[0]):
            # Último evento es una salida que termina al día siguiente
            dest = start_rec[1]
            if dest != "MAD":
                plane_days.add(day)
        elif end_rec:
            dest = end_rec[1]
            if dest != "MAD":
                plane_days.add(day)

    return year, month, by_day, dest_codes, plane_days


# -------- Airports map ---------

def load_airports_map(json_path: Path) -> Dict[str, str]:
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
    """Devuelve un mapa IATA -> (lat, lon) a partir de airports.json estándar.

    Ignora entradas sin coordenadas válidas.
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


# -------- PDF drawing (to bytes) ---------

_SYMBOL_FONT_NAME: str | None = None


def _get_symbol_font() -> str | None:
    global _SYMBOL_FONT_NAME
    if _SYMBOL_FONT_NAME:
        return _SYMBOL_FONT_NAME
    candidates = [
        Path(r"/usr/share/fonts/truetype/seguisym.ttf"),
        Path(r"C:/Windows/Fonts/seguisym.ttf"),
        Path(r"C:/Windows/Fonts/seguiemj.ttf"),
        Path(r"C:/Windows/Fonts/arialuni.ttf"),
    ]
    for p in candidates:
        try:
            if p.exists():
                fname = f"Sym_{p.stem}"
                try:
                    pdfmetrics.getFont(fname)
                    _SYMBOL_FONT_NAME = fname
                    return _SYMBOL_FONT_NAME
                except Exception:
                    pass
                pdfmetrics.registerFont(TTFont(fname, str(p)))
                _SYMBOL_FONT_NAME = fname
                return _SYMBOL_FONT_NAME
        except Exception:
            continue
    return None


def draw_month_calendar_pdf_bytes(
    year: int,
    month: int,
    by_day: Dict[int, List[str]],
    legend_codes: set[str] | None = None,
    airports_map: Dict[str, str] | None = None,
    plane_days: set[int] | None = None,
) -> bytes:
    buf = io.BytesIO()

    page_w, page_h = landscape(A4)
    c = canvas.Canvas(buf, pagesize=(page_w, page_h))

    margin = 36
    legend_h = 70
    legend_gap = 20
    grid_w = page_w - 2 * margin
    grid_h = page_h - 2 * margin - legend_h - legend_gap

    # Title
    title = f"{MONTHS_ES[month]} {year}"
    c.setFont("Helvetica-Bold", 20)
    c.drawString(margin, page_h - margin + 6, title)

    # Header weekdays
    import calendar as pycal
    cal = pycal.Calendar(firstweekday=0)
    weeks = cal.monthdayscalendar(year, month)

    c.setFont("Helvetica-Bold", 10)
    col_w = grid_w / 7.0
    row_h = grid_h / len(weeks)
    for i, wd in enumerate(WEEKDAYS_ES):
        x = margin + i * col_w + 2
        y = page_h - margin - 12
        c.drawString(x, y, wd)
    c.setStrokeColor(colors.black)
    c.line(margin, page_h - margin - 16, margin + grid_w, page_h - margin - 16)

    # Plane icon
    def draw_plane_icon(cx: float, cy: float, size: float = 8.0, orient: str = "right") -> None:
        font_name = _get_symbol_font()
        if font_name:
            c.saveState()
            c.setFillColor(colors.red)
            emoji = "✈"
            font_size = max(8, int(size * 1.6))
            c.setFont(font_name, font_size)
            text_w = c.stringWidth(emoji, font_name, font_size)
            c.drawString(cx - text_w / 2, cy - font_size * 0.6, emoji)
            c.restoreState()
            return
        # vector fallback
        c.setStrokeColor(colors.red)
        c.setLineWidth(1)
        if orient == "right":
            c.line(cx - size * 0.4, cy, cx + size * 0.6, cy)
            c.line(cx - size * 0.1, cy, cx - size * 0.4, cy + size * 0.25)
            c.line(cx - size * 0.1, cy, cx - size * 0.4, cy - size * 0.25)
            c.line(cx - size * 0.45, cy, cx - size * 0.55, cy + size * 0.18)
            c.line(cx - size * 0.45, cy, cx - size * 0.55, cy - size * 0.18)
        else:
            c.line(cx, cy + size * 0.6, cx, cy - size * 0.4)
            c.line(cx, cy - size * 0.1, cx + size * 0.25, cy - size * 0.4)
            c.line(cx, cy - size * 0.1, cx - size * 0.25, cy - size * 0.4)
            c.line(cx, cy - size * 0.45, cx + size * 0.18, cy - size * 0.55)
            c.line(cx, cy - size * 0.45, cx - size * 0.18, cy - size * 0.55)
        c.setStrokeColor(colors.black)

    # Cells
    top_y = page_h - margin - 18
    for row_idx, week in enumerate(weeks):
        for col_idx, day in enumerate(week):
            x0 = margin + col_idx * col_w
            y0 = top_y - (row_idx + 1) * row_h
            c.setStrokeColor(colors.black)
            c.rect(x0, y0, col_w, row_h, stroke=1, fill=0)

            if day == 0:
                continue

            c.setFont("Helvetica", 9)
            c.setFillColor(colors.grey)
            c.drawString(x0 + 3, y0 + row_h - 12, str(day))
            c.setFillColor(colors.black)

            lines = by_day.get(day, [])

            text_x = x0 + 3
            text_y = y0 + row_h - 26
            c.setFont("Helvetica", 8)
            line_height = 10
            min_needed = 5 * line_height + 4
            if (row_h - 26) < min_needed:
                c.setFont("Helvetica", 7)
                line_height = 9

            for s in lines:
                max_chars = 42
                if len(s) > max_chars:
                    s = s[: max_chars - 1] + "…"
                c.drawString(text_x, text_y, s)
                text_y -= line_height
                if text_y < y0 + 4:
                    break

            if plane_days and (day in plane_days):
                day_icon_y = y0 + row_h - 14
                if col_idx < 6:
                    px = x0 + col_w - 1
                    py = day_icon_y
                    draw_plane_icon(px, py, size=9, orient="right")
                else:
                    px = x0 + col_w - 1
                    py = day_icon_y
                    draw_plane_icon(px, py, size=9, orient="down")

    # Legend
    if legend_codes:
        legend_y0 = margin
        legend_x0 = margin
        legend_w = page_w - 2 * margin

        c.setFont("Helvetica-Bold", 10)
        title_h = 10
        title_gap = 14
        title_y = legend_y0 + legend_h - title_h
        c.drawString(legend_x0, title_y, "Leyenda destinos")

        items = []
        airports_map = airports_map or {}
        for code in sorted(legend_codes):
            name = airports_map.get(code, "")
            items.append(f"{code}: {name}" if name else f"{code}")

        c.setFont("Helvetica", 8)
        col_padding = 15
        col_count = 4
        col_w = (legend_w - (col_count - 1) * col_padding) / col_count
        line_h = 15
        top_reserved = title_h + title_gap
        max_lines_per_col = int((legend_h - top_reserved) // line_h)
        while max_lines_per_col * col_count < len(items) and col_count < 6:
            col_count += 1
            col_w = (legend_w - (col_count - 1) * col_padding) / col_count
            max_lines_per_col = int((legend_h - top_reserved) // line_h)

        for idx, text in enumerate(items[: max_lines_per_col * col_count]):
            col = idx // max_lines_per_col
            row = idx % max_lines_per_col
            x = legend_x0 + col * (col_w + col_padding)
            y = legend_y0 + legend_h - top_reserved - row * line_h
            c.drawString(x, y, text)

    # Footer
    footer_text = "Hora local de Badajoz"
    c.setFont("Helvetica", 8)
    c.setFillColor(colors.lightgrey)
    footer_w = c.stringWidth(footer_text, "Helvetica", 8)
    c.drawString((page_w - footer_w) / 2, 12, footer_text)
    c.setFillColor(colors.black)

    c.showPage()
    c.save()

    buf.seek(0)
    return buf.getvalue()


# -------- Streamlit UI ---------

def main() -> None:
    st.set_page_config(page_title="Calendario de vuelos", layout="wide")
    st.title("Programaciones de vuelo")
    st.caption("Selecciona el mes del que quieres ver la progra.")

    # --- Estadísticas guardadas (por defecto mes actual) ---
    st.subheader("Visualización")
    today = date.today()
    col_sel1, col_sel2 = st.columns([1.2, 2])
    with col_sel1:
        stats_user_display = st.selectbox("Usuario", options=["Pedro", "Bea"], index=0, key="stats_user")
        stats_user_id = _normalize(stats_user_display)
    with col_sel2:
        quick = st.radio(
            "Periodo rápido",
            options=["Mes anterior", "Mes actual", "Próximo mes"],
            index=1,
            horizontal=True,
            help="Cambia rápidamente el mes con un clic"
        )
    # Mes/año por defecto a partir del selector rápido
    def _shift_month(y: int, m: int, delta: int) -> Tuple[int, int]:
        m2, y2 = m + delta, y
        while m2 < 1:
            m2 += 12
            y2 -= 1
        while m2 > 12:
            m2 -= 12
            y2 += 1
        return y2, m2

    base_year, base_month = today.year, today.month
    if quick == "Mes anterior":
        base_year, base_month = _shift_month(base_year, base_month, -1)
    elif quick == "Próximo mes":
        base_year, base_month = _shift_month(base_year, base_month, 1)

    # Selector lento opcional
    show_manual = st.checkbox("Seleccionar mes manualmente (lento)", value=False)
    stats_year, stats_month = base_year, base_month
    month_names = [MONTHS_ES[m] for m in range(1, 13)]
    if show_manual:
        mcol1, mcol2 = st.columns([1, 1])
        with mcol1:
            stats_year = st.number_input("Año", min_value=2000, max_value=2100, value=base_year, step=1, key="stats_year")
        with mcol2:
            stats_month_name = st.selectbox("Mes", options=month_names, index=base_month - 1, key="stats_month")
            stats_month = month_names.index(stats_month_name) + 1

    # Posibilidad de ver estadísticas (opcional para ahorrar carga)
    show_stats = st.checkbox("Ver estadísticas (gráfico y mapa)", value=False)

    # Intentar cargar de Firestore el doc del usuario/mes escogido
    def _parse_stats_from_by_day(by_day_any: Dict[Union[int, str], List[str]]):
        import re
        # Normalizar claves a int
        by_day_local: Dict[int, List[str]] = {}
        for k, v in by_day_any.items():
            try:
                ik = int(k)
            except Exception:
                continue
            if isinstance(v, list):
                by_day_local[ik] = [str(x) for x in v]

        # Extraer destinos y rutas de las líneas
        regex_full = re.compile(r"\b([A-Z]{3})\s*-\s*([A-Z]{3})\b")
        regex_end = re.compile(r"^\s*([A-Z]{3})\s*-\s*")
        dest_counter = Counter()
        route_set: set[Tuple[str, str]] = set()

        for _day, lines in by_day_local.items():
            for line in lines:
                m = regex_full.search(line)
                if m:
                    orig, dest = m.group(1), m.group(2)
                    route_set.add((orig, dest))
                    dest_counter[dest] += 1
                else:
                    m2 = regex_end.search(line)
                    if m2:
                        dest = m2.group(1)
                        dest_counter[dest] += 1

        # Contar legs reales: cada línea completa cuenta 1, y los vuelos que cruzan de día se cuentan por la línea de cierre ("DEST - hh:mm)")
        legs_count = sum(dest_counter.values())
        return by_day_local, dest_counter, route_set, legs_count

    def _compute_day_events(by_day_local: Dict[int, List[str]]):
        import re
        # Devuelve:
        #  - last_end_map: {day: ((endH,endM), dest)} con el fin más tardío del día
        #  - last_start_only: {day: (startH,startM)} para salidas que NO terminan ese día (líneas "ORIG (hh:mm-")
        regex_full = re.compile(r"^\s*([A-Z]{3})\s*-\s*([A-Z]{3})\s*\(\s*(\d{1,2}):(\d{2})\s*-\s*(\d{1,2}):(\d{2})\s*\)\s*$")
        regex_endonly = re.compile(r"^\s*([A-Z]{3})\s*-\s*(\d{1,2}):(\d{2})\)\s*$")
        regex_startonly = re.compile(r"^\s*([A-Z]{3})\s*\(\s*(\d{1,2}):(\d{2})\s*-\s*$")
        last_end_map: Dict[int, Tuple[Tuple[int, int], str]] = {}
        last_start_only: Dict[int, Tuple[int, int]] = {}
        for day, lines in by_day_local.items():
            best_end: Tuple[Tuple[int, int], str] | None = None
            best_start_only: Tuple[int, int] | None = None
            for line in lines:
                m_full = regex_full.match(line)
                if m_full:
                    dest = m_full.group(2)
                    # sh, sm = int(m_full.group(3)), int(m_full.group(4))  # start not needed here
                    eh, em = int(m_full.group(5)), int(m_full.group(6))
                    tk = (eh, em)
                    if best_end is None or tk > best_end[0]:
                        best_end = (tk, dest)
                    continue
                m_end = regex_endonly.match(line)
                if m_end:
                    dest = m_end.group(1)
                    eh, em = int(m_end.group(2)), int(m_end.group(3))
                    tk = (eh, em)
                    if best_end is None or tk > best_end[0]:
                        best_end = (tk, dest)
                    continue
                m_start = regex_startonly.match(line)
                if m_start:
                    sh, sm = int(m_start.group(2)), int(m_start.group(3))
                    tk = (sh, sm)
                    if best_start_only is None or tk > best_start_only:
                        best_start_only = tk
            if best_end:
                last_end_map[day] = best_end
            if best_start_only:
                last_start_only[day] = best_start_only
        return last_end_map, last_start_only

    stats_loaded = False
    try:
        from google.cloud import firestore  # type: ignore
        from google.oauth2 import service_account  # type: ignore

        sa_info = st.secrets.get("gcp_service_account", None)
        project_id = st.secrets.get("gcp_project", None)
        if not sa_info:
            flat_keys = ("type", "project_id", "private_key", "client_email", "token_uri")
            if all(k in st.secrets for k in flat_keys):
                sa_info = {k: st.secrets[k] for k in st.secrets.keys() if isinstance(k, str)}
                if not project_id:
                    project_id = st.secrets.get("project_id")

        if sa_info and project_id:
            creds = service_account.Credentials.from_service_account_info(dict(sa_info))
            db = firestore.Client(credentials=creds, project=str(project_id))
            doc_id = f"{stats_user_id}-{int(stats_year)}-{int(stats_month):02d}"
            snap = db.collection("calendars").document(doc_id).get()
            if snap.exists:
                payload = snap.to_dict() or {}
                by_day_payload = payload.get("by_day", {})
                by_day_local, dest_counter_s, route_set_s, legs_count = _parse_stats_from_by_day(by_day_payload)

                # 1) Botón de descarga de PDF (lo primero tras el selector)
                airports_map = load_airports_map(AIRPORTS_JSON)
                dest_codes = set(dest_counter_s.keys())
                last_end_map, last_start_only = _compute_day_events(by_day_local)
                # Reglas de icono simplificadas:
                # 1) Último vuelo del día termina en el día N y fuera de MAD -> icono en N
                # 2) Último vuelo del día termina en el día N+1 y fuera de MAD -> icono en N
                # Implementación: si la última salida (start-only) es posterior al último fin del día,
                # se considera "vuelo que termina al día siguiente"; tomamos el primer aterrizaje del día N+1.
                def earliest_end_for(day: int):
                    # devuelve (h,m,dest) o None para el primer aterrizaje del día
                    import re
                    regex_full = re.compile(r"^\s*[A-Z]{3}\s*-\s*([A-Z]{3})\s*\(\s*\d{1,2}:\d{2}\s*-\s*(\d{1,2}):(\d{2})\s*\)\s*$")
                    regex_endonly = re.compile(r"^\s*([A-Z]{3})\s*-\s*(\d{1,2}):(\d{2})\)\s*$")
                    lines = by_day_local.get(day, [])
                    best: tuple[int, int, str] | None = None
                    for line in lines:
                        m = regex_full.match(line)
                        if m:
                            dest = m.group(1)
                            eh, em = int(m.group(2)), int(m.group(3))
                            tk = (eh, em)
                            if best is None or tk < (best[0], best[1]):
                                best = (eh, em, dest)
                            continue
                        m2 = regex_endonly.match(line)
                        if m2:
                            dest = m2.group(1)
                            eh, em = int(m2.group(2)), int(m2.group(3))
                            tk = (eh, em)
                            if best is None or tk < (best[0], best[1]):
                                best = (eh, em, dest)
                    return best

                plane_days = set()
                for day in sorted(set(list(last_end_map.keys()) + list(last_start_only.keys()))):
                    t_end_dest = last_end_map.get(day)  # ((h,m), dest)
                    t_start = last_start_only.get(day)  # (h,m)
                    if t_start and (not t_end_dest or t_start > t_end_dest[0]):
                        # caso 2: termina N+1 -> mirar primer aterrizaje de N+1
                        nxt = earliest_end_for(day + 1)
                        if nxt and nxt[2] != "MAD":
                            plane_days.add(day)
                    elif t_end_dest:
                        # caso 1: termina en N
                        if t_end_dest[1] != "MAD":
                            plane_days.add(day)

                pdf_bytes_stats = draw_month_calendar_pdf_bytes(
                    int(stats_year),
                    int(stats_month),
                    by_day_local,
                    legend_codes=dest_codes,
                    airports_map=airports_map,
                    plane_days=plane_days,
                )
                st.download_button(
                    label=f"Descargar PDF - {MONTHS_ES[int(stats_month)]} {int(stats_year)}",
                    data=pdf_bytes_stats,
                    file_name=f"calendar_{int(stats_year)}_{int(stats_month):02d}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )

                # 2) Estadísticas opcionales
                if show_stats:
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Vuelos (legs)", f"{legs_count}")
                    c2.metric("Días con vuelo", f"{len(by_day_local)}")
                    c3.metric("Destinos únicos", f"{len(set(dest_counter_s.keys()))}")

                    if dest_counter_s:
                        import pandas as pd
                        filtered = [(d, c) for d, c in dest_counter_s.most_common() if d != "MAD"][:10]
                        df_top = pd.DataFrame(filtered, columns=["Destino", "Vuelos"]) if filtered else pd.DataFrame(columns=["Destino", "Vuelos"])
                        st.bar_chart(df_top.set_index("Destino"))

                    coords_map = load_airport_coords(AIRPORTS_JSON)
                    arc_data = []
                    center_lat, center_lon, n_cent = 40.0, -3.7, 0
                    for (orig, dest) in sorted(route_set_s):
                        if orig in coords_map and dest in coords_map:
                            o_lat, o_lon = coords_map[orig]
                            d_lat, d_lon = coords_map[dest]
                            arc_data.append({
                                "from_code": orig,
                                "to_code": dest,
                                "from_lat": o_lat,
                                "from_lon": o_lon,
                                "to_lat": d_lat,
                                "to_lon": d_lon,
                            })
                            center_lat += (o_lat + d_lat)
                            center_lon += (o_lon + d_lon)
                            n_cent += 2
                    if n_cent:
                        center_lat /= (n_cent + 1)
                        center_lon /= (n_cent + 1)

                    if arc_data:
                        layer = pdk.Layer(
                            "ArcLayer",
                            arc_data,
                            get_source_position="[from_lon, from_lat]",
                            get_target_position="[to_lon, to_lat]",
                            get_source_color=[255, 0, 0, 160],
                            get_target_color=[0, 102, 204, 160],
                            get_width=2,
                            pickable=True,
                        )
                        view_state = pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=3.5)
                        st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip={"text": "{from_code} → {to_code}"}))
                    else:
                        st.info("No hay rutas con coordenadas disponibles para mostrar el mapa.")

                stats_loaded = True
            else:
                st.info("No hay datos guardados para ese usuario y mes en Firestore.")
        else:
            st.info("Configura secretos de Firestore para cargar estadísticas guardadas.")
    except Exception as _e:
        # No bloquea la app si Firestore no está instalado/configurado
        st.info("No se pudo cargar Firestore o no está configurado; puedes seguir usando la app.")

    st.markdown("---")

    # --- Actualizar datos (subida de CSV) ---
    with st.expander("Actualizar datos (subir CSV)", expanded=False):
        # Identificación mínima para guardar en Firestore
        user_display = st.selectbox("Usuario", options=["Pedro", "Bea"], index=0, key="upload_user")
        user_id = _normalize(user_display)
        save_to_firestore = st.checkbox("Guardar resultado procesado en Firestore", value=False, key="upload_save")

        uploaded = st.file_uploader("Selecciona tu progra.csv", type=["csv"], accept_multiple_files=False)

    if uploaded is not None:
        data = uploaded.getvalue()
        try:
            rows = read_ib_flights_from_csv_bytes(data)
        except Exception as e:
            st.error(f"No se pudo leer el CSV: {e}")
            return

        try:
            year, month, by_day, dest_codes, plane_days = build_calendar_lines(rows)
        except Exception as e:
            st.error(str(e))
            return

        airports_map = load_airports_map(AIRPORTS_JSON)

        pdf_bytes = draw_month_calendar_pdf_bytes(
            year,
            month,
            by_day,
            legend_codes=dest_codes,
            airports_map=airports_map,
            plane_days=plane_days,
        )

        st.success(f"PDF generado para {MONTHS_ES[month]} {year}")
        st.download_button(
            label="Descargar PDF",
            data=pdf_bytes,
            file_name=f"calendar_{year}_{month:02d}.pdf",
            mime="application/pdf",
        )

        # Guardado mínimo en Firestore (solo identificación y by_day)
        if save_to_firestore and user_id.strip() and uploaded is not None:
            try:
                # Importar Firestore y credenciales solo si se va a guardar (evita fallos si no están instalados)
                from google.cloud import firestore
                from google.oauth2 import service_account

                # Construir cliente Firestore desde secrets (acepta dos formatos):
                # 1) Formato recomendado: st.secrets['gcp_service_account'] + st.secrets['gcp_project']
                # 2) Formato "plano" (JSON pegado al nivel raíz) con los campos del service account y 'project_id'
                sa_info = st.secrets.get("gcp_service_account", None)
                project_id = st.secrets.get("gcp_project", None)

                if not sa_info:
                    # Intento con formato plano
                    flat_keys = ("type", "project_id", "private_key", "client_email", "token_uri")
                    if all(k in st.secrets for k in flat_keys):
                        sa_info = {k: st.secrets[k] for k in st.secrets.keys() if isinstance(k, str)}
                        if not project_id:
                            project_id = st.secrets.get("project_id")

                if not sa_info or not project_id:
                    # Opción de pegar el JSON del service account directamente (solo para esta sesión)
                    st.warning("Secrets incompletos: define gcp_service_account + gcp_project en secrets o pega el JSON del service account aquí abajo.")
                    with st.expander("Configurar Firestore ahora (pegar JSON del Service Account)"):
                        import json
                        json_text = st.text_area(
                            "Pega el contenido completo del archivo JSON del Service Account",
                            value="",
                            height=200,
                            help="No se guarda en disco; se usa solo durante esta ejecución."
                        )
                        if json_text.strip():
                            try:
                                parsed = json.loads(json_text)
                                if isinstance(parsed, dict) and parsed.get("type") == "service_account":
                                    sa_info = parsed
                                    if not project_id:
                                        project_id = parsed.get("project_id")
                                else:
                                    st.error("El JSON pegado no parece ser una clave de Service Account válida.")
                            except Exception as ex:
                                st.error(f"JSON inválido: {ex}")

                if not sa_info or not project_id:
                    st.info("Faltan credenciales: define gcp_service_account + gcp_project en secrets o pega un JSON válido en el cuadro anterior.")
                else:
                    creds = service_account.Credentials.from_service_account_info(dict(sa_info))
                    db = firestore.Client(credentials=creds, project=project_id)

                    # Firestore no admite claves numéricas en mapas; convertir días a strings
                    by_day_str_keys: Dict[str, List[str]] = {str(k): v for k, v in by_day.items()}
                    doc_id = f"{user_id.strip()}-{year}-{month:02d}"
                    doc_ref = db.collection("calendars").document(doc_id)
                    payload = {
                        "user_id": user_id.strip(),
                        "year": int(year),
                        "month": int(month),
                        "by_day": by_day_str_keys,
                    }
                    doc_ref.set(payload, merge=True)
                    st.success(f"Guardado en Firestore: {doc_id}")
            except Exception as e:
                st.error(f"No se pudo guardar en Firestore: {e}")

            # --- Estadísticas básicas (del CSV subido) ---
            # Filtrar filas del mes/año actual
            month_rows = [r for r in rows if r.start_date.year == year and r.start_date.month == month]
            dest_counter = Counter()
            route_set = set()
            for r in month_rows:
                parts = _extract_route_from_subject(r.subject)
                if not parts:
                    continue
                dest_counter[parts.dest] += 1
                route_set.add((parts.origin, parts.dest))

            col1, col2, col3 = st.columns(3)
            col1.metric("Vuelos (legs)", f"{len(month_rows)}")
            col2.metric("Días con vuelo", f"{len(by_day)}")
            col3.metric("Destinos únicos", f"{len(set(dest_counter.keys()))}")

            if dest_counter:
                st.subheader("Top destinos del mes")
                import pandas as pd
                # Excluir MAD del ranking para centrarlo en destinos fuera de base
                filtered = [(d, c) for d, c in dest_counter.most_common() if d != "MAD"][:10]
                df_top = pd.DataFrame(filtered, columns=["Destino", "Vuelos"]) if filtered else pd.DataFrame(columns=["Destino", "Vuelos"])
                st.bar_chart(df_top.set_index("Destino"))

            # --- Mapa de rutas ---
            st.subheader("Mapa de rutas del mes")
            coords_map = load_airport_coords(AIRPORTS_JSON)
            arc_data = []
            center_lat, center_lon, n_cent = 40.0, -3.7, 0  # centrado aprox. España por defecto
            for (orig, dest) in sorted(route_set):
                if orig in coords_map and dest in coords_map:
                    o_lat, o_lon = coords_map[orig]
                    d_lat, d_lon = coords_map[dest]
                    arc_data.append({
                        "from_code": orig,
                        "to_code": dest,
                        "from_lat": o_lat,
                        "from_lon": o_lon,
                        "to_lat": d_lat,
                        "to_lon": d_lon,
                    })
                    # media simple para viewport
                    center_lat += (o_lat + d_lat)
                    center_lon += (o_lon + d_lon)
                    n_cent += 2
            if n_cent:
                center_lat /= (n_cent + 1)
                center_lon /= (n_cent + 1)

            if arc_data:
                layer = pdk.Layer(
                    "ArcLayer",
                    arc_data,
                    get_source_position="[from_lon, from_lat]",
                    get_target_position="[to_lon, to_lat]",
                    get_source_color=[255, 0, 0, 160],
                    get_target_color=[0, 102, 204, 160],
                    get_width=2,
                    pickable=True,
                )
                view_state = pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=3.5)
                st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip={"text": "{from_code} → {to_code}"}))
            else:
                st.info("No hay rutas con coordenadas disponibles para mostrar el mapa.")

    st.markdown("---")
    st.caption("Tus archivos no se almacenan; el PDF se genera bajo demanda y se descarga. Si activas Firestore, solo se guarda identificación y by_day.")


if __name__ == "__main__":
    main()
