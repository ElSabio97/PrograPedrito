"""Airline flight scheduling calendar Streamlit app.

Main UI application that orchestrates CSV parsing, calendar building, PDF generation,
and optional Firestore persistence.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Dict, List, Tuple, Union
from collections import Counter

import streamlit as st
import pandas as pd
import calendar as pycal

from csv_parser import read_ib_flights_from_csv_bytes, extract_route_from_subject
from calendar_builder import build_calendar_lines, hhmm_to_tuple
from airports import load_airports_map, load_airport_coords
from pdf_writer import draw_month_calendar_pdf_bytes, MONTHS_ES
from map_renderer import render_routes_map

# Paths
APP_DIR = Path(__file__).parent
AIRPORTS_JSON = APP_DIR / "airports.json"


# -------- Helper functions for Streamlit UI --------

def _normalize(s: str) -> str:
    """Normalize string for case-insensitive comparison."""
    import unicodedata
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower().strip()
    s = " ".join(s.split())
    return s


def _shift_month(y: int, m: int, delta: int) -> Tuple[int, int]:
    """Shift month by delta, wrapping years."""
    m2, y2 = m + delta, y
    while m2 < 1:
        m2 += 12
        y2 -= 1
    while m2 > 12:
        m2 -= 12
        y2 += 1
    return y2, m2


def _to_min(hhmm: str) -> int:
    """Convert HH:MM to minutes."""
    h, m = hhmm_to_tuple(hhmm)
    return h * 60 + m


def _render_calendar_and_map(by_day: Dict[int, List[str]], route_counts_month: Dict[Tuple[str, str], int], year: int, month: int) -> None:
    """Render a simple month calendar (buttons) below the map and filter routes by selected day.

    - Days with flights are shown with a green square, otherwise red.
    - Clicking a day filters the map to only routes for that day.
    """
    # Initialize selected_day in session state
    key = f"selected_day_{year}_{month}"
    if key not in st.session_state:
        st.session_state[key] = None

    st.markdown("---")
    st.subheader("Calendario del mes (clic en un día para filtrar rutas)")

    weeks = pycal.Calendar(firstweekday=0).monthdayscalendar(year, month)
    cols = None
    for week in weeks:
        cols = st.columns(7)
        for i, day in enumerate(week):
            if day == 0:
                cols[i].write("")
                continue
            has = day in (by_day or {})
            emoji = "🟩" if has else "🟥"
            label = f"{day} {emoji}"
            if cols[i].button(label, key=f"day_{year}_{month}_{day}"):
                # toggle selection
                if st.session_state.get(key) == day:
                    st.session_state[key] = None
                else:
                    st.session_state[key] = day

    sel = st.session_state.get(key)
    if sel:
        st.info(f"Mostrando rutas para el día {sel}")
        # compute route counts for selected day
        rc: Dict[Tuple[str, str], int] = {}
        import re
        regex_full = re.compile(r"\b([A-Z]{3})\s*-\s*([A-Z]{3})\b")
        lines = by_day.get(int(sel), []) if by_day else []
        for line in lines:
            m = regex_full.search(line)
            if m:
                r = (m.group(1), m.group(2))
                rc[r] = rc.get(r, 0) + 1
        render_routes_map(rc, AIRPORTS_JSON)
    else:
        st.info("Mostrando rutas de todo el mes")
        render_routes_map(route_counts_month or {}, AIRPORTS_JSON)



def _parse_stats_from_by_day(by_day_any: Dict[Union[int, str], List[str]]):
    """Extract statistics from by_day calendar structure."""
    import re
    by_day_local: Dict[int, List[str]] = {}
    for k, v in by_day_any.items():
        try:
            ik = int(k)
        except Exception:
            continue
        if isinstance(v, list):
            by_day_local[ik] = [str(x) for x in v]

    regex_full = re.compile(r"\b([A-Z]{3})\s*-\s*([A-Z]{3})\b")
    regex_end = re.compile(r"^\s*([A-Z]{3})\s*-\s*")
    dest_counter = Counter()
    route_counts: Dict[Tuple[str, str], int] = {}

    for _day, lines in by_day_local.items():
        for line in lines:
            m = regex_full.search(line)
            if m:
                orig, dest = m.group(1), m.group(2)
                route_counts[(orig, dest)] = route_counts.get((orig, dest), 0) + 1
                dest_counter[dest] += 1
            else:
                m2 = regex_end.search(line)
                if m2:
                    dest = m2.group(1)
                    dest_counter[dest] += 1

    legs_count = sum(dest_counter.values())
    return by_day_local, dest_counter, route_counts, legs_count


def _compute_day_events(by_day_local: Dict[int, List[str]]):
    """Compute last end/start times for each day."""
    import re
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


def _total_minutes_from_by_day(by_day_local: Dict[int, List[str]]) -> int:
    """Calculate total flight minutes from by_day structure."""
    import re
    regex_full = re.compile(r"^\s*[A-Z]{3}\s*-\s*[A-Z]{3}\s*\(\s*(\d{1,2}):(\d{2})\s*-\s*(\d{1,2}):(\d{2})\s*\)\s*$")
    regex_startonly = re.compile(r"^\s*[A-Z]{3}\s*\(\s*(\d{1,2}):(\d{2})\s*-\s*$")
    regex_endonly = re.compile(r"^\s*[A-Z]{3}\s*-\s*(\d{1,2}):(\d{2})\)\s*$")

    total = 0
    starts_by_day: Dict[int, List[int]] = {}
    ends_by_day: Dict[int, List[int]] = {}

    for day, lines in by_day_local.items():
        for line in lines:
            m_full = regex_full.match(line)
            if m_full:
                sh, sm, eh, em = map(int, (m_full.group(1), m_full.group(2), m_full.group(3), m_full.group(4)))
                total += max(0, _to_min(f"{eh}:{em}") - _to_min(f"{sh}:{sm}"))
                continue
            m_s = regex_startonly.match(line)
            if m_s:
                sh, sm = int(m_s.group(1)), int(m_s.group(2))
                starts_by_day.setdefault(day, []).append(_to_min(f"{sh}:{sm}"))
                continue
            m_e = regex_endonly.match(line)
            if m_e:
                eh, em = int(m_e.group(1)), int(m_e.group(2))
                ends_by_day.setdefault(day, []).append(_to_min(f"{eh}:{em}"))

    for day in sorted(starts_by_day.keys()):
        next_day = day + 1
        if next_day not in ends_by_day:
            continue
        s_list = sorted(starts_by_day[day])
        e_list = sorted(ends_by_day[next_day])
        for sh_min, eh_min in zip(s_list, e_list):
            overnight = (24 * 60 - sh_min) + eh_min
            if overnight > 0:
                total += overnight
    return total


def _try_load_firestore_stats(stats_user_id: str, stats_year: int, stats_month: int):
    """Attempt to load statistics from Firestore (optional)."""
    try:
        from google.cloud import firestore
        from google.oauth2 import service_account

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
            return snap.exists, db if snap.exists else None, snap.to_dict() if snap.exists else None
        else:
            st.info("Configura secretos de Firestore para cargar estadísticas guardadas.")
            return False, None, None
    except Exception:
        st.info("No se pudo cargar Firestore o no está configurado; puedes seguir usando la app.")
        return False, None, None


# -------- Main Streamlit app --------

def main() -> None:
    st.set_page_config(page_title="Calendario de vuelos", layout="wide")
    st.title("Programaciones de welo")
    st.caption("Selecciona el mes del que quieres ver la progra.")

    # === Visualization section ===
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

    base_year, base_month = today.year, today.month
    if quick == "Mes anterior":
        base_year, base_month = _shift_month(base_year, base_month, -1)
    elif quick == "Próximo mes":
        base_year, base_month = _shift_month(base_year, base_month, 1)

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

    show_stats = st.checkbox("Ver estadísticas (gráfico y mapa)", value=False)

    # === Load from Firestore (optional) ===
    firestore_loaded = False
    firestore_exists, _db, firestore_payload = _try_load_firestore_stats(stats_user_id, stats_year, stats_month)
    
    if firestore_exists and firestore_payload:
        by_day_payload = firestore_payload.get("by_day", {})
        by_day_local, dest_counter_s, route_counts_s, legs_count = _parse_stats_from_by_day(by_day_payload)

        airports_map = load_airports_map(AIRPORTS_JSON)
        dest_codes = set(dest_counter_s.keys())
        last_end_map, last_start_only = _compute_day_events(by_day_local)

        def earliest_end_for(day: int):
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
            t_end_dest = last_end_map.get(day)
            t_start = last_start_only.get(day)
            if t_start and (not t_end_dest or t_start > t_end_dest[0]):
                nxt = earliest_end_for(day + 1)
                if nxt and nxt[2] != "MAD":
                    plane_days.add(day)
            elif t_end_dest:
                if t_end_dest[1] != "MAD":
                    plane_days.add(day)

        pdf_bytes_stats = draw_month_calendar_pdf_bytes(
            int(stats_year),
            int(stats_month),
            by_day_local,
            legend_codes=dest_codes,
            airports_map=airports_map,
            plane_days=plane_days,
            app_dir=APP_DIR,
        )
        st.download_button(
            label=f"Descargar PDF - {MONTHS_ES[int(stats_month)]} {int(stats_year)}",
            data=pdf_bytes_stats,
            file_name=f"calendar_{int(stats_year)}_{int(stats_month):02d}.pdf",
            mime="application/pdf",
            use_container_width=True,
        )

        if show_stats:
            total_min = _total_minutes_from_by_day(by_day_local)
            total_h = total_min // 60
            rem_m = total_min % 60
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Vuelos (legs)", f"{legs_count}")
            c2.metric("Días con vuelo", f"{len(by_day_local)}")
            c3.metric("Destinos únicos", f"{len(set(dest_counter_s.keys()))}")
            c4.metric("Horas de vuelo", f"{total_h} h {rem_m:02d} m")

            if dest_counter_s:
                filtered = [(d, c) for d, c in dest_counter_s.most_common() if d != "MAD"][:10]
                df_top = pd.DataFrame(filtered, columns=["Destino", "Vuelos"]) if filtered else pd.DataFrame(columns=["Destino", "Vuelos"])
                st.bar_chart(df_top.set_index("Destino"))

            st.subheader("Mapa de rutas del mes")
            # Render calendar + map with per-day filtering
            _render_calendar_and_map(by_day_local, route_counts_s if isinstance(route_counts_s, dict) else {}, int(stats_year), int(stats_month))

        firestore_loaded = True

    st.markdown("---")

    # === CSV Upload section ===
    with st.expander("Actualizar datos (subir CSV)", expanded=False):
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
            app_dir=APP_DIR,
        )

        st.success(f"PDF generado para {MONTHS_ES[month]} {year}")
        st.download_button(
            label="Descargar PDF",
            data=pdf_bytes,
            file_name=f"calendar_{year}_{month:02d}.pdf",
            mime="application/pdf",
        )

        month_rows = [r for r in rows if r.start_date.year == year and r.start_date.month == month]
        dest_counter = Counter()
        route_counts: Dict[Tuple[str, str], int] = {}
        for r in month_rows:
            parts = extract_route_from_subject(r.subject)
            if not parts:
                continue
            dest_counter[parts.dest] += 1
            route_key = (parts.origin, parts.dest)
            route_counts[route_key] = route_counts.get(route_key, 0) + 1

        total_min_csv = 0
        for r in month_rows:
            s = _to_min(r.start_time)
            e = _to_min(r.end_time)
            if r.end_date == r.start_date:
                total_min_csv += max(0, e - s)
            else:
                total_min_csv += (24 * 60 - s) + e
        th, tm = divmod(total_min_csv, 60)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Vuelos (legs)", f"{len(month_rows)}")
        col2.metric("Días con vuelo", f"{len(by_day)}")
        col3.metric("Destinos únicos", f"{len(set(dest_counter.keys()))}")
        col4.metric("Horas de vuelo", f"{th} h {tm:02d} m")

        if dest_counter:
            st.subheader("Top destinos del mes")
            filtered = [(d, c) for d, c in dest_counter.most_common() if d != "MAD"][:10]
            df_top = pd.DataFrame(filtered, columns=["Destino", "Vuelos"]) if filtered else pd.DataFrame(columns=["Destino", "Vuelos"])
            st.bar_chart(df_top.set_index("Destino"))

        st.subheader("Mapa de rutas del mes")
        # Render calendar + map with per-day filtering for uploaded CSV
        _render_calendar_and_map(by_day, route_counts, int(year), int(month))

        # === Firestore save (optional) ===
        if save_to_firestore and user_id.strip() and uploaded is not None:
            try:
                from google.cloud import firestore
                from google.oauth2 import service_account

                sa_info = st.secrets.get("gcp_service_account", None)
                project_id = st.secrets.get("gcp_project", None)

                if not sa_info:
                    flat_keys = ("type", "project_id", "private_key", "client_email", "token_uri")
                    if all(k in st.secrets for k in flat_keys):
                        sa_info = {k: st.secrets[k] for k in st.secrets.keys() if isinstance(k, str)}
                        if not project_id:
                            project_id = st.secrets.get("project_id")

                if not sa_info or not project_id:
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

    st.markdown("---")
    st.caption("Tus archivos no se almacenan; el PDF se genera bajo demanda y se descarga. Si activas Firestore, solo se guarda identificación y by_day.")


if __name__ == "__main__":
    main()
