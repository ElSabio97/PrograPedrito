"""PDF calendar generation using ReportLab."""

from __future__ import annotations

import calendar as pycal
import io
from pathlib import Path
from typing import Dict, List

import streamlit as st
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# UI labels
WEEKDAYS_ES = ["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"]
MONTHS_ES = {
    1: "Enero", 2: "Febrero", 3: "Marzo", 4: "Abril",
    5: "Mayo", 6: "Junio", 7: "Julio", 8: "Agosto",
    9: "Septiembre", 10: "Octubre", 11: "Noviembre", 12: "Diciembre",
}

_SYMBOL_FONT_NAME: str | None = None


def get_symbol_font(app_dir: Path) -> str | None:
    """Load symbol font for plane icons (cached globally)."""
    global _SYMBOL_FONT_NAME
    if _SYMBOL_FONT_NAME:
        return _SYMBOL_FONT_NAME
    
    candidates = [
        app_dir / "NotoSansSymbols2-Regular.ttf",
        app_dir / "Noto Sans Symbols 2 Regular.ttf",
        app_dir / "Noto Sans Symbols 2.ttf",
        app_dir / "seguisym.ttf",
        app_dir / "fonts" / "NotoSansSymbols2-Regular.ttf",
        app_dir / "fonts" / "Noto Sans Symbols 2 Regular.ttf",
        app_dir / "fonts" / "Noto Sans Symbols 2.ttf",
        app_dir / "fonts" / "seguisym.ttf",
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
    app_dir: Path | None = None,
) -> bytes:
    """Generate PDF calendar for a month with flights and legend.
    
    Args:
        year, month: Calendar month
        by_day: {day: [flight_lines]}
        legend_codes: unique destination codes
        airports_map: {IATA: city_name}
        plane_days: days where plane remains away
        app_dir: app directory for font lookup
    """
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

    # Weekday headers
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

    # Draw plane icon
    def draw_plane_icon(cx: float, cy: float, size: float = 8.0) -> None:
        if not app_dir:
            return
        font_name = get_symbol_font(app_dir)
        if not font_name:
            return
        c.saveState()
        c.setFillColor(colors.red)
        emoji = "✈️"
        font_size = max(8, int(size * 1.6))
        c.setFont(font_name, font_size)
        text_w = c.stringWidth(emoji, font_name, font_size)
        c.drawString(cx - text_w / 2 + 0.5, cy - font_size * 0.6, emoji)
        c.restoreState()

    # Draw calendar cells
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
                px = x0 + col_w - 1
                py = day_icon_y
                draw_plane_icon(px, py, size=9)

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
