"""Map visualization using pydeck."""

from __future__ import annotations

import random
from typing import Dict, Set, Tuple, Union

import streamlit as st
import pydeck as pdk

from airports import load_airport_coords


def render_routes_map(
    route_input: Union[Set[Tuple[str, str]], Dict[Tuple[str, str], int]],
    airports_json_path,
) -> None:
    """Render an interactive routes map using pydeck ArcLayer with random route colors.
    
    Each route gets a unique random color assigned each time the map is rendered.
    Route frequency counts are displayed on top of each arc.
    
    Args:
        route_input: either a set of (origin, dest) tuples or a dict mapping routes to counts
        airports_json_path: path to airports.json for coordinates
    """
    coords_map = load_airport_coords(airports_json_path)
    
    # Convert route_input to a dictionary of counts
    if isinstance(route_input, set):
        route_counts = {route: 1 for route in route_input}
    else:
        route_counts = route_input
    
    arc_data = []
    center_lat, center_lon, n_cent = 40.0, -3.7, 0  # Spain default
    
    # Create a mapping of routes to random colors
    route_colors: dict[Tuple[str, str], Tuple[int, int, int, int]] = {}
    for route in sorted(route_counts.keys()):
        # Generate random RGB color with 160 alpha (semi-transparent)
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)
        route_colors[route] = (r, g, b, 160)
    
    for (orig, dest) in sorted(route_counts.keys()):
        if orig in coords_map and dest in coords_map:
            o_lat, o_lon = coords_map[orig]
            d_lat, d_lon = coords_map[dest]
            color = route_colors[(orig, dest)]
            count = route_counts[(orig, dest)]
            arc_data.append({
                "from_code": orig,
                "to_code": dest,
                "from_lat": o_lat,
                "from_lon": o_lon,
                "to_lat": d_lat,
                "to_lon": d_lon,
                "color": list(color),
                "count": count,
            })
            center_lat += (o_lat + d_lat)
            center_lon += (o_lon + d_lon)
            n_cent += 2
    
    if n_cent:
        center_lat /= (n_cent + 1)
        center_lon /= (n_cent + 1)

    if arc_data:
        # Arc layer for the routes
        arc_layer = pdk.Layer(
            "ArcLayer",
            arc_data,
            get_source_position="[from_lon, from_lat]",
            get_target_position="[to_lon, to_lat]",
            get_source_color="color",
            get_target_color="color",
            get_width=2,
            pickable=True,
        )
        
        # Text layer to display route counts
        text_layer = pdk.Layer(
            "TextLayer",
            arc_data,
            get_position="[(from_lon + to_lon) / 2, (from_lat + to_lat) / 2]",
            get_text="count",
            get_size=16,
            get_color=[0, 0, 0, 255],
            get_angle=0,
            get_text_anchor="'middle'",
            get_alignment_baseline="'center'",
            pickable=True,
        )
        
        view_state = pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=3.5)
        st.pydeck_chart(pdk.Deck(
            layers=[arc_layer, text_layer],
            initial_view_state=view_state,
            tooltip={"text": "{from_code} → {to_code}: {count} vuelos"},
        ))
    else:
        st.info("No hay rutas con coordenadas disponibles para mostrar el mapa.")
