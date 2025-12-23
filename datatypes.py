"""Data classes and types for flight scheduling app."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date


@dataclass
class FlightRow:
    """Represents a single flight row from CSV."""
    subject: str
    start_date: date
    start_time: str  # HH:MM
    end_date: date
    end_time: str  # HH:MM


@dataclass
class FlightParts:
    """Represents origin and destination from a flight subject."""
    origin: str
    dest: str
