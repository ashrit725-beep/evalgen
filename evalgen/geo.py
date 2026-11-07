"""Geometry and units."""
from __future__ import annotations

import math

EARTH_RADIUS_KM = 6371.0
KM_PER_MILE = 1.609344
UNIT_WORDS = {"km": "km", "mi": "miles"}


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance on a sphere of radius 6371 km (the rule stated in every prompt)."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(min(1.0, math.sqrt(a)))


def to_km(value: float, unit: str) -> float:
    if unit == "km":
        return value
    if unit == "mi":
        return value * KM_PER_MILE
    raise ValueError(f"unknown unit {unit!r}")


def from_km(value_km: float, unit: str) -> float:
    if unit == "km":
        return value_km
    if unit == "mi":
        return value_km / KM_PER_MILE
    raise ValueError(f"unknown unit {unit!r}")


def destination(lat: float, lon: float, distance_km: float, bearing_deg: float) -> tuple[float, float]:
    """Point reached by travelling distance_km from (lat, lon) along the given bearing."""
    d = distance_km / EARTH_RADIUS_KM
    b = math.radians(bearing_deg)
    p1, l1 = math.radians(lat), math.radians(lon)
    p2 = math.asin(math.sin(p1) * math.cos(d) + math.cos(p1) * math.sin(d) * math.cos(b))
    l2 = l1 + math.atan2(math.sin(b) * math.sin(d) * math.cos(p1), math.cos(d) - math.sin(p1) * math.sin(p2))
    return math.degrees(p2), (math.degrees(l2) + 540.0) % 360.0 - 180.0


COMPASS16 = ("N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW")


def initial_bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compass bearing (degrees from north) from point 1 towards point 2."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    x = math.sin(dl) * math.cos(p2)
    y = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return (math.degrees(math.atan2(x, y)) + 360.0) % 360.0


def compass_point(bearing_deg: float) -> str:
    return COMPASS16[int((bearing_deg + 11.25) // 22.5) % 16]


def utc_offset_hours(lon: float) -> int:
    """Nominal 'local time' offset used in digests: the whole hour nearest to solar time.

    It is not a real time zone; every line that uses it states the offset explicitly.
    """
    return max(-12, min(14, int(round(lon / 15.0))))
