"""Delayed-resolution forecast questions, asked only of earthquake instances read from the USGS feed
(--source usgs, or --source live with earthquakes in the mix, or a frozen USGS feed replayed).

A forecast asked of a feed just fetched cannot be memorised, scraped or fine-tuned for, because its
answer does not exist yet: "Will an M4.5+ earthquake occur within R km of L in the 24 hours after this
digest?" It is stored as *pending* with an explicit resolution rule; ``cli.py resolve`` grades it later
against the USGS catalogue (FDSN event web service) and the report shows a Brier score per model. A
frozen feed replayed later asks about the day after that feed was compiled, which may have ended: its
answer is then already in the catalogue (generate warns of it), and the question tests recall, not
forecasting.
"""
from __future__ import annotations

import math
import random
from typing import Any, Callable, Sequence

import requests

from . import sources
from .geo import destination
from .models import Event
from .sources import USER_AGENT, failure_text, read_feed_json
from .util import utc_iso

FDSN_URL = "https://earthquake.usgs.gov/fdsnws/event/1/query"
MIN_MAG = 4.5
HORIZON_S = 24 * 3600
GRACE_S = 2 * 3600  # wait a little for late catalogue entries before resolving


def make_forecast(events: Sequence[Event], rng: random.Random, compiled_at: int, window_hours: int) -> dict:
    anchor = max(events, key=lambda e: (e.mag, e.id)) if rng.random() < 0.5 else rng.choice(events)
    lat, lon = destination(anchor.lat, anchor.lon, rng.uniform(0.0, 30.0), rng.uniform(0.0, 360.0))
    return {
        "qid": "F1",
        "answer_type": "probability",
        "rule": {
            "min_mag": MIN_MAG,
            "lat": round(lat, 3),
            "lon": round(lon, 3),
            "radius_km": rng.choice((150, 250, 400, 600)),
            "label": f"point {anchor.network.upper()}-F{rng.randrange(10, 100)}",
            "start": compiled_at,
            "end": compiled_at + HORIZON_S,
            "digest_window_hours": window_hours,
            "source": FDSN_URL,
        },
        "status": "pending",
        "outcome": None,
        "resolved_at": None,
        "evidence": [],
    }


def forecast_text(forecast: dict) -> str:
    r = forecast["rule"]
    coords = f"{abs(r['lat']):.3f}°{'N' if r['lat'] >= 0 else 'S'}, {abs(r['lon']):.3f}°{'E' if r['lon'] >= 0 else 'W'}"
    return (f"Forecast (graded later against the USGS catalogue): what is the probability that an earthquake "
            f"of magnitude {r['min_mag']:g} or greater will occur within {r['radius_km']} km of {r['label']} "
            f"({coords}) between {utc_iso(r['start'])} and {utc_iso(r['end'])}, i.e. in the 24 hours after this "
            f"digest was compiled?")


def fetch_fdsn(rule: dict, timeout: float = 30.0) -> list[dict]:
    """Query the USGS FDSN event service for earthquakes matching a forecast's resolution rule, as every
    live request is made (sources.http_get): the features of its GeoJSON answer, and none for an answer
    without content (status 204, the service's "no data"). An answer that is not a GeoJSON object with a
    list of features, each an object with its properties, raises ValueError (as does JSON nested too
    deeply to read), so that resolve leaves the forecast pending."""
    params = {
        "format": "geojson",
        "eventtype": "earthquake",
        "starttime": utc_iso(rule["start"]).rstrip("Z"),
        "endtime": utc_iso(rule["end"]).rstrip("Z"),
        "minmagnitude": rule["min_mag"],
        "latitude": rule["lat"],
        "longitude": rule["lon"],
        "maxradiuskm": rule["radius_km"],
    }
    response = sources.http_get(FDSN_URL, params=params, timeout=timeout, headers={"User-Agent": USER_AGENT})
    response.raise_for_status()
    if response.status_code == 204:
        return []
    payload = read_feed_json(response.content, "the FDSN answer")
    features = payload.get("features", []) if isinstance(payload, dict) else None
    if not isinstance(features, list) or not all(
            isinstance(x, dict) and isinstance(x.get("properties", {}), dict) for x in features):
        raise ValueError("not an FDSN GeoJSON answer: it is not an object with a list of features, each with its "
                         "properties")
    return features


def _magnitude(feature: Any) -> float:
    """A catalogue feature's magnitude, or -inf when it gives none that is a finite number."""
    props = feature.get("properties") if isinstance(feature, dict) else None
    mag = props.get("mag") if isinstance(props, dict) else None
    if isinstance(mag, bool) or not isinstance(mag, (int, float)) or not math.isfinite(mag):
        return -math.inf
    return float(mag)


def resolve_forecasts(forecasts: list[dict], now: int,
                      fetch: Callable[[dict], list[dict]] = fetch_fdsn) -> dict:
    """Resolve every forecast whose window (plus a grace period) has passed. Mutates `forecasts`. A forecast
    whose catalogue query fails, or answers with what cannot be read, stays pending, with the error."""
    summary: dict = {"resolved": 0, "pending": 0, "newly_resolved": 0, "errors": []}
    for f in forecasts:
        if f["status"] == "resolved":
            summary["resolved"] += 1
            continue
        if now < f["rule"]["end"] + GRACE_S:
            summary["pending"] += 1
            continue
        try:
            features = fetch(f["rule"])
        except (requests.RequestException, ValueError) as exc:
            summary["errors"].append(f"{f.get('instance_id')}: {failure_text(exc)}")
            summary["pending"] += 1
            continue
        hits = [x for x in features if _magnitude(x) >= f["rule"]["min_mag"]]
        f["outcome"] = 1 if hits else 0
        f["evidence"] = [x.get("id") for x in hits]
        f["status"] = "resolved"
        f["resolved_at"] = utc_iso(now)
        summary["resolved"] += 1
        summary["newly_resolved"] += 1
    return summary
