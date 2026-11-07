"""Render an instance as a messy, news-style text digest.

Messy on purpose, but always reversible by a careful reader:
  * four line styles mixed in one digest: bulletin, prose, wire, log
  * depths in km, miles, or both; coordinates as 38.812°N, (38.812, -122.301) or 38°48.72'N
  * six time formats, including local time with an explicitly stated UTC offset
  * reports shuffled, grouped by network, or sorted by time; notices interleaved or at the end
  * chatter lines that carry no data
The traps themselves (re-reports, retractions, corrections) are planted by traps.py; this module
only decides how things look. A metamorphic variant is the same reports drawn with another style.
"""
from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone

from .geo import KM_PER_MILE, utc_offset_hours
from .models import Notice, Report

MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
MONTHS_LONG = ("January", "February", "March", "April", "May", "June", "July", "August",
               "September", "October", "November", "December")
WEEKDAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")

UTC_TIME_FORMATS = ("iso", "iso_space", "rfc", "prose_utc")
TIME_FORMATS = UTC_TIME_FORMATS + ("local12", "local24")

# A rendering style picks one preset per dimension; each preset is a weighted mix.
DEPTH_PRESETS = {
    "mostly_km": {"km": 0.7, "mi": 0.15, "both": 0.15},
    "mostly_mi": {"km": 0.15, "mi": 0.7, "both": 0.15},
    "mixed": {"km": 0.4, "mi": 0.4, "both": 0.2},
}
TIME_PRESETS = {
    "utc_heavy": {"iso": 0.3, "iso_space": 0.2, "rfc": 0.2, "prose_utc": 0.2, "local12": 0.05, "local24": 0.05},
    "local_heavy": {"iso": 0.1, "iso_space": 0.1, "rfc": 0.1, "prose_utc": 0.1, "local12": 0.3, "local24": 0.3},
    "mixed": {fmt: 1.0 for fmt in TIME_FORMATS},
}
COORD_PRESETS = {
    "hemi": {"hemi": 0.7, "signed": 0.2, "ddm": 0.1},
    "signed": {"hemi": 0.2, "signed": 0.7, "ddm": 0.1},
    "mixed": {"hemi": 0.4, "signed": 0.3, "ddm": 0.3},
}
LINE_PRESETS = {
    "bulletin": {"bulletin": 0.55, "prose": 0.15, "wire": 0.15, "log": 0.15},
    "prose": {"bulletin": 0.15, "prose": 0.55, "wire": 0.15, "log": 0.15},
    "mixed": {"bulletin": 0.25, "prose": 0.25, "wire": 0.25, "log": 0.25},
}
LAYOUTS = ("by_network", "by_time", "shuffled")
NOTICE_PLACEMENTS = ("interleaved", "end")

HEADERS = (
    "SEISMIC ACTIVITY DIGEST (edition {edition})\n"
    "Compiled {compiled}; covers the {window} before compilation. Reports are reproduced as received.",
    "Earthquake wire roundup: the {window} ending {compiled}\n"
    "Sources: regional and global seismic networks. Edition {edition}.",
    "== QUAKE DIGEST {edition} ==\n"
    "Window: the {window} up to {compiled}. Entries are listed as filed by each network.",
)
SECTION_HEADS = ("-- {net} network --", "From the {net} network:", "== {net} feed ==")
CHATTER = (
    "Station {station} was offline from {h1} to {h2} UTC for maintenance; its readings are not included.",
    "No tsunami threat was issued for any event in this digest.",
    "Felt reports are still being collected and are not summarised here.",
    "Editor's note: entries are reproduced exactly as each network filed them, time conventions included.",
    "Reminder: networks report on different magnitude scales (md, ml, mb, mw); values are not converted.",
)


# ------------------------------------------------------------------------- field formatters


def fmt_mag(mag: float) -> str:
    """Magnitudes are printed exactly as stored: 4.3, 1.27, 2.0 (never rounded further)."""
    text = f"{mag:.2f}"
    return text[:-1] if text.endswith("0") else text


def fmt_time(ts: int, fmt: str, lon: float = 0.0) -> str:
    dt = datetime.fromtimestamp(ts, tz=timezone.utc)
    if fmt == "iso":
        return f"{dt.year:04d}-{dt.month:02d}-{dt.day:02d}T{dt.hour:02d}:{dt.minute:02d}:{dt.second:02d}Z"
    if fmt == "iso_space":
        return f"{dt.year:04d}-{dt.month:02d}-{dt.day:02d} {dt.hour:02d}:{dt.minute:02d}:{dt.second:02d} UTC"
    if fmt == "rfc":
        return (f"{WEEKDAYS[dt.weekday()]}, {dt.day:02d} {MONTHS[dt.month - 1]} {dt.year} "
                f"{dt.hour:02d}:{dt.minute:02d}:{dt.second:02d} +0000")
    if fmt == "prose_utc":  # minute precision: seconds are truncated, never rounded
        return f"{dt.hour:02d}:{dt.minute:02d} UTC on {dt.day} {MONTHS_LONG[dt.month - 1]} {dt.year}"
    offset = utc_offset_hours(lon)
    local = dt + timedelta(hours=offset)
    zone = f"UTC{'+' if offset >= 0 else '-'}{abs(offset):02d}:00"
    if fmt == "local12":
        hour12 = local.hour % 12 or 12
        ampm = "AM" if local.hour < 12 else "PM"
        return f"{hour12}:{local.minute:02d} {ampm} local time ({zone}) on {MONTHS[local.month - 1]} {local.day}, {local.year}"
    if fmt == "local24":
        return f"{local.hour:02d}:{local.minute:02d} ({zone}) {local.day} {MONTHS[local.month - 1]} {local.year}"
    raise ValueError(f"unknown time format {fmt!r}")


def _ddm(value: float, positive: str, negative: str) -> str:
    hundredths = round(abs(value) * 6000)  # hundredths of an arc-minute
    degrees, rest = divmod(hundredths, 6000)
    return f"{degrees}°{rest / 100:05.2f}'{positive if value >= 0 else negative}"


def fmt_coords(lat: float, lon: float, fmt: str) -> str:
    if fmt == "hemi":
        return f"{abs(lat):.3f}°{'N' if lat >= 0 else 'S'}, {abs(lon):.3f}°{'E' if lon >= 0 else 'W'}"
    if fmt == "signed":
        return f"({lat:.3f}, {lon:.3f})"
    if fmt == "ddm":
        return f"{_ddm(lat, 'N', 'S')} {_ddm(lon, 'E', 'W')}"
    raise ValueError(f"unknown coordinate format {fmt!r}")


def fmt_depth(depth_km: float, fmt: str) -> str:
    km = f"{depth_km:.1f} km"
    mi = f"{depth_km / KM_PER_MILE:.1f} mi"
    if fmt == "km":
        return km
    if fmt == "mi":
        return mi
    return f"{km} ({mi})"


# -------------------------------------------------------------------------- line renderers


def render_report(r: Report, line_style: str, *, time_fmt: str, coord_fmt: str, depth_fmt: str, mag_form: int) -> str:
    when = fmt_time(r.time, time_fmt, r.lon)
    depth = fmt_depth(r.depth_km, depth_fmt)
    mag = fmt_mag(r.mag)
    net = r.network.upper()
    if line_style == "bulletin":
        mag_text = f"M{mag} ({r.mag_type})" if mag_form == 0 else f"{r.mag_type.upper()} {mag}"
        parts = [f"[{net}] {r.ref}", when, r.region, fmt_coords(r.lat, r.lon, coord_fmt), f"depth {depth}", mag_text, r.status]
        if r.same_as:
            parts.append(f"same event as {r.same_as}")
        if r.place:
            parts.append(f"near: {r.place}")
        return " | ".join(parts)
    if line_style == "prose":
        text = (f"{when} — A magnitude {mag} ({r.mag_type}) earthquake struck {r.region}; "
                f"epicentre {fmt_coords(r.lat, r.lon, coord_fmt)}, depth {depth}. "
                f"The {net} network lists it as {r.ref} ({r.status}).")
        if r.same_as:
            text += f" This is the same earthquake as {r.same_as}."
        if r.place:
            text += f" Location note: {r.place}."
        return text
    if line_style == "wire":
        text = (f"{net} WIRE [{when}] {r.mag_type.upper()} {mag} quake in {r.region} @ "
                f"{fmt_coords(r.lat, r.lon, coord_fmt)}; {depth} deep. Ref {r.ref}, {r.status}.")
        if r.same_as:
            text += f" Dup of {r.same_as}."
        return text
    if line_style == "log":
        text = (f"{when} ref={r.ref} net={r.network} mag={mag} type={r.mag_type} region=\"{r.region}\" "
                f"loc={r.lat:.3f},{r.lon:.3f} depth={depth} status={r.status}")
        if r.same_as:
            text += f" same_as={r.same_as}"
        return text
    raise ValueError(f"unknown line style {line_style!r}")


def render_notice(n: Notice, form: int, time_fmt: str) -> str:
    when = fmt_time(n.time, time_fmt)
    if n.kind == "retraction":
        return (
            f"NOTICE [{when}]: report {n.target} has been RETRACTED ({n.reason}). Disregard it.",
            f"RETRACTION [{when}]: {n.target} was not an earthquake ({n.reason}); ignore that report.",
            f"UPDATE [{when}]: {n.target} withdrawn ({n.reason}).",
        )[form]
    new, old = fmt_mag(n.new_mag), fmt_mag(n.old_mag)
    return (
        f"CORRECTION [{when}]: the magnitude of {n.target} is revised to {new} ({n.mag_type}); it was first given as {old}.",
        f"UPDATE [{when}]: magnitude for {n.target} revised from {old} to {new} {n.mag_type}.",
        f"Revised [{when}]: {n.target} magnitude now {n.mag_type.upper()} {new} (was {old}).",
    )[form]


# ------------------------------------------------------------------------------- assembly


def choose_style(rng: random.Random, avoid: dict | None = None) -> dict:
    """Pick presets for one rendering. Metamorphic variants avoid the original's presets."""

    def pick(key: str, options) -> str:
        names = sorted(options)
        if avoid is not None:
            names = [n for n in names if n != avoid[key]] or names
        return rng.choice(names)

    return {
        "depth": pick("depth", DEPTH_PRESETS),
        "time": pick("time", TIME_PRESETS),
        "coords": pick("coords", COORD_PRESETS),
        "lines": pick("lines", LINE_PRESETS),
        "layout": pick("layout", LAYOUTS),
        "notices": rng.choice(NOTICE_PLACEMENTS),
        "header": rng.randrange(len(HEADERS)),
    }


def _draw(rng: random.Random, table: dict[str, float]) -> str:
    names = list(table)
    return rng.choices(names, weights=[table[n] for n in names], k=1)[0]


def render_digest(reports: list[Report], notices: list[Notice], chatter: list[str], *, compiled_at: int,
                  window_hours: int, style: dict, rng: random.Random) -> str:
    lines: list[tuple[Report, str]] = []
    for r in reports:
        text = render_report(
            r,
            _draw(rng, LINE_PRESETS[style["lines"]]),
            time_fmt=_draw(rng, TIME_PRESETS[style["time"]]),
            coord_fmt=_draw(rng, COORD_PRESETS[style["coords"]]),
            depth_fmt=_draw(rng, DEPTH_PRESETS[style["depth"]]),
            mag_form=rng.randrange(2),
        )
        lines.append((r, text))

    body: list[str] = []
    if style["layout"] == "by_time":
        lines.sort(key=lambda pair: (pair[0].time, pair[0].ref, pair[0].network))
        body = [text for _, text in lines]
    elif style["layout"] == "by_network":
        networks = sorted({r.network for r, _ in lines})
        rng.shuffle(networks)
        for net in networks:
            body.append(rng.choice(SECTION_HEADS).format(net=net.upper()))
            group = [text for r, text in lines if r.network == net]
            rng.shuffle(group)
            body.extend(group)
    else:
        rng.shuffle(lines)
        body = [text for _, text in lines]

    notice_lines = [render_notice(n, rng.randrange(3), rng.choice(UTC_TIME_FORMATS)) for n in notices]
    if style["notices"] == "interleaved":
        for text in notice_lines:
            body.insert(rng.randrange(len(body) + 1), text)
    elif notice_lines:
        rng.shuffle(notice_lines)
        body += ["", "Corrections and retractions:"] + notice_lines

    for text in chatter:
        body.insert(rng.randrange(len(body) + 1), text)

    header = HEADERS[style["header"]].format(
        edition=f"{rng.randrange(100, 1000)}-{rng.choice('ABCDEFGHJK')}",
        compiled=fmt_time(compiled_at, rng.choice(UTC_TIME_FORMATS)),
        window="24 hours" if window_hours == 24 else f"{window_hours} hour{'s' if window_hours != 1 else ''}",
    )
    return "\n".join([header, ""] + body)


def make_chatter(rng: random.Random, compiled_at: int, networks: list[str]) -> list[str]:
    """One to three lines of harmless noise (no IDs, magnitudes or coordinates)."""
    picks = rng.sample(range(len(CHATTER)), rng.randint(1, 3))
    out = []
    for i in sorted(picks):
        start = compiled_at - rng.randrange(2, 20) * 3600
        h1 = datetime.fromtimestamp(start, tz=timezone.utc).strftime("%H:%M")
        h2 = datetime.fromtimestamp(start + rng.randrange(20, 120) * 60, tz=timezone.utc).strftime("%H:%M")
        station = f"{rng.choice(networks).upper()}.{''.join(rng.choice('ABCDEFGHKLMNPRSTW') for _ in range(3))}"
        out.append(CHATTER[i].format(station=station, h1=h1, h2=h2))
    return out
