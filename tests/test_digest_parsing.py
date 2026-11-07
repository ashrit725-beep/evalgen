"""The messy digest is messy but lossless: the text-only parser recovers every report and notice."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from evalgen.digest import TIME_FORMATS, fmt_coords, fmt_depth, fmt_time
from evalgen.solver import parse_coords, parse_depth_km, parse_digest, parse_time


@pytest.mark.parametrize("fmt", TIME_FORMATS)
@pytest.mark.parametrize("lon", [-174.6, -122.8, 0.0, 141.6])
def test_every_time_format_round_trips(fmt, lon):
    for ts in (int(datetime(2026, 1, 1, 0, 0, 5, tzinfo=timezone.utc).timestamp()),
               int(datetime(2026, 2, 28, 23, 59, 59, tzinfo=timezone.utc).timestamp()),
               int(datetime(2026, 9, 14, 12, 30, 41, tzinfo=timezone.utc).timestamp())):
        parsed = parse_time(fmt_time(ts, fmt, lon))
        exact = fmt in ("iso", "iso_space", "rfc")
        assert parsed == (ts if exact else ts - ts % 60), (fmt, fmt_time(ts, fmt, lon))


@pytest.mark.parametrize("fmt", ["hemi", "signed", "ddm"])
@pytest.mark.parametrize("lat, lon", [(38.8123, -122.8011), (-33.2, -71.7), (0.0004, 179.9996), (60.5, -150.0)])
def test_coordinates_round_trip(fmt, lat, lon):
    got = parse_coords(fmt_coords(lat, lon, fmt))
    assert abs(got[0] - lat) < 6e-4 and abs(got[1] - lon) < 6e-4


@pytest.mark.parametrize("fmt", ["km", "mi", "both"])
def test_depth_round_trips_within_display_precision(fmt):
    for depth in (0.4, 6.2, 35.0, 287.3, -0.7):
        assert abs(parse_depth_km(fmt_depth(depth, fmt)) - depth) < 0.09


def _identity(r):
    return (r.ref, r.network, r.status, r.same_as)


def test_parser_recovers_every_report_and_notice(instances):
    for inst in instances:
        truth = {(r.ref, r.network, r.status): r for r in inst.reports}
        for rendering in inst.renderings:
            data = parse_digest(rendering.digest)
            assert sorted(map(_identity, data.reports)) == sorted(map(_identity, inst.reports))
            assert sorted((n.kind, n.target) for n in data.notices) == sorted((n.kind, n.target) for n in inst.notices)
            for got in data.reports:
                want = truth[(got.ref, got.network, got.status)]
                assert got.mag == want.mag and got.mag_type == want.mag_type and got.region == want.region
                assert 0 <= want.time - got.time < 60  # minute-precision formats truncate seconds
                assert abs(got.lat - want.lat) < 6e-4 and abs(got.lon - want.lon) < 6e-4
                assert abs(got.depth_km - want.depth_km) < 0.09
            for notice in data.notices:
                if notice.kind == "correction":
                    want = next(n for n in inst.notices if n.kind == "correction" and n.target == notice.target)
                    assert notice.new_mag == want.new_mag


def test_digests_are_messy(instances):
    """Each digest mixes units, time formats and line styles (requirement: realistic noise)."""
    for inst in instances:
        text = "\n".join(r.digest for r in inst.renderings)
        assert " km" in text and " mi" in text
        assert "local time" in text or "(UTC" in text
        assert sum(marker in text for marker in ("] ", " — A magnitude", " WIRE [", " ref=")) >= 3
