# Live evaluation report: `usgs-live-seed5`

Every instance in this run was generated fresh from the seed below, every answer key was computed by code from the frozen snapshot, and every score was computed automatically. No item was labelled by hand.

## Run

- **Master seed:** `5` (commitment `7d766e00cd25e027a3e7f7da…`); every derived seed is listed in `manifest.json`.
- **Data source:** usgs (https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_day.geojson); 202 events in the snapshot pool; window: 24 h ending 2026-11-03T22:18:53Z.
- **Snapshot SHA-256:** `813b93566c1b728a7bb5cbef87284d8f63d894c2c8df9f1aa3177948f91ac672`; raw feed SHA-256 `cef1e67615c9737ac1e413617a429938c0d36a1eb38d6d464dcfd2e41a376335`
- **Note:** collapsed 3 duplicate feed solution(s) of the same quake (us7000tv2m, aka2026vtxzvj, aka2026vudyyh)
- **Instances:** 20; **questions:** 82 (41 from public templates, 41 from new-this-run templates); **renderings per instance:** 3 (original + 2 metamorphic variants); **prompts:** 60.
- **Traps planted:** 93 (duplicate 53, retraction 20, conflicting magnitude 20); 87 are load-bearing: mishandling them changes the answers of 39 of the 82 questions (100 trap–question pairs). The rest stay in as plain noise.

## Model accuracy (original renderings)

| model | kind | accuracy | 95% CI | mean instance score | accuracy, all renderings | JSON strict / lenient / failed (API errors) |
|---|---|---|---|---|---|---|
| oracle | reference | 100.0% (82/82) | 96–100% | 100.0% | 100.0% | 60 / 0 / 0 (0) |
| naive | reference | 54.9% (45/82) | 44–65% | 55.0% | 54.9% | 60 / 0 / 0 (0) |
| mock | simulated | 70.7% (58/82) | 60–79% | 71.1% | 69.5% | 46 / 14 / 0 (0) |

`oracle` reads only the digest text and applies the reading rules, so it should score 100%. That shows every item is answerable from the text. `naive` treats every line as a separate quake and ignores notices. `mock` is a *simulated* model with built-in benchmark contamination, used to exercise the metrics without API keys; it is not a language model.

## Metamorphic consistency

Each instance is re-rendered with the same snapshot but different units, line order, formats and question wording. Correct answers cannot change, so any change in a model's answer is an inconsistency. This needs no ground truth.

| model | consistent answer pairs | 95% CI | questions consistent across all variants |
|---|---|---|---|
| oracle | 100.0% (164/164) | 98–100% | 100.0% (82/82) |
| naive | 100.0% (164/164) | 98–100% | 100.0% (82/82) |
| mock | 55.5% (91/164) | 48–63% | 37.8% (31/82) |

## Seen vs new templates

Public templates are published (a model may have trained on them). Each new template is composed for this run as a twin of one public template: the same final operation and number of filters, but a different filter combination, and each twin pair is used equally often. The *adjusted* gap compares the pools like for like, within strata of equal code-computed difficulty (how many traps change the answer, and whether the filters matter), so that the pools' different trap exposure is not mistaken for overfitting. A large positive adjusted gap flags overfitting to the published skeletons (flag: > 10 points with a 95% CI that excludes 0). It is a flag, not proof.

| model | public (seen) | new this run | raw gap (95% CI) | adjusted gap (95% CI) | flag |
|---|---|---|---|---|---|
| oracle | 100.0% (41/41) | 100.0% (41/41) | +0.0 pts (+0 to +0) | +0.0 pts (-12 to +12) | — |
| naive | 61.0% (25/41) | 48.8% (20/41) | +12.2 pts (-9 to +34) | +1.3 pts (-11 to +14) | — |
| mock | 87.8% (36/41) | 53.7% (22/41) | +34.1 pts (+16 to +52) | +33.9 pts (+14 to +54) | ⚠ overfitting? |

## Trap catch rate

Counted over (load-bearing trap, affected question, rendering) triples. *Caught* = answered correctly; *fell* = gave exactly an answer a reader produces by missing that trap (alone or together with other traps).

| model | caught (overall) | duplicate | retraction | conflicting magnitude | fell for the trap |
|---|---|---|---|---|---|
| oracle | 100.0% (300/300) | 100.0% | 100.0% | 100.0% | 0.0% (0/300) |
| naive | 2.0% (6/300) | 0.0% | 0.0% | 9.5% | 98.0% (294/300) |
| mock | 71.7% (215/300) | 71.9% | 72.7% | 69.8% | 22.3% (67/300) |

## Accuracy by answer type (original renderings)

| model | count | event_id | magnitude | depth | distance | region | utc_time |
|---|---|---|---|---|---|---|---|
| oracle | 100.0% (26/26) | 100.0% (20/20) | 100.0% (7/7) | 100.0% (12/12) | 100.0% (9/9) | 100.0% (5/5) | 100.0% (3/3) |
| naive | 15.4% (4/26) | 65.0% (13/20) | 71.4% (5/7) | 75.0% (9/12) | 66.7% (6/9) | 100.0% (5/5) | 100.0% (3/3) |
| mock | 76.9% (20/26) | 60.0% (12/20) | 85.7% (6/7) | 83.3% (10/12) | 66.7% (6/9) | 60.0% (3/5) | 33.3% (1/3) |

## Generator audit

- **Automated self-check:** for 246/246 question-renderings, the digest-only oracle agreed with the code-computed key. Applying the reading rules to each digest recovered the snapshot exactly for 20/20 instances.
- **Human spot-check sample:** 2 of 82 generated items (2.4%; target 2.0%) → `audit/audit_sample.jsonl` (checklist: `audit/audit_sample.md`).
- **Verdicts:** 0/2 reviewed so far. Fill in `verdict` (ok / error) in the JSONL file and re-run `cli.py report` to get an estimated generator error rate.

## Delayed-resolution forecasts (USGS mode)

20 forecast question(s); 20 still pending. Run `python cli.py resolve runs/usgs-live-seed5` after the window closes (+2 h grace) to grade them.

| model | forecasts answered (v0) | resolved | Brier score (lower is better) |
|---|---|---|---|
| oracle | 20/20 | 0 | pending |
| naive | 20/20 | 0 | pending |
| mock | 20/20 | 0 | pending |

## Templates

| id | pool | twin of | skeleton |
|---|---|---|---|
| P01 | public | — | dedupe → mag_at_least → count |
| P02 | public | — | dedupe → within_distance → count |
| P03 | public | — | dedupe → in_region → argmax_mag |
| P04 | public | — | dedupe → mag_at_least → max_depth |
| P05 | public | — | dedupe → mag_at_least → compare_regions |
| P06 | public | — | dedupe → depth_shallower_than → time_between → count |
| P07 | public | — | dedupe → in_region → nearest_to |
| P08 | public | — | dedupe → in_region → time_of_largest |
| P09 | public | — | dedupe → within_distance → rank_mag |
| P10 | public | — | dedupe → depth_at_least → max_mag |
| P11 | public | — | dedupe → time_between → min_depth |
| P12 | public | — | dedupe → mag_below → nearest_distance |
| N01 | new | P01 | dedupe → depth_shallower_than → count |
| N02 | new | P02 | dedupe → mag_below → count |
| N03 | new | P03 | dedupe → depth_shallower_than → argmax_mag |
| N04 | new | P04 | dedupe → time_between → max_depth |
| N05 | new | P05 | dedupe → depth_at_least → compare_regions |
| N06 | new | P06 | dedupe → depth_at_least → mag_at_least → count |
| N07 | new | P07 | dedupe → depth_at_least → nearest_to |
| N08 | new | P08 | dedupe → mag_below → time_of_largest |
| N09 | new | P09 | dedupe → mag_below → rank_mag |
| N10 | new | P10 | dedupe → in_region → max_mag |
| N11 | new | P11 | dedupe → depth_at_least → min_depth |
| N12 | new | P12 | dedupe → depth_shallower_than → nearest_distance |

## Per-instance scores (original rendering)

<details><summary>Show table</summary>

| instance | questions | load-bearing traps | oracle | naive | mock |
|---|---|---|---|---|---|
| inst-000 | 4 | 4 | 100.0% | 50.0% | 75.0% |
| inst-001 | 5 | 5 | 100.0% | 60.0% | 60.0% |
| inst-002 | 5 | 5 | 100.0% | 60.0% | 60.0% |
| inst-003 | 4 | 4 | 100.0% | 50.0% | 50.0% |
| inst-004 | 3 | 4 | 100.0% | 33.3% | 66.7% |
| inst-005 | 3 | 5 | 100.0% | 66.7% | 66.7% |
| inst-006 | 4 | 5 | 100.0% | 25.0% | 75.0% |
| inst-007 | 3 | 5 | 100.0% | 66.7% | 66.7% |
| inst-008 | 4 | 4 | 100.0% | 75.0% | 100.0% |
| inst-009 | 3 | 4 | 100.0% | 66.7% | 66.7% |
| inst-010 | 4 | 3 | 100.0% | 50.0% | 75.0% |
| inst-011 | 4 | 5 | 100.0% | 50.0% | 75.0% |
| inst-012 | 5 | 5 | 100.0% | 40.0% | 80.0% |
| inst-013 | 5 | 4 | 100.0% | 80.0% | 60.0% |
| inst-014 | 5 | 4 | 100.0% | 60.0% | 80.0% |
| inst-015 | 4 | 5 | 100.0% | 75.0% | 75.0% |
| inst-016 | 5 | 5 | 100.0% | 40.0% | 60.0% |
| inst-017 | 3 | 3 | 100.0% | 66.7% | 100.0% |
| inst-018 | 4 | 3 | 100.0% | 25.0% | 50.0% |
| inst-019 | 5 | 5 | 100.0% | 60.0% | 80.0% |

</details>

## Example instances

### Example 1: `inst-001` (28 earthquakes, 33 report lines, 1 notice, 5 questions)

**Digest (original rendering `v0`, exactly as sent to the models):**

```text
== QUAKE DIGEST 361-G ==
Window: the 24 hours up to 22:18 UTC on 3 November 2026. Entries are listed as filed by each network.

[AK] aka2026vtnclo | Tue, 03 Nov 2026 04:12:02 +0000 | Alaska | 59.320°N, 152.887°W | depth 56.8 mi | M2.0 (ml) | preliminary | near: 55 km W of Nanwalek, Alaska
2026-11-03T05:40:18Z — A magnitude 4.8 (mwr) earthquake struck Japan; epicentre 35.659°N, 140.648°E, depth 32.5 mi. The US network lists it as us7000tv3q (reviewed). Location note: 6 km S of Asahi, Japan.
[CI] ci41572064 | 2026-11-03T18:27:31Z | Mexico | 32.431°N, 115.180°W | depth 15.6 km (9.7 mi) | M4.89 (mw) | reviewed | near: 9 km N of Delta, B.C., MX
2026-11-03 13:53:32 UTC ref=nc75462037 net=nc mag=2.07 type=md region="California" loc=38.846,-122.832 depth=0.4 mi status=reviewed
UU WIRE [Tue, 03 Nov 2026 10:19:10 +0000] MD 0.79 quake in Utah @ (38.516, -112.915); 1.7 mi deep. Ref uu80162566, reviewed.
[AK] aka2026vudsgu | 2026-11-03T12:37:46Z | Alaska | (59.546, -152.837) | depth 96.7 km | ML 2.3 | reviewed | near: 56 km WNW of Nanwalek, Alaska
2026-11-03T16:56:42Z — A magnitude 2.4 (ml) earthquake struck Alaska; epicentre 59.849°N, 152.757°W, depth 62.2 mi. The AK network lists it as aka2026vumpow (reviewed). Location note: 52 km W of Anchor Point, Alaska.
AK WIRE [Tue, 03 Nov 2026 13:27:33 +0000] ML 1.9 quake in Alaska @ 61.695°N, 151.630°W; 49.7 mi deep. Ref aka2026vufqcp, reviewed.
2026-11-03T21:45:05Z ref=uu80162686 net=uu mag=1.56 type=md region="Utah" loc=38.513,-112.919 depth=1.5 mi status=reviewed
[AK] aka2026vtkchh | 02:41 UTC on 3 November 2026 | Alaska | 60.710°N, 151.509°W | depth 49.0 mi | ML 2.8 | reviewed | near: 12 km W of Nikiski, Alaska
US WIRE [2026-11-03T16:38:55Z] ML 2.5 quake in Alaska @ 51.874°N, 169.447°W; 6.2 mi deep. Ref us7000tvs7, reviewed.
08:01 UTC on 3 November 2026 — A magnitude 1.95 (ml) earthquake struck Utah; epicentre 38.284°N, 112.285°W, depth 5.1 mi. The UU network lists it as uu80162541 (reviewed). Location note: 7 km NW of Junction, Utah.
01:48 (UTC+09:00) 4 Nov 2026 — A magnitude 5.4 (mww) earthquake struck Japan; epicentre 24.114°N, 141.662°E, depth 141.6 km (88.0 mi). The AT network lists it as us7000tvie (preliminary). Location note: Volcano Islands, Japan region.
[UU] uu80162481 | 2026-11-02T23:22:41Z | Utah | 38.512°N, 112.916°W | depth 3.3 km | M0.47 (md) | reviewed | near: 15 km NNE of Milford, Utah
AK WIRE [2026-11-03 04:12:02 UTC] ML 2.4 quake in Alaska @ (59.320, -152.887); 56.8 mi deep. Ref aka2026vtnclo, reviewed.
[HV] hv75068191 | 2026-11-03 16:49:25 UTC | Hawaii | 19.225°N, 155.403°W | depth 30.8 km (19.1 mi) | M1.7 (md) | reviewed | near: 8 km ENE of Pāhala, Hawaii
UU WIRE [2026-11-03T18:52:33Z] ML 1.27 quake in Utah @ 38.519°N, 112.914°W; 4.9 km (3.0 mi) deep. Ref uu80162661, reviewed.
[HV] hv75068221 | 2026-11-03T20:50:14Z | Hawaii | 19.428°N, 155.507°W | depth 5.3 mi | M2.08 (md) | reviewed | near: 25 km N of Pāhala, Hawaii
[PT] aka2026vtkchh | 2026-11-03 02:41:40 UTC | Alaska | 60.749°N, 151.437°W | depth 48.5 mi | M3.1 (ml) | preliminary | near: 9 km WNW of Nikiski, Alaska
Felt reports are still being collected and are not summarised here.
2026-11-03 10:15:10 UTC ref=aka2026vtzfhv net=ak mag=1.7 type=ml region="Alaska" loc=55.945,-159.606 depth=3.1 mi status=reviewed
[UU] uu80162616 | Tue, 03 Nov 2026 13:47:46 +0000 | Utah | (38.517, -112.917) | depth 1.9 mi | M2.02 (ml) | reviewed | near: 15 km NNE of Milford, Utah
[NC] nc75462147 | 2026-11-03T20:15:29Z | California | 38.800°N, 122.767°W | depth 4.2 km (2.6 mi) | MD 0.75 | reviewed | near: 3 km NNW of The Geysers, CA
Reminder: networks report on different magnitude scales (md, ml, mb, mw); values are not converted.
2026-11-03T02:05:09Z ref=ci41571720 net=ci mag=1.81 type=ml region="Mexico" loc=32.429,-115.165 depth=9.8 mi status=reviewed
[UU] uu80162486 | Mon, 02 Nov 2026 23:26:35 +0000 | Utah | (38.514, -112.914) | depth 2.1 mi | M2.58 (ml) | reviewed | near: 15 km NNE of Milford, Utah
Tue, 03 Nov 2026 03:15:55 +0000 ref=ci41571760 net=ci mag=0.56 type=ml region="California" loc=33.478,-116.882 depth=7.4 mi status=reviewed
[US] us7000tvx5 | 2026-11-03T16:06:39Z | Japan | (35.758, 140.988) | depth 52.5 km | M4.7 (mwr) | reviewed | near: 31 km E of Asahi, Japan
[TX] tx2026vrbasw | 2026-11-03 17:34:24 UTC | Texas | 32.108°N, 101.911°W | depth 5.3 mi | ML 2.3 | reviewed | near: 11 km WSW of Stanton, Texas
Tue, 03 Nov 2026 16:48:13 +0000 ref=us7000tvie net=us mag=5.2 type=mww region="Japan" loc=24.144,141.673 depth=86.7 mi status=reviewed
[CI] ci41572256 | 22:06 UTC on 3 November 2026 | California | 33.236°N, 116.028°W | depth 2.7 mi | ML 0.99 | reviewed | near: 10 km SW of Salton City, CA
[UU] uu80162586 | 2026-11-03T11:47:59Z | Utah | 39.306°N, 111.216°W | depth -1.4 km (-0.9 mi) | M1.03 (ml) | reviewed | near: 16 km WNW of Orangeville, Utah
Editor's note: entries are reproduced exactly as each network filed them, time conventions included.
[CI] ci41571784 | 2026-11-03T04:01:07Z | California | 33.482°N, 116.482°W | depth 12.8 km (8.0 mi) | ML 1.09 | reviewed | near: 20 km ESE of Anza, CA
[CI] ci41572024 | 2026-11-03 15:27:12 UTC | California | 33.540°N, 116.705°W | depth 4.2 mi | M0.0 (ml) | reviewed | near: 3 km WSW of Anza, CA
[EW] ew35089965 | 14:39 (UTC+09:00) 3 Nov 2026 | Japan | 35.701°N, 140.635°E | depth 51.2 km (31.8 mi) | M5.0 (mwr) | preliminary | same event as us7000tv3q | near: 2 km SW of Asahi, Japan
RETRACTION [16:55 UTC on 3 November 2026]: us7000tvx5 was not an earthquake (mine blast); ignore that report.
```

**Questions, code-computed key and model answers (`v0`):**

| Q | template | question | key | oracle | naive | mock |
|---|---|---|---|---|---|---|
| Q1 | P05 (public) | Counting only earthquakes with magnitude at least 2.1, which region has more of them: Alaska or Japan? Answer "tie" if the counts are equal. | Alaska | Alaska ✓ | Alaska ✓ | Alaska ✓ |
| Q2 | P03 (public) | Among the earthquakes in California, which one had the largest magnitude? Give its event ID. | nc75462037 | nc75462037 ✓ | nc75462037 ✓ | nc75462037 ✓ |
| Q3 | N06 (new) | What is the number of separate earthquakes with a depth of 18 miles or more and of magnitude 2.1 or greater? | 6 | 6 ✓ | 10 ✗ | 7 ✗ |
| Q4 | P10 (public) | Among the earthquakes at least 6 km deep, what was the highest magnitude recorded? | 5.2 | 5.2 ✓ | 5.4 ✗ | 5.2 ✓ |
| Q5 | N08 (new) | When, in UTC (HH:MM), was the strongest of the earthquakes with magnitude below 1.9 recorded? | 02:05 | 02:05 ✓ | 02:05 ✓ | 01:25 ✗ |

**Operation chains (the machine-readable meaning of each question):**

- Q1: `dedupe → mag ≥ 2.1 → compare_regions(Alaska vs Japan)`
- Q2: `dedupe → region = California → argmax_mag`
- Q3: `dedupe → depth ≥ 18 mi → mag ≥ 2.1 → count`
- Q4: `dedupe → depth ≥ 6 km → max_mag`
- Q5: `dedupe → mag < 1.9 → time_of_largest`

**Traps in this digest:**

- T1 *duplicate* (same_id): AT re-reported us7000tvie under the same event ID: preliminary M5.4, depth 141.6 km (reviewed: M5.2, 139.5 km). If mishandled: Q3 → 7, Q4 → 5.4.
- T2 *duplicate* (cross_ref): EW re-reported us7000tv3q as ew35089965 with a 'same event as' note: preliminary M5, depth 51.2 km (reviewed: M4.8, 52.3 km). If mishandled: Q3 → 7.
- T3 *duplicate* (same_id): PT re-reported aka2026vtkchh under the same event ID: preliminary M3.1, depth 78.1 km (reviewed: M2.8, 78.9 km). If mishandled: Q3 → 7.
- T4 *retraction* (phantom): phantom report us7000tvx5 (M4.7, Japan) later retracted: mine blast. If mishandled: Q3 → 7.
- T5 *conflicting magnitude* (second_report): aka2026vtnclo also reported as M2 (preliminary, AK) beside the reviewed M2.4. If mishandled: Q3 → 5.

<details><summary>Metamorphic variant `v1` of the same instance (same key)</summary>

```text
Earthquake wire roundup: the 24 hours ending 2026-11-03 22:18:53 UTC
Sources: regional and global seismic networks. Edition 591-H.

Editor's note: entries are reproduced exactly as each network filed them, time conventions included.
23:22 UTC on 2 November 2026 — A magnitude 0.47 (md) earthquake struck Utah; epicentre (38.512, -112.916), depth 3.3 km (2.1 mi). The UU network lists it as uu80162481 (reviewed). Location note: 15 km NNE of Milford, Utah.
[UU] uu80162486 | 2026-11-02 23:26:35 UTC | Utah | (38.514, -112.914) | depth 3.4 km | ML 2.58 | reviewed | near: 15 km NNE of Milford, Utah
6:05 PM local time (UTC-08:00) on Nov 2, 2026 — A magnitude 1.81 (ml) earthquake struck Mexico; epicentre 32.429°N, 115.165°W, depth 15.8 km (9.8 mi). The CI network lists it as ci41571720 (reviewed). Location note: 9 km NNE of Delta, B.C., MX.
4:41 PM local time (UTC-10:00) on Nov 2, 2026 — A magnitude 3.1 (ml) earthquake struck Alaska; epicentre (60.749, -151.437), depth 48.5 mi. The PT network lists it as aka2026vtkchh (preliminary). Location note: 9 km WNW of Nikiski, Alaska.
Tue, 03 Nov 2026 02:41:55 +0000 — A magnitude 2.8 (ml) earthquake struck Alaska; epicentre (60.710, -151.509), depth 78.9 km. The AK network lists it as aka2026vtkchh (reviewed). Location note: 12 km W of Nikiski, Alaska.
7:15 PM local time (UTC-08:00) on Nov 2, 2026 ref=ci41571760 net=ci mag=0.56 type=ml region="California" loc=33.478,-116.882 depth=11.9 km status=reviewed
8:01 PM local time (UTC-08:00) on Nov 2, 2026 — A magnitude 1.09 (ml) earthquake struck California; epicentre (33.482, -116.482), depth 12.8 km. The CI network lists it as ci41571784 (reviewed). Location note: 20 km ESE of Anza, CA.
6:12 PM local time (UTC-10:00) on Nov 2, 2026 — A magnitude 2.4 (ml) earthquake struck Alaska; epicentre 59.320°N, 152.887°W, depth 56.8 mi. The AK network lists it as aka2026vtnclo (reviewed). Location note: 55 km W of Nanwalek, Alaska.
AK WIRE [04:12 UTC on 3 November 2026] ML 2.0 quake in Alaska @ (59.320, -152.887); 91.4 km deep. Ref aka2026vtnclo, preliminary.
2:39 PM local time (UTC+09:00) on Nov 3, 2026 — A magnitude 5.0 (mwr) earthquake struck Japan; epicentre 35.701°N, 140.635°E, depth 51.2 km (31.8 mi). The EW network lists it as ew35089965 (preliminary). This is the same earthquake as us7000tv3q. Location note: 2 km SW of Asahi, Japan.
14:40 (UTC+09:00) 3 Nov 2026 — A magnitude 4.8 (mwr) earthquake struck Japan; epicentre 35.659°N, 140.648°E, depth 52.3 km. The US network lists it as us7000tv3q (reviewed). Location note: 6 km S of Asahi, Japan.
01:01 (UTC-07:00) 3 Nov 2026 — A magnitude 1.95 (ml) earthquake struck Utah; epicentre (38.284, -112.285), depth 5.1 mi. The UU network lists it as uu80162541 (reviewed). Location note: 7 km NW of Junction, Utah.
2026-11-03 10:15:10 UTC ref=aka2026vtzfhv net=ak mag=1.7 type=ml region="Alaska" loc=55.945,-159.606 depth=5.0 km (3.1 mi) status=reviewed
2:19 AM local time (UTC-08:00) on Nov 3, 2026 — A magnitude 0.79 (md) earthquake struck Utah; epicentre (38.516, -112.915), depth 1.7 mi. The UU network lists it as uu80162566 (reviewed). Location note: 15 km NNE of Milford, Utah.
4:47 AM local time (UTC-07:00) on Nov 3, 2026 — A magnitude 1.03 (ml) earthquake struck Utah; epicentre 39.306°N, 111.216°W, depth -1.4 km. The UU network lists it as uu80162586 (reviewed). Location note: 16 km WNW of Orangeville, Utah.
2:37 AM local time (UTC-10:00) on Nov 3, 2026 — A magnitude 2.3 (ml) earthquake struck Alaska; epicentre (59.546, -152.837), depth 96.7 km (60.1 mi). The AK network lists it as aka2026vudsgu (reviewed). Location note: 56 km WNW of Nanwalek, Alaska.
03:27 (UTC-10:00) 3 Nov 2026 — A magnitude 1.9 (ml) earthquake struck Alaska; epicentre (61.695, -151.630), depth 49.7 mi. The AK network lists it as aka2026vufqcp (reviewed). Location note: 35 km SSW of Skwentna, Alaska.
05:47 (UTC-08:00) 3 Nov 2026 ref=uu80162616 net=uu mag=2.02 type=ml region="Utah" loc=38.517,-112.917 depth=3.0 km (1.9 mi) status=reviewed
Tue, 03 Nov 2026 13:53:32 +0000 — A magnitude 2.07 (md) earthquake struck California; epicentre 38.846°N, 122.832°W, depth 0.6 km (0.4 mi). The NC network lists it as nc75462037 (reviewed). Location note: 10 km WNW of Cobb, CA.
07:27 (UTC-08:00) 3 Nov 2026 — A magnitude 0.0 (ml) earthquake struck California; epicentre 33°32.39'N 116°42.30'W, depth 6.7 km. The CI network lists it as ci41572024 (reviewed). Location note: 3 km WSW of Anza, CA.
Reminder: networks report on different magnitude scales (md, ml, mb, mw); values are not converted.
US WIRE [01:06 (UTC+09:00) 4 Nov 2026] MWR 4.7 quake in Japan @ 35.758°N, 140.988°E; 52.5 km deep. Ref us7000tvx5, reviewed.
16:38 UTC on 3 November 2026 ref=us7000tvs7 net=us mag=2.5 type=ml region="Alaska" loc=51.874,-169.447 depth=10.0 km status=reviewed
01:48 (UTC+09:00) 4 Nov 2026 — A magnitude 5.4 (mww) earthquake struck Japan; epicentre (24.114, 141.662), depth 141.6 km. The AT network lists it as us7000tvie (preliminary). Location note: Volcano Islands, Japan region.
16:48 UTC on 3 November 2026 — A magnitude 5.2 (mww) earthquake struck Japan; epicentre (24.144, 141.673), depth 139.5 km. The US network lists it as us7000tvie (reviewed). Location note: Volcano Islands, Japan region.
HV WIRE [06:49 (UTC-10:00) 3 Nov 2026] MD 1.7 quake in Hawaii @ 19°13.52'N 155°24.17'W; 30.8 km deep. Ref hv75068191, reviewed.
UPDATE [16:55 UTC on 3 November 2026]: us7000tvx5 withdrawn (mine blast).
AK WIRE [06:56 (UTC-10:00) 3 Nov 2026] ML 2.4 quake in Alaska @ 59.849°N, 152.757°W; 100.1 km (62.2 mi) deep. Ref aka2026vumpow, reviewed.
10:34 (UTC-07:00) 3 Nov 2026 — A magnitude 2.3 (ml) earthquake struck Texas; epicentre (32.108, -101.911), depth 8.5 km. The TX network lists it as tx2026vrbasw (reviewed). Location note: 11 km WSW of Stanton, Texas.
18:27 UTC on 3 November 2026 ref=ci41572064 net=ci mag=4.89 type=mw region="Mexico" loc=32.431,-115.180 depth=15.6 km status=reviewed
10:52 AM local time (UTC-08:00) on Nov 3, 2026 — A magnitude 1.27 (ml) earthquake struck Utah; epicentre (38.519, -112.914), depth 3.0 mi. The UU network lists it as uu80162661 (reviewed). Location note: 15 km NNE of Milford, Utah.
Felt reports are still being collected and are not summarised here.
12:15 (UTC-08:00) 3 Nov 2026 ref=nc75462147 net=nc mag=0.75 type=md region="California" loc=38.800,-122.767 depth=2.6 mi status=reviewed
10:50 AM local time (UTC-10:00) on Nov 3, 2026 ref=hv75068221 net=hv mag=2.08 type=md region="Hawaii" loc=19.428,-155.507 depth=5.3 mi status=reviewed
2026-11-03 21:45:05 UTC — A magnitude 1.56 (md) earthquake struck Utah; epicentre (38.513, -112.919), depth 2.4 km. The UU network lists it as uu80162686 (reviewed). Location note: 15 km NNE of Milford, Utah.
2:06 PM local time (UTC-08:00) on Nov 3, 2026 — A magnitude 0.99 (ml) earthquake struck California; epicentre 33.236°N, 116.028°W, depth 4.3 km. The CI network lists it as ci41572256 (reviewed). Location note: 10 km SW of Salton City, CA.
```

- Q1: Counting only earthquakes with magnitude at least 2.1, which region has more of them: Alaska or Japan? Answer "tie" if the counts are equal.
- Q2: Among the earthquakes whose region is listed as California, which one had the largest magnitude? Give its event ID.
- Q3: What is the number of separate earthquakes with a depth of 29.0 km or more and with magnitude at least 2.1?
- Q4: What was the largest magnitude among the earthquakes with a depth of 6 km or more?
- Q5: At what time (UTC, HH:MM) did the largest of the earthquakes smaller than magnitude 1.9 occur?

</details>

### Example 2: `inst-002` (26 earthquakes, 30 report lines, 2 notices, 5 questions)

**Digest (original rendering `v0`, exactly as sent to the models):**

```text
SEISMIC ACTIVITY DIGEST (edition 232-F)
Compiled 2026-11-03T22:18:53Z; covers the 24 hours before compilation. Reports are reproduced as received.

[UU] uu80162486 | 15:26 (UTC-08:00) 2 Nov 2026 | Utah | 38°30.84'N 112°54.82'W | depth 3.4 km (2.1 mi) | M2.58 (ml) | reviewed | near: 15 km NNE of Milford, Utah
15:27 (UTC-08:00) 2 Nov 2026 — A magnitude 2.28 (ml) earthquake struck Utah; epicentre 38°32.08'N 112°54.31'W, depth 6.1 km. The EW network lists it as uu80162486 (preliminary). Location note: 17 km NNE of Milford, Utah.
[AK] aka2026vteirb | 1:49 PM local time (UTC-10:00) on Nov 2, 2026 | Alaska | 60.776°N, 142.808°W | depth 5.0 km | ML 0.9 | reviewed | near: 73 km S of McCarthy, Alaska
[AK] aka2026vtfofw | 15:25 (UTC-09:00) 2 Nov 2026 | Alaska | 59.987°N, 141.493°W | depth 5.0 km (3.1 mi) | ML 2.1 | reviewed | near: 110 km WNW of Yakutat, Alaska
2026-11-03T00:27:08Z ref=us7000tv1v net=us mag=4.2 type=mb region="China" loc=32.292,93.562 depth=10.0 km status=reviewed
CORRECTION [2026-11-03 20:22:22 UTC]: the magnitude of ci41572064 is revised to 4.89 (mw); it was first given as 1.96.
[AK] aka2026vtfqar | 2026-11-03T00:27:57Z | Alaska | (56.872, -155.591) | depth 25.3 km | ML 3.4 | reviewed | near: 86 km W of Akhiok, Alaska
[CI] ci41571712 | 5:33 PM local time (UTC-08:00) on Nov 2, 2026 | Mexico | 32.417°N, 115.166°W | depth 12.1 mi | M2.75 (ml) | reviewed | near: 7 km NNE of Delta, B.C., MX
PT WIRE [19:02 (UTC-08:00) 2 Nov 2026] ML 3.58 quake in Mexico @ (32.384, -115.170); 12.8 mi deep. Ref ci41571744, preliminary.
[CI] ci41571744 | 7:02 PM local time (UTC-08:00) on Nov 2, 2026 | Mexico | (32.404, -115.165) | depth 20.3 km | M3.28 (ml) | reviewed | near: 6 km NNE of Delta, B.C., MX
[PR] pr71538373 | 2026-11-03 03:34:01 UTC | Puerto Rico | (18.689, -66.443) | depth 32.5 km | MD 2.54 | reviewed | near: 25 km N of Tierras Nuevas Poniente, Puerto Rico
[CI] ci41571816 | 21:20 (UTC-08:00) 2 Nov 2026 | California | 32°56.63'N 115°32.52'W | depth 5.2 mi | M1.15 (ml) | reviewed | near: 4 km SSW of Brawley, CA
Editor's note: entries are reproduced exactly as each network filed them, time conventions included.
[CI] ci41571904 | 08:08 UTC on 3 November 2026 | Mexico | 32.420°N, 115.166°W | depth 11.6 km | M1.66 (ml) | reviewed | near: 8 km NNE of Delta, B.C., MX
[AK] aka2026vtvtat | 10:31 PM local time (UTC-10:00) on Nov 2, 2026 | Alaska | 61.631°N, 150.521°W | depth 51.8 km | M2.5 (ml) | reviewed | near: 9 km N of Susitna, Alaska
NOTICE [2026-11-03T18:25:35Z]: report us7000tviy has been RETRACTED (sonic boom). Disregard it.
US WIRE [13:03 (UTC+02:00) 3 Nov 2026] MB 4.9 quake in Greece @ (34.325, 26.571); 45.7 km deep. Ref us7000tv51, reviewed.
[AK] aka2026vudflk | 02:15 (UTC-10:00) 3 Nov 2026 | Alaska | (61.394, -151.208) | depth 70.1 km | M2.5 (ml) | reviewed | near: 28 km NNW of Beluga, Alaska
[US] us7000tva7 | 23:37 (UTC+08:00) 3 Nov 2026 | Indonesia | (-0.230, 123.153) | depth 136.7 km | M4.8 (mb) | reviewed | near: 85 km S of Gorontalo, Indonesia
[TX] tx2026vqyakk | 08:59 (UTC-07:00) 3 Nov 2026 | Texas | (28.828, -98.975) | depth 7.4 km | ML 2.0 | reviewed | near: 13 km ESE of Pearsall, Texas
4:48 AM local time (UTC-12:00) on Nov 3, 2026 ref=us7000tvsl net=us mag=3.6 type=ml region="Alaska" loc=51.431,-175.959 depth=35.0 km status=reviewed
Felt reports are still being collected and are not summarised here.
[AK] aka2026vunrap | 7:28 AM local time (UTC-10:00) on Nov 3, 2026 | Alaska | 60.269°N, 153.087°W | depth 155.8 km | M2.5 (ml) | reviewed | near: 68 km E of Port Alsworth, Alaska
[TX] tx2026vrbasw | 10:34 AM local time (UTC-07:00) on Nov 3, 2026 | Texas | 32.108°N, 101.911°W | depth 8.5 km | M2.3 (ml) | reviewed | near: 11 km WSW of Stanton, Texas
23:57 (UTC+06:00) 3 Nov 2026 ref=us7000tviy net=us mag=4.1 type=mb region="China" loc=32.419,93.602 depth=5.8 km status=reviewed
AK WIRE [08:11 (UTC-10:00) 3 Nov 2026] ML 2.9 quake in Alaska @ 60.987°N, 150.717°W; 60.3 km deep. Ref aka2026vupcsv, reviewed.
CI WIRE [18:27 UTC on 3 November 2026] MW 1.96 quake in Mexico @ 32.431°N, 115.180°W; 15.6 km deep. Ref ci41572064, reviewed.
2026-11-03T18:42:13Z — A magnitude 2.19 (ml) earthquake struck Mexico; epicentre (32.429, -115.164), depth 9.9 mi. The CI network lists it as ci41572072 (reviewed). Location note: 9 km NNE of Delta, B.C., MX.
Tue, 03 Nov 2026 19:35:20 +0000 ref=ci41572128 net=ci mag=2.09 type=ml region="Mexico" loc=32.405,-115.163 depth=9.6 mi status=reviewed
[CI] ci41572136 | 2026-11-03T19:35:51Z | Mexico | (32.409, -115.158) | depth 9.6 mi | ML 2.5 | reviewed | near: 7 km NNE of Delta, B.C., MX
AT WIRE [Tue, 03 Nov 2026 19:36:11 +0000] ML 2.2 quake in Mexico @ (32.374, -115.197); 16.3 km deep. Ref ci41572136, preliminary.
[CI] ci10259790 | Tue, 03 Nov 2026 19:49:18 +0000 | Mexico | (32.405, -115.192) | depth 15.1 km | M1.34 (ml) | reviewed | near: 6 km N of Delta, B.C., MX
[CI] ci41572184 | 12:16 PM local time (UTC-08:00) on Nov 3, 2026 | Mexico | 32.440°N, 115.169°W | depth 14.5 km | ML 2.16 | reviewed | near: 10 km NNE of Delta, B.C., MX
[CI] ci41572192 | 2026-11-03 20:17:49 UTC | Mexico | 32.435°N, 115.176°W | depth 12.1 km | M1.98 (ml) | reviewed | near: 9 km N of Delta, B.C., MX
```

**Questions, code-computed key and model answers (`v0`):**

| Q | template | question | key | oracle | naive | mock |
|---|---|---|---|---|---|---|
| Q1 | N11 (new) | Among the earthquakes with a depth of 6.6 miles or more, how deep was the shallowest one, in km? Round to 1 decimal place. | 11.6 | 11.6 ✓ | 11.6 ✓ | 11.6 ✓ |
| Q2 | N01 (new) | What is the number of separate earthquakes shallower than 8 miles? | 9 | 9 ✓ | 11 ✗ | 9 ✓ |
| Q3 | P01 (public) | Count the distinct earthquakes with magnitude at least 2.1. | 19 | 19 ✓ | 22 ✗ | 20 ✗ |
| Q4 | P03 (public) | Among the earthquakes whose region is listed as Alaska, which one had the largest magnitude? Give its event ID. | us7000tvsl | us7000tvsl ✓ | us7000tvsl ✓ | us7000tv1v ✗ |
| Q5 | P09 (public) | Rank the earthquakes no more than 300 km from station CI-97 (32.526°N, 115.348°W) by magnitude, largest first: which event is second? Give its event ID. | ci41571744 | ci41571744 ✓ | ci41571744 ✓ | ci41571744 ✓ |

**Operation chains (the machine-readable meaning of each question):**

- Q1: `dedupe → depth ≥ 6.6 mi → min_depth → convert → km → round(1)`
- Q2: `dedupe → depth < 8 mi → count`
- Q3: `dedupe → mag ≥ 2.1 → count`
- Q4: `dedupe → region = Alaska → argmax_mag`
- Q5: `dedupe → within 300 km of (32.526, -115.348) → rank_mag(k=2)`

**Traps in this digest:**

- T1 *duplicate* (same_id): PT re-reported ci41571744 under the same event ID: preliminary M3.58, depth 20.6 km (reviewed: M3.28, 20.3 km). If mishandled: Q3 → 20.
- T2 *duplicate* (same_id): AT re-reported ci41572136 under the same event ID: preliminary M2.2, depth 16.3 km (reviewed: M2.5, 15.5 km). If mishandled: Q3 → 20.
- T3 *duplicate* (same_id): EW re-reported uu80162486 under the same event ID: preliminary M2.28, depth 6.1 km (reviewed: M2.58, 3.4 km). If mishandled: Q2 → 10, Q3 → 20.
- T4 *retraction* (phantom): phantom report us7000tviy (M4.1, China) later retracted: sonic boom. If mishandled: Q2 → 10, Q3 → 20.
- T5 *conflicting magnitude* (correction): ci41572064 filed as M1.96 on its reviewed report, then CORRECTED to M4.89. If mishandled: Q3 → 18, Q5 → ci41571712.

<details><summary>Metamorphic variant `v1` of the same instance (same key)</summary>

```text
SEISMIC ACTIVITY DIGEST (edition 336-C)
Compiled Tue, 03 Nov 2026 22:18:53 +0000; covers the 24 hours before compilation. Reports are reproduced as received.

-- AT network --
Editor's note: entries are reproduced exactly as each network filed them, time conventions included.
11:36 AM local time (UTC-08:00) on Nov 3, 2026 — A magnitude 2.2 (ml) earthquake struck Mexico; epicentre 32.374°N, 115.197°W, depth 16.3 km (10.1 mi). The AT network lists it as ci41572136 (preliminary). Location note: 2 km NNW of Delta, B.C., MX.
CORRECTION [20:22 UTC on 3 November 2026]: the magnitude of ci41572064 is revised to 4.89 (mw); it was first given as 1.96.
-- AK network --
00:25 UTC on 3 November 2026 — A magnitude 2.1 (ml) earthquake struck Alaska; epicentre (59.987, -141.493), depth 5.0 km (3.1 mi). The AK network lists it as aka2026vtfofw (reviewed). Location note: 110 km WNW of Yakutat, Alaska.
[AK] aka2026vtvtat | Tue, 03 Nov 2026 08:31:06 +0000 | Alaska | 61.631°N, 150.521°W | depth 32.2 mi | ML 2.5 | reviewed | near: 9 km N of Susitna, Alaska
00:27 UTC on 3 November 2026 — A magnitude 3.4 (ml) earthquake struck Alaska; epicentre 56.872°N, 155.591°W, depth 15.7 mi. The AK network lists it as aka2026vtfqar (reviewed). Location note: 86 km W of Akhiok, Alaska.
2026-11-02T23:49:20Z — A magnitude 0.9 (ml) earthquake struck Alaska; epicentre 60°46.56'N 142°48.48'W, depth 3.1 mi. The AK network lists it as aka2026vteirb (reviewed). Location note: 73 km S of McCarthy, Alaska.
2026-11-03T17:28:41Z — A magnitude 2.5 (ml) earthquake struck Alaska; epicentre (60.269, -153.087), depth 155.8 km. The AK network lists it as aka2026vunrap (reviewed). Location note: 68 km E of Port Alsworth, Alaska.
AK WIRE [08:11 (UTC-10:00) 3 Nov 2026] ML 2.9 quake in Alaska @ 60°59.22'N 150°43.02'W; 37.5 mi deep. Ref aka2026vupcsv, reviewed.
12:15 UTC on 3 November 2026 — A magnitude 2.5 (ml) earthquake struck Alaska; epicentre (61.394, -151.208), depth 70.1 km (43.6 mi). The AK network lists it as aka2026vudflk (reviewed). Location note: 28 km NNW of Beluga, Alaska.
From the EW network:
2026-11-02T23:27:06Z — A magnitude 2.28 (ml) earthquake struck Utah; epicentre 38.535°N, 112.905°W, depth 3.8 mi. The EW network lists it as uu80162486 (preliminary). Location note: 17 km NNE of Milford, Utah.
== PR feed ==
2026-11-03 03:34:01 UTC — A magnitude 2.54 (md) earthquake struck Puerto Rico; epicentre (18.689, -66.443), depth 32.5 km (20.2 mi). The PR network lists it as pr71538373 (reviewed). Location note: 25 km N of Tierras Nuevas Poniente, Puerto Rico.
-- US network --
2026-11-03 00:27:08 UTC — A magnitude 4.2 (mb) earthquake struck China; epicentre 32.292°N, 93.562°E, depth 6.2 mi. The US network lists it as us7000tv1v (reviewed). Location note: 168 km ENE of Nagqu, China.
2026-11-03T16:48:57Z ref=us7000tvsl net=us mag=3.6 type=ml region="Alaska" loc=51.431,-175.959 depth=21.7 mi status=reviewed
[US] us7000tva7 | 2026-11-03 15:37:51 UTC | Indonesia | 0°13.77'S 123°09.19'E | depth 84.9 mi | MB 4.8 | reviewed | near: 85 km S of Gorontalo, Indonesia
17:57 UTC on 3 November 2026 — A magnitude 4.1 (mb) earthquake struck China; epicentre 32.419°N, 93.602°E, depth 5.8 km. The US network lists it as us7000tviy (reviewed). Location note: 177 km ENE of Nagqu, China.
US WIRE [Tue, 03 Nov 2026 11:03:47 +0000] MB 4.9 quake in Greece @ 34.325°N, 26.571°E; 45.7 km (28.4 mi) deep. Ref us7000tv51, reviewed.
== TX feed ==
Tue, 03 Nov 2026 15:59:46 +0000 ref=tx2026vqyakk net=tx mag=2.0 type=ml region="Texas" loc=28.828,-98.975 depth=4.6 mi status=reviewed
2026-11-03T17:34:24Z — A magnitude 2.3 (ml) earthquake struck Texas; epicentre 32°06.48'N 101°54.66'W, depth 5.3 mi. The TX network lists it as tx2026vrbasw (reviewed). Location note: 11 km WSW of Stanton, Texas.
== CI feed ==
[CI] ci41572064 | 10:27 AM local time (UTC-08:00) on Nov 3, 2026 | Mexico | 32°25.85'N 115°10.78'W | depth 15.6 km | MW 1.96 | reviewed | near: 9 km N of Delta, B.C., MX
Tue, 03 Nov 2026 20:16:39 +0000 — A magnitude 2.16 (ml) earthquake struck Mexico; epicentre (32.440, -115.169), depth 14.5 km. The CI network lists it as ci41572184 (reviewed). Location note: 10 km NNE of Delta, B.C., MX.
Tue, 03 Nov 2026 05:20:32 +0000 — A magnitude 1.15 (ml) earthquake struck California; epicentre (32.944, -115.542), depth 5.2 mi. The CI network lists it as ci41571816 (reviewed). Location note: 4 km SSW of Brawley, CA.
2026-11-03 03:02:49 UTC — A magnitude 3.28 (ml) earthquake struck Mexico; epicentre 32°24.26'N 115°09.89'W, depth 12.6 mi. The CI network lists it as ci41571744 (reviewed). Location note: 6 km NNE of Delta, B.C., MX.
19:35 UTC on 3 November 2026 — A magnitude 2.09 (ml) earthquake struck Mexico; epicentre 32.405°N, 115.163°W, depth 9.6 mi. The CI network lists it as ci41572128 (reviewed). Location note: 6 km NNE of Delta, B.C., MX.
11:49 AM local time (UTC-08:00) on Nov 3, 2026 — A magnitude 1.34 (ml) earthquake struck Mexico; epicentre (32.405, -115.192), depth 9.4 mi. The CI network lists it as ci10259790 (reviewed). Location note: 6 km N of Delta, B.C., MX.
01:33 UTC on 3 November 2026 — A magnitude 2.75 (ml) earthquake struck Mexico; epicentre 32.417°N, 115.166°W, depth 12.1 mi. The CI network lists it as ci41571712 (reviewed). Location note: 7 km NNE of Delta, B.C., MX.
UPDATE [2026-11-03 18:25:35 UTC]: us7000tviy withdrawn (sonic boom).
CI WIRE [08:08 UTC on 3 November 2026] ML 1.66 quake in Mexico @ (32.420, -115.166); 7.2 mi deep. Ref ci41571904, reviewed.
[CI] ci41572072 | 18:42 UTC on 3 November 2026 | Mexico | (32.429, -115.164) | depth 9.9 mi | M2.19 (ml) | reviewed | near: 9 km NNE of Delta, B.C., MX
11:35 AM local time (UTC-08:00) on Nov 3, 2026 — A magnitude 2.5 (ml) earthquake struck Mexico; epicentre 32°24.54'N 115°09.47'W, depth 9.6 mi. The CI network lists it as ci41572136 (reviewed). Location note: 7 km NNE of Delta, B.C., MX.
2026-11-03T20:17:49Z — A magnitude 1.98 (ml) earthquake struck Mexico; epicentre 32°26.09'N 115°10.58'W, depth 7.5 mi. The CI network lists it as ci41572192 (reviewed). Location note: 9 km N of Delta, B.C., MX.
From the PT network:
03:02 UTC on 3 November 2026 — A magnitude 3.58 (ml) earthquake struck Mexico; epicentre 32°23.05'N 115°10.18'W, depth 12.8 mi. The PT network lists it as ci41571744 (preliminary). Location note: 4 km NNE of Delta, B.C., MX.
== UU feed ==
Felt reports are still being collected and are not summarised here.
Mon, 02 Nov 2026 23:26:35 +0000 — A magnitude 2.58 (ml) earthquake struck Utah; epicentre 38.514°N, 112.914°W, depth 2.1 mi. The UU network lists it as uu80162486 (reviewed). Location note: 15 km NNE of Milford, Utah.
```

- Q1: Among the earthquakes with a depth of 10.6 km or more, how deep was the shallowest one, in km? Round to 1 decimal place.
- Q2: What is the number of separate earthquakes shallower than 12.9 km?
- Q3: Count the distinct earthquakes of magnitude 2.1 or greater.
- Q4: Among the earthquakes in Alaska, which one had the largest magnitude? Give its event ID.
- Q5: Rank the earthquakes no more than 300 km from station CI-97 at (32.526, -115.348) by magnitude, largest first: which event is second? Give its event ID.

</details>
