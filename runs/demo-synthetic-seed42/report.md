# Live evaluation report: `demo-synthetic-seed42`

Every instance in this run was generated fresh from the seed below, every answer key was computed by code from the frozen snapshot, and every score was computed automatically. No item was labelled by hand.

## Run

- **Generated with:** evalgen 0.1.0 on 2026-11-03T22:18:54Z (the reading rules and templates are versioned with the code).
- **Master seed:** `42` (commitment `5a73024a42fc8355a4687c27…`); every derived seed is listed in `manifest.json`.
- **Data source:** synthetic; 360 events in the snapshot pool; window: 24 h ending 2026-09-15T06:00:00Z.
- **Snapshot SHA-256:** `2e5ff1139c47ef21e5501d3213f398544430b87e93a6e977a01c107fd42a219d`
- **⚠ Note:** the evalgen code has changed since this run was generated (with evalgen 0.1.0, code `277dab3e900e5e98…`). This report scores the stored prompts and replies; regenerating the run from its seed reproduces it exactly only with the code that generated it.
- **Instances:** 20; **questions:** 80 (40 from public templates, 40 from new-this-run templates); **renderings per instance:** 3 (original + 2 metamorphic variants); **prompts:** 60.
- **Traps planted:** 91 (duplicate 51, retraction 20, conflicting magnitude 20); 82 are load-bearing: mishandling them changes the answers of 40 of the 80 questions (91 trap–question pairs). The rest stay in as plain noise.

## Model accuracy (original renderings)

| model | kind | accuracy | 95% CI | mean instance score | accuracy, all renderings | JSON strict / lenient / failed (API errors) |
|---|---|---|---|---|---|---|
| oracle | reference | 100.0% (80/80) | 95–100% | 100.0% | 100.0% | 60 / 0 / 0 (0) |
| naive | reference | 53.8% (43/80) | 43–64% | 53.3% | 53.8% | 60 / 0 / 0 (0) |
| mock | simulated | 75.0% (60/80) | 65–83% | 74.5% | 70.4% | 54 / 6 / 0 (0) |
| anthropic:claude-opus-5-5 | api | 100.0% (80/80) | 95–100% | 100.0% | 99.6% | 60 / 0 / 0 (0) |

`oracle` reads only the digest text and applies the reading rules, so it should score 100%. That shows every item is answerable from the text. `naive` treats every line as a separate quake and ignores notices. `mock` is a *simulated* model with built-in benchmark contamination, used to exercise the metrics without API keys; it is not a language model.

## API usage

Tokens as each API reported them, for the replies that were scored (an attempt that failed or was cut off and then asked again is not counted). Output tokens include any thinking before the answer.

| model | replies | input tokens | output tokens | median output tokens per reply | median latency |
|---|---|---|---|---|---|
| anthropic:claude-opus-5-5 | 60 | 210,601 | 80,475 | 1,315 | 11 s |

## Metamorphic consistency

Each instance is re-rendered with the same snapshot but different units, line order, formats and question wording. Correct answers cannot change, so any change in a model's answer is an inconsistency. This needs no ground truth.

| model | consistent answer pairs | 95% CI | questions consistent across all variants |
|---|---|---|---|
| oracle | 100.0% (160/160) | 98–100% | 100.0% (80/80) |
| naive | 100.0% (160/160) | 98–100% | 100.0% (80/80) |
| mock | 53.1% (85/160) | 45–61% | 37.5% (30/80) |
| anthropic:claude-opus-5-5 | 99.4% (159/160) | 97–100% | 98.8% (79/80) |

## Seen vs new templates

Public templates are published (a model may have trained on them). Each new template is composed for this run as a twin of one public template: the same final operation and number of filters, but a different filter combination, and each twin pair is used equally often. The *adjusted* gap compares the pools like for like, within strata of equal code-computed difficulty (how many traps change the answer, and whether the filters matter), so that the pools' different trap exposure is not mistaken for overfitting. A large positive adjusted gap flags overfitting to the published skeletons (flag: > 10 points with a 95% CI that excludes 0). It is a flag, not proof.

| model | public (seen) | new this run | raw gap (95% CI) | adjusted gap (95% CI) | flag |
|---|---|---|---|---|---|
| oracle | 100.0% (40/40) | 100.0% (40/40) | +0.0 pts (+0 to +0) | +0.0 pts (-10 to +10) | — |
| naive | 57.5% (23/40) | 50.0% (20/40) | +7.5 pts (-14 to +29) | +2.5 pts (-10 to +15) | — |
| mock | 85.0% (34/40) | 65.0% (26/40) | +20.0 pts (+2 to +38) | +15.3 pts (-3 to +33) | — |
| anthropic:claude-opus-5-5 | 100.0% (40/40) | 100.0% (40/40) | +0.0 pts (+0 to +0) | +0.0 pts (-10 to +10) | — |

## Trap catch rate

Counted over (load-bearing trap, affected question, rendering) triples. *Caught* = answered correctly; *fell* = gave exactly an answer a reader produces by missing that trap (alone or together with other traps).

| model | caught (overall) | duplicate | retraction | conflicting magnitude | fell for the trap |
|---|---|---|---|---|---|
| oracle | 100.0% (273/273) | 100.0% | 100.0% | 100.0% | 0.0% (0/273) |
| naive | 4.4% (12/273) | 3.9% | 5.3% | 4.8% | 95.6% (261/273) |
| mock | 72.2% (197/273) | 74.5% | 75.4% | 63.5% | 19.8% (54/273) |
| anthropic:claude-opus-5-5 | 99.6% (272/273) | 99.3% | 100.0% | 100.0% | 0.0% (0/273) |

## Accuracy by answer type (original renderings)

| model | count | event_id | magnitude | depth | distance | region | utc_time |
|---|---|---|---|---|---|---|---|
| oracle | 100.0% (17/17) | 100.0% (23/23) | 100.0% (9/9) | 100.0% (14/14) | 100.0% (5/5) | 100.0% (6/6) | 100.0% (6/6) |
| naive | 17.6% (3/17) | 47.8% (11/23) | 55.6% (5/9) | 64.3% (9/14) | 100.0% (5/5) | 83.3% (5/6) | 83.3% (5/6) |
| mock | 76.5% (13/17) | 73.9% (17/23) | 77.8% (7/9) | 78.6% (11/14) | 60.0% (3/5) | 66.7% (4/6) | 83.3% (5/6) |
| anthropic:claude-opus-5-5 | 100.0% (17/17) | 100.0% (23/23) | 100.0% (9/9) | 100.0% (14/14) | 100.0% (5/5) | 100.0% (6/6) | 100.0% (6/6) |

## Generator audit

- **Automated self-check:** for 240/240 question-renderings, the digest-only oracle agreed with the code-computed key. Applying the reading rules to each digest recovered the snapshot exactly for 20/20 instances.
- **Human spot-check sample:** 2 of 80 generated items (2.5%; target 2.0%) → `audit/audit_sample.jsonl` (checklist: `audit/audit_sample.md`).
- **Verdicts:** 0/2 reviewed so far. Fill in `verdict` (ok / error) in the JSONL file and re-run `cli.py report` to get an estimated generator error rate.

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
| N01 | new | P01 | dedupe → time_between → count |
| N02 | new | P02 | dedupe → in_region → count |
| N03 | new | P03 | dedupe → within_distance → argmax_mag |
| N04 | new | P04 | dedupe → time_between → max_depth |
| N05 | new | P05 | dedupe → depth_at_least → compare_regions |
| N06 | new | P06 | dedupe → mag_at_least → depth_at_least → count |
| N07 | new | P07 | dedupe → depth_at_least → nearest_to |
| N08 | new | P08 | dedupe → time_between → time_of_largest |
| N09 | new | P09 | dedupe → mag_below → rank_mag |
| N10 | new | P10 | dedupe → within_distance → max_mag |
| N11 | new | P11 | dedupe → mag_at_least → min_depth |
| N12 | new | P12 | dedupe → mag_at_least → nearest_distance |

## Per-instance scores (original rendering)

<details><summary>Show table</summary>

| instance | questions | load-bearing traps | oracle | naive | mock | anthropic:claude-opus-5-5 |
|---|---|---|---|---|---|---|
| inst-000 | 5 | 4 | 100.0% | 60.0% | 100.0% | 100.0% |
| inst-001 | 5 | 4 | 100.0% | 40.0% | 100.0% | 100.0% |
| inst-002 | 3 | 5 | 100.0% | 33.3% | 33.3% | 100.0% |
| inst-003 | 4 | 4 | 100.0% | 50.0% | 75.0% | 100.0% |
| inst-004 | 5 | 5 | 100.0% | 60.0% | 40.0% | 100.0% |
| inst-005 | 4 | 3 | 100.0% | 50.0% | 100.0% | 100.0% |
| inst-006 | 5 | 4 | 100.0% | 60.0% | 60.0% | 100.0% |
| inst-007 | 5 | 5 | 100.0% | 80.0% | 80.0% | 100.0% |
| inst-008 | 4 | 4 | 100.0% | 75.0% | 75.0% | 100.0% |
| inst-009 | 3 | 5 | 100.0% | 33.3% | 66.7% | 100.0% |
| inst-010 | 3 | 3 | 100.0% | 33.3% | 100.0% | 100.0% |
| inst-011 | 4 | 4 | 100.0% | 0.0% | 100.0% | 100.0% |
| inst-012 | 5 | 5 | 100.0% | 40.0% | 60.0% | 100.0% |
| inst-013 | 5 | 4 | 100.0% | 60.0% | 100.0% | 100.0% |
| inst-014 | 3 | 4 | 100.0% | 66.7% | 100.0% | 100.0% |
| inst-015 | 3 | 1 | 100.0% | 66.7% | 33.3% | 100.0% |
| inst-016 | 3 | 4 | 100.0% | 66.7% | 66.7% | 100.0% |
| inst-017 | 4 | 4 | 100.0% | 50.0% | 50.0% | 100.0% |
| inst-018 | 3 | 5 | 100.0% | 66.7% | 100.0% | 100.0% |
| inst-019 | 4 | 5 | 100.0% | 75.0% | 50.0% | 100.0% |

</details>

## Example instances

### Example 1: `inst-004` (23 earthquakes, 28 report lines, 1 notice, 5 questions)

**Digest (original rendering `v0`, exactly as sent to the models):**

```text
Earthquake wire roundup: the 24 hours ending Tue, 15 Sep 2026 06:00:00 +0000
Sources: regional and global seismic networks. Edition 857-A.

Mon, 14 Sep 2026 06:20:20 +0000 — A magnitude 1.3 (md) earthquake struck Puerto Rico; epicentre 18°03.19'N 66°50.66'W, depth 6.5 km. The PR network lists it as pr57584067 (reviewed).
06:45 UTC on 14 September 2026 — A magnitude 4.0 (mb) earthquake struck Central Chile; epicentre 34.374°S, 72.353°W, depth 64.7 km. The US network lists it as us6510j5sy (reviewed).
[US] us7894bl0f | 07:35 UTC on 14 September 2026 | Central Chile | 32°10.28'S 73°14.61'W | depth 81.0 km | MB 4.2 | reviewed
22:43 (UTC-10:00) 13 Sep 2026 ref=hv61367843 net=at mag=2.5 type=md region="Hawaii" loc=19.375,-155.311 depth=37.9 km status=preliminary
Station NC.PBR was offline from 12:00 to 12:44 UTC for maintenance; its readings are not included.
10:43 PM local time (UTC-10:00) on Sep 13, 2026 — A magnitude 2.8 (md) earthquake struck Hawaii; epicentre 19.400°N, 155.300°W, depth 36.5 km. The HV network lists it as hv61367843 (reviewed).
1:15 AM local time (UTC-08:00) on Sep 14, 2026 — A magnitude 1.3 (md) earthquake struck Northern California; epicentre (38.614, -122.911), depth 4.7 mi. The NC network lists it as nc84330092 (reviewed).
[HV] hv75592581 | 12:39 AM local time (UTC-10:00) on Sep 14, 2026 | Hawaii | (19.361, -155.420) | depth 13.8 km (8.6 mi) | MD 1.0 | reviewed
12:39 UTC on 14 September 2026 — A magnitude 1.8 (md) earthquake struck Puerto Rico; epicentre (17.968, -67.603), depth 15.8 km. The PR network lists it as pr22281028 (reviewed).
Mon, 14 Sep 2026 12:48:28 +0000 — A magnitude 5.2 (mb) earthquake struck Honshu, Japan; epicentre 35.325°N, 140.342°E, depth 80.3 km. The US network lists it as us6111meob (reviewed).
US WIRE [13:51 UTC on 14 September 2026] MB 4.2 quake in Central Chile @ 33.641°S, 72.143°W; 110.2 km deep. Ref us6577wo0k, reviewed.
HV WIRE [04:05 (UTC-10:00) 14 Sep 2026] MD 3.2 quake in Hawaii @ (19.383, -155.292); 23.1 km deep. Ref hv83297115, reviewed.
HV WIRE [05:08 (UTC-10:00) 14 Sep 2026] MD 1.1 quake in Hawaii @ 20°04.85'N 155°17.45'W; 4.3 mi deep. Ref hv68175627, reviewed.
5:09 AM local time (UTC-10:00) on Sep 14, 2026 — A magnitude 0.9 (md) earthquake struck Hawaii; epicentre (20.070, -155.286), depth 8.3 km (5.2 mi). The PT network lists it as hv68175627 (preliminary).
11:56 (UTC-05:00) 14 Sep 2026 — A magnitude 5.3 (mb) earthquake struck Central Chile; epicentre 33.464°S, 73.584°W, depth 50.3 mi. The US network lists it as us77588lhm (reviewed).
Mon, 14 Sep 2026 17:48:09 +0000 ref=at47434712 net=at mag=1.8 type=ml region="Nevada" loc=38.311,-117.868 depth=5.5 km status=preliminary same_as=nn80947618
09:48 (UTC-08:00) 14 Sep 2026 ref=nn80947618 net=nn mag=1.5 type=ml region="Nevada" loc=38.304,-117.945 depth=6.9 km status=reviewed
12:52 PM local time (UTC-05:00) on Sep 14, 2026 — A magnitude 5.1 (mb) earthquake struck Central Chile; epicentre 34.228°S, 72.034°W, depth 60.1 km. The US network lists it as us7884jbg0 (reviewed).
11:13 (UTC-08:00) 14 Sep 2026 — A magnitude 1.1 (md) earthquake struck Northern California; epicentre 39°09.86'N 122°20.43'W, depth 1.5 km (0.9 mi). The NC network lists it as nc24064840 (reviewed).
2026-09-14T20:28:56Z — A magnitude 2.0 (ml) earthquake struck Southern California; epicentre 33.973°N, 117.240°W, depth 12.7 km. The CI network lists it as ci63518918 (reviewed).
16:36 (UTC-05:00) 14 Sep 2026 ref=us6759vakn net=us mag=4.0 type=mb region="Central Chile" loc=-33.327,-71.530 depth=101.8 km (63.3 mi) status=reviewed
22:26 UTC on 14 September 2026 — A magnitude 1.0 (md) earthquake struck Puerto Rico; epicentre 17.973°N, 66.837°W, depth 18.3 km. The PR network lists it as pr14639407 (reviewed).
22:46 UTC on 14 September 2026 — A magnitude 2.9 (md) earthquake struck Puerto Rico; epicentre 18.300°N, 66.834°W, depth 18.6 km. The EW network lists it as pr82937817 (preliminary).
18:46 (UTC-04:00) 14 Sep 2026 — A magnitude 3.0 (md) earthquake struck Puerto Rico; epicentre 18.300°N, 66.834°W, depth 18.6 km. The PR network lists it as pr82937817 (reviewed).
8:04 PM local time (UTC-04:00) on Sep 14, 2026 — A magnitude 1.3 (md) earthquake struck Puerto Rico; epicentre 18.278°N, 67.079°W, depth 26.0 km. The PR network lists it as pr96255606 (reviewed).
US WIRE [10:24 (UTC+09:00) 15 Sep 2026] MB 4.2 quake in Honshu, Japan @ (36.093, 141.799); 30.4 km deep. Ref us63988qsy, reviewed.
UPDATE [2026-09-14 18:29:10 UTC]: us7884jbg0 withdrawn (mine blast).
2026-09-15 01:29:22 UTC ref=hv66757965 net=hv mag=1.5 type=md region="Hawaii" loc=19.393,-155.246 depth=21.3 mi status=reviewed
7:18 PM local time (UTC-08:00) on Sep 14, 2026 — A magnitude 1.0 (md) earthquake struck Northern California; epicentre 39°07.86'N 123°54.67'W, depth 5.4 km. The NC network lists it as nc20476354 (reviewed).
[US] us6377e32b | 2026-09-15 04:10:03 UTC | Honshu, Japan | 38°21.64'N 141°56.26'E | depth 17.6 km | MB 5.4 | reviewed
```

**Questions, code-computed key and model answers (`v0`):**

| Q | template | question | key | oracle | naive | mock | anthropic:claude-opus-5-5 |
|---|---|---|---|---|---|---|---|
| Q1 | N01 (new) | How many distinct earthquakes that were recorded from 2026-09-14 14:15 UTC up to 2026-09-14 18:05 UTC are there? | 3 | 3 ✓ | 6 ✗ | 3 ✓ | 3 ✓ |
| Q2 | P04 (public) | What was the greatest depth, in miles, among the earthquakes of magnitude 4.2 or more? Round to 1 decimal place. | 68.5 | 68.5 ✓ | 68.5 ✓ | 87.1 ✗ | 68.5 ✓ |
| Q3 | N10 (new) | Among the earthquakes within 125 miles of station HV-88 (19.650°N, 155.491°W), what was the highest magnitude recorded? | 3.2 | 3.2 ✓ | 3.2 ✓ | 2.9 ✗ | 3.2 ✓ |
| Q4 | N09 (new) | Rank the earthquakes smaller than magnitude 3 by magnitude, largest first: which event is second? Give its event ID. | ci63518918 | ci63518918 ✓ | hv61367843 ✗ | ci63518918 ✓ | ci63518918 ✓ |
| Q5 | P05 (public) | Which region had more earthquakes of magnitude 1.3 or more: Honshu, Japan or Northern California? Say "tie" if they are equal. | Honshu, Japan | Honshu, Japan ✓ | Honshu, Japan ✓ | Northern California ✗ | Honshu, Japan ✓ |

**Operation chains (the machine-readable meaning of each question):**

- Q1: `dedupe → time in [2026-09-14T14:15:00Z, 2026-09-14T18:05:00Z) → count`
- Q2: `dedupe → mag ≥ 4.2 → max_depth → convert → mi → round(1)`
- Q3: `dedupe → within 125 mi of (19.650, -155.491) → max_mag`
- Q4: `dedupe → mag < 3 → rank_mag(k=2)`
- Q5: `dedupe → mag ≥ 1.3 → compare_regions(Honshu, Japan vs Northern California)`

**Traps in this digest:**

- T1 *duplicate* (cross_ref): AT re-reported nn80947618 as at47434712 with a 'same event as' note: preliminary M1.8, depth 5.5 km (reviewed: M1.5, 6.9 km). If mishandled: Q1 → 4.
- T2 *duplicate* (same_id): PT re-reported hv68175627 under the same event ID: preliminary M0.9, depth 8.3 km (reviewed: M1.1, 7 km). If mishandled: Q1 → 4.
- T3 *duplicate* (same_id): AT re-reported hv61367843 under the same event ID: preliminary M2.5, depth 37.9 km (reviewed: M2.8, 36.5 km). If mishandled: Q4 → hv61367843.
- T4 *retraction* (phantom): phantom report us7884jbg0 (M5.1, Central Chile) later retracted: mine blast. If mishandled: Q1 → 4.
- T5 *conflicting magnitude* (second_report): pr82937817 also reported as M2.9 (preliminary, EW) beside the reviewed M3. If mishandled: Q4 → hv61367843.

<details><summary>Metamorphic variant `v1` of the same instance (same key)</summary>

```text
== QUAKE DIGEST 152-G ==
Window: the 24 hours up to 06:00 UTC on 15 September 2026. Entries are listed as filed by each network.

-- NC network --
[NC] nc20476354 | 19:18 (UTC-08:00) 14 Sep 2026 | Northern California | (39.131, -123.911) | depth 5.4 km | M1.0 (md) | reviewed
2026-09-14T19:13:22Z — A magnitude 1.1 (md) earthquake struck Northern California; epicentre 39.164°N, 122.341°W, depth 1.5 km (0.9 mi). The NC network lists it as nc24064840 (reviewed).
2026-09-14T09:15:23Z ref=nc84330092 net=nc mag=1.3 type=md region="Northern California" loc=38.614,-122.911 depth=4.7 mi status=reviewed
-- EW network --
22:46 UTC on 14 September 2026 — A magnitude 2.9 (md) earthquake struck Puerto Rico; epicentre 18.300°N, 66.834°W, depth 18.6 km (11.6 mi). The EW network lists it as pr82937817 (preliminary).
== CI feed ==
Mon, 14 Sep 2026 20:28:56 +0000 — A magnitude 2.0 (ml) earthquake struck Southern California; epicentre 33°58.40'N 117°14.42'W, depth 12.7 km (7.9 mi). The CI network lists it as ci63518918 (reviewed).
== HV feed ==
[HV] hv83297115 | 14:05 UTC on 14 September 2026 | Hawaii | 19°23.00'N 155°17.51'W | depth 14.4 mi | M3.2 (md) | reviewed
Mon, 14 Sep 2026 15:08:52 +0000 ref=hv68175627 net=hv mag=1.1 type=md region="Hawaii" loc=20.081,-155.291 depth=4.3 mi status=reviewed
[HV] hv75592581 | 2026-09-14T10:39:44Z | Hawaii | 19.361°N, 155.420°W | depth 8.6 mi | M1.0 (md) | reviewed
[HV] hv66757965 | 01:29 UTC on 15 September 2026 | Hawaii | 19°23.61'N 155°14.77'W | depth 34.2 km | M1.5 (md) | reviewed
[HV] hv61367843 | Mon, 14 Sep 2026 08:43:54 +0000 | Hawaii | 19.400°N, 155.300°W | depth 22.7 mi | M2.8 (md) | reviewed
== AT feed ==
[AT] hv61367843 | 2026-09-14 08:43:34 UTC | Hawaii | 19.375°N, 155.311°W | depth 37.9 km (23.5 mi) | MD 2.5 | preliminary
[AT] at47434712 | 2026-09-14T17:48:09Z | Nevada | 38.311°N, 117.868°W | depth 3.4 mi | ML 1.8 | preliminary | same event as nn80947618
From the PR network:
NOTICE [2026-09-14T18:29:10Z]: report us7884jbg0 has been RETRACTED (mine blast). Disregard it.
[PR] pr22281028 | Mon, 14 Sep 2026 12:39:27 +0000 | Puerto Rico | 17.968°N, 67.603°W | depth 15.8 km | MD 1.8 | reviewed
[PR] pr14639407 | 2026-09-14T22:26:55Z | Puerto Rico | 17.973°N, 66.837°W | depth 18.3 km (11.4 mi) | MD 1.0 | reviewed
Station NC.PBR was offline from 12:00 to 12:44 UTC for maintenance; its readings are not included.
06:20 UTC on 14 September 2026 ref=pr57584067 net=pr mag=1.3 type=md region="Puerto Rico" loc=18.053,-66.844 depth=6.5 km (4.0 mi) status=reviewed
[PR] pr96255606 | Tue, 15 Sep 2026 00:04:09 +0000 | Puerto Rico | (18.278, -67.079) | depth 16.2 mi | M1.3 (md) | reviewed
Mon, 14 Sep 2026 22:46:39 +0000 ref=pr82937817 net=pr mag=3.0 type=md region="Puerto Rico" loc=18.300,-66.834 depth=18.6 km status=reviewed
== PT feed ==
2026-09-14 15:09:27 UTC ref=hv68175627 net=pt mag=0.9 type=md region="Hawaii" loc=20.070,-155.286 depth=8.3 km (5.2 mi) status=preliminary
== NN feed ==
NN WIRE [2026-09-14 17:48:33 UTC] ML 1.5 quake in Nevada @ 38.304°N, 117.945°W; 4.3 mi deep. Ref nn80947618, reviewed.
From the US network:
[US] us77588lhm | Mon, 14 Sep 2026 16:56:38 +0000 | Central Chile | 33.464°S, 73.584°W | depth 50.3 mi | MB 5.3 | reviewed
[US] us6577wo0k | 2026-09-14T13:51:53Z | Central Chile | 33.641°S, 72.143°W | depth 68.5 mi | M4.2 (mb) | reviewed
[US] us6510j5sy | 1:45 AM local time (UTC-05:00) on Sep 14, 2026 | Central Chile | 34.374°S, 72.353°W | depth 64.7 km | M4.0 (mb) | reviewed
[US] us6377e32b | 04:10 UTC on 15 September 2026 | Honshu, Japan | (38.361, 141.938) | depth 10.9 mi | M5.4 (mb) | reviewed
[US] us7894bl0f | 07:35 UTC on 14 September 2026 | Central Chile | 32.171°S, 73.243°W | depth 81.0 km | M4.2 (mb) | reviewed
[US] us7884jbg0 | 2026-09-14T17:52:41Z | Central Chile | 34.228°S, 72.034°W | depth 37.3 mi | MB 5.1 | reviewed
[US] us6111meob | Mon, 14 Sep 2026 12:48:28 +0000 | Honshu, Japan | 35.325°N, 140.342°E | depth 49.9 mi | MB 5.2 | reviewed
US WIRE [2026-09-14T21:36:25Z] MB 4.0 quake in Central Chile @ 33.327°S, 71.530°W; 101.8 km deep. Ref us6759vakn, reviewed.
2026-09-15T01:24:38Z — A magnitude 4.2 (mb) earthquake struck Honshu, Japan; epicentre (36.093, 141.799), depth 30.4 km (18.9 mi). The US network lists it as us63988qsy (reviewed).
```

- Q1: How many distinct earthquakes that occurred between 2026-09-14 14:15 UTC and 2026-09-14 18:05 UTC are there?
- Q2: What was the greatest depth, in miles, among the earthquakes with magnitude at least 4.2? Round to 1 decimal place.
- Q3: What was the largest magnitude among the earthquakes no more than 125 miles from station HV-88 (19.650°N, 155.491°W)?
- Q4: Among the earthquakes with magnitude below 3, which one has the second-largest magnitude? Give its event ID.
- Q5: Counting only earthquakes of magnitude 1.3 or greater, which region has more of them: Honshu, Japan or Northern California? Answer "tie" if the counts are equal.

</details>

### Example 2: `inst-007` (24 earthquakes, 29 report lines, 1 notice, 5 questions)

**Digest (original rendering `v0`, exactly as sent to the models):**

```text
SEISMIC ACTIVITY DIGEST (edition 724-G)
Compiled 06:00 UTC on 15 September 2026; covers the 24 hours before compilation. Reports are reproduced as received.

PR WIRE [2026-09-14T06:20:20Z] MD 1.3 quake in Puerto Rico @ 18°03.19'N 66°50.66'W; 6.5 km (4.0 mi) deep. Ref pr57584067, reviewed.
US WIRE [2026-09-14 06:35:15 UTC] MB 4.1 quake in Tonga @ (-20.814, -175.498); 109.1 mi deep. Ref us6257eovm, reviewed.
2026-09-14T07:31:39Z — A magnitude 4.1 (mb) earthquake struck Honshu, Japan; epicentre 37.176°N, 141.997°E, depth 77.3 km. The US network lists it as us7781do06 (reviewed).
[HV] hv61367843 | 2026-09-14 08:43:54 UTC | Hawaii | (19.400, -155.300) | depth 22.7 mi | M2.8 (md) | reviewed
2026-09-14T08:43:54Z — A magnitude 3.6 (md) earthquake struck Hawaii; epicentre 19.400°N, 155.300°W, depth 22.7 mi. The US network lists it as hv61367843 (preliminary).
US WIRE [2026-09-14T09:17:35Z] MB 4.3 quake in Honshu, Japan @ 36.429°N, 141.325°E; 32.6 mi deep. Ref us7140wc40, reviewed.
2026-09-14T11:11:22Z ref=nn73242246 net=nn mag=1.0 type=ml region="Nevada" loc=37.883,-118.334 depth=1.1 mi status=reviewed
Mon, 14 Sep 2026 11:45:45 +0000 ref=us7475uj6u net=us mag=4.2 type=mb region="Honshu, Japan" loc=37.058,141.839 depth=14.2 km status=reviewed
12:31 UTC on 14 September 2026 ref=pr12798603 net=pr mag=1.9 type=md region="Puerto Rico" loc=17.970,-66.977 depth=4.2 km status=reviewed
No tsunami threat was issued for any event in this digest.
[PR] pr22281028 | 2026-09-14 12:39:27 UTC | Puerto Rico | (17.968, -67.603) | depth 15.8 km | M1.8 (md) | reviewed
US WIRE [2026-09-14T12:48:28Z] MB 5.2 quake in Honshu, Japan @ 35°19.49'N 140°20.50'E; 49.9 mi deep. Ref us6111meob, reviewed.
2026-09-14T12:48:48Z — A magnitude 5.1 (mb) earthquake struck Honshu, Japan; epicentre (35.349, 140.386), depth 77.3 km. The AT network lists it as us6111meob (preliminary).
US WIRE [15:01 UTC on 14 September 2026] MB 4.2 quake in Honshu, Japan @ 37°45.23'N 141°30.32'E; 80.8 km deep. Ref us7024hywc, reviewed.
2026-09-14T16:16:49Z — A magnitude 1.8 (md) earthquake struck Hawaii; epicentre 19.399°N, 155.307°W, depth 37.9 km. The HV network lists it as hv69675565 (reviewed).
17:35 UTC on 14 September 2026 ref=nc13675307 net=nc mag=1.4 type=md region="Northern California" loc=38.977,-122.848 depth=4.8 mi status=reviewed
18:22 UTC on 14 September 2026 ref=nn85588387 net=nn mag=3.4 type=ml region="Nevada" loc=38.653,-117.975 depth=8.3 km (5.2 mi) status=reviewed
20:18 UTC on 14 September 2026 ref=us7245nhmb net=us mag=5.0 type=mb region="Honshu, Japan" loc=37.397,141.727 depth=10.5 km status=reviewed
[HV] hv60607320 | 21:02 UTC on 14 September 2026 | Hawaii | 19°22.70'N 155°11.83'W | depth 17.4 km | MD 1.8 | reviewed
2026-09-14T21:29:56Z ref=us7074sz48 net=us mag=4.8 type=mb region="Tonga" loc=-21.436,-174.182 depth=173.0 mi status=reviewed
2026-09-14T22:32:39Z ref=us6711p1ht net=us mag=4.5 type=mb region="Honshu, Japan" loc=38.080,141.416 depth=11.9 km (7.4 mi) status=reviewed
2026-09-14T22:43:56Z — A magnitude 1.7 (md) earthquake struck Northern California; epicentre 38.906°N, 122.584°W, depth 1.0 km. The NC network lists it as nc51399576 (reviewed).
PR WIRE [22:46 UTC on 14 September 2026] MD 3.0 quake in Puerto Rico @ 18.300°N, 66.834°W; 18.6 km (11.6 mi) deep. Ref pr82937817, reviewed.
2026-09-15T01:39:01Z ref=nn59427457 net=nn mag=1.2 type=ml region="Nevada" loc=38.613,-119.090 depth=1.4 mi status=reviewed
2026-09-15T03:42:12Z ref=us6711a2wk net=us mag=4.9 type=mb region="Honshu, Japan" loc=36.504,142.175 depth=6.4 mi status=reviewed
PT WIRE [2026-09-15 03:42:30 UTC] MB 4.8 quake in Honshu, Japan @ 36.433°N, 142.166°E; 10.2 km (6.3 mi) deep. Ref pt32682811, preliminary. Dup of us6711a2wk.
2026-09-15 03:58:56 UTC — A magnitude 2.0 (ml) earthquake struck Nevada; epicentre (38.559, -117.963), depth 1.1 mi. The NN network lists it as nn95709561 (reviewed).
2026-09-15 04:10:03 UTC — A magnitude 5.4 (mb) earthquake struck Honshu, Japan; epicentre 38.361°N, 141.938°E, depth 10.9 mi. The US network lists it as us6377e32b (reviewed).
Reminder: networks report on different magnitude scales (md, ml, mb, mw); values are not converted.
2026-09-15T04:45:05Z ref=us6751tr1y net=us mag=4.2 type=mb region="Tonga" loc=-20.457,-173.740 depth=197.1 km (122.5 mi) status=reviewed
[EW] ew28349705 | 2026-09-15T04:45:30Z | Tonga | 20°25.15'S 173°45.07'W | depth 194.7 km (121.0 mi) | MB 3.9 | preliminary | same event as us6751tr1y

Corrections and retractions:
UPDATE [Mon, 14 Sep 2026 13:13:09 +0000]: us7475uj6u withdrawn (sonic boom).
```

**Questions, code-computed key and model answers (`v0`):**

| Q | template | question | key | oracle | naive | mock | anthropic:claude-opus-5-5 |
|---|---|---|---|---|---|---|---|
| Q1 | N11 (new) | Among the earthquakes of magnitude 3.4 or more, how deep was the shallowest one, in miles? Round to 1 decimal place. | 5.2 | 5.2 ✓ | 5.2 ✓ | 5.2 ✓ | 5.2 ✓ |
| Q2 | P11 (public) | What was the smallest depth, in km, among the earthquakes that occurred between 2026-09-14 13:30 UTC and 2026-09-15 01:20 UTC? Round to 1 decimal place. | 1.0 | 1.0 ✓ | 1.0 ✓ | 1.0 ✓ | 1.0 ✓ |
| Q3 | P03 (public) | What is the event ID of the strongest of the earthquakes in Nevada? | nn85588387 | nn85588387 ✓ | nn85588387 ✓ | nn85588387 ✓ | nn85588387 ✓ |
| Q4 | P01 (public) | How many distinct earthquakes of magnitude 3.4 or more are there? | 12 | 12 ✓ | 17 ✗ | 12 ✓ | 12 ✓ |
| Q5 | N04 (new) | Among the earthquakes that occurred between 16:50 UTC on 14 September 2026 and 20:55 UTC on 14 September 2026, how deep was the deepest one, in miles? Round to 1 decimal place. | 6.5 | 6.5 ✓ | 6.5 ✓ | 4.5 ✗ | 6.5 ✓ |

**Operation chains (the machine-readable meaning of each question):**

- Q1: `dedupe → mag ≥ 3.4 → min_depth → convert → mi → round(1)`
- Q2: `dedupe → time in [2026-09-14T13:30:00Z, 2026-09-15T01:20:00Z) → min_depth → convert → km → round(1)`
- Q3: `dedupe → region = Nevada → argmax_mag`
- Q4: `dedupe → mag ≥ 3.4 → count`
- Q5: `dedupe → time in [2026-09-14T16:50:00Z, 2026-09-14T20:55:00Z) → max_depth → convert → mi → round(1)`

**Traps in this digest:**

- T1 *duplicate* (cross_ref): EW re-reported us6751tr1y as ew28349705 with a 'same event as' note: preliminary M3.9, depth 194.7 km (reviewed: M4.2, 197.1 km). If mishandled: Q4 → 13.
- T2 *duplicate* (cross_ref): PT re-reported us6711a2wk as pt32682811 with a 'same event as' note: preliminary M4.8, depth 10.2 km (reviewed: M4.9, 10.3 km). If mishandled: Q4 → 13.
- T3 *duplicate* (same_id): AT re-reported us6111meob under the same event ID: preliminary M5.1, depth 77.3 km (reviewed: M5.2, 80.3 km). If mishandled: Q4 → 13.
- T4 *retraction* (phantom): phantom report us7475uj6u (M4.2, Honshu, Japan) later retracted: sonic boom. If mishandled: Q4 → 13.
- T5 *conflicting magnitude* (second_report): hv61367843 also reported as M3.6 (preliminary, US) beside the reviewed M2.8. If mishandled: Q4 → 13.

<details><summary>Metamorphic variant `v1` of the same instance (same key)</summary>

```text
SEISMIC ACTIVITY DIGEST (edition 799-A)
Compiled Tue, 15 Sep 2026 06:00:00 +0000; covers the 24 hours before compilation. Reports are reproduced as received.

== US feed ==
05:18 (UTC+09:00) 15 Sep 2026 — A magnitude 5.0 (mb) earthquake struck Honshu, Japan; epicentre 37.397°N, 141.727°E, depth 6.5 mi. The US network lists it as us7245nhmb (reviewed).
9:48 PM local time (UTC+09:00) on Sep 14, 2026 ref=us6111meob net=us mag=5.2 type=mb region="Honshu, Japan" loc=35.325,140.342 depth=80.3 km (49.9 mi) status=reviewed
2026-09-14 08:43:54 UTC ref=hv61367843 net=us mag=3.6 type=md region="Hawaii" loc=19.400,-155.300 depth=22.7 mi status=preliminary
09:29 (UTC-12:00) 14 Sep 2026 ref=us7074sz48 net=us mag=4.8 type=mb region="Tonga" loc=-21.436,-174.182 depth=173.0 mi status=reviewed
[US] us6711a2wk | Tue, 15 Sep 2026 03:42:12 +0000 | Honshu, Japan | (36.504, 142.175) | depth 6.4 mi | M4.9 (mb) | reviewed
12:01 AM local time (UTC+09:00) on Sep 15, 2026 ref=us7024hywc net=us mag=4.2 type=mb region="Honshu, Japan" loc=37.754,141.505 depth=80.8 km (50.2 mi) status=reviewed
2026-09-14 22:32:39 UTC ref=us6711p1ht net=us mag=4.5 type=mb region="Honshu, Japan" loc=38.080,141.416 depth=7.4 mi status=reviewed
2026-09-15T04:45:05Z — A magnitude 4.2 (mb) earthquake struck Tonga; epicentre 20.457°S, 173.740°W, depth 122.5 mi. The US network lists it as us6751tr1y (reviewed).
1:10 PM local time (UTC+09:00) on Sep 15, 2026 ref=us6377e32b net=us mag=5.4 type=mb region="Honshu, Japan" loc=38.361,141.938 depth=17.6 km status=reviewed
[US] us7781do06 | 16:31 (UTC+09:00) 14 Sep 2026 | Honshu, Japan | 37.176°N, 141.997°E | depth 48.0 mi | MB 4.1 | reviewed
06:35 UTC on 14 September 2026 ref=us6257eovm net=us mag=4.1 type=mb region="Tonga" loc=-20.814,-175.498 depth=109.1 mi status=reviewed
2026-09-14T11:45:45Z — A magnitude 4.2 (mb) earthquake struck Honshu, Japan; epicentre 37.058°N, 141.839°E, depth 8.8 mi. The US network lists it as us7475uj6u (reviewed).
2026-09-14 09:17:35 UTC — A magnitude 4.3 (mb) earthquake struck Honshu, Japan; epicentre 36.429°N, 141.325°E, depth 32.6 mi. The US network lists it as us7140wc40 (reviewed).
From the PT network:
03:42 UTC on 15 September 2026 — A magnitude 4.8 (mb) earthquake struck Honshu, Japan; epicentre 36.433°N, 142.166°E, depth 6.3 mi. The PT network lists it as pt32682811 (preliminary). This is the same earthquake as us6711a2wk.
From the PR network:
[PR] pr12798603 | 2026-09-14T12:31:05Z | Puerto Rico | 17.970°N, 66.977°W | depth 4.2 km | MD 1.9 | reviewed
Mon, 14 Sep 2026 22:46:39 +0000 — A magnitude 3.0 (md) earthquake struck Puerto Rico; epicentre 18.300°N, 66.834°W, depth 11.6 mi. The PR network lists it as pr82937817 (reviewed).
2026-09-14 06:20:20 UTC — A magnitude 1.3 (md) earthquake struck Puerto Rico; epicentre 18.053°N, 66.844°W, depth 4.0 mi. The PR network lists it as pr57584067 (reviewed).
2026-09-14 12:39:27 UTC ref=pr22281028 net=pr mag=1.8 type=md region="Puerto Rico" loc=17.968,-67.603 depth=15.8 km status=reviewed
-- EW network --
UPDATE [13:13 UTC on 14 September 2026]: us7475uj6u withdrawn (sonic boom).
4:45 PM local time (UTC-12:00) on Sep 14, 2026 — A magnitude 3.9 (mb) earthquake struck Tonga; epicentre 20.419°S, 173.751°W, depth 121.0 mi. The EW network lists it as ew28349705 (preliminary). This is the same earthquake as us6751tr1y.
== NC feed ==
9:35 AM local time (UTC-08:00) on Sep 14, 2026 — A magnitude 1.4 (md) earthquake struck Northern California; epicentre 38.977°N, 122.848°W, depth 4.8 mi. The NC network lists it as nc13675307 (reviewed).
[NC] nc51399576 | Mon, 14 Sep 2026 22:43:56 +0000 | Northern California | 38.906°N, 122.584°W | depth 1.0 km | M1.7 (md) | reviewed
-- AT network --
9:48 PM local time (UTC+09:00) on Sep 14, 2026 ref=us6111meob net=at mag=5.1 type=mb region="Honshu, Japan" loc=35.349,140.386 depth=77.3 km (48.0 mi) status=preliminary
-- NN network --
10:22 AM local time (UTC-08:00) on Sep 14, 2026 — A magnitude 3.4 (ml) earthquake struck Nevada; epicentre 38.653°N, 117.975°W, depth 5.2 mi. The NN network lists it as nn85588387 (reviewed).
7:58 PM local time (UTC-08:00) on Sep 14, 2026 — A magnitude 2.0 (ml) earthquake struck Nevada; epicentre 38.559°N, 117.963°W, depth 1.1 mi. The NN network lists it as nn95709561 (reviewed).
5:39 PM local time (UTC-08:00) on Sep 14, 2026 — A magnitude 1.2 (ml) earthquake struck Nevada; epicentre 38.613°N, 119.090°W, depth 1.4 mi. The NN network lists it as nn59427457 (reviewed).
NN WIRE [03:11 (UTC-08:00) 14 Sep 2026] ML 1.0 quake in Nevada @ 37.883°N, 118.334°W; 1.1 mi deep. Ref nn73242246, reviewed.
== HV feed ==
11:02 AM local time (UTC-10:00) on Sep 14, 2026 — A magnitude 1.8 (md) earthquake struck Hawaii; epicentre 19°22.70'N 155°11.83'W, depth 10.8 mi. The HV network lists it as hv60607320 (reviewed).
Reminder: networks report on different magnitude scales (md, ml, mb, mw); values are not converted.
[HV] hv69675565 | 2026-09-14T16:16:49Z | Hawaii | 19.399°N, 155.307°W | depth 37.9 km | M1.8 (md) | reviewed
08:43 UTC on 14 September 2026 ref=hv61367843 net=hv mag=2.8 type=md region="Hawaii" loc=19.400,-155.300 depth=36.5 km (22.7 mi) status=reviewed
No tsunami threat was issued for any event in this digest.
```

- Q1: What was the smallest depth, in miles, among the earthquakes of magnitude 3.4 or more? Round to 1 decimal place.
- Q2: Among the earthquakes that were recorded from 2026-09-14 13:30 UTC up to 2026-09-15 01:20 UTC, how deep was the shallowest one, in km? Round to 1 decimal place.
- Q3: What is the event ID of the strongest of the earthquakes located in Nevada?
- Q4: Count the distinct earthquakes of magnitude 3.4 or greater.
- Q5: Among the earthquakes that occurred between 2026-09-14 16:50 UTC and 2026-09-14 20:55 UTC, how deep was the deepest one, in miles? Round to 1 decimal place.

</details>
