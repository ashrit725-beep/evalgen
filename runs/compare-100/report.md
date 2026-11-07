# Live evaluation report: `compare-100`

Every instance in this run was generated fresh from the seed below, every answer key was computed by code from the frozen snapshot, and every score was computed automatically. No item was labelled by hand.

## Run

- **Generated with:** evalgen 0.1.0 on 2026-11-03T23:57:41Z (the reading rules and templates are versioned with the code).
- **Master seed:** `13762304563634797857` (commitment `4475452b7ef612b1d1402a20…`); every derived seed is listed in `manifest.json`.
- **Data source:** synthetic; 360 events in the snapshot pool; window: 24 h ending 2026-02-08T06:00:00Z.
- **Snapshot SHA-256:** `444b4412fb67ae2bd6ff94b2b355ce0f085480c8d8795efe3bb67c6c2d8bee64`
- **⚠ Note:** the evalgen code has changed since this run was generated (with evalgen 0.1.0, code `2252be93d87ad04c…`). This report scores the stored prompts and replies; regenerating the run from its seed reproduces it exactly only with the code that generated it.
- **Instances:** 100; **questions:** 411 (208 from public templates, 203 from new-this-run templates); **renderings per instance:** 3 (original + 2 metamorphic variants); **prompts:** 300.
- **Traps planted:** 445 (duplicate 245, retraction 100, conflicting magnitude 100); 417 are load-bearing: mishandling them changes the answers of 213 of the 411 questions (469 trap–question pairs). The rest stay in as plain noise.

## Model accuracy (original renderings)

| model | kind | accuracy | 95% CI | mean instance score | accuracy, all renderings | JSON strict / lenient / failed (API errors) |
|---|---|---|---|---|---|---|
| oracle | reference | 100.0% (411/411) | 99–100% | 100.0% | 100.0% | 300 / 0 / 0 (0) |
| naive | reference | 50.6% (208/411) | 46–55% | 49.2% | 50.6% | 300 / 0 / 0 (0) |
| mock | simulated | 76.9% (316/411) | 73–81% | 76.3% | 73.6% | 240 / 60 / 0 (0) |
| anthropic:claude-haiku-5-5 | api | 99.0% (407/411) | 98–100% | 98.9% | 98.3% | 300 / 0 / 0 (0) |
| anthropic:claude-opus-5-5 | api | 99.8% (410/411) | 99–100% | 99.8% | 99.8% | 300 / 0 / 0 (0) |

`oracle` reads only the digest text and applies the reading rules, so it should score 100%. That shows every item is answerable from the text. `naive` treats every line as a separate quake and ignores notices. `mock` is a *simulated* model with built-in benchmark contamination, used to exercise the metrics without API keys; it is not a language model.

## Head to head (paired)

The API models answered the same questions, so they are compared question by question, with questions as the independent units. Only the questions that exactly one model got right say which is better; the exact McNemar test asks whether those split evenly. The first row uses the original renderings. The second counts a question as right only if every rendering of it was answered correctly, so it also rewards robustness to surface changes (units, formats, line order, wording).

| A | B | right on | questions | both right | only A right | only B right | both wrong | A minus B (95% CI) | McNemar p |
|---|---|---|---|---|---|---|---|---|---|
| anthropic:claude-opus-5-5 | anthropic:claude-haiku-5-5 | original rendering | 411 | 406 | 4 | 1 | 0 | +0.7 pts (-0.4 to +1.9) | 0.375 |
| anthropic:claude-opus-5-5 | anthropic:claude-haiku-5-5 | every rendering | 411 | 389 | 20 | 2 | 0 | +4.4 pts (+2.1 to +6.6) | 1.2e-04 |

## API usage

Tokens as each API reported them, for the replies that were scored (an attempt that failed or was cut off and then asked again is not counted). Output tokens include any thinking before the answer.

| model | replies | input tokens | output tokens | median output tokens per reply | median latency |
|---|---|---|---|---|---|
| anthropic:claude-haiku-5-5 | 300 | 1,041,118 | 1,438,004 | 4,599 | 18 s |
| anthropic:claude-opus-5-5 | 300 | 1,041,118 | 439,377 | 1,424 | 12 s |

## Metamorphic consistency

Each instance is re-rendered with the same snapshot but different units, line order, formats and question wording. Correct answers cannot change, so any change in a model's answer is an inconsistency. This needs no ground truth.

| model | consistent answer pairs | 95% CI | questions consistent across all variants |
|---|---|---|---|
| oracle | 100.0% (822/822) | 100–100% | 100.0% (411/411) |
| naive | 100.0% (822/822) | 100–100% | 100.0% (411/411) |
| mock | 59.1% (486/822) | 56–62% | 41.4% (170/411) |
| anthropic:claude-haiku-5-5 | 97.2% (799/822) | 96–98% | 95.1% (391/411) |
| anthropic:claude-opus-5-5 | 99.6% (819/822) | 99–100% | 99.5% (409/411) |

## Seen vs new templates

Public templates are published (a model may have trained on them). Each new template is composed for this run as a twin of one public template: the same final operation and number of filters, but a different filter combination, and each twin pair is used equally often. The *adjusted* gap compares the pools like for like, within strata of equal code-computed difficulty (how many traps change the answer, and whether the filters matter), so that the pools' different trap exposure is not mistaken for overfitting. A large positive adjusted gap flags overfitting to the published skeletons (flag: > 10 points with a 95% CI that excludes 0). It is a flag, not proof.

| model | public (seen) | new this run | raw gap (95% CI) | adjusted gap (95% CI) | flag |
|---|---|---|---|---|---|
| oracle | 100.0% (208/208) | 100.0% (203/203) | +0.0 pts (+0 to +0) | +0.0 pts (-2 to +2) | — |
| naive | 52.9% (110/208) | 48.3% (98/203) | +4.6 pts (-5 to +14) | +0.7 pts (-3 to +4) | — |
| mock | 90.9% (189/208) | 62.6% (127/203) | +28.3 pts (+21 to +36) | +28.8 pts (+21 to +37) | ⚠ overfitting? |
| anthropic:claude-haiku-5-5 | 99.0% (206/208) | 99.0% (201/203) | +0.0 pts (-2 to +2) | -0.2 pts (-3 to +3) | — |
| anthropic:claude-opus-5-5 | 100.0% (208/208) | 99.5% (202/203) | +0.5 pts (-0 to +1) | +0.5 pts (-2 to +3) | — |

## Trap catch rate

Counted over (load-bearing trap, affected question, rendering) triples. *Caught* = answered correctly; *fell* = gave exactly an answer a reader produces by missing that trap (alone or together with other traps).

| model | caught (overall) | duplicate | retraction | conflicting magnitude | fell for the trap |
|---|---|---|---|---|---|
| oracle | 100.0% (1407/1407) | 100.0% | 100.0% | 100.0% | 0.0% (0/1407) |
| naive | 2.8% (39/1407) | 1.2% | 0.0% | 9.3% | 97.2% (1368/1407) |
| mock | 72.3% (1017/1407) | 72.2% | 70.6% | 74.1% | 20.9% (294/1407) |
| anthropic:claude-haiku-5-5 | 96.2% (1353/1407) | 96.2% | 96.0% | 96.3% | 0.9% (12/1407) |
| anthropic:claude-opus-5-5 | 100.0% (1407/1407) | 100.0% | 100.0% | 100.0% | 0.0% (0/1407) |

## Accuracy by answer type (original renderings)

| model | count | event_id | magnitude | depth | distance | region | utc_time |
|---|---|---|---|---|---|---|---|
| oracle | 100.0% (98/98) | 100.0% (107/107) | 100.0% (29/29) | 100.0% (76/76) | 100.0% (47/47) | 100.0% (34/34) | 100.0% (20/20) |
| naive | 13.3% (13/98) | 54.2% (58/107) | 51.7% (15/29) | 63.2% (48/76) | 66.0% (31/47) | 85.3% (29/34) | 70.0% (14/20) |
| mock | 85.7% (84/98) | 73.8% (79/107) | 82.8% (24/29) | 68.4% (52/76) | 85.1% (40/47) | 70.6% (24/34) | 65.0% (13/20) |
| anthropic:claude-haiku-5-5 | 96.9% (95/98) | 100.0% (107/107) | 100.0% (29/29) | 100.0% (76/76) | 97.9% (46/47) | 100.0% (34/34) | 100.0% (20/20) |
| anthropic:claude-opus-5-5 | 99.0% (97/98) | 100.0% (107/107) | 100.0% (29/29) | 100.0% (76/76) | 100.0% (47/47) | 100.0% (34/34) | 100.0% (20/20) |

## Generator audit

- **Automated self-check:** for 1233/1233 question-renderings, the digest-only oracle agreed with the code-computed key. Applying the reading rules to each digest recovered the snapshot exactly for 100/100 instances.
- **Human spot-check sample:** 9 of 411 generated items (2.2%; target 2.0%) → `audit/audit_sample.jsonl` (checklist: `audit/audit_sample.md`).
- **Verdicts:** 0/9 reviewed so far. Fill in `verdict` (ok / error) in the JSONL file and re-run `cli.py report` to get an estimated generator error rate.

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
| N03 | new | P03 | dedupe → depth_shallower_than → argmax_mag |
| N04 | new | P04 | dedupe → in_region → max_depth |
| N05 | new | P05 | dedupe → mag_below → compare_regions |
| N06 | new | P06 | dedupe → mag_at_least → depth_at_least → count |
| N07 | new | P07 | dedupe → time_between → nearest_to |
| N08 | new | P08 | dedupe → depth_at_least → time_of_largest |
| N09 | new | P09 | dedupe → mag_below → rank_mag |
| N10 | new | P10 | dedupe → mag_below → max_mag |
| N11 | new | P11 | dedupe → mag_below → min_depth |
| N12 | new | P12 | dedupe → mag_at_least → nearest_distance |

## Per-instance scores (original rendering)

<details><summary>Show table</summary>

| instance | questions | load-bearing traps | oracle | naive | mock | anthropic:claude-haiku-5-5 | anthropic:claude-opus-5-5 |
|---|---|---|---|---|---|---|---|
| inst-000 | 5 | 4 | 100.0% | 60.0% | 80.0% | 100.0% | 100.0% |
| inst-001 | 5 | 4 | 100.0% | 40.0% | 80.0% | 100.0% | 100.0% |
| inst-002 | 4 | 5 | 100.0% | 50.0% | 75.0% | 100.0% | 100.0% |
| inst-003 | 3 | 5 | 100.0% | 33.3% | 100.0% | 100.0% | 100.0% |
| inst-004 | 4 | 4 | 100.0% | 50.0% | 75.0% | 100.0% | 100.0% |
| inst-005 | 4 | 5 | 100.0% | 50.0% | 50.0% | 100.0% | 100.0% |
| inst-006 | 3 | 3 | 100.0% | 0.0% | 100.0% | 100.0% | 100.0% |
| inst-007 | 3 | 1 | 100.0% | 66.7% | 66.7% | 100.0% | 100.0% |
| inst-008 | 5 | 4 | 100.0% | 40.0% | 80.0% | 100.0% | 100.0% |
| inst-009 | 4 | 3 | 100.0% | 25.0% | 25.0% | 100.0% | 100.0% |
| inst-010 | 4 | 4 | 100.0% | 50.0% | 100.0% | 100.0% | 100.0% |
| inst-011 | 5 | 5 | 100.0% | 60.0% | 40.0% | 100.0% | 100.0% |
| inst-012 | 5 | 5 | 100.0% | 20.0% | 80.0% | 100.0% | 100.0% |
| inst-013 | 4 | 5 | 100.0% | 25.0% | 75.0% | 100.0% | 100.0% |
| inst-014 | 3 | 4 | 100.0% | 33.3% | 100.0% | 100.0% | 100.0% |
| inst-015 | 4 | 4 | 100.0% | 50.0% | 75.0% | 100.0% | 100.0% |
| inst-016 | 3 | 5 | 100.0% | 66.7% | 66.7% | 100.0% | 100.0% |
| inst-017 | 4 | 4 | 100.0% | 50.0% | 100.0% | 100.0% | 100.0% |
| inst-018 | 5 | 1 | 100.0% | 80.0% | 80.0% | 100.0% | 100.0% |
| inst-019 | 4 | 4 | 100.0% | 50.0% | 75.0% | 100.0% | 100.0% |
| inst-020 | 4 | 5 | 100.0% | 75.0% | 75.0% | 100.0% | 100.0% |
| inst-021 | 5 | 4 | 100.0% | 80.0% | 80.0% | 100.0% | 100.0% |
| inst-022 | 3 | 4 | 100.0% | 0.0% | 66.7% | 100.0% | 100.0% |
| inst-023 | 4 | 5 | 100.0% | 50.0% | 50.0% | 100.0% | 100.0% |
| inst-024 | 5 | 4 | 100.0% | 20.0% | 60.0% | 100.0% | 100.0% |
| inst-025 | 5 | 5 | 100.0% | 40.0% | 80.0% | 80.0% | 100.0% |
| inst-026 | 4 | 4 | 100.0% | 50.0% | 100.0% | 100.0% | 100.0% |
| inst-027 | 3 | 4 | 100.0% | 33.3% | 33.3% | 100.0% | 100.0% |
| inst-028 | 3 | 5 | 100.0% | 66.7% | 66.7% | 100.0% | 100.0% |
| inst-029 | 5 | 5 | 100.0% | 60.0% | 60.0% | 100.0% | 100.0% |
| inst-030 | 4 | 4 | 100.0% | 50.0% | 75.0% | 100.0% | 100.0% |
| inst-031 | 4 | 5 | 100.0% | 50.0% | 75.0% | 100.0% | 100.0% |
| inst-032 | 5 | 5 | 100.0% | 80.0% | 100.0% | 100.0% | 100.0% |
| inst-033 | 4 | 4 | 100.0% | 25.0% | 75.0% | 100.0% | 100.0% |
| inst-034 | 5 | 4 | 100.0% | 60.0% | 100.0% | 100.0% | 100.0% |
| inst-035 | 5 | 4 | 100.0% | 40.0% | 80.0% | 100.0% | 100.0% |
| inst-036 | 5 | 4 | 100.0% | 80.0% | 100.0% | 100.0% | 100.0% |
| inst-037 | 5 | 4 | 100.0% | 40.0% | 80.0% | 100.0% | 100.0% |
| inst-038 | 3 | 5 | 100.0% | 0.0% | 100.0% | 100.0% | 100.0% |
| inst-039 | 4 | 4 | 100.0% | 50.0% | 75.0% | 100.0% | 100.0% |
| inst-040 | 5 | 4 | 100.0% | 80.0% | 80.0% | 100.0% | 100.0% |
| inst-041 | 3 | 4 | 100.0% | 33.3% | 100.0% | 100.0% | 100.0% |
| inst-042 | 4 | 4 | 100.0% | 50.0% | 75.0% | 100.0% | 100.0% |
| inst-043 | 4 | 4 | 100.0% | 75.0% | 100.0% | 100.0% | 100.0% |
| inst-044 | 5 | 5 | 100.0% | 60.0% | 60.0% | 100.0% | 100.0% |
| inst-045 | 3 | 5 | 100.0% | 33.3% | 66.7% | 100.0% | 100.0% |
| inst-046 | 3 | 5 | 100.0% | 33.3% | 66.7% | 100.0% | 100.0% |
| inst-047 | 3 | 2 | 100.0% | 33.3% | 66.7% | 100.0% | 100.0% |
| inst-048 | 5 | 4 | 100.0% | 40.0% | 80.0% | 100.0% | 100.0% |
| inst-049 | 5 | 5 | 100.0% | 80.0% | 60.0% | 100.0% | 100.0% |
| inst-050 | 3 | 5 | 100.0% | 66.7% | 100.0% | 100.0% | 100.0% |
| inst-051 | 5 | 4 | 100.0% | 80.0% | 100.0% | 100.0% | 80.0% |
| inst-052 | 5 | 4 | 100.0% | 40.0% | 60.0% | 100.0% | 100.0% |
| inst-053 | 5 | 4 | 100.0% | 40.0% | 40.0% | 100.0% | 100.0% |
| inst-054 | 4 | 4 | 100.0% | 25.0% | 0.0% | 100.0% | 100.0% |
| inst-055 | 4 | 5 | 100.0% | 50.0% | 75.0% | 100.0% | 100.0% |
| inst-056 | 4 | 5 | 100.0% | 50.0% | 100.0% | 100.0% | 100.0% |
| inst-057 | 3 | 3 | 100.0% | 33.3% | 33.3% | 100.0% | 100.0% |
| inst-058 | 4 | 4 | 100.0% | 25.0% | 75.0% | 100.0% | 100.0% |
| inst-059 | 4 | 4 | 100.0% | 25.0% | 75.0% | 100.0% | 100.0% |
| inst-060 | 5 | 5 | 100.0% | 60.0% | 100.0% | 100.0% | 100.0% |
| inst-061 | 5 | 4 | 100.0% | 40.0% | 80.0% | 100.0% | 100.0% |
| inst-062 | 3 | 5 | 100.0% | 33.3% | 100.0% | 66.7% | 100.0% |
| inst-063 | 3 | 4 | 100.0% | 33.3% | 66.7% | 100.0% | 100.0% |
| inst-064 | 4 | 4 | 100.0% | 50.0% | 75.0% | 100.0% | 100.0% |
| inst-065 | 4 | 5 | 100.0% | 50.0% | 75.0% | 100.0% | 100.0% |
| inst-066 | 3 | 2 | 100.0% | 66.7% | 66.7% | 100.0% | 100.0% |
| inst-067 | 5 | 2 | 100.0% | 80.0% | 100.0% | 100.0% | 100.0% |
| inst-068 | 3 | 5 | 100.0% | 33.3% | 66.7% | 100.0% | 100.0% |
| inst-069 | 5 | 5 | 100.0% | 20.0% | 100.0% | 100.0% | 100.0% |
| inst-070 | 4 | 4 | 100.0% | 75.0% | 100.0% | 100.0% | 100.0% |
| inst-071 | 4 | 5 | 100.0% | 50.0% | 75.0% | 100.0% | 100.0% |
| inst-072 | 5 | 5 | 100.0% | 60.0% | 60.0% | 100.0% | 100.0% |
| inst-073 | 4 | 5 | 100.0% | 25.0% | 100.0% | 100.0% | 100.0% |
| inst-074 | 5 | 4 | 100.0% | 60.0% | 100.0% | 100.0% | 100.0% |
| inst-075 | 5 | 4 | 100.0% | 60.0% | 80.0% | 100.0% | 100.0% |
| inst-076 | 3 | 4 | 100.0% | 33.3% | 33.3% | 100.0% | 100.0% |
| inst-077 | 5 | 4 | 100.0% | 60.0% | 100.0% | 100.0% | 100.0% |
| inst-078 | 4 | 5 | 100.0% | 50.0% | 75.0% | 100.0% | 100.0% |
| inst-079 | 3 | 4 | 100.0% | 33.3% | 100.0% | 100.0% | 100.0% |
| inst-080 | 5 | 4 | 100.0% | 60.0% | 80.0% | 100.0% | 100.0% |
| inst-081 | 5 | 5 | 100.0% | 60.0% | 100.0% | 100.0% | 100.0% |
| inst-082 | 5 | 4 | 100.0% | 60.0% | 100.0% | 100.0% | 100.0% |
| inst-083 | 4 | 5 | 100.0% | 50.0% | 75.0% | 100.0% | 100.0% |
| inst-084 | 4 | 4 | 100.0% | 75.0% | 75.0% | 100.0% | 100.0% |
| inst-085 | 5 | 4 | 100.0% | 60.0% | 60.0% | 100.0% | 100.0% |
| inst-086 | 3 | 4 | 100.0% | 66.7% | 66.7% | 100.0% | 100.0% |
| inst-087 | 3 | 2 | 100.0% | 33.3% | 66.7% | 100.0% | 100.0% |
| inst-088 | 5 | 4 | 100.0% | 60.0% | 40.0% | 80.0% | 100.0% |
| inst-089 | 4 | 5 | 100.0% | 75.0% | 100.0% | 100.0% | 100.0% |
| inst-090 | 5 | 5 | 100.0% | 60.0% | 100.0% | 100.0% | 100.0% |
| inst-091 | 5 | 5 | 100.0% | 80.0% | 100.0% | 100.0% | 100.0% |
| inst-092 | 3 | 4 | 100.0% | 66.7% | 66.7% | 66.7% | 100.0% |
| inst-093 | 3 | 3 | 100.0% | 0.0% | 100.0% | 100.0% | 100.0% |
| inst-094 | 5 | 3 | 100.0% | 60.0% | 80.0% | 100.0% | 100.0% |
| inst-095 | 3 | 4 | 100.0% | 66.7% | 33.3% | 100.0% | 100.0% |
| inst-096 | 3 | 4 | 100.0% | 33.3% | 66.7% | 100.0% | 100.0% |
| inst-097 | 3 | 4 | 100.0% | 66.7% | 66.7% | 100.0% | 100.0% |
| inst-098 | 5 | 4 | 100.0% | 60.0% | 60.0% | 100.0% | 100.0% |
| inst-099 | 4 | 4 | 100.0% | 50.0% | 75.0% | 100.0% | 100.0% |

</details>

## Example instances

### Example 1: `inst-011` (22 earthquakes, 27 report lines, 1 notice, 5 questions)

**Digest (original rendering `v0`, exactly as sent to the models):**

```text
== QUAKE DIGEST 365-H ==
Window: the 24 hours up to 2026-02-08T06:00:00Z. Entries are listed as filed by each network.

[NC] nc41072473 | 2026-02-07T06:38:18Z | Northern California | 39.377°N, 122.852°W | depth 1.1 km | M1.1 (md) | reviewed
AK WIRE [22:38 (UTC-10:00) 6 Feb 2026] ML 0.9 quake in Southern Alaska @ 61.134°N, 153.006°W; 17.2 km deep. Ref ak97220386, reviewed.
US WIRE [09:34 UTC on 7 February 2026] MB 4.2 quake in Central Chile @ 33°08.78'S 71°18.96'W; 33.0 km deep. Ref us7966hqsu, reviewed.
6:14 AM local time (UTC-05:00) on Feb 7, 2026 — A magnitude 1.5 (mb) earthquake struck Central Chile; epicentre 33°07.47'S 71°54.03'W, depth 110.6 km. The AT network lists it as us673420ol (preliminary).
Station AK.ATF was offline from 19:00 to 19:37 UTC for maintenance; its readings are not included.
[US] us673420ol | 6:14 AM local time (UTC-05:00) on Feb 7, 2026 | Central Chile | 33.124°S, 71.900°W | depth 110.6 km | MB 4.2 | reviewed
PT WIRE [3:26 AM local time (UTC-08:00) on Feb 7, 2026] ML 0.9 quake in Nevada @ 39°04.90'N 117°38.71'W; 0.0 km deep. Ref pt43617452, preliminary. Dup of nn19271017.
[NN] nn19271017 | Sat, 07 Feb 2026 11:27:12 +0000 | Nevada | 39°05.34'N 117°39.46'W | depth 1.0 km | ML 1.0 | reviewed
[US] us6380i1z6 | 6:34 AM local time (UTC-05:00) on Feb 7, 2026 | Central Chile | 33.391°S, 71.569°W | depth 106.1 km | MB 4.3 | reviewed
[AK] ak34487352 | 2026-02-07 12:32:50 UTC | Southern Alaska | (61.287, -153.503) | depth 27.9 km | M1.3 (ml) | reviewed
Editor's note: entries are reproduced exactly as each network filed them, time conventions included.
Reminder: networks report on different magnitude scales (md, ml, mb, mw); values are not converted.
[AT] at68470955 | Sat, 07 Feb 2026 12:56:24 +0000 | Northern California | 38.996°N, 122.778°W | depth 6.8 km | M2.5 (md) | preliminary | same event as nc50798754
[NC] nc50798754 | 12:56 UTC on 7 February 2026 | Northern California | 39°00.49'N 122°50.11'W | depth 6.1 km (3.8 mi) | MD 2.7 | reviewed
2026-02-07T14:17:08Z — A magnitude 1.6 (md) earthquake struck Northern California; epicentre 38°53.35'N 122°06.75'W, depth 6.3 km. The US network lists it as us603418b5 (preliminary). This is the same earthquake as nc53309104.
[NC] nc53309104 | 6:17 AM local time (UTC-08:00) on Feb 7, 2026 | Northern California | 38.874°N, 122.100°W | depth 8.4 km (5.2 mi) | M1.9 (md) | reviewed
[AK] ak27924237 | 05:47 (UTC-10:00) 7 Feb 2026 | Southern Alaska | 60°40.07'N 151°59.76'W | depth 89.6 km | M1.6 (ml) | reviewed
AK WIRE [5:56 AM local time (UTC-10:00) on Feb 7, 2026] ML 1.1 quake in Southern Alaska @ 62°26.80'N 149°26.68'W; 7.2 km deep. Ref ak39042597, reviewed.
16:41 UTC on 7 February 2026 — A magnitude 1.8 (ml) earthquake struck Southern California; epicentre 33°46.09'N 117°55.78'W, depth 2.3 km. The CI network lists it as ci23138832 (reviewed).
[AK] ak91178001 | 08:16 (UTC-10:00) 7 Feb 2026 | Southern Alaska | (59.157, -152.439) | depth 10.2 km | M1.9 (ml) | reviewed
[US] us62314pnl | 06:53 (UTC-12:00) 7 Feb 2026 | Tonga | (-20.698, -175.399) | depth 14.7 km (9.1 mi) | M4.1 (mb) | reviewed
[NC] nc98646422 | 19:48 UTC on 7 February 2026 | Northern California | (38.802, -122.869) | depth 7.6 km (4.7 mi) | M1.0 (md) | reviewed
[US] us7977rs80 | 04:58 (UTC+09:00) 8 Feb 2026 | Honshu, Japan | (37.303, 141.707) | depth 17.0 mi | MB 4.1 | reviewed
20:30 UTC on 7 February 2026 ref=us628128p1 net=us mag=5.4 type=mb region="Honshu, Japan" loc=37.303,140.410 depth=31.0 mi status=reviewed
[US] us6095jd7d | 2026-02-07T22:07:53Z | Tonga | 20°07.94'S 174°31.85'W | depth 297.7 km (185.0 mi) | MB 5.1 | reviewed
8:51 AM local time (UTC+09:00) on Feb 8, 2026 — A magnitude 4.5 (mb) earthquake struck Honshu, Japan; epicentre 35.897°N, 142.391°E, depth 62.4 km. The US network lists it as us74526poc (reviewed).
2026-02-08 02:31:24 UTC ref=ak18634012 net=ak mag=1.5 type=ml region="Southern Alaska" loc=62.043,-152.384 depth=14.4 mi status=reviewed
[US] us7729qt8h | 1:15 PM local time (UTC+09:00) on Feb 8, 2026 | Honshu, Japan | 37.415°N, 141.687°E | depth 15.9 km | MB 5.3 | reviewed
2026-02-08T05:40:14Z — A magnitude 5.4 (mb) earthquake struck Honshu, Japan; epicentre 37.100°N, 140.859°E, depth 66.5 km. The US network lists it as us6734s263 (reviewed).
UPDATE [2026-02-07T12:41:03Z]: us6380i1z6 withdrawn (automatic solution with no real event).
[US] us6967srs9 | 05:56 UTC on 8 February 2026 | Central Chile | 33.066°S, 71.740°W | depth 111.1 km | M4.4 (mb) | reviewed
```

**Questions, code-computed key and model answers (`v0`):**

| Q | template | question | key | oracle | naive | mock | anthropic:claude-haiku-5-5 | anthropic:claude-opus-5-5 |
|---|---|---|---|---|---|---|---|---|
| Q1 | N11 (new) | Among the earthquakes smaller than magnitude 4.1, how deep was the shallowest one, in km? Round to 1 decimal place. | 1.0 | 1.0 ✓ | 0.0 ✗ | 3.0 ✗ | 1.0 ✓ | 1.0 ✓ |
| Q2 | N01 (new) | What is the number of separate earthquakes that occurred between 10:45 UTC on 7 February 2026 and 16:20 UTC on 7 February 2026? | 7 | 7 ✓ | 12 ✗ | 8 ✗ | 7 ✓ | 7 ✓ |
| Q3 | P11 (public) | Among the earthquakes that were recorded from 16:10 UTC on 7 February 2026 up to 00:30 UTC on 8 February 2026, how deep was the shallowest one, in miles? Round to 1 decimal place. | 1.4 | 1.4 ✓ | 1.4 ✓ | 1.4 ✓ | 1.4 ✓ | 1.4 ✓ |
| Q4 | P07 (public) | Among the earthquakes located in Northern California, give the event ID of the one nearest to station NN-14 at (39.107, -117.481). | nc53309104 | nc53309104 ✓ | nc53309104 ✓ | nc53309104 ✓ | nc53309104 ✓ | nc53309104 ✓ |
| Q5 | N05 (new) | Counting only earthquakes smaller than magnitude 1.6, which region has more of them: Central Chile or Southern Alaska? Answer "tie" if the counts are equal. | Southern Alaska | Southern Alaska ✓ | Southern Alaska ✓ | Central Chile ✗ | Southern Alaska ✓ | Southern Alaska ✓ |

**Operation chains (the machine-readable meaning of each question):**

- Q1: `dedupe → mag < 4.1 → min_depth → convert → km → round(1)`
- Q2: `dedupe → time in [2026-02-07T10:45:00Z, 2026-02-07T16:20:00Z) → count`
- Q3: `dedupe → time in [2026-02-07T16:10:00Z, 2026-02-08T00:30:00Z) → min_depth → convert → mi → round(1)`
- Q4: `dedupe → region = Northern California → nearest_to(39.107, -117.481)`
- Q5: `dedupe → mag < 1.6 → compare_regions(Central Chile vs Southern Alaska)`

**Traps in this digest:**

- T1 *duplicate* (cross_ref): AT re-reported nc50798754 as at68470955 with a 'same event as' note: preliminary M2.5, depth 6.8 km (reviewed: M2.7, 6.1 km). If mishandled: Q2 → 8.
- T2 *duplicate* (cross_ref): PT re-reported nn19271017 as pt43617452 with a 'same event as' note: preliminary M0.9, depth 0 km (reviewed: M1, 1 km). If mishandled: Q1 → 0.0, Q2 → 8.
- T3 *duplicate* (cross_ref): US re-reported nc53309104 as us603418b5 with a 'same event as' note: preliminary M1.6, depth 6.3 km (reviewed: M1.9, 8.4 km). If mishandled: Q2 → 8.
- T4 *retraction* (phantom): phantom report us6380i1z6 (M4.3, Central Chile) later retracted: automatic solution with no real event. If mishandled: Q2 → 8.
- T5 *conflicting magnitude* (second_report): us673420ol also reported as M1.5 (preliminary, AT) beside the reviewed M4.2. If mishandled: Q2 → 8.

<details><summary>Metamorphic variant `v1` of the same instance (same key)</summary>

```text
== QUAKE DIGEST 808-F ==
Window: the 24 hours up to 2026-02-08 06:00:00 UTC. Entries are listed as filed by each network.

== NC feed ==
Station AK.ATF was offline from 19:00 to 19:37 UTC for maintenance; its readings are not included.
06:17 (UTC-08:00) 7 Feb 2026 — A magnitude 1.9 (md) earthquake struck Northern California; epicentre 38.874°N, 122.100°W, depth 5.2 mi. The NC network lists it as nc53309104 (reviewed).
Sat, 07 Feb 2026 06:38:18 +0000 — A magnitude 1.1 (md) earthquake struck Northern California; epicentre 39.377°N, 122.852°W, depth 0.7 mi. The NC network lists it as nc41072473 (reviewed).
2026-02-07T19:48:09Z — A magnitude 1.0 (md) earthquake struck Northern California; epicentre 38.802°N, 122.869°W, depth 7.6 km. The NC network lists it as nc98646422 (reviewed).
4:56 AM local time (UTC-08:00) on Feb 7, 2026 — A magnitude 2.7 (md) earthquake struck Northern California; epicentre 39.008°N, 122.835°W, depth 6.1 km. The NC network lists it as nc50798754 (reviewed).
== AK feed ==
AK WIRE [05:56 (UTC-10:00) 7 Feb 2026] ML 1.1 quake in Southern Alaska @ 62.447°N, 149.445°W; 4.5 mi deep. Ref ak39042597, reviewed.
2026-02-07T12:32:50Z — A magnitude 1.3 (ml) earthquake struck Southern Alaska; epicentre 61.287°N, 153.503°W, depth 27.9 km (17.3 mi). The AK network lists it as ak34487352 (reviewed).
15:47 UTC on 7 February 2026 ref=ak27924237 net=ak mag=1.6 type=ml region="Southern Alaska" loc=60.668,-151.996 depth=89.6 km (55.7 mi) status=reviewed
[AK] ak18634012 | 16:31 (UTC-10:00) 7 Feb 2026 | Southern Alaska | 62.043°N, 152.384°W | depth 23.1 km | ML 1.5 | reviewed
10:38 PM local time (UTC-10:00) on Feb 6, 2026 — A magnitude 0.9 (ml) earthquake struck Southern Alaska; epicentre 61.134°N, 153.006°W, depth 10.7 mi. The AK network lists it as ak97220386 (reviewed).
AK WIRE [2026-02-07 18:16:08 UTC] ML 1.9 quake in Southern Alaska @ 59.157°N, 152.439°W; 10.2 km (6.3 mi) deep. Ref ak91178001, reviewed.
== US feed ==
04:58 (UTC+09:00) 8 Feb 2026 — A magnitude 4.1 (mb) earthquake struck Honshu, Japan; epicentre 37.303°N, 141.707°E, depth 27.3 km. The US network lists it as us7977rs80 (reviewed).
Editor's note: entries are reproduced exactly as each network filed them, time conventions included.
13:15 (UTC+09:00) 8 Feb 2026 — A magnitude 5.3 (mb) earthquake struck Honshu, Japan; epicentre 37.415°N, 141.687°E, depth 9.9 mi. The US network lists it as us7729qt8h (reviewed).
Sat, 07 Feb 2026 11:14:44 +0000 — A magnitude 4.2 (mb) earthquake struck Central Chile; epicentre (-33.124, -71.900), depth 68.7 mi. The US network lists it as us673420ol (reviewed).
2026-02-08 05:40:14 UTC — A magnitude 5.4 (mb) earthquake struck Honshu, Japan; epicentre 37.100°N, 140.859°E, depth 41.3 mi. The US network lists it as us6734s263 (reviewed).
12:56 AM local time (UTC-05:00) on Feb 8, 2026 — A magnitude 4.4 (mb) earthquake struck Central Chile; epicentre 33.066°S, 71.740°W, depth 111.1 km. The US network lists it as us6967srs9 (reviewed).
06:34 (UTC-05:00) 7 Feb 2026 — A magnitude 4.3 (mb) earthquake struck Central Chile; epicentre 33.391°S, 71.569°W, depth 106.1 km (65.9 mi). The US network lists it as us6380i1z6 (reviewed).
Sat, 07 Feb 2026 20:30:08 +0000 — A magnitude 5.4 (mb) earthquake struck Honshu, Japan; epicentre 37°18.20'N 140°24.61'E, depth 49.9 km (31.0 mi). The US network lists it as us628128p1 (reviewed).
06:53 (UTC-12:00) 7 Feb 2026 — A magnitude 4.1 (mb) earthquake struck Tonga; epicentre 20°41.90'S 175°23.93'W, depth 14.7 km (9.1 mi). The US network lists it as us62314pnl (reviewed).
US WIRE [4:34 AM local time (UTC-05:00) on Feb 7, 2026] MB 4.2 quake in Central Chile @ 33°08.78'S 71°18.96'W; 20.5 mi deep. Ref us7966hqsu, reviewed.
Reminder: networks report on different magnitude scales (md, ml, mb, mw); values are not converted.
2026-02-07T14:17:08Z — A magnitude 1.6 (md) earthquake struck Northern California; epicentre 38.889°N, 122.112°W, depth 3.9 mi. The US network lists it as us603418b5 (preliminary). This is the same earthquake as nc53309104.
US WIRE [10:07 AM local time (UTC-12:00) on Feb 7, 2026] MB 5.1 quake in Tonga @ 20.132°S, 174.531°W; 185.0 mi deep. Ref us6095jd7d, reviewed.
UPDATE [2026-02-07T12:41:03Z]: us6380i1z6 withdrawn (automatic solution with no real event).
US WIRE [2026-02-07T23:51:00Z] MB 4.5 quake in Honshu, Japan @ 35.897°N, 142.391°E; 38.8 mi deep. Ref us74526poc, reviewed.
From the PT network:
2026-02-07T11:26:38Z — A magnitude 0.9 (ml) earthquake struck Nevada; epicentre 39.082°N, 117.645°W, depth 0.0 mi. The PT network lists it as pt43617452 (preliminary). This is the same earthquake as nn19271017.
== CI feed ==
Sat, 07 Feb 2026 16:41:37 +0000 ref=ci23138832 net=ci mag=1.8 type=ml region="Southern California" loc=33.768,-117.930 depth=1.4 mi status=reviewed
From the AT network:
11:14 UTC on 7 February 2026 — A magnitude 1.5 (mb) earthquake struck Central Chile; epicentre 33.124°S, 71.900°W, depth 68.7 mi. The AT network lists it as us673420ol (preliminary).
4:56 AM local time (UTC-08:00) on Feb 7, 2026 ref=at68470955 net=at mag=2.5 type=md region="Northern California" loc=38.996,-122.778 depth=4.2 mi status=preliminary same_as=nc50798754
From the NN network:
2026-02-07 11:27:12 UTC ref=nn19271017 net=nn mag=1.0 type=ml region="Nevada" loc=39.089,-117.658 depth=0.6 mi status=reviewed
```

- Q1: Among the earthquakes smaller than magnitude 4.1, how deep was the shallowest one, in km? Round to 1 decimal place.
- Q2: How many distinct earthquakes that occurred between 10:45 UTC on 7 February 2026 and 16:20 UTC on 7 February 2026 are there?
- Q3: Among the earthquakes that were recorded from 16:10 UTC on 7 February 2026 up to 00:30 UTC on 8 February 2026, how deep was the shallowest one, in miles? Round to 1 decimal place.
- Q4: Which of the earthquakes in Northern California was closest to station NN-14 at (39.107, -117.481)? Give its event ID.
- Q5: Counting only earthquakes smaller than magnitude 1.6, which region has more of them: Central Chile or Southern Alaska? Answer "tie" if the counts are equal.

</details>

### Example 2: `inst-012` (24 earthquakes, 29 report lines, 1 notice, 5 questions)

**Digest (original rendering `v0`, exactly as sent to the models):**

```text
SEISMIC ACTIVITY DIGEST (edition 454-E)
Compiled 06:00 UTC on 8 February 2026; covers the 24 hours before compilation. Reports are reproduced as received.

== HV feed ==
Sat, 07 Feb 2026 21:46:16 +0000 ref=hv72951955 net=hv mag=6.5 type=md region="Hawaii" loc=19.510,-154.851 depth=5.8 km status=reviewed
HV WIRE [11:02 AM local time (UTC-10:00) on Feb 7, 2026] MD 2.1 quake in Hawaii @ (19.390, -155.309); 37.2 km deep. Ref hv72846463, reviewed.
[HV] hv30150735 | 02:39 (UTC-10:00) 7 Feb 2026 | Hawaii | (19.459, -155.103) | depth 6.3 km | M1.3 (md) | reviewed
03:30 (UTC-10:00) 7 Feb 2026 — A magnitude 0.9 (md) earthquake struck Hawaii; epicentre (18.977, -155.124), depth 5.8 km. The HV network lists it as hv56074840 (reviewed).
== NC feed ==
11:04 (UTC-08:00) 7 Feb 2026 — A magnitude 2.8 (md) earthquake struck Northern California; epicentre (38.118, -123.093), depth 2.0 km. The NC network lists it as nc99332164 (reviewed).
-- PT network --
10:07 (UTC-05:00) 7 Feb 2026 — A magnitude 4.9 (mb) earthquake struck Central Chile; epicentre (-32.264, -74.578), depth 89.5 km. The PT network lists it as us7307xxgc (preliminary).
-- NN network --
[NN] nn27496364 | 12:07 AM local time (UTC-08:00) on Feb 7, 2026 | Nevada | (39.168, -117.586) | depth 13.1 km | M2.4 (ml) | reviewed
-- EW network --
2026-02-07T13:42:58Z — A magnitude 2.2 (ml) earthquake struck Utah; epicentre (39.400, -111.428), depth 3.2 km. The EW network lists it as ew83305412 (preliminary). This is the same earthquake as uu87408363.
Felt reports are still being collected and are not summarised here.
2026-02-07T18:29:55Z — A magnitude 5.5 (ml) earthquake struck Southern Alaska; epicentre (59.775, -149.628), depth 66.9 km. The EW network lists it as ew85171858 (preliminary). This is the same earthquake as ak18554701.
== US feed ==
2026-02-07T14:54:04Z — A magnitude 4.3 (mb) earthquake struck Central Chile; epicentre (-32.960, -71.839), depth 24.6 km. The US network lists it as us708612jg (reviewed).
4:48 AM local time (UTC+09:00) on Feb 8, 2026 — A magnitude 5.2 (mb) earthquake struck Honshu, Japan; epicentre 37.678°N, 142.249°E, depth 14.2 km. The US network lists it as us76099pu4 (reviewed).
Sat, 07 Feb 2026 20:05:52 +0000 — A magnitude 4.2 (mb) earthquake struck Tonga; epicentre 20.005°S, 174.333°W, depth 183.2 km. The US network lists it as us645351qi (reviewed).
Sat, 07 Feb 2026 22:07:53 +0000 — A magnitude 5.1 (mb) earthquake struck Tonga; epicentre (-20.132, -174.531), depth 185.0 mi. The US network lists it as us6095jd7d (reviewed).
10:07 (UTC-05:00) 7 Feb 2026 — A magnitude 4.1 (mb) earthquake struck Central Chile; epicentre 32°15.81'S 74°34.66'W, depth 89.5 km. The US network lists it as us7307xxgc (reviewed).
[US] us6973uc0o | 03:34 (UTC-12:00) 7 Feb 2026 | Tonga | (-19.759, -174.004) | depth 112.0 km | M4.1 (mb) | reviewed
Sun, 08 Feb 2026 04:15:58 +0000 ref=us7729qt8h net=us mag=5.3 type=mb region="Honshu, Japan" loc=37.415,141.687 depth=15.9 km status=reviewed
2026-02-07T09:48:38Z — A magnitude 4.2 (mb) earthquake struck Central Chile; epicentre (-34.918, -72.060), depth 54.5 mi. The US network lists it as us7804dhsg (reviewed).
11:14 UTC on 7 February 2026 — A magnitude 4.2 (mb) earthquake struck Central Chile; epicentre (-33.124, -71.900), depth 110.6 km. The US network lists it as us673420ol (reviewed).
08:58 (UTC-05:00) 7 Feb 2026 — A magnitude 4.6 (mb) earthquake struck Central Chile; epicentre (-32.714, -72.515), depth 77.0 km (47.8 mi). The US network lists it as us70521nf6 (reviewed).
[US] us6084287d | 9:11 AM local time (UTC-12:00) on Feb 7, 2026 | Tonga | (-19.702, -174.837) | depth 133.5 km | MB 4.2 | reviewed
05:56 UTC on 8 February 2026 — A magnitude 4.4 (mb) earthquake struck Central Chile; epicentre (-33.066, -71.740), depth 111.1 km (69.0 mi). The US network lists it as us6967srs9 (reviewed).
09:06 (UTC-05:00) 7 Feb 2026 ref=us6670vunm net=us mag=4.6 type=mb region="Central Chile" loc=-32.640,-71.688 depth=25.6 km status=reviewed
11:34 UTC on 7 February 2026 — A magnitude 4.5 (mb) earthquake struck Tonga; epicentre 21.187°S, 174.245°W, depth 25.3 mi. The US network lists it as us739180wt (reviewed).
-- UU network --
[UU] uu41250808 | 23:18 UTC on 7 February 2026 | Utah | 39.834°N, 111.852°W | depth 6.4 km | ML 0.8 | reviewed
13:43 UTC on 7 February 2026 — A magnitude 2.3 (ml) earthquake struck Utah; epicentre 39.371°N, 111.436°W, depth 5.9 km (3.7 mi). The UU network lists it as uu87408363 (reviewed).
2:53 PM local time (UTC-07:00) on Feb 7, 2026 ref=uu53124785 net=uu mag=1.1 type=ml region="Utah" loc=39.736,-112.118 depth=9.0 km status=reviewed
NOTICE [2026-02-07T14:25:07Z]: report us6670vunm has been RETRACTED (sonic boom). Disregard it.
[UU] uu36821212 | 2026-02-08T00:24:38Z | Utah | (38.917, -111.806) | depth 4.6 km | M1.0 (ml) | reviewed
From the AK network:
08:30 (UTC-10:00) 7 Feb 2026 — A magnitude 5.2 (ml) earthquake struck Southern Alaska; epicentre (59.714, -149.702), depth 66.1 km. The AK network lists it as ak18554701 (reviewed).
-- AT network --
6:14 AM local time (UTC-05:00) on Feb 7, 2026 — A magnitude 4.5 (mb) earthquake struck Central Chile; epicentre (-33.153, -71.839), depth 112.2 km. The AT network lists it as at15196390 (preliminary). This is the same earthquake as us673420ol.
```

**Questions, code-computed key and model answers (`v0`):**

| Q | template | question | key | oracle | naive | mock | anthropic:claude-haiku-5-5 | anthropic:claude-opus-5-5 |
|---|---|---|---|---|---|---|---|---|
| Q1 | P03 (public) | What is the event ID of the strongest of the earthquakes in Central Chile? | us70521nf6 | us70521nf6 ✓ | us7307xxgc ✗ | us70521nf6 ✓ | us70521nf6 ✓ | us70521nf6 ✓ |
| Q2 | P10 (public) | What was the largest magnitude among the earthquakes at least 22 miles deep? | 5.2 | 5.2 ✓ | 5.5 ✗ | 5.2 ✓ | 5.2 ✓ | 5.2 ✓ |
| Q3 | N07 (new) | Which of the earthquakes that were recorded from 2026-02-08 00:00 UTC up to 2026-02-08 03:25 UTC was closest to station UU-11 (39.096°N, 111.711°W)? Give its event ID. | uu36821212 | uu36821212 ✓ | uu36821212 ✓ | us7729qt8h ✗ | uu36821212 ✓ | uu36821212 ✓ |
| Q4 | N12 (new) | Among the earthquakes with magnitude at least 4.5, what is the distance in miles from station US-15 at (-33.136, -71.836) to the nearest one? Round to 1 decimal place. | 49.0 | 49.0 ✓ | 1.2 ✗ | 49.0 ✓ | 49.0 ✓ | 49.0 ✓ |
| Q5 | P11 (public) | Among the earthquakes that were recorded from 2026-02-07 09:25 UTC up to 2026-02-07 15:55 UTC, how deep was the shallowest one, in km? Round to 1 decimal place. | 5.8 | 5.8 ✓ | 3.2 ✗ | 5.8 ✓ | 5.8 ✓ | 5.8 ✓ |

**Operation chains (the machine-readable meaning of each question):**

- Q1: `dedupe → region = Central Chile → argmax_mag`
- Q2: `dedupe → depth ≥ 22 mi → max_mag`
- Q3: `dedupe → time in [2026-02-08T00:00:00Z, 2026-02-08T03:25:00Z) → nearest_to(39.096, -111.711)`
- Q4: `dedupe → mag ≥ 4.5 → nearest_distance(-33.136, -71.836) → convert → mi → round(1)`
- Q5: `dedupe → time in [2026-02-07T09:25:00Z, 2026-02-07T15:55:00Z) → min_depth → convert → km → round(1)`

**Traps in this digest:**

- T1 *duplicate* (cross_ref): EW re-reported ak18554701 as ew85171858 with a 'same event as' note: preliminary M5.5, depth 66.9 km (reviewed: M5.2, 66.1 km). If mishandled: Q2 → 5.5.
- T2 *duplicate* (cross_ref): AT re-reported us673420ol as at15196390 with a 'same event as' note: preliminary M4.5, depth 112.2 km (reviewed: M4.2, 110.6 km). If mishandled: Q4 → 1.2.
- T3 *duplicate* (cross_ref): EW re-reported uu87408363 as ew83305412 with a 'same event as' note: preliminary M2.2, depth 3.2 km (reviewed: M2.3, 5.9 km). If mishandled: Q5 → 3.2.
- T4 *retraction* (phantom): phantom report us6670vunm (M4.6, Central Chile) later retracted: sonic boom. If mishandled: Q4 → 35.3.
- T5 *conflicting magnitude* (second_report): us7307xxgc also reported as M4.9 (preliminary, PT) beside the reviewed M4.1. If mishandled: Q1 → us7307xxgc.

<details><summary>Metamorphic variant `v1` of the same instance (same key)</summary>

```text
== QUAKE DIGEST 439-D ==
Window: the 24 hours up to Sun, 08 Feb 2026 06:00:00 +0000. Entries are listed as filed by each network.

UU WIRE [Sat, 07 Feb 2026 23:18:10 +0000] ML 0.8 quake in Utah @ 39.834°N, 111.852°W; 6.4 km deep. Ref uu41250808, reviewed.
[UU] uu53124785 | 2026-02-07T21:53:45Z | Utah | 39.736°N, 112.118°W | depth 5.6 mi | M1.1 (ml) | reviewed
2026-02-07T21:11:10Z — A magnitude 4.2 (mb) earthquake struck Tonga; epicentre 19.702°S, 174.837°W, depth 133.5 km (83.0 mi). The US network lists it as us6084287d (reviewed).
US WIRE [2026-02-07 11:34:06 UTC] MB 4.5 quake in Tonga @ (-21.187, -174.245); 25.3 mi deep. Ref us739180wt, reviewed.
[PT] us7307xxgc | 10:07 AM local time (UTC-05:00) on Feb 7, 2026 | Central Chile | (-32.264, -74.578) | depth 55.6 mi | MB 4.9 | preliminary
[EW] ew83305412 | 2026-02-07 13:42:58 UTC | Utah | (39.400, -111.428) | depth 2.0 mi | M2.2 (ml) | preliminary | same event as uu87408363
2026-02-07 13:58:42 UTC ref=us70521nf6 net=us mag=4.6 type=mb region="Central Chile" loc=-32.714,-72.515 depth=47.8 mi status=reviewed
05:56 UTC on 8 February 2026 — A magnitude 4.4 (mb) earthquake struck Central Chile; epicentre (-33.066, -71.740), depth 69.0 mi. The US network lists it as us6967srs9 (reviewed).
[AK] ak18554701 | 2026-02-07 18:30:13 UTC | Southern Alaska | 59.714°N, 149.702°W | depth 41.1 mi | M5.2 (ml) | reviewed
NN WIRE [Sat, 07 Feb 2026 08:07:39 +0000] ML 2.4 quake in Nevada @ 39°10.08'N 117°35.14'W; 8.1 mi deep. Ref nn27496364, reviewed.
HV WIRE [12:39 UTC on 7 February 2026] MD 1.3 quake in Hawaii @ 19.459°N, 155.103°W; 3.9 mi deep. Ref hv30150735, reviewed.
NC WIRE [2026-02-07T19:04:51Z] MD 2.8 quake in Northern California @ 38.118°N, 123.093°W; 1.2 mi deep. Ref nc99332164, reviewed.
US WIRE [2026-02-07T15:34:37Z] MB 4.1 quake in Tonga @ 19.759°S, 174.004°W; 69.6 mi deep. Ref us6973uc0o, reviewed.
US WIRE [2026-02-07 22:07:53 UTC] MB 5.1 quake in Tonga @ (-20.132, -174.531); 297.7 km deep. Ref us6095jd7d, reviewed.
14:54 UTC on 7 February 2026 ref=us708612jg net=us mag=4.3 type=mb region="Central Chile" loc=-32.960,-71.839 depth=15.3 mi status=reviewed
[US] us645351qi | 20:05 UTC on 7 February 2026 | Tonga | 20.005°S, 174.333°W | depth 113.8 mi | MB 4.2 | reviewed
[US] us7729qt8h | 04:15 UTC on 8 February 2026 | Honshu, Japan | (37.415, 141.687) | depth 9.9 mi | MB 5.3 | reviewed
15:07 UTC on 7 February 2026 — A magnitude 4.1 (mb) earthquake struck Central Chile; epicentre 32.264°S, 74.578°W, depth 55.6 mi. The US network lists it as us7307xxgc (reviewed).
UU WIRE [2026-02-07T13:43:21Z] ML 2.3 quake in Utah @ 39.371°N, 111.436°W; 3.7 mi deep. Ref uu87408363, reviewed.
2026-02-07T13:30:31Z ref=hv56074840 net=hv mag=0.9 type=md region="Hawaii" loc=18.977,-155.124 depth=5.8 km (3.6 mi) status=reviewed
2026-02-07 14:06:01 UTC — A magnitude 4.6 (mb) earthquake struck Central Chile; epicentre 32.640°S, 71.688°W, depth 15.9 mi. The US network lists it as us6670vunm (reviewed).
2026-02-07T11:14:44Z — A magnitude 4.2 (mb) earthquake struck Central Chile; epicentre 33.124°S, 71.900°W, depth 68.7 mi. The US network lists it as us673420ol (reviewed).
2026-02-08 00:24:38 UTC ref=uu36821212 net=uu mag=1.0 type=ml region="Utah" loc=38.917,-111.806 depth=4.6 km (2.9 mi) status=reviewed
2026-02-07T21:02:42Z — A magnitude 2.1 (md) earthquake struck Hawaii; epicentre (19.390, -155.309), depth 23.1 mi. The HV network lists it as hv72846463 (reviewed).
[AT] at15196390 | 2026-02-07T11:14:21Z | Central Chile | 33.153°S, 71.839°W | depth 69.7 mi | M4.5 (mb) | preliminary | same event as us673420ol
11:46 AM local time (UTC-10:00) on Feb 7, 2026 — A magnitude 6.5 (md) earthquake struck Hawaii; epicentre (19.510, -154.851), depth 3.6 mi. The HV network lists it as hv72951955 (reviewed).
Sat, 07 Feb 2026 19:48:32 +0000 — A magnitude 5.2 (mb) earthquake struck Honshu, Japan; epicentre 37.678°N, 142.249°E, depth 14.2 km (8.8 mi). The US network lists it as us76099pu4 (reviewed).
Felt reports are still being collected and are not summarised here.
US WIRE [Sat, 07 Feb 2026 09:48:38 +0000] MB 4.2 quake in Central Chile @ 34.918°S, 72.060°W; 54.5 mi deep. Ref us7804dhsg, reviewed.
[EW] ew85171858 | 2026-02-07T18:29:55Z | Southern Alaska | 59.775°N, 149.628°W | depth 66.9 km | ML 5.5 | preliminary | same event as ak18554701

Corrections and retractions:
RETRACTION [2026-02-07 14:25:07 UTC]: us6670vunm was not an earthquake (sonic boom); ignore that report.
```

- Q1: Among the earthquakes in Central Chile, which one had the largest magnitude? Give its event ID.
- Q2: Among the earthquakes with a depth of 22 miles or more, what was the highest magnitude recorded?
- Q3: Among the earthquakes that were recorded from 2026-02-08 00:00 UTC up to 2026-02-08 03:25 UTC, give the event ID of the one nearest to station UU-11 (39.096°N, 111.711°W).
- Q4: How far from station US-15 at (-33.136, -71.836), in miles, was the closest of the earthquakes of magnitude 4.5 or greater? Round to 1 decimal place.
- Q5: What was the smallest depth, in km, among the earthquakes that were recorded from 09:25 UTC on 7 February 2026 up to 15:55 UTC on 7 February 2026? Round to 1 decimal place.

</details>
