# Live evaluation report: `compare-100-v2`

Every instance in this run was generated fresh from the seed below, every answer key was computed by code from the frozen snapshot, and every score was computed automatically. No item was labelled by hand.

## Run

- **Generated with:** evalgen 0.2.0 (the reading rules and templates are versioned with the code).
- **Master seed:** `16222556419173536875` (commitment `366a1a1fdfa065e85b3bd4cd…`); every derived seed is listed in `manifest.json`.
- **Data source:** synthetic; 360 events in the snapshot pool; window: 24 h ending 2026-04-13T06:00:00Z.
- **Snapshot SHA-256:** `92ee4459278d550321bc1d01aa1668b1a3aadd2bbf3cd7a03eb18a4c93e1539b`
- **Instances:** 100; **questions:** 397 (197 from public templates, 200 from new-this-run templates); **renderings per instance:** 3 (original + 2 metamorphic variants); **prompts:** 300.
- **Traps planted:** 445 (duplicate 245, retraction 100, conflicting magnitude 100); 405 are load-bearing: mishandling them changes the answers of 203 of the 397 questions (446 trap–question pairs). The rest stay in as plain noise.

## Model accuracy (original renderings)

| model | kind | accuracy | 95% CI | mean instance score | accuracy, all renderings | JSON strict / lenient / failed (API errors) |
|---|---|---|---|---|---|---|
| oracle | reference | 100.0% (397/397) | 99–100% | 100.0% | 100.0% | 300 / 0 / 0 (0) |
| naive | reference | 49.6% (197/397) | 45–55% | 48.6% | 49.6% | 300 / 0 / 0 (0) |
| mock | simulated | 73.0% (290/397) | 68–77% | 73.2% | 70.7% | 234 / 66 / 0 (0) |
| anthropic:claude-haiku-5-5 | api | 97.5% (387/397) | 95–99% | 97.6% | 98.3% | 300 / 0 / 0 (0) |
| anthropic:claude-opus-5-5 | api | 100.0% (397/397) | 99–100% | 100.0% | 100.0% | 300 / 0 / 0 (0) |

`oracle` reads only the digest text and applies the reading rules, so it should score 100%. That shows every item is answerable from the text. `naive` treats every line as a separate quake and ignores notices. `mock` is a *simulated* model with built-in benchmark contamination, used to exercise the metrics without API keys; it is not a language model.

## Head to head (paired)

The API models answered the same questions, so they are compared question by question, with questions as the independent units. Only the questions that exactly one model got right say which is better; the exact McNemar test asks whether those split evenly. The first row uses the original renderings. The second counts a question as right only if every rendering of it was answered correctly, so it also rewards robustness to surface changes (units, formats, line order, wording).

| A | B | right on | questions | both right | only A right | only B right | both wrong | A minus B (95% CI) | McNemar p |
|---|---|---|---|---|---|---|---|---|---|
| anthropic:claude-opus-5-5 | anthropic:claude-haiku-5-5 | original rendering | 397 | 387 | 10 | 0 | 0 | +2.5 pts (+0.9 to +4.1) | 0.002 |
| anthropic:claude-opus-5-5 | anthropic:claude-haiku-5-5 | every rendering | 397 | 377 | 20 | 0 | 0 | +5.0 pts (+2.8 to +7.2) | 1.9e-06 |

## API usage

Tokens as each API reported them, for the replies that were scored (an attempt that failed or was cut off and then asked again is not counted). Output tokens include any thinking before the answer.

| model | replies | input tokens | output tokens | median output tokens per reply | median latency |
|---|---|---|---|---|---|
| anthropic:claude-haiku-5-5 | 300 | 1,170,002 | 1,558,755 | 4,928 | 19 s |
| anthropic:claude-opus-5-5 | 300 | 1,170,002 | 463,085 | 1,413 | 12 s |

## Metamorphic consistency

Each instance is re-rendered with the same snapshot but different units, line order, formats and question wording. Correct answers cannot change, so any change in a model's answer is an inconsistency. This needs no ground truth.

| model | consistent answer pairs | 95% CI | questions consistent across all variants |
|---|---|---|---|
| oracle | 100.0% (794/794) | 100–100% | 100.0% (397/397) |
| naive | 100.0% (794/794) | 100–100% | 100.0% (397/397) |
| mock | 55.4% (440/794) | 52–59% | 36.8% (146/397) |
| anthropic:claude-haiku-5-5 | 96.2% (764/794) | 95–97% | 95.0% (377/397) |
| anthropic:claude-opus-5-5 | 100.0% (794/794) | 100–100% | 100.0% (397/397) |

## Seen vs new templates

Public templates are published (a model may have trained on them). Each new template is composed for this run as a twin of one public template: the same final operation and number of filters, but a different filter combination, and each twin pair is used equally often. The *adjusted* gap compares the pools like for like, within strata of equal code-computed difficulty (how many traps change the answer, and whether the filters matter), so that the pools' different trap exposure is not mistaken for overfitting. A large positive adjusted gap flags overfitting to the published skeletons (flag: > 10 points with a 95% CI that excludes 0). It is a flag, not proof.

| model | public (seen) | new this run | raw gap (95% CI) | adjusted gap (95% CI) | flag |
|---|---|---|---|---|---|
| oracle | 100.0% (197/197) | 100.0% (200/200) | +0.0 pts (+0 to +0) | +0.0 pts (-3 to +3) | — |
| naive | 51.8% (102/197) | 47.5% (95/200) | +4.3 pts (-6 to +14) | -0.2 pts (-3 to +3) | — |
| mock | 87.8% (173/197) | 58.5% (117/200) | +29.3 pts (+21 to +38) | +30.1 pts (+22 to +38) | ⚠ overfitting? |
| anthropic:claude-haiku-5-5 | 97.0% (191/197) | 98.0% (196/200) | -1.0 pts (-4 to +2) | -1.6 pts (-6 to +2) | — |
| anthropic:claude-opus-5-5 | 100.0% (197/197) | 100.0% (200/200) | +0.0 pts (+0 to +0) | +0.0 pts (-3 to +3) | — |

## Trap catch rate

Counted over (load-bearing trap, affected question, rendering) triples. *Caught* = answered correctly; *fell* = gave exactly an answer a reader produces by missing that trap (alone or together with other traps).

| model | caught (overall) | duplicate | retraction | conflicting magnitude | fell for the trap |
|---|---|---|---|---|---|
| oracle | 100.0% (1338/1338) | 100.0% | 100.0% | 100.0% | 0.0% (0/1338) |
| naive | 0.9% (12/1338) | 0.0% | 0.0% | 4.0% | 99.1% (1326/1338) |
| mock | 70.3% (941/1338) | 70.7% | 68.9% | 71.0% | 23.2% (310/1338) |
| anthropic:claude-haiku-5-5 | 97.6% (1306/1338) | 97.5% | 98.4% | 97.0% | 1.0% (14/1338) |
| anthropic:claude-opus-5-5 | 100.0% (1338/1338) | 100.0% | 100.0% | 100.0% | 0.0% (0/1338) |

## Accuracy by answer type (original renderings)

| model | count | event_id | magnitude | depth | distance | region | utc_time |
|---|---|---|---|---|---|---|---|
| oracle | 100.0% (117/117) | 100.0% (71/71) | 100.0% (36/36) | 100.0% (66/66) | 100.0% (41/41) | 100.0% (45/45) | 100.0% (21/21) |
| naive | 8.5% (10/117) | 47.9% (34/71) | 63.9% (23/36) | 68.2% (45/66) | 73.2% (30/41) | 84.4% (38/45) | 81.0% (17/21) |
| mock | 77.8% (91/117) | 69.0% (49/71) | 69.4% (25/36) | 72.7% (48/66) | 70.7% (29/41) | 71.1% (32/45) | 76.2% (16/21) |
| anthropic:claude-haiku-5-5 | 94.9% (111/117) | 100.0% (71/71) | 97.2% (35/36) | 95.5% (63/66) | 100.0% (41/41) | 100.0% (45/45) | 100.0% (21/21) |
| anthropic:claude-opus-5-5 | 100.0% (117/117) | 100.0% (71/71) | 100.0% (36/36) | 100.0% (66/66) | 100.0% (41/41) | 100.0% (45/45) | 100.0% (21/21) |

## Generator audit

- **Automated self-check:** for 1191/1191 question-renderings, the digest-only oracle agreed with the code-computed key. Applying the reading rules to each digest recovered the snapshot exactly for 100/100 instances.
- **Human spot-check sample:** 8 of 397 generated items (2.0%; target 2.0%) → `audit/audit_sample.jsonl` (checklist: `audit/audit_sample.md`).
- **Verdicts:** 0/8 reviewed so far. Fill in `verdict` (ok / error) in the JSONL file and re-run `cli.py report` to get an estimated generator error rate.

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
| N02 | new | P02 | dedupe → depth_at_least → count |
| N03 | new | P03 | dedupe → depth_shallower_than → argmax_mag |
| N04 | new | P04 | dedupe → within_distance → max_depth |
| N05 | new | P05 | dedupe → depth_at_least → compare_regions |
| N06 | new | P06 | dedupe → within_distance → time_between → count |
| N07 | new | P07 | dedupe → mag_below → nearest_to |
| N08 | new | P08 | dedupe → within_distance → time_of_largest |
| N09 | new | P09 | dedupe → in_region → rank_mag |
| N10 | new | P10 | dedupe → mag_below → max_mag |
| N11 | new | P11 | dedupe → depth_at_least → min_depth |
| N12 | new | P12 | dedupe → time_between → nearest_distance |

## Per-instance scores (original rendering)

<details><summary>Show table</summary>

| instance | questions | load-bearing traps | oracle | naive | mock | anthropic:claude-haiku-5-5 | anthropic:claude-opus-5-5 |
|---|---|---|---|---|---|---|---|
| inst-000 | 3 | 5 | 100.0% | 66.7% | 100.0% | 66.7% | 100.0% |
| inst-001 | 4 | 3 | 100.0% | 50.0% | 75.0% | 100.0% | 100.0% |
| inst-002 | 4 | 4 | 100.0% | 75.0% | 75.0% | 100.0% | 100.0% |
| inst-003 | 5 | 5 | 100.0% | 80.0% | 100.0% | 100.0% | 100.0% |
| inst-004 | 5 | 5 | 100.0% | 40.0% | 80.0% | 100.0% | 100.0% |
| inst-005 | 4 | 4 | 100.0% | 75.0% | 100.0% | 100.0% | 100.0% |
| inst-006 | 5 | 5 | 100.0% | 60.0% | 80.0% | 100.0% | 100.0% |
| inst-007 | 5 | 4 | 100.0% | 60.0% | 80.0% | 100.0% | 100.0% |
| inst-008 | 5 | 4 | 100.0% | 60.0% | 60.0% | 100.0% | 100.0% |
| inst-009 | 4 | 5 | 100.0% | 50.0% | 75.0% | 100.0% | 100.0% |
| inst-010 | 4 | 3 | 100.0% | 50.0% | 50.0% | 100.0% | 100.0% |
| inst-011 | 3 | 4 | 100.0% | 66.7% | 33.3% | 100.0% | 100.0% |
| inst-012 | 3 | 4 | 100.0% | 33.3% | 66.7% | 100.0% | 100.0% |
| inst-013 | 5 | 5 | 100.0% | 60.0% | 80.0% | 100.0% | 100.0% |
| inst-014 | 3 | 4 | 100.0% | 33.3% | 100.0% | 100.0% | 100.0% |
| inst-015 | 5 | 3 | 100.0% | 40.0% | 60.0% | 100.0% | 100.0% |
| inst-016 | 3 | 4 | 100.0% | 66.7% | 66.7% | 100.0% | 100.0% |
| inst-017 | 4 | 5 | 100.0% | 50.0% | 75.0% | 75.0% | 100.0% |
| inst-018 | 5 | 5 | 100.0% | 40.0% | 60.0% | 100.0% | 100.0% |
| inst-019 | 5 | 4 | 100.0% | 60.0% | 80.0% | 100.0% | 100.0% |
| inst-020 | 3 | 4 | 100.0% | 66.7% | 100.0% | 100.0% | 100.0% |
| inst-021 | 4 | 3 | 100.0% | 50.0% | 100.0% | 100.0% | 100.0% |
| inst-022 | 4 | 4 | 100.0% | 25.0% | 100.0% | 100.0% | 100.0% |
| inst-023 | 3 | 5 | 100.0% | 66.7% | 66.7% | 100.0% | 100.0% |
| inst-024 | 4 | 5 | 100.0% | 50.0% | 75.0% | 100.0% | 100.0% |
| inst-025 | 4 | 4 | 100.0% | 50.0% | 75.0% | 100.0% | 100.0% |
| inst-026 | 5 | 5 | 100.0% | 20.0% | 60.0% | 100.0% | 100.0% |
| inst-027 | 4 | 4 | 100.0% | 50.0% | 100.0% | 100.0% | 100.0% |
| inst-028 | 4 | 4 | 100.0% | 25.0% | 50.0% | 100.0% | 100.0% |
| inst-029 | 4 | 5 | 100.0% | 0.0% | 50.0% | 100.0% | 100.0% |
| inst-030 | 4 | 5 | 100.0% | 25.0% | 75.0% | 100.0% | 100.0% |
| inst-031 | 5 | 5 | 100.0% | 60.0% | 80.0% | 40.0% | 100.0% |
| inst-032 | 3 | 3 | 100.0% | 33.3% | 66.7% | 100.0% | 100.0% |
| inst-033 | 3 | 3 | 100.0% | 66.7% | 66.7% | 100.0% | 100.0% |
| inst-034 | 4 | 5 | 100.0% | 75.0% | 75.0% | 100.0% | 100.0% |
| inst-035 | 3 | 4 | 100.0% | 66.7% | 100.0% | 100.0% | 100.0% |
| inst-036 | 5 | 4 | 100.0% | 40.0% | 80.0% | 100.0% | 100.0% |
| inst-037 | 3 | 2 | 100.0% | 33.3% | 0.0% | 100.0% | 100.0% |
| inst-038 | 3 | 4 | 100.0% | 33.3% | 100.0% | 100.0% | 100.0% |
| inst-039 | 4 | 3 | 100.0% | 50.0% | 75.0% | 100.0% | 100.0% |
| inst-040 | 4 | 4 | 100.0% | 75.0% | 75.0% | 100.0% | 100.0% |
| inst-041 | 3 | 3 | 100.0% | 0.0% | 100.0% | 100.0% | 100.0% |
| inst-042 | 3 | 5 | 100.0% | 66.7% | 100.0% | 100.0% | 100.0% |
| inst-043 | 4 | 4 | 100.0% | 50.0% | 50.0% | 100.0% | 100.0% |
| inst-044 | 5 | 4 | 100.0% | 80.0% | 60.0% | 100.0% | 100.0% |
| inst-045 | 3 | 4 | 100.0% | 33.3% | 100.0% | 100.0% | 100.0% |
| inst-046 | 5 | 5 | 100.0% | 80.0% | 60.0% | 100.0% | 100.0% |
| inst-047 | 4 | 4 | 100.0% | 25.0% | 75.0% | 100.0% | 100.0% |
| inst-048 | 5 | 4 | 100.0% | 40.0% | 80.0% | 100.0% | 100.0% |
| inst-049 | 5 | 5 | 100.0% | 60.0% | 80.0% | 80.0% | 100.0% |
| inst-050 | 3 | 5 | 100.0% | 66.7% | 33.3% | 100.0% | 100.0% |
| inst-051 | 3 | 5 | 100.0% | 0.0% | 100.0% | 100.0% | 100.0% |
| inst-052 | 4 | 4 | 100.0% | 50.0% | 75.0% | 100.0% | 100.0% |
| inst-053 | 3 | 4 | 100.0% | 33.3% | 100.0% | 66.7% | 100.0% |
| inst-054 | 4 | 5 | 100.0% | 75.0% | 50.0% | 100.0% | 100.0% |
| inst-055 | 4 | 4 | 100.0% | 50.0% | 50.0% | 100.0% | 100.0% |
| inst-056 | 3 | 4 | 100.0% | 0.0% | 100.0% | 100.0% | 100.0% |
| inst-057 | 5 | 4 | 100.0% | 80.0% | 80.0% | 100.0% | 100.0% |
| inst-058 | 5 | 5 | 100.0% | 60.0% | 80.0% | 80.0% | 100.0% |
| inst-059 | 5 | 5 | 100.0% | 40.0% | 80.0% | 100.0% | 100.0% |
| inst-060 | 5 | 4 | 100.0% | 60.0% | 40.0% | 100.0% | 100.0% |
| inst-061 | 4 | 4 | 100.0% | 50.0% | 75.0% | 100.0% | 100.0% |
| inst-062 | 5 | 5 | 100.0% | 40.0% | 60.0% | 100.0% | 100.0% |
| inst-063 | 4 | 2 | 100.0% | 50.0% | 100.0% | 75.0% | 100.0% |
| inst-064 | 5 | 4 | 100.0% | 40.0% | 80.0% | 100.0% | 100.0% |
| inst-065 | 3 | 2 | 100.0% | 66.7% | 66.7% | 100.0% | 100.0% |
| inst-066 | 4 | 2 | 100.0% | 50.0% | 25.0% | 100.0% | 100.0% |
| inst-067 | 5 | 4 | 100.0% | 40.0% | 80.0% | 100.0% | 100.0% |
| inst-068 | 4 | 1 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| inst-069 | 4 | 3 | 100.0% | 50.0% | 100.0% | 100.0% | 100.0% |
| inst-070 | 5 | 5 | 100.0% | 20.0% | 60.0% | 80.0% | 100.0% |
| inst-071 | 3 | 2 | 100.0% | 66.7% | 100.0% | 100.0% | 100.0% |
| inst-072 | 3 | 5 | 100.0% | 33.3% | 100.0% | 100.0% | 100.0% |
| inst-073 | 5 | 4 | 100.0% | 60.0% | 60.0% | 100.0% | 100.0% |
| inst-074 | 3 | 5 | 100.0% | 33.3% | 66.7% | 100.0% | 100.0% |
| inst-075 | 4 | 4 | 100.0% | 25.0% | 25.0% | 100.0% | 100.0% |
| inst-076 | 3 | 2 | 100.0% | 33.3% | 66.7% | 100.0% | 100.0% |
| inst-077 | 3 | 4 | 100.0% | 33.3% | 66.7% | 100.0% | 100.0% |
| inst-078 | 4 | 4 | 100.0% | 75.0% | 100.0% | 100.0% | 100.0% |
| inst-079 | 5 | 4 | 100.0% | 60.0% | 100.0% | 100.0% | 100.0% |
| inst-080 | 4 | 4 | 100.0% | 50.0% | 75.0% | 100.0% | 100.0% |
| inst-081 | 4 | 5 | 100.0% | 50.0% | 75.0% | 100.0% | 100.0% |
| inst-082 | 5 | 4 | 100.0% | 60.0% | 100.0% | 100.0% | 100.0% |
| inst-083 | 3 | 5 | 100.0% | 33.3% | 100.0% | 100.0% | 100.0% |
| inst-084 | 3 | 3 | 100.0% | 33.3% | 33.3% | 100.0% | 100.0% |
| inst-085 | 3 | 3 | 100.0% | 0.0% | 66.7% | 100.0% | 100.0% |
| inst-086 | 3 | 4 | 100.0% | 33.3% | 33.3% | 100.0% | 100.0% |
| inst-087 | 3 | 5 | 100.0% | 0.0% | 100.0% | 100.0% | 100.0% |
| inst-088 | 5 | 4 | 100.0% | 20.0% | 80.0% | 100.0% | 100.0% |
| inst-089 | 3 | 4 | 100.0% | 66.7% | 66.7% | 100.0% | 100.0% |
| inst-090 | 4 | 4 | 100.0% | 75.0% | 75.0% | 100.0% | 100.0% |
| inst-091 | 3 | 1 | 100.0% | 66.7% | 66.7% | 100.0% | 100.0% |
| inst-092 | 3 | 5 | 100.0% | 66.7% | 66.7% | 100.0% | 100.0% |
| inst-093 | 5 | 4 | 100.0% | 80.0% | 60.0% | 100.0% | 100.0% |
| inst-094 | 3 | 4 | 100.0% | 33.3% | 0.0% | 100.0% | 100.0% |
| inst-095 | 5 | 4 | 100.0% | 60.0% | 40.0% | 100.0% | 100.0% |
| inst-096 | 5 | 5 | 100.0% | 60.0% | 40.0% | 100.0% | 100.0% |
| inst-097 | 5 | 5 | 100.0% | 80.0% | 80.0% | 100.0% | 100.0% |
| inst-098 | 3 | 4 | 100.0% | 33.3% | 100.0% | 100.0% | 100.0% |
| inst-099 | 4 | 5 | 100.0% | 0.0% | 75.0% | 100.0% | 100.0% |

</details>

## Example instances

### Example 1: `inst-003` (23 earthquakes, 28 report lines, 1 notice, 5 questions)

**Digest (original rendering `v0`, exactly as sent to the models):**

```text
Earthquake wire roundup: the 24 hours ending 2026-04-13 06:00:00 UTC
Sources: regional and global seismic networks. Edition 481-F.

== US feed ==
10:10 AM local time (UTC-12:00) on Apr 12, 2026 — A magnitude 4.2 (mb) earthquake struck Tonga; epicentre (-20.840, -175.480), depth 65.6 mi. The US network lists it as us780574g3 (reviewed).
2026-04-12 22:55:17 UTC ref=us7128wt0a net=us mag=4.2 type=mb region="Tonga" loc=-21.399,-175.065 depth=75.2 km status=reviewed
[US] us77429wiq | 11:44 UTC on 12 April 2026 | Tonga | (-20.629, -176.820) | depth 18.1 km | MB 5.4 | reviewed
11:40 UTC on 12 April 2026 — A magnitude 6.0 (mww) earthquake struck Honshu, Japan; epicentre (36.820, 141.957), depth 17.8 km. The US network lists it as us7074p0wg (reviewed).
3:55 PM local time (UTC-12:00) on Apr 12, 2026 — A magnitude 4.5 (mb) earthquake struck Tonga; epicentre (-20.173, -174.589), depth 180.9 km. The US network lists it as us70374idw (reviewed).
22:25 (UTC-10:00) 11 Apr 2026 ref=ak51330636 net=us mag=3.5 type=ml region="Southern Alaska" loc=60.373,-154.022 depth=32.9 km status=preliminary
05:35 UTC on 13 April 2026 — A magnitude 4.6 (mb) earthquake struck Central Chile; epicentre 33.655°S, 70.653°W, depth 28.6 km (17.8 mi). The US network lists it as us6084crko (reviewed).
8:31 PM local time (UTC+09:00) on Apr 12, 2026 ref=us753189vy net=us mag=4.6 type=mww region="Honshu, Japan" loc=36.712,141.905 depth=13.9 km status=reviewed
US WIRE [2026-04-13T00:29:53Z] MB 4.6 quake in Tonga @ (-19.492, -174.503); 121.0 mi deep. Ref us7728oyth, reviewed.
From the AT network:
2026-04-12 10:43:31 UTC — A magnitude 1.5 (ml) earthquake struck Nevada; epicentre (38.897, -118.801), depth 16.2 km. The AT network lists it as at93307771 (preliminary). This is the same earthquake as nn43624946.
-- PR network --
Sun, 12 Apr 2026 18:30:35 +0000 — A magnitude 1.8 (md) earthquake struck Puerto Rico; epicentre (18.055, -66.833), depth 5.1 km (3.2 mi). The PR network lists it as pr96881023 (reviewed).
== NN feed ==
NN WIRE [Sun, 12 Apr 2026 08:57:30 +0000] ML 1.5 quake in Nevada @ 38.676°N, 117.278°W; 12.5 km deep. Ref nn30103988, reviewed.
2026-04-12 18:34:21 UTC ref=nn12417876 net=nn mag=1.0 type=ml region="Nevada" loc=38.972,-118.326 depth=12.7 km status=reviewed
Sun, 12 Apr 2026 07:09:02 +0000 — A magnitude 1.1 (ml) earthquake struck Nevada; epicentre (38.638, -118.177), depth 7.9 km (4.9 mi). The NN network lists it as nn48989083 (reviewed).
02:42 (UTC-08:00) 12 Apr 2026 ref=nn43624946 net=nn mag=1.6 type=ml region="Nevada" loc=38.901,-118.813 depth=14.0 km (8.7 mi) status=reviewed
07:14 (UTC-08:00) 12 Apr 2026 — A magnitude 1.3 (ml) earthquake struck Nevada; epicentre (38.195, -117.804), depth 13.2 km. The NN network lists it as nn50991702 (reviewed).
== HV feed ==
13:49 UTC on 12 April 2026 — A magnitude 1.8 (md) earthquake struck Hawaii; epicentre 19°07.26'N 155°35.70'W, depth 5.0 km. The HV network lists it as hv57177349 (reviewed).
From the CI network:
2026-04-12 14:18:44 UTC ref=ci39107273 net=ci mag=1.9 type=ml region="Southern California" loc=34.328,-117.255 depth=2.4 km status=reviewed
-- EW network --
EW WIRE [1:21 AM local time (UTC-10:00) on Apr 12, 2026] ML 1.9 quake in Southern Alaska @ (60.077, -152.046); 54.4 km deep. Ref ak75595810, preliminary.
-- NC network --
NC WIRE [00:17 (UTC-08:00) 12 Apr 2026] MD 2.7 quake in Northern California @ (38.942, -123.224); 5.3 km (3.3 mi) deep. Ref nc62863253, reviewed.
Editor's note: entries are reproduced exactly as each network filed them, time conventions included.
== PT feed ==
10:46 PM local time (UTC-10:00) on Apr 11, 2026 — A magnitude 4.1 (ml) earthquake struck Southern Alaska; epicentre (60.273, -155.441), depth 24.7 km (15.3 mi). The PT network lists it as pt20530402 (preliminary). This is the same earthquake as ak96782286.
== AK feed ==
[AK] ak62259644 | 2026-04-13T04:25:16Z | Southern Alaska | (59.578, -154.722) | depth 27.5 mi | ML 0.9 | reviewed
2026-04-12 11:21:43 UTC — A magnitude 1.0 (ml) earthquake struck Southern Alaska; epicentre (60.077, -152.046), depth 54.4 km. The AK network lists it as ak75595810 (reviewed).
[AK] ak51330636 | 2026-04-12 08:25:38 UTC | Southern Alaska | 60.341°N, 154.005°W | depth 30.7 km | M3.6 (ml) | reviewed
2026-04-12T11:00:13Z — A magnitude 1.1 (ml) earthquake struck Southern Alaska; epicentre (60.545, -151.867), depth 102.7 km. The AK network lists it as ak97660683 (reviewed).
Station US.CMH was offline from 13:00 to 14:36 UTC for maintenance; its readings are not included.
[AK] ak62486964 | Sun, 12 Apr 2026 19:22:20 +0000 | Southern Alaska | 61.176°N, 151.677°W | depth 8.7 km | ML 1.4 | reviewed
2026-04-12 08:46:23 UTC — A magnitude 4.0 (ml) earthquake struck Southern Alaska; epicentre 60.333°N, 155.385°W, depth 22.8 km. The AK network lists it as ak96782286 (reviewed).
[AK] ak89400569 | 04:07 UTC on 13 April 2026 | Southern Alaska | 59°42.30'N 151°27.54'W | depth 19.3 km | M2.4 (ml) | reviewed

Corrections and retractions:
NOTICE [2026-04-12T12:25:24Z]: report us753189vy has been RETRACTED (mine blast). Disregard it.
```

**Questions, code-computed key and model answers (`v0`):**

| Q | template | question | key | oracle | naive | mock | anthropic:claude-haiku-5-5 | anthropic:claude-opus-5-5 |
|---|---|---|---|---|---|---|---|---|
| Q1 | N08 (new) | When, in UTC (HH:MM), was the strongest of the earthquakes within 75 miles of station AK-85 (61.231°N, 152.402°W) recorded? | 19:22 | 19:22 ✓ | 19:22 ✓ | 19:22 ✓ | 19:22 ✓ | 19:22 ✓ |
| Q2 | P11 (public) | What was the smallest depth, in km, among the earthquakes that were recorded from 12:10 UTC on 12 April 2026 up to 23:00 UTC on 12 April 2026? Round to 1 decimal place. | 2.4 | 2.4 ✓ | 2.4 ✓ | 2.4 ✓ | 2.4 ✓ | 2.4 ✓ |
| Q3 | P04 (public) | Among the earthquakes with magnitude at least 1.8, how deep was the deepest one, in km? Round to 1 decimal place. | 194.7 | 194.7 ✓ | 194.7 ✓ | 194.7 ✓ | 194.7 ✓ | 194.7 ✓ |
| Q4 | P01 (public) | Count the distinct earthquakes of magnitude 1.5 or greater. | 16 | 16 ✓ | 21 ✗ | 16 ✓ | 16 ✓ | 16 ✓ |
| Q5 | N07 (new) | Among the earthquakes with magnitude below 1.3, give the event ID of the one nearest to station AK-51 (60.163°N, 151.119°W). | ak75595810 | ak75595810 ✓ | ak75595810 ✓ | ak75595810 ✓ | ak75595810 ✓ | ak75595810 ✓ |

**Operation chains (the machine-readable meaning of each question):**

- Q1: `dedupe → within 75 mi of (61.231, -152.402) → time_of_largest`
- Q2: `dedupe → time in [2026-04-12T12:10:00Z, 2026-04-12T23:00:00Z) → min_depth → convert → km → round(1)`
- Q3: `dedupe → mag ≥ 1.8 → max_depth → convert → km → round(1)`
- Q4: `dedupe → mag ≥ 1.5 → count`
- Q5: `dedupe → mag < 1.3 → nearest_to(60.163, -151.119)`

**Traps in this digest:**

- T1 *duplicate* (same_id): US re-reported ak51330636 under the same event ID: preliminary M3.5, depth 32.9 km (reviewed: M3.6, 30.7 km). If mishandled: Q4 → 17.
- T2 *duplicate* (cross_ref): AT re-reported nn43624946 as at93307771 with a 'same event as' note: preliminary M1.5, depth 16.2 km (reviewed: M1.6, 14 km). If mishandled: Q4 → 17.
- T3 *duplicate* (cross_ref): PT re-reported ak96782286 as pt20530402 with a 'same event as' note: preliminary M4.1, depth 24.7 km (reviewed: M4, 22.8 km). If mishandled: Q4 → 17.
- T4 *retraction* (phantom): phantom report us753189vy (M4.6, Honshu, Japan) later retracted: mine blast. If mishandled: Q4 → 17.
- T5 *conflicting magnitude* (second_report): ak75595810 also reported as M1.9 (preliminary, EW) beside the reviewed M1. If mishandled: Q4 → 17, Q5 → ak97660683.

<details><summary>Metamorphic variant `v1` of the same instance (same key)</summary>

```text
== QUAKE DIGEST 500-K ==
Window: the 24 hours up to Mon, 13 Apr 2026 06:00:00 +0000. Entries are listed as filed by each network.

2026-04-12 11:40:23 UTC — A magnitude 6.0 (mww) earthquake struck Honshu, Japan; epicentre 36.820°N, 141.957°E, depth 17.8 km. The US network lists it as us7074p0wg (reviewed).
NN WIRE [12:57 AM local time (UTC-08:00) on Apr 12, 2026] ML 1.5 quake in Nevada @ 38.676°N, 117.278°W; 7.8 mi deep. Ref nn30103988, reviewed.
6:25 PM local time (UTC-10:00) on Apr 12, 2026 ref=ak62259644 net=ak mag=0.9 type=ml region="Southern Alaska" loc=59.578,-154.722 depth=27.5 mi status=reviewed
US WIRE [22:25 (UTC-10:00) 11 Apr 2026] ML 3.5 quake in Southern Alaska @ 60.373°N, 154.022°W; 20.4 mi deep. Ref ak51330636, preliminary.
AK WIRE [1:21 AM local time (UTC-10:00) on Apr 12, 2026] ML 1.0 quake in Southern Alaska @ 60.077°N, 152.046°W; 33.8 mi deep. Ref ak75595810, reviewed.
RETRACTION [12:25 UTC on 12 April 2026]: us753189vy was not an earthquake (mine blast); ignore that report.
3:55 PM local time (UTC-12:00) on Apr 12, 2026 — A magnitude 4.5 (mb) earthquake struck Tonga; epicentre (-20.173, -174.589), depth 112.4 mi. The US network lists it as us70374idw (reviewed).
[US] us753189vy | 8:31 PM local time (UTC+09:00) on Apr 12, 2026 | Honshu, Japan | 36°42.72'N 141°54.30'E | depth 8.6 mi | MWW 4.6 | reviewed
AK WIRE [10:46 PM local time (UTC-10:00) on Apr 11, 2026] ML 4.0 quake in Southern Alaska @ 60.333°N, 155.385°W; 14.2 mi deep. Ref ak96782286, reviewed.
2026-04-12 10:42:58 UTC — A magnitude 1.6 (ml) earthquake struck Nevada; epicentre 38.901°N, 118.813°W, depth 8.7 mi. The NN network lists it as nn43624946 (reviewed).
3:49 AM local time (UTC-10:00) on Apr 12, 2026 ref=hv57177349 net=hv mag=1.8 type=md region="Hawaii" loc=19.121,-155.595 depth=3.1 mi status=reviewed
2:43 AM local time (UTC-08:00) on Apr 12, 2026 — A magnitude 1.5 (ml) earthquake struck Nevada; epicentre 38.897°N, 118.801°W, depth 16.2 km (10.1 mi). The AT network lists it as at93307771 (preliminary). This is the same earthquake as nn43624946.
Station US.CMH was offline from 13:00 to 14:36 UTC for maintenance; its readings are not included.
10:55 (UTC-12:00) 12 Apr 2026 — A magnitude 4.2 (mb) earthquake struck Tonga; epicentre 21.399°S, 175.065°W, depth 46.7 mi. The US network lists it as us7128wt0a (reviewed).
NN WIRE [2026-04-12T15:14:56Z] ML 1.3 quake in Nevada @ 38.195°N, 117.804°W; 8.2 mi deep. Ref nn50991702, reviewed.
2026-04-12 08:25:38 UTC ref=ak51330636 net=ak mag=3.6 type=ml region="Southern Alaska" loc=60.341,-154.005 depth=19.1 mi status=reviewed
12:17 AM local time (UTC-08:00) on Apr 12, 2026 — A magnitude 2.7 (md) earthquake struck Northern California; epicentre 38.942°N, 123.224°W, depth 3.3 mi. The NC network lists it as nc62863253 (reviewed).
[PT] pt20530402 | 10:46 PM local time (UTC-10:00) on Apr 11, 2026 | Southern Alaska | 60.273°N, 155.441°W | depth 15.3 mi | M4.1 (ml) | preliminary | same event as ak96782286
NN WIRE [10:34 (UTC-08:00) 12 Apr 2026] ML 1.0 quake in Nevada @ (38.972, -118.326); 7.9 mi deep. Ref nn12417876, reviewed.
[AK] ak62486964 | 19:22 UTC on 12 April 2026 | Southern Alaska | 61.176°N, 151.677°W | depth 8.7 km (5.4 mi) | ML 1.4 | reviewed
2026-04-13T00:29:53Z ref=us7728oyth net=us mag=4.6 type=mb region="Tonga" loc=-19.492,-174.503 depth=121.0 mi status=reviewed
[EW] ak75595810 | 01:21 (UTC-10:00) 12 Apr 2026 | Southern Alaska | 60.077°N, 152.046°W | depth 33.8 mi | ML 1.9 | preliminary
2026-04-13 05:35:35 UTC ref=us6084crko net=us mag=4.6 type=mb region="Central Chile" loc=-33.655,-70.653 depth=17.8 mi status=reviewed
US WIRE [10:10 (UTC-12:00) 12 Apr 2026] MB 4.2 quake in Tonga @ 20°50.40'S 175°28.80'W; 65.6 mi deep. Ref us780574g3, reviewed.
11:09 PM local time (UTC-08:00) on Apr 11, 2026 — A magnitude 1.1 (ml) earthquake struck Nevada; epicentre 38°38.28'N 118°10.62'W, depth 4.9 mi. The NN network lists it as nn48989083 (reviewed).
PR WIRE [14:30 (UTC-04:00) 12 Apr 2026] MD 1.8 quake in Puerto Rico @ (18.055, -66.833); 3.2 mi deep. Ref pr96881023, reviewed.
1:00 AM local time (UTC-10:00) on Apr 12, 2026 — A magnitude 1.1 (ml) earthquake struck Southern Alaska; epicentre 60.545°N, 151.867°W, depth 63.8 mi. The AK network lists it as ak97660683 (reviewed).
AK WIRE [Mon, 13 Apr 2026 04:07:59 +0000] ML 2.4 quake in Southern Alaska @ 59.705°N, 151.459°W; 12.0 mi deep. Ref ak89400569, reviewed.
23:44 (UTC-12:00) 11 Apr 2026 — A magnitude 5.4 (mb) earthquake struck Tonga; epicentre 20.629°S, 176.820°W, depth 11.2 mi. The US network lists it as us77429wiq (reviewed).
[CI] ci39107273 | 06:18 (UTC-08:00) 12 Apr 2026 | Southern California | 34.328°N, 117.255°W | depth 1.5 mi | M1.9 (ml) | reviewed
Editor's note: entries are reproduced exactly as each network filed them, time conventions included.
```

- Q1: At what time (UTC, HH:MM) did the largest of the earthquakes no more than 75 miles from station AK-85 (61.231°N, 152.402°W) occur?
- Q2: What was the smallest depth, in km, among the earthquakes that were recorded from 2026-04-12 12:10 UTC up to 2026-04-12 23:00 UTC? Round to 1 decimal place.
- Q3: Among the earthquakes with magnitude at least 1.8, how deep was the deepest one, in km? Round to 1 decimal place.
- Q4: Count the distinct earthquakes of magnitude 1.5 or more.
- Q5: Among the earthquakes with magnitude below 1.3, give the event ID of the one nearest to station AK-51 (60.163°N, 151.119°W).

</details>

### Example 2: `inst-004` (23 earthquakes, 28 report lines, 1 notice, 5 questions)

**Digest (original rendering `v0`, exactly as sent to the models):**

```text
SEISMIC ACTIVITY DIGEST (edition 644-J)
Compiled Mon, 13 Apr 2026 06:00:00 +0000; covers the 24 hours before compilation. Reports are reproduced as received.

Sun, 12 Apr 2026 08:27:11 +0000 — A magnitude 5.4 (mw) earthquake struck Hawaii; epicentre 19.405°N, 155.279°W, depth 0.1 km (0.1 mi). The EW network lists it as ew93062955 (preliminary). This is the same earthquake as hv57799647.
[AT] us6507h7kn | 12:12 PM local time (UTC+09:00) on Apr 13, 2026 | Honshu, Japan | (37.334, 141.409) | depth 69.0 km | MB 4.6 | preliminary
NC WIRE [08:17 UTC on 12 April 2026] MD 1.2 quake in Northern California @ 38.514°N, 122.754°W; 7.1 mi deep. Ref nc99579054, reviewed.
US WIRE [09:37 (UTC-05:00) 12 Apr 2026] MB 4.2 quake in Central Chile @ 33.834°S, 70.690°W; 21.2 mi deep. Ref us68032lh6, reviewed.
14:42 (UTC-04:00) 12 Apr 2026 ref=pr13717605 net=pr mag=1.3 type=md region="Puerto Rico" loc=18.323,-66.709 depth=2.2 mi status=reviewed
[AK] ak83586299 | 13:22 (UTC-10:00) 12 Apr 2026 | Southern Alaska | (60.702, -151.638) | depth 98.8 km (61.4 mi) | ML 1.1 | reviewed
19:35 (UTC-10:00) 12 Apr 2026 — A magnitude 1.0 (md) earthquake struck Hawaii; epicentre 19.221°N, 155.203°W, depth 13.2 km. The HV network lists it as hv17879852 (reviewed).
2026-04-13T00:57:44Z — A magnitude 1.1 (md) earthquake struck Puerto Rico; epicentre (18.059, -66.854), depth 17.0 mi. The PR network lists it as pr46139773 (reviewed).
NC WIRE [08:02 (UTC-08:00) 12 Apr 2026] MD 2.0 quake in Northern California @ (38.802, -122.938); 1.8 mi deep. Ref nc45177335, preliminary.
Felt reports are still being collected and are not summarised here.
HV WIRE [16:47 (UTC-10:00) 12 Apr 2026] MD 0.8 quake in Hawaii @ 19.336°N, 155.319°W; 0.4 mi deep. Ref hv82291455, reviewed.
10:19 AM local time (UTC+09:00) on Apr 13, 2026 — A magnitude 4.5 (mb) earthquake struck Honshu, Japan; epicentre 36.205°N, 140.859°E, depth 18.3 mi. The US network lists it as us7972msjn (reviewed).
[HV] hv48315658 | 8:02 AM local time (UTC-10:00) on Apr 12, 2026 | Hawaii | 19°05.10'N 154°43.02'W | depth 1.3 km | MD 0.9 | reviewed
[AK] ak73646881 | 16:46 (UTC-10:00) 12 Apr 2026 | Southern Alaska | 59.942°N, 151.884°W | depth 11.2 km | M1.2 (ml) | reviewed
7:09 PM local time (UTC-10:00) on Apr 12, 2026 — A magnitude 3.5 (ml) earthquake struck Southern Alaska; epicentre 61.563°N, 151.086°W, depth 54.6 mi. The AK network lists it as ak51444988 (reviewed).
[HV] hv57799647 | 2026-04-12 08:27:30 UTC | Hawaii | (19.409, -155.328) | depth 0.9 km | M5.1 (mw) | reviewed
12:48 AM local time (UTC-10:00) on Apr 12, 2026 — A magnitude 1.1 (ml) earthquake struck Southern Alaska; epicentre 60.643°N, 152.388°W, depth 32.5 km (20.2 mi). The AK network lists it as ak82817824 (reviewed).
HV WIRE [2026-04-12T13:49:54Z] MD 1.8 quake in Hawaii @ 19.121°N, 155.595°W; 5.0 km (3.1 mi) deep. Ref hv57177349, reviewed.
NC WIRE [Sun, 12 Apr 2026 16:02:22 +0000] MD 1.2 quake in Northern California @ 38.802°N, 122.938°W; 2.9 km deep. Ref nc45177335, reviewed.
19:41 (UTC-04:00) 12 Apr 2026 — A magnitude 2.5 (md) earthquake struck Puerto Rico; epicentre (18.319, -66.770), depth 4.5 km (2.8 mi). The PR network lists it as pr10083501 (reviewed).
Sun, 12 Apr 2026 17:50:41 +0000 ref=uu84607107 net=uu mag=1.0 type=ml region="Utah" loc=39.286,-111.684 depth=2.7 mi status=reviewed
11:47 PM local time (UTC-04:00) on Apr 12, 2026 ref=pr21452414 net=pr mag=1.5 type=md region="Puerto Rico" loc=18.139,-66.796 depth=17.6 km status=reviewed
8:05 AM local time (UTC-10:00) on Apr 12, 2026 ref=ak53937112 net=ak mag=1.0 type=ml region="Southern Alaska" loc=59.892,-150.906 depth=50.2 km status=reviewed
US WIRE [2026-04-13 03:11:56 UTC] MB 4.3 quake in Honshu, Japan @ 37.325°N, 141.490°E; 71.7 km deep. Ref us6507h7kn, reviewed.
22:46 (UTC-10:00) 11 Apr 2026 — A magnitude 1.0 (ml) earthquake struck Southern Alaska; epicentre (60.296, -151.189), depth 15.4 mi. The AK network lists it as ak86711893 (reviewed).
03:01 (UTC-10:00) 12 Apr 2026 ref=ak38693693 net=ak mag=1.5 type=ml region="Southern Alaska" loc=61.408,-154.766 depth=5.8 km (3.6 mi) status=reviewed
[PR] pr79834763 | 4:36 PM local time (UTC-04:00) on Apr 12, 2026 | Puerto Rico | (18.211, -66.848) | depth 7.2 mi | MD 0.8 | reviewed
[US] us698159t5 | Sun, 12 Apr 2026 16:35:39 +0000 | Honshu, Japan | (39.704, 142.434) | depth 37.1 km | MWW 5.6 | reviewed
[EW] pr10083501 | 7:41 PM local time (UTC-04:00) on Apr 12, 2026 | Puerto Rico | 18.315°N, 66.788°W | depth 7.2 km (4.5 mi) | M2.7 (md) | preliminary

Corrections and retractions:
NOTICE [2026-04-12T19:07:17Z]: report pr13717605 has been RETRACTED (sonic boom). Disregard it.
```

**Questions, code-computed key and model answers (`v0`):**

| Q | template | question | key | oracle | naive | mock | anthropic:claude-haiku-5-5 | anthropic:claude-opus-5-5 |
|---|---|---|---|---|---|---|---|---|
| Q1 | P10 (public) | What was the largest magnitude among the earthquakes at least 16.3 miles deep? | 5.6 | 5.6 ✓ | 5.6 ✓ | 5.6 ✓ | 5.6 ✓ | 5.6 ✓ |
| Q2 | P01 (public) | Count the distinct earthquakes of magnitude 1.8 or greater. | 8 | 8 ✓ | 12 ✗ | 8 ✓ | 8 ✓ | 8 ✓ |
| Q3 | N09 (new) | Rank the earthquakes whose region is listed as Hawaii by magnitude, largest first: which event is second? Give its event ID. | hv57177349 | hv57177349 ✓ | hv57799647 ✗ | hv57177349 ✓ | hv57177349 ✓ | hv57177349 ✓ |
| Q4 | N10 (new) | What was the largest magnitude among the earthquakes with magnitude below 1.5? | 1.2 | 1.2 ✓ | 1.3 ✗ | 1.3 ✗ | 1.2 ✓ | 1.2 ✓ |
| Q5 | N07 (new) | Among the earthquakes with magnitude below 1.1, give the event ID of the one nearest to station AK-35 at (60.258, -151.093). | ak86711893 | ak86711893 ✓ | ak86711893 ✓ | ak86711893 ✓ | ak86711893 ✓ | ak86711893 ✓ |

**Operation chains (the machine-readable meaning of each question):**

- Q1: `dedupe → depth ≥ 16.3 mi → max_mag`
- Q2: `dedupe → mag ≥ 1.8 → count`
- Q3: `dedupe → region = Hawaii → rank_mag(k=2)`
- Q4: `dedupe → mag < 1.5 → max_mag`
- Q5: `dedupe → mag < 1.1 → nearest_to(60.258, -151.093)`

**Traps in this digest:**

- T1 *duplicate* (same_id): AT re-reported us6507h7kn under the same event ID: preliminary M4.6, depth 69 km (reviewed: M4.3, 71.7 km). If mishandled: Q2 → 9.
- T2 *duplicate* (same_id): EW re-reported pr10083501 under the same event ID: preliminary M2.7, depth 7.2 km (reviewed: M2.5, 4.5 km). If mishandled: Q2 → 9.
- T3 *duplicate* (cross_ref): EW re-reported hv57799647 as ew93062955 with a 'same event as' note: preliminary M5.4, depth 0.1 km (reviewed: M5.1, 0.9 km). If mishandled: Q2 → 9, Q3 → hv57799647.
- T4 *retraction* (phantom): phantom report pr13717605 (M1.3, Puerto Rico) later retracted: sonic boom. If mishandled: Q4 → 1.3.
- T5 *conflicting magnitude* (second_report): nc45177335 also reported as M2 (preliminary, NC) beside the reviewed M1.2. If mishandled: Q2 → 9.

<details><summary>Metamorphic variant `v1` of the same instance (same key)</summary>

```text
Earthquake wire roundup: the 24 hours ending Mon, 13 Apr 2026 06:00:00 +0000
Sources: regional and global seismic networks. Edition 119-F.

[NC] nc99579054 | 12:17 AM local time (UTC-08:00) on Apr 12, 2026 | Northern California | 38°30.84'N 122°45.24'W | depth 7.1 mi | M1.2 (md) | reviewed
[EW] ew93062955 | Sun, 12 Apr 2026 08:27:11 +0000 | Hawaii | (19.405, -155.279) | depth 0.1 mi | M5.4 (mw) | preliminary | same event as hv57799647
[HV] hv57799647 | 22:27 (UTC-10:00) 11 Apr 2026 | Hawaii | (19.409, -155.328) | depth 0.9 km (0.6 mi) | M5.1 (mw) | reviewed
AK WIRE [2026-04-12T08:46:10Z] ML 1.0 quake in Southern Alaska @ 60.296°N, 151.189°W; 15.4 mi deep. Ref ak86711893, reviewed.
12:48 AM local time (UTC-10:00) on Apr 12, 2026 — A magnitude 1.1 (ml) earthquake struck Southern Alaska; epicentre (60.643, -152.388), depth 20.2 mi. The AK network lists it as ak82817824 (reviewed).
[AK] ak38693693 | Sun, 12 Apr 2026 13:01:17 +0000 | Southern Alaska | 61°24.48'N 154°45.96'W | depth 3.6 mi | M1.5 (ml) | reviewed
03:49 (UTC-10:00) 12 Apr 2026 — A magnitude 1.8 (md) earthquake struck Hawaii; epicentre (19.121, -155.595), depth 5.0 km. The HV network lists it as hv57177349 (reviewed).
[US] us68032lh6 | 09:37 (UTC-05:00) 12 Apr 2026 | Central Chile | 33.834°S, 70.690°W | depth 34.1 km | MB 4.2 | reviewed
2026-04-12T16:02:22Z — A magnitude 1.2 (md) earthquake struck Northern California; epicentre (38.802, -122.938), depth 2.9 km (1.8 mi). The NC network lists it as nc45177335 (reviewed).
[NC] nc45177335 | 8:02 AM local time (UTC-08:00) on Apr 12, 2026 | Northern California | (38.802, -122.938) | depth 1.8 mi | MD 2.0 | preliminary
01:35 (UTC+09:00) 13 Apr 2026 — A magnitude 5.6 (mww) earthquake struck Honshu, Japan; epicentre 39°42.24'N 142°26.04'E, depth 37.1 km. The US network lists it as us698159t5 (reviewed).
UU WIRE [2026-04-12 17:50:41 UTC] ML 1.0 quake in Utah @ 39°17.16'N 111°41.04'W; 2.7 mi deep. Ref uu84607107, reviewed.
[HV] hv48315658 | Sun, 12 Apr 2026 18:02:40 +0000 | Hawaii | (19.085, -154.717) | depth 0.8 mi | M0.9 (md) | reviewed
Felt reports are still being collected and are not summarised here.
18:05 UTC on 12 April 2026 ref=ak53937112 net=ak mag=1.0 type=ml region="Southern Alaska" loc=59.892,-150.906 depth=31.2 mi status=reviewed
2:42 PM local time (UTC-04:00) on Apr 12, 2026 — A magnitude 1.3 (md) earthquake struck Puerto Rico; epicentre (18.323, -66.709), depth 2.2 mi. The PR network lists it as pr13717605 (reviewed).
[PR] pr79834763 | 2026-04-12T20:36:54Z | Puerto Rico | 18°12.66'N 66°50.88'W | depth 11.6 km (7.2 mi) | M0.8 (md) | reviewed
[AK] ak83586299 | 1:22 PM local time (UTC-10:00) on Apr 12, 2026 | Southern Alaska | (60.702, -151.638) | depth 61.4 mi | M1.1 (ml) | reviewed
[EW] pr10083501 | 2026-04-12T23:41:03Z | Puerto Rico | (18.315, -66.788) | depth 4.5 mi | MD 2.7 | preliminary
2026-04-12 23:41:19 UTC — A magnitude 2.5 (md) earthquake struck Puerto Rico; epicentre 18.319°N, 66.770°W, depth 4.5 km (2.8 mi). The PR network lists it as pr10083501 (reviewed).
[PR] pr46139773 | Mon, 13 Apr 2026 00:57:44 +0000 | Puerto Rico | (18.059, -66.854) | depth 27.4 km (17.0 mi) | MD 1.1 | reviewed
[US] us7972msjn | 2026-04-13 01:19:16 UTC | Honshu, Japan | (36.205, 140.859) | depth 18.3 mi | MB 4.5 | reviewed
2026-04-13 02:46:13 UTC — A magnitude 1.2 (ml) earthquake struck Southern Alaska; epicentre (59.942, -151.884), depth 11.2 km (7.0 mi). The AK network lists it as ak73646881 (reviewed).
16:47 (UTC-10:00) 12 Apr 2026 ref=hv82291455 net=hv mag=0.8 type=md region="Hawaii" loc=19.336,-155.319 depth=0.4 mi status=reviewed
[US] us6507h7kn | 12:11 PM local time (UTC+09:00) on Apr 13, 2026 | Honshu, Japan | (37.325, 141.490) | depth 44.6 mi | MB 4.3 | reviewed
UPDATE [2026-04-12T19:07:17Z]: pr13717605 withdrawn (sonic boom).
12:12 PM local time (UTC+09:00) on Apr 13, 2026 ref=us6507h7kn net=at mag=4.6 type=mb region="Honshu, Japan" loc=37.334,141.409 depth=42.9 mi status=preliminary
[PR] pr21452414 | Mon, 13 Apr 2026 03:47:26 +0000 | Puerto Rico | (18.139, -66.796) | depth 17.6 km | M1.5 (md) | reviewed
[AK] ak51444988 | 7:09 PM local time (UTC-10:00) on Apr 12, 2026 | Southern Alaska | (61.563, -151.086) | depth 54.6 mi | M3.5 (ml) | reviewed
[HV] hv17879852 | 2026-04-13T05:35:16Z | Hawaii | 19°13.26'N 155°12.18'W | depth 8.2 mi | M1.0 (md) | reviewed
```

- Q1: Among the earthquakes at least 16.3 miles deep, what was the highest magnitude recorded?
- Q2: Count the distinct earthquakes of magnitude 1.8 or more.
- Q3: Rank the earthquakes whose region is listed as Hawaii by magnitude, largest first: which event is second? Give its event ID.
- Q4: Among the earthquakes smaller than magnitude 1.5, what was the highest magnitude recorded?
- Q5: Among the earthquakes smaller than magnitude 1.1, give the event ID of the one nearest to station AK-35 (60.258°N, 151.093°W).

</details>
