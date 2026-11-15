# evalgen 0.3.0: validation numbers at its final code commit b5a8d59

Compiled 2026-11-14 from the measured outputs in this folder (calibration, trap shortcuts, identity; the demo report is not committed),
the run folders kept outside the repository (live replay), the blind-reader counts and the checks named below. Every figure carries its source
in brackets; nothing is taken from a document. Where a figure of the earlier 4179c0c round is needed for comparison,
its source is the 4179c0c output file, named in full. Figures not recomputed at b5a8d59 are marked as such (section 14).

## Provenance

- **Code.** Commit `b5a8d59e370b82343f5f83b6bed04d35b2363202` ("Final audit fixes: review corrections", branch
  multi-source). Tree code fingerprint `11d06dc71115a544687f2f1d9568cd8954cefb5f3c207335f8a9f7507c70409f`:
  every one of the 870 calibration result rows carries it as `code_sha256` and `commit` = b5a8d59
  [`calibration/calib-*/results.jsonl`; `calibration/calib-1-40/checks.json` (`tree_code_fingerprint`)]. The extracted tree
  `tree` is `git archive b5a8d59` (114 files), re-extracted for section 9 and identical (a second extraction, compared with `diff -rq`; not committed).
- **Tests.** 1328 passed, 0 failed, 264.9 s [`tests-final.log`; `repair_result.json`].
- **Clean checkout.** The code was extracted from the commit with `git archive`; nothing was committed or pushed for the measurement.
- **Other commits named.** 4179c0c (`4179c0ce3cb2de5e6c7874d30df12937b88ef66b`, the previous final round,
  code_sha256 `3ef7d22f...`); 8e63aee (code_sha256 `71e4a019...`); 8ef8657 (live run, code_sha256 `c281a51a...`);
  b9f29b4 (0.2.0, code_sha256 `dc14c8a0...`) [`runs/live-smoke-2026-11-12/manifest.json`; the 0.2.0 demo manifest is not committed].
- **Calibration seeds and configs.** Random mix: seeds 1-120 (120 generation runs x 7 models = 840 rows). Confounded
  mix: seeds 1-5, 41-45, 81-85 (15 runs x 2 models = 30 rows). 870 rows, 135 generation runs, 100 instances each,
  2 variants (3 renderings). Each slice has its own config (`calibration/calib-*/config.json`, `config-confounded.json`):
  random mix holds out the weather domain and the email and radio genres.
- **Calibration commands** [`run_calib.sh`]:
  ```
  generate --mix random --private-config CONFIG --seed S --instances 100 --variants 2 --out RUN
  run RUN --private-config CONFIG --models oracle,naive,mock,mock:held_out_domains=0.15,mock:held_out_domains=0.3,mock:held_out_genres=0.15,mock:held_out_genres=0.3
  score RUN --private-config CONFIG ;  report RUN --private-config CONFIG
  # confounded: --mix neo:email,weather:radio,earthquakes:classic,invented:news, --models mock,mock:held_out_domains=0.3
  ```
  Under `nice -n 10`, `python3 -B`, every HTTP(S) and ALL proxy at 127.0.0.1:9, API-key variables unset.
- **Pooling and statistics.** `calibration/pool_check.py` on `calibration/`: re-run here; `calibration/pool_check.txt` is its output
  (870 rows, 840 random, 30 confounded). Wilson 95% for rates over runs. Means and seed-pooled rates: a seed-cluster
  percentile bootstrap (10,000 resamples of the 120 seeds, numpy seed 20261114). pool_check.py gives no bootstrap for means; those intervals come from a separate checker that is not in the repository. The self-check, all-renderings accuracy and recovery come from the calibration checks in this folder.
- **Trap shortcuts.** Seeds 5001-5084 (84 seeds): L = legacy (plain generate), G = every built-in domain with the genre
  rotating by seed, R = `--mix random`, C = each domain in classic; 3 instances x 3 renderings per run. Trees:
  `tree` (b5a8d59; manifests code_sha256 `11d06dc7...`, checked on 400 of them). Scripts:
  the shortcut scripts (not committed); the
  scripts folder is `shortcuts/scripts/`. Outputs: `shortcuts/` (tables.md, summary.json, *-paired.txt,
  *-auc-fail.txt, *-sc-table.txt, L-runs-compare.json, L-check-same.json). The `report.md` there is stale (section 14).
- **Demo.** `cli.py demo --mix random --private-config <the demo's private config, not committed> --seed 42 --instances 40 --out <output folder> --force`,
  then `verify-config`: rc 0, "ok" (the demo report and its manifest are not committed).
- **Identity.** `demo --seed 42 --instances 20` and `generate --seed 16222556419173536875 --instances 100`, run in 0.2.0 (b9f29b4) and at b5a8d59 with the same `--out` names and compared file by file (`identity-result.txt`; the comparison script is not committed).
- **Live replay.** Section 9.

## Headline

At b5a8d59, over 120 random-mix calibration seeds of 100 instances each, the adjusted source and genre gaps raised 2 false flags in 720 checks of the three null models (0.3%, Wilson 95% [0.1, 1.0]), flagged a 30-point held-out-domain penalty in 95.8% [90.6, 98.2] of runs and a 15-point one in 50.0% [41.2, 58.8], and flagged 30- and 15-point held-out-genre penalties in 100% [96.9, 100] and 90.8% [84.3, 94.8]; the adjusted intervals contained the injected gap in 93.5% [91.0, 95.4] of penalised runs.
On 252 legacy 0.2.0-style digests, the best rule-free reader, picked after the fact from the readers in summary.json, closes 38.9% [35.3, 42.9] of the naive-to-oracle gap against 41.1% [37.3, 44.9] for the status rule (paired lead 2.2 [-0.6, +4.9]), while on 1,008 digests of every domain and genre the best rule-free reader closes 0.0% [-0.7, +0.8] against 37.2% [34.5, 39.8] for the status rule.
The 0.2.0 regression holds: the 100-instance legacy generation and the 20-instance legacy demo at b5a8d59 are byte-identical to 0.2.0's 677 files once the version, code fingerprint, generation time, seed nonce and commitment, and the lines that name them are masked; a fresh live run at b5a8d59 (2026-11-15) replays offline with identical instances, prompts, audit and forecast files (section 9).

## 0. Calibration setup

| item | value | source |
|---|---|---|
| generation runs / report rows | 135 runs; 870 rows (random 120 x 7 = 840; confounded 15 x 2 = 30) | `pool_check.txt` |
| per run | 100 instances, 3 renderings (2 variants), 377-421 original-rendering questions | `results.jsonl` (`accuracy.n`: min 377, max 421) |
| held out, random mix | weather (domain; public: earthquakes, invented, neo); email and radio (genres; public: chat, classic, form, news, table) | `calib-*/config.json` |
| models | oracle; naive reader; mock (built-in template contamination, no held-out effect); four penalised mocks at -0.15 or -0.30 on held-out domain instances or held-out genre renderings | `run_calib.sh` |
| generator self-check | 161,958 of 161,958 agree (1,007,454 / 7 random + 36,072 / 2 confounded, each generation counted once); 1131-1263 checks a run | `results.jsonl` (`self_check`), `calib_more.py` |
| oracle | 100.0% on originals in all 120 runs; 100.00% on all renderings | `results.jsonl`, `calib_more.py` |
| failed steps | 0 | `checks.json` (every run `ok`) |

## 1. False flags under the null (random mix, 120 seeds)

Flag = the report's own flag (adjusted gap above 10 points with a 95% interval excluding 0, and comparable rows on each side).

| model | template gap | source gap | genre gap |
|---|---|---|---|
| oracle | 0/120 = 0.0% [0.0, 3.1] | 0/120 = 0.0% [0.0, 3.1] | 0/120 = 0.0% [0.0, 3.1] |
| naive reader | 0/120 = 0.0% [0.0, 3.1] | 1/120 = 0.8% [0.1, 4.6] | 0/120 = 0.0% [0.0, 3.1] |
| plain mock | 120/120 = 100% [96.9, 100] (designed contamination: a true flag) | 1/120 = 0.8% [0.1, 4.6] | 0/120 = 0.0% [0.0, 3.1] |

Pooled, oracle + naive + plain mock (Wilson; seed-cluster bootstrap in brackets):
- source gap: 2/360 = 0.6% [0.2, 2.0] (bootstrap [0.0, 1.4]);
- genre gap: 0/360 = 0.0% [0.0, 1.1] (bootstrap [0.0, 0.0]);
- source + genre: 2/720 = 0.3% [0.1, 1.0] (Wilson on 720 checks: [0.08, 1.01]);
- template gap, oracle + naive + mock: 120/360 = 33.3% [28.7, 38.4], all of it the plain mock's designed contamination; the oracle and naive template gaps flagged 0/240.

Source: `pool_check.txt` (flag rates and pooled rows), `calib_stats.txt`. The only source flags are the naive reader on seed 71 (adjusted +13.2 [+0.7, +25.6]) and the plain mock on seed 20 (adjusted +20.2 [+6.7, +33.7]); the 4179c0c round had the naive seed 71 (+12.6 [+0.3, +25.0]) and the plain mock on seed 35 (+14.4 [+2.5, +26.3]).

## 2. Power and cross-flags

| penalised mock | targeted gap | flagged (k/120) [Wilson 95%] | other gap flagged |
|---|---|---|---|
| held-out domains -0.15 | source | 60/120 = 50.0% [41.2, 58.8] | genre 0/120 |
| held-out domains -0.30 | source | 115/120 = 95.8% [90.6, 98.2] | genre 0/120 |
| held-out genres -0.15 | genre | 109/120 = 90.8% [84.3, 94.8] | source 2/120 [0.5, 5.9] |
| held-out genres -0.30 | genre | 120/120 = 100% [96.9, 100] | source 2/120 [0.5, 5.9] |

The template gap is flagged in every run of every penalised mock (120/120) by design, so it is not in this table.

- Cross-flags: domain penalties flagged the genre gap in 0 of 240 runs; genre penalties flagged the source gap in 4 of 240 runs (seed 20 under both penalties, where the plain mock also flagged the source gap; seed 33 at -0.15 and seed 96 at -0.30, beyond the plain mock). Across both kinds: 4 of 480 runs.
- The plain mock's own source flags: 1/120 (seed 20). Source: `pool_check.txt`; seed lists: `calib_extra.py`.
- Realised injected gap (penalised minus plain on the targeted rows; mean over seeds, seed-cluster 95%): -0.15 domains -14.45 [-15.20, -13.70]; -0.30 domains -29.40 [-30.39, -28.41]; -0.15 genres -14.89 [-15.34, -14.44]; -0.30 genres -29.52 [-30.04, -28.99]. Source: `calib_stats.txt`.
- Mean adjusted targeted gap: +13.70 [+12.49, +14.91]; +28.46 [+27.05, +29.89]; +14.91 [+14.28, +15.54]; +29.52 [+28.90, +30.16].
- Recovery (mean shift of the targeted adjusted gap against the plain mock on the same seed, divided by the mean injected gap): 1.004 [0.959, 1.050] (domains -0.15); 0.995 [0.968, 1.022] (domains -0.30); 0.988 [0.973, 1.004] (genres -0.15); 0.994 [0.984, 1.003] (genres -0.30). Source: `calib_more.py` (seed bootstrap).

## 3. Coverage of the adjusted 95% intervals against the injected gap

The adjusted interval of the targeted gap contains -injected (the realised injected gap, section 2):

| penalised mock | covers -injected (k/120) [Wilson 95%] |
|---|---|
| held-out domains -0.15 (source) | 113/120 = 94.2% [88.4, 97.1] |
| held-out domains -0.30 (source) | 114/120 = 95.0% [89.5, 97.7] |
| held-out genres -0.15 (genre) | 111/120 = 92.5% [86.4, 96.0] |
| held-out genres -0.30 (genre) | 111/120 = 92.5% [86.4, 96.0] |
| pooled, four penalised mocks | 449/480 = 93.5% [91.0, 95.4] (seed-cluster bootstrap [90.8, 95.8]) |

Source: `pool_check.txt`, `calib_extra.py`. The expectation of the gap over the mock's draws (design-expected coverage, 94.8% [92.3, 96.9] in 4179c0c) was not recomputed at b5a8d59 (section 14).

## 4. Interval half-widths at 100 instances, and the like-for-like share

Half-width = (ci_high - ci_low) / 2 per run, averaged over 120 seeds; seed-cluster 95% in brackets; source: `calib_stats.txt`.

| model | template gap | source gap | genre gap |
|---|---|---|---|
| five mocks, average (plain, -0.15/-0.30 domains, -0.15/-0.30 genres) | 8.42 [8.37, 8.46] | 13.97 [13.74, 14.21] | 6.53 [6.46, 6.60] |
| plain mock | 7.95 [7.90, 8.00] | 13.70 [13.46, 13.96] | 6.44 [6.36, 6.52] |
| naive reader | 6.39 [6.33, 6.44] | 13.22 [12.97, 13.48] | 5.37 [5.31, 5.42] |
| oracle | 2.45 [2.42, 2.48] | 12.10 [11.90, 12.30] | 4.65 [4.60, 4.71] |

- Like-for-like share (rows compared like for like, mean over seeds): source 55.0% [53.9, 56.2] (run range 36.2-71.9%); genre 87.9% [87.4, 88.3] (78.7-94.0%); template 100%.

## 5. The confounded mix (15 seeds; plain mock and the mock with -0.30 on held-out domains)

- Template gap: flagged in 15/15 runs for both models (designed contamination; raw template mean +31.1 and +31.2 points; adjusted +31.3 and +31.6).
- Source and genre gaps: adjusted n/a in 30 of 30 runs (status n/a, spread check failed), never flagged.
- Raw means: plain mock source -1.1, genre -0.4; -0.30 domain mock source +28.6, genre +30.2 (the raw gaps move with the penalty; the adjusted ones are not estimable).
- Source: `calib_stats.txt` (`confounded`), `results.jsonl` (config-confounded runs).

## 6. Accuracy and cross-genre consistency (originals; mean over 120 seeds, seed-cluster 95%)

| model | accuracy | run range | all renderings | cross-genre consistency |
|---|---|---|---|---|
| oracle | 100.0 [100.0, 100.0] | 100.0 | 100.00 | 100.0 |
| naive reader | 60.2 [59.7, 60.6] | 55.0-66.4 | 60.15 | 100.0 |
| plain mock | 75.8 [75.4, 76.2] | 70.6-81.7 | 71.93 | 56.1 [55.8, 56.4] |
| mock, -0.15 domains | 72.2 [71.8, 72.6] | 66.8-77.5 | 68.38 | 52.4 [52.1, 52.7] |
| mock, -0.30 domains | 68.5 [68.1, 69.0] | 62.9-74.9 | 64.64 | 50.2 [49.8, 50.5] |
| mock, -0.15 genres | 71.5 [71.1, 72.0] | 65.1-77.5 | 67.68 | 51.0 [50.7, 51.3] |
| mock, -0.30 genres | 67.3 [66.8, 67.8] | 58.9-73.9 | 63.48 | 46.4 [46.1, 46.7] |

Source: `calib_stats.txt` (accuracy, cross-genre), `calib_more.py` (all renderings); `pool_check.txt`.

## 7. Trap shortcuts: legacy digests (L) against the earlier commits, and 0.3.0 digests (G, R, C) at b5a8d59

Each cell: share of the naive-to-oracle accuracy gap closed, % [95% CI over instances] (tables.md; the status rule and the paired lead are the task's columns). The best combination and the best rule-free reader are picked after the fact in each slice. Rule-free readers read no status, ID, note or notice.

### 7a. L (legacy path; 252 instances, 3,039 questions; naive 0.500, oracle 1.000)

| reader | b5a8d59 | 4179c0c | 8e63aee |
|---|---|---|---|
| proximity 40 s / 8.5 km (commoner source kept) | +24.7 [+21.4, +28.2] | +24.7 [+21.4, +28.2] | +24.7 [+21.4, +28.2] |
| proximity, same group within 1 h | +37.8 [+34.1, +41.4] | +37.8 [+34.1, +41.4] | +37.8 [+34.1, +41.4] |
| relay-source drop (us, at, pt, ew) | -22.1 [-30.3, -14.1] | -22.1 [-30.3, -14.1] | -22.1 [-30.3, -14.1] |
| relay drop, at/pt/ew only | +28.5 [+24.9, +32.4] | +28.5 [+24.9, +32.4] | +28.5 [+24.9, +32.4] |
| precision (neo) | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] |
| best combination (post hoc) | +38.9 [+35.3, +42.9] (c_g3600+relay_x) | same | same |
| best rule-free reader (post hoc) | +38.9 [+35.3, +42.9] (c_g3600+relay_x) | same | same |
| status rule | +41.1 [+37.3, +44.9] | +41.1 [+37.3, +44.9] | +41.1 [+37.3, +44.9] |
| status minus best rule-free (paired) | +2.2 [-0.6, +4.9]; ratio 0.95 [0.88, 1.01] | same | same |

- L matched the earlier measurements: at b5a8d59 its runs equal the 4179c0c runs in every file except the manifests (1428 files identical; the 84 manifests differ only in `code_sha256`, `generated_at_utc`, `seed_commitment` and `seed_nonce`), and its outputs (sc-summary, table text, AUC files and pickled rows) equal the 8e63aee outputs. The comparison files are not committed.
- L failing AUC cells (unintended features; interval rule / point rule / cells), all reports: 182 / 147 / 600 (ID-derived features excluded: 175 / 141); per domain 184 / 147 / 600; per genre 184 / 147 / 600. Identical at 8e63aee and 4179c0c (`L-auc-fail.txt` in each folder).

### 7b. G, R and C at b5a8d59 (with 4179c0c for comparison)

| sample (instances, questions; naive) | prox 40 s/8.5 km | relay drop | best combination | best rule-free (post hoc) | status rule | status minus best rule-free |
|---|---|---|---|---|---|---|
| G (1,008; 12,156; 0.602), b5a8d59 | -0.7 [-1.6, +0.0] | -38.0 [-42.5, -33.6] | -0.7 [-1.6, +0.0] (c_prox_src+neo) | +0.0 [-0.7, +0.8] (vsim_t3600) | +37.2 [+34.5, +39.8] | +37.2 [+34.5, +39.9] |
| G, 4179c0c | -0.9 [-1.8, -0.1] | -37.8 [-42.4, -33.0] | -0.9 [-1.8, -0.1] | +0.0 [-0.7, +0.8] (vsim_t3600) | +38.2 [+35.5, +40.8] | +38.1 [+35.4, +40.7] |
| R (252; 3,039; 0.622), b5a8d59 | -1.0 [-3.1, +1.0] | -43.6 [-55.0, -34.1] | -1.0 [-3.1, +1.0] (c_prox_src+neo) | +0.0 [0, 0] (neo_prec_drop) | +38.6 [+32.7, +44.3] | +38.6 [+32.7, +44.3] |
| R, 4179c0c | -1.2 [-3.2, +0.8] | -41.1 [-51.4, -31.8] | -1.2 [-3.2, +0.8] | +0.0 [0, 0] (neo_prec_drop) | +37.5 [+31.9, +43.2] | +37.5 [+31.9, +43.2] |
| C (1,008; 12,156; 0.602), b5a8d59 | -0.6 [-1.4, +0.1] | -38.0 [-42.7, -33.4] | -0.6 [-1.4, +0.1] (c_prox_src+neo) | +0.0 [0, 0] (neo_prec_drop) | +37.2 [+34.4, +39.8] | +37.2 [+34.4, +39.8] |
| C, 4179c0c | -0.8 [-1.6, +0.0] | -37.8 [-42.5, -33.1] | -0.8 [-1.6, +0.0] | +0.0 [0, 0] (neo_prec_drop) | +38.2 [+35.5, +40.6] | +38.2 [+35.5, +40.6] |

Sources: `shortcuts/tables.md` (b5a8d59) and the 4179c0c tables (not committed). Both use the same scripts (2000 resamples). The G/R/C values moved because the generator changed between the two commits (G's naive accuracy 0.603 at 4179c0c, 0.602 at b5a8d59; the L table did not move).

- G per domain, best rule-free: weather +1.4 [-1.1, +3.8] (vsim_t3600), neo +0.2 [+0.0, +0.8], invented +0.9 [-0.9, +2.6], earthquakes +0.0. Per genre: table +1.9 [-0.0, +4.3], form +1.3 [-0.5, +3.4], radio +1.7 [-0.7, +3.9], email +0.4 [-0.8, +1.9], classic, news and chat +0.0. Status rule per domain: weather +50.0 [+44.2, +55.6], earthquakes +39.6 [+34.1, +45.3], neo +30.8 [+25.3, +36.2], invented +31.0 [+26.2, +36.2] (`tables.md`, G block).
- R per domain, best rule-free: weather +2.0 [+0.0, +5.2] (vsim_03), the largest in R.
- G per domain x genre cell (36 instances each), largest best rule-free: weather/radio +7.6 [-4.7, +17.7] (prox_g3600); weather/table +6.7 [0.0, +14.4]; weather/classic +6.4 [-4.3, +17.0]. No cell's best rule-free interval lies above 0 (all 40 rows of the G block of `tables.md` have a lower bound at or below 0.0).
- The 40 s merge loses 0.6-1.0 points on G, R and C; the relay-source drop loses 38-44; the rule-free readers that change a digest close less than nothing. The status rule plus the 40 s merge, 36.5% on G at 4179c0c (DESIGN.md), has no row at b5a8d59 (section 14).

### 7c. Failing unintended-feature AUC cells (cells = slice x planted subset x feature; intended features left out)

Interval rule (95% CI excludes 0.5 and reaches outside [0.35, 0.65]) / point rule (estimate outside [0.35, 0.65]) / cells:

| sample | all reports | per domain | per genre | per domain x genre cell |
|---|---|---|---|---|
| L (b5a8d59) | 182 / 147 / 600 | 184 / 147 / 600 | 184 / 147 / 600 | not cut |
| G (b5a8d59) | 3 / 0 / 1,314 | 15 / 4 / 2,515 | 186 / 29 / 8,929 | 617 / 83 / 17,120 |
| G (4179c0c) | 2 / 0 / 1,314 | 13 / 4 / 2,515 | 185 / 22 / 8,919 | 589 / 77 / 17,115 |
| R (b5a8d59) | 20 / 2 / 1,290 | 50 / 3 / 2,468 | 254 / 81 / 6,800 | not cut |
| R (4179c0c) | 19 / 2 / 1,290 | 44 / 3 / 2,468 | 261 / 93 / 6,771 | not cut |
| C (b5a8d59) | 2 / 0 / 1,314 | 14 / 2 / 2,515 | 3 / 0 / 1,314 | not cut |
| C (4179c0c) | 2 / 0 / 1,314 | 14 / 5 / 2,515 | not cut | not cut |

- Strongest G cells at b5a8d59 (all reports, `G-auc-fail.txt`): second report `x4_grp_tz` 0.601 [0.515, 0.682] n=35 (interval rule); correction `x4_neo_sib_ddist` 0.417 [0.347, 0.487] n=64 (interval rule); earthquake second report `x4_grp_tz` 0.695 [0.560, 0.811] n=19 (both rules); weather duplicate-with-same-ID `x3_decide_gap` 0.347 [0.266, 0.418] n=50 (both); earthquake phantoms `x3_decide_gap` 0.349 [0.262, 0.447] n=16 (both); weather duplicates `x3_decide_gap` 0.349 [0.288, 0.409] n=73 (both).
- R strongest (all): `dup_xref x3_wx_dwind` 0.317 [0.229, 0.418] n=31 (both); `corr x4_grp_tz` 0.678 [0.555, 0.784] n=23 (both). C's strongest unintended feature (all; not a failing cell): `phantom x4_wx_dtemp_both` 0.623 [0.481, 0.760] n=11 (`C-auc-top.txt`). Source: `tables.md`, `summary.json` (`auc_fail_counts`, `auc_strongest`), `*-auc-fail.txt`.

## 8. Blind careful readers (at 4179c0c; the prompts were not re-read at b5a8d59)

- 28 reader cells = 7 genres x 4 domains (earthquakes, invented, neo, weather); each cell is one reader's answer to one prompt file, 14 cells in each of two rounds.
- 118 questions answered; 118 of 118 matched the answer key under `evalgen.scoring.is_correct` (14 checker passes: 0 slips, 0 ambiguities, 0 bugs; every checker `ok` = true). Per genre: chat 16, classic 17, email 17, form 17, news 19, radio 14, table 18 answers (the checker counts agree).
- The checkers recorded 24 non-blocking problem notes (nits on wording: e.g. "message follows" in radio intros; "recorded" in chat). They did not affect a key.
- Sources: the careful-reader counts in `blind-readers.md` (recomputed from the checkers' outputs, which are not committed).
- The checks ran on the 4179c0c tree (two extracted trees, byte-identical to `git archive 4179c0c`); nothing was re-read at b5a8d59.

## 9. Live runs and replays

- **A fresh live run at b5a8d59 (2026-11-15).** `generate --mix "weather:classic,neo:table,earthquakes:news" --source live --seed 8435451682288510821 --instances 6 --variants 1` from the final tree (rc 0; the run folder is not committed). It fetched 546 NWS observations (24 stations, 24 h window), 61 JPL close approaches and 199 USGS day-feed events; no source fell back to synthetic data; the oracle self-check was 56/56, 6/6 resolution invariant. Manifest: `runs/live-smoke-2026-11-15/manifest.json` (committed).
- **Offline replay of that run at b5a8d59** (`--source live --snapshot-dir <run>/snapshot`, the same seed and mix, every request routed to an unusable proxy; rc 0). Byte comparison: all six `instances/*.json`, all twelve `prompts/*/v*.txt`, both `audit/*` files and `forecasts.json` are identical, and so are the raw feeds and `events.json` in `snapshot/`. Each `snapshot/*/snapshot.json` differs only in its replay `note`; `manifest.json` differs in `run_id`, `seed_commitment`, `seed_nonce`, `generated_at_utc`, its `replay` block, `config.snapshot_dir` and the three snapshot notes. `live_replay_identical` = true for the instances, prompts, audit and forecasts (the sentence the docs now use).
- **The earlier committed run (8ef8657, 2026-11-12, `runs/live-smoke-2026-11-12/manifest.json`).** Its snapshot replayed at 8ef8657 reproduces all 28 non-provenance files; replayed at b5a8d59 it does not (26 of 32 files differ), because the generator changed between the two commits (the instance domains and the prompts differ). Measured in the earlier round; the docs say so.

## 10. The final demo (seed 42, 40 instances, random mix; the demo report is not committed)

- Generation: 40 instances, 157 questions, 3 renderings each; oracle self-check 471/471 renderings; the digest-only reading recovered the snapshot for 40/40 instances; audit sample 4 of 157 (2.55%; report.json stores 0.02), 0 of 4 reviewed. `verify-config`: ok.
- Accuracy (original renderings; 95% CI as printed in `report.md`): oracle 100.0% (157/157) [98, 100]; naive 59.2% (93/157) [51, 67]; mock 74.5% (117/157) [67, 81]. All renderings: oracle 100%, naive 59.2%, mock 71.1%.
- Metamorphic consistency: oracle 100.0% (314/314), naive 100.0% (314/314), mock 57.0% (179/314) [51, 62]. Cross-genre: oracle 100.0% (471/471), naive 100.0%, mock 56.1% (264/471) [52, 60].
- Template gap (public vs new templates, adjusted): oracle +0.0 (-6, +6); naive -3.8 (-14, +6), not flagged; mock +32.9 (+20, +46), flagged ("overfitting?").
- Source gap (adjusted; held-out weather): oracle +0.0 (-24, +24); naive -5.9 (-32, +20); mock -1.4 (-31, +28); no flags. Genre gap (held-out email and radio): oracle +0.0 (-9, +9); naive +3.4 (-6, +13); mock +3.9 (-10, +18); no flags.
- Trap catch (699 trap-question renderings: 233 affected trap-question pairs x 3 renderings): oracle 100.0%; naive 12.9% (90/699); mock 70.5% (493/699).
- Source: the demo report, not committed (sections "Model accuracy", "Metamorphic consistency", "Cross-genre consistency", "Seen vs new templates", "Public vs held-out", "Trap catch rate", "Generator audit").

## 11. The 0.2.0 identity (`identity`)

- Pinned legacy generation, seed 16222556419173536875, 100 instances: 406 files including the `.evalgen-run` marker (405 without). Legacy demo, seed 42, 20 instances: 271 files. Total 677 files in each tree.
- Byte-for-byte: 677 files compared; 4 differ, all manifests and the demo's `report.json` and `report.md`, in the version, code fingerprint, generation time, seed nonce and commitment, and the "Generated with" line.
- With those lines masked (`compare.py` masks the JSON keys and the `commitment`/`run_id` lines; the bullet line "Generated with" is masked too): **677 of 677 identical**.
- Pinned manifest keys that differ from 0.2.0: `code_sha256`, `evalgen_version`, `generated_at_utc`, `seed_commitment`, `seed_nonce`; no instance, prompt, score or audit file differs.

## 12. Tests and reproduction

- Full suite at b5a8d59: 1328 passed, 0 failed, 264.9 s. The README and DESIGN quote 866 tests (section 14 and `numbers-changes.md`).

## 13. Caveats

- The held-out sets are built in and public, and the models are simulated; the calibration measures the metrics, not a model's generalisation. The mock's template contamination is designed (30 points), so its template flags are true flags.
- Trap shortcuts are post hoc: the best combination and the best rule-free reader are chosen after the fact in each slice, which favours them. Shares are on the naive-to-oracle gap over every rendering.
- The legacy path is measured on 252 digests (seeds 5001-5084) and matches 4179c0c and 8e63aee exactly; off the legacy path the 0.3.0 digests are new at b5a8d59 (generator changed).

## 14. Unresolved and stale items (not resolved in this measurement)

1. **Live replay not identical at b5a8d59** (section 9). The docs' statement that the offline replay reproduces every instance, prompt and audit file holds at 8ef8657 only.
2. **Stale labels in `shortcuts/`.** the shortcut report written with the measurement is the 4179c0c narrative (superseded; not committed): its title and "Comparison with the 8e63aee measurement" section say 4179c0c, the comparison columns labelled "at 4179c0c" carry b5a8d59 values, and its prose quotes 4179c0c figures (G status 38.2, R 37.5, C 38.2) that disagree with its own tables (37.2, 38.6, 37.2). `out/summary.json` `comparison` entries labelled 4179c0c hold b5a8d59 values (G naive 0.6022, against 0.603 in the 4179c0c tables). The `*-auc-fail.txt` headers and `shortcuts/README.txt` also say 4179c0c. The tables (tables.md; summary.json `samples`) are b5a8d59 and are the source used here.
3. **Status rule plus the 40 s merge on G (36.5% at 4179c0c, DESIGN.md:936) is not reproduced at b5a8d59**: no such row exists in the outputs; the only combination with the status rule is `status_neo_prec_drop`, +0.0.
4. **Design-expected quantities not recomputed**: the design-expected coverage (94.8%, 95.9% of 1,800), the design SD and SD/SE ratios, the design bias (-0.30 [-0.59, -0.03], SD 1.6; the first 40 instances, SD 4.4; genre SD 0.4), and the design-expected template gap. These need `design_expect_all.py` and `design_sd_all.py` from the 4179c0c synthesis, which are not among this measurement's inputs. The nearest realised quantity: plain-mock adjusted source gap -0.80 [-1.97, +0.37] (`calib_stats.txt`).
5. **40-instance calibration not recomputed** (the 4179c0c design_bias_40 run): the source interval at 40 instances (about +-29) and the 20-instance demo of the example config (+-48 at 29% coverage) are 4179c0c figures.
6. **10-minute merge (35.6%, legacy)**: quoted by DESIGN.md:1691-1693; no 10-minute row exists in the b5a8d59 L outputs (1-hour: 37.8%, reproduced).
7. DESIGN.md:1849-1850 says the best rule-free reader closes "at most +2.0% [+0.0, +5.2] in a domain". At b5a8d59 that maximum still holds over the three off-legacy samples (R weather +2.0); G alone peaks at +1.4 [-1.1, +3.8] (weather). Not a change.
8. **pool_check.py has no bootstrap for means**, contrary to the bootstrap convention used elsewhere; the means above come from a separate checker that is not in the repository.
9. The 4179c0c figure "95.0% [92.1, 97.5]" (456/480) for pooled coverage is 93.5% [91.0, 95.4] (449/480) at b5a8d59. The cause is not isolated here; pool_check.py and calib_extra.py agree on 449/480.
