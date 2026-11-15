# Opus 5.5 vs Haiku 5.5 on careful reading

This repository holds **evalgen**, a generator of reading tests that are new on every run, and a
study that uses it to compare Claude Opus 5.5 with Claude Haiku 5.5. A seeded generator writes a
messy digest of earthquake reports and questions about it, computes the answer key in code, and
scores the models' replies automatically. Because every test item is generated fresh, no model can
have seen it before.

## The question

Claude Haiku 5.5 costs about 1/16 as much as Claude Opus 5.5 for the same work here. Does it hold
up on careful-reading tasks: reconciling reports from several networks, converting time zones and
units, and applying rules about duplicates, corrections and retractions? And when some of its
errors trace back to loosely worded rules, does tightening the rules fix them?

## Setup

Each prompt is a digest of 20–28 earthquakes (23–33 report lines, counting re-reports and retracted
reports), followed by 3–5 questions that chain filters, deduplication, conversions, counts and
rankings. The digest is deliberately messy: shuffled order,
local times with UTC offsets, km and miles, three coordinate formats, re-reports of the same quake
by other networks, retracted reports, and magnitude corrections. Models answer in strict JSON.

Every test instance is shown in **3 renderings**: the original and 2 variants that change the units,
formats, line order and wording but not the answers. Two runs of **100 fresh instances** each:

| run | reading rules | questions | prompts per model | seed |
|---|---|---|---|---|
| [`compare-100`](runs/compare-100/report.md) | 0.1.0 | 411 | 300 | 13762304563634797857 |
| [`compare-100-v2`](runs/compare-100-v2/report.md) | 0.2.0 | 397 | 300 | 16222556419173536875 |

Version 0.2.0 tightened the rules after a review of every wrong answer in the first run. It names
every wording of a correction or retraction notice (such as "UPDATE … withdrawn"), says that
magnitudes on different scales are compared as plain numbers, and states how time windows, signed
depths, region names and lines that are not reports are to be read.

Both models ran through the Anthropic API with adaptive thinking, default effort and a
16,000-token output limit. The adapter asks for temperature 0, but these models accept only the
default temperature, so it falls back to that.

## Results

### Headline accuracy

| | Opus 5.5, 0.1.0 | Haiku 5.5, 0.1.0 | Opus 5.5, 0.2.0 | Haiku 5.5, 0.2.0 |
|---|---|---|---|---|
| Accuracy, original renderings | 99.8% (410/411) | 99.0% (407/411) | **100%** (397/397) | 97.5% (387/397) |
| Right on all 3 renderings | 99.5% (409/411) | 95.1% (391/411) | **100%** (397/397) | 95.0% (377/397) |
| Consistency across renderings | 99.6% | 97.2% | 100% | 96.2% |
| Traps handled correctly | 100% | 96.2% | 100% | 97.6% |
| Wrong answers, all 3 renderings | 3 | 21 | 0 | 20 |
| Cost of the run | $12.95 | $0.82 | $13.94 | $0.90 |

Paired on the same questions (Opus minus Haiku, exact McNemar test):

- **0.1.0 run:** +0.7 points on the original renderings (95% CI −0.4 to +1.9, p = 0.38); +4.4 points
  on all three renderings (+2.1 to +6.6, p = 0.0001).
- **0.2.0 run:** +2.5 points on the original renderings (+0.9 to +4.1, p = 0.002); +5.0 points on all
  three renderings (+2.8 to +7.2, p = 0.000002).

### Haiku's misses by cause

| most likely cause | 0.1.0 rules | 0.2.0 rules |
|---|---|---|
| Local time → UTC conversion | 6 | 6 |
| A qualifying line skipped | 6 | 5 |
| A re-report counted as a second quake | 2 | 2 |
| A retracted report counted | 2 | 0 |
| A retraction applied to a look-alike report | 0 | 1 |
| Only one magnitude scale counted | 2 | 0 |
| "At least" read as "more than" | 1 | 0 |
| A correction or reviewed value not applied | 1 | 1 |
| A km/mile slip | 1 | 2 |
| Other: a rank slip, an out-of-window quake, an overnight window misread | 0 | 3 |
| **Total** | **21** | **20** |

Opus missed 3 answers under 0.1.0 (all local-time conversions) and none under 0.2.0.

Three independent Claude Opus 5.5 agents reviewed every wrong answer: one diagnosed it, a skeptic
without the ground truth tried to show the model's answer was defensible, and a third checked the
diagnosis. All answer keys were confirmed correct, and no wrong answer was judged defensible. The
details are in each run's miss analysis: [0.1.0](runs/compare-100/miss_analysis.md) and
[0.2.0](runs/compare-100-v2/miss_analysis.md).

## Takeaways

- **The rules removed the errors they targeted.** Counting only one magnitude scale went from 2
  misses to 0, and counting a retracted report from 2 to 0, though misses this rare, on different
  questions, cannot prove that the models read the new wording.
- **Haiku's total barely moved: 21 → 20.** Most of its misses are careful-reading slips, such as
  local-time conversions, skipped lines, double-counted re-reports and unit slips, which no rule
  wording prevents.
- **Pooled gap: 4.7 points (95% CI +3.1 to +6.2).** Over both runs (808 questions), Opus answered
  806 correctly on all three renderings and Haiku 768 (only Opus right 40 times, only Haiku twice;
  p ≈ 4 × 10⁻¹⁰). On a single rendering the gap is 1.6 points (+0.6 to +2.6). The reformatted
  variants are what expose it.
- **At about 1/16 of the cost**, Haiku makes roughly one extra error per 20 questions when its
  answers have to hold up across reformatted versions of the same digest.

## Limitations

- **One task type.** These are careful-reading questions over earthquake digests; the results say
  nothing direct about other skills.
- **Synthetic data.** The earthquakes, networks and reports are generated: plausible, but simpler
  than real catalogues. evalgen can also build tests from the live USGS feed, but the comparison
  used synthetic data.
- **Two runs on different questions.** Each run drew fresh instances, so the before-and-after
  comparison of the rules mixes the effect of the rules with sampling variation.
- **The causes are inferred.** The models' reasoning is hidden, so the cause of each miss is the
  review agents' most likely explanation, with a confidence level in the miss analysis.
- **Default settings only.** One effort level and one run per model per rule version.
- **Trap handling is not evidence that a model read the rules.** In digests like these the planted
  lines can be told from real ones by where they lie and which network filed them: on 252 fresh
  digests written as the 0.2.0 run's were (seeds 5001–5084, with the same result at the final 0.3.0
  commit), a reader that merges reports of one region lying within an hour and 8.5 km of each other
  and drops those of the networks that file no events of their own, reading no status, note or
  notice, closes 38.9% of the gap between a reader who counts every line and one who applies every
  rule, nearly the 41.1% closed by the status rule that the rules intend (keep only the REVIEWED
  reports). evalgen 0.3.0 removes this tell from runs with any `--mix` but `legacy`, the default,
  which keeps these digests byte for byte: off it, on 1,008 digests of every domain and genre, the best
  of the 27 such readers (13 single rules and 14 combinations) closes 0.0% of that gap (95% CI −0.7 to +0.8), against 37.2% for the status rule
  (see the limitations in [`docs/DESIGN.md`](docs/DESIGN.md)).

## Reproduce it

Install the dependencies (requests and pytest; Python 3.10 or later), run the 1328 tests (about 4
minutes), and run a full cycle offline, with no API key and no cost:

```bash
python3 -m pip install -r requirements.txt
python3 -m pytest -q
python3 cli.py demo --out runs/demo-local
```

To rerun the 0.2.0 comparison, first regenerate it. This reproduces the run byte for byte: the
same instances and the same 300 prompts per model.

```bash
python3 cli.py generate --seed 16222556419173536875 --instances 100 --out runs/rerun-v2
```

Then read your Anthropic API key without echoing it. In bash:

```bash
read -rs -p "Anthropic API key: " ANTHROPIC_API_KEY && export ANTHROPIC_API_KEY
```

In zsh:

```zsh
read -s "ANTHROPIC_API_KEY?Anthropic API key: " && export ANTHROPIC_API_KEY
```

Reading the key this way keeps it out of your shell history; `export ANTHROPIC_API_KEY=…` typed at
a prompt does not. Finally, run both models, score the replies and write `runs/rerun-v2/report.md`:

```bash
python3 cli.py run runs/rerun-v2 --models anthropic:claude-opus-5-5,anthropic:claude-haiku-5-5 --workers 4 --max-tokens 16000
python3 cli.py score runs/rerun-v2
python3 cli.py report runs/rerun-v2
```

For a new comparison on fresh questions, leave out `--seed`: evalgen draws a new seed and records
it in `manifest.json`.

**Cost: about $15 per run.** Opus costs about $14 and Haiku about $0.90, at list prices of $4 / $20
(Opus 5.5) and $0.10 / $0.50 (Haiku 5.5) per million input / output tokens. Each model reads about
1.2 million input tokens per run; Opus writes about 0.46 million output tokens and Haiku about 1.56
million, mostly thinking. The run takes about 40 minutes with the two models one after the other,
or about 25 with one `run` command per model in parallel.

## 0.3.0: multiple sources and genres

The study above used evalgen 0.2.0, in which every test is about earthquakes and is written in one
family of digest styles. A review pointed out that an adversary who knows that one source, the USGS
feed, could tune a model to it. evalgen 0.3.0 therefore draws the source and the kind of document
too. It adds three domains (weather observations from NWS stations, close approaches of near-Earth
objects from JPL, and an invented domain whose names, labels and units are drawn afresh for each
instance) and six genres (news articles, email threads, chat logs, tables, report forms and radio
transcripts). An evaluator can hold domains and genres out in private plugins, listed by path in a
private config that the run commits to by hash; a plugin listed with the SHA-256 of its bytes too is
pinned by that commitment, and one listed by path alone is not, so give every real plugin its
SHA-256. It covers that one file, so a plugin must also be self-contained: it reads no other file, the
environment, the clock or the network, and imports only from evalgen and the standard library. Regenerating a
revealed run executes the plugin code in-process with the verifier's rights: the SHA-256 pins which code runs,
not that it is safe, so read the plugins first or regenerate in a sandbox. The report then compares
accuracy on the public and the held-out sets (the source gap and
the genre gap) and checks that answers agree across genres.
Answer keys are still computed by code, and the text-only oracle still scores 100% on every domain
in every genre.

The new options are `--mix` (`legacy`, the default; `random`; or a list such as
`weather,neo:email`), `--private-config` (the held-out domains and genres), `--source live` and
`--snapshot-dir` (each domain's live source, frozen and replayed), the `verify-config` command
(which checks a revealed config against the commitment published before the evaluation, `--commitment HEX`,
or against a run) and held-out penalties for the simulated model
(`mock:held_out=0.3`). This runs every built-in domain and genre offline:

```bash
python3 cli.py demo --mix random
```

Adding `--private-config configs/private.example.json` shows the held-out metrics, with the
test-only toy domain and genre held out (in `runs/demo-heldout-random-synthetic-seed42`, beside the
demo without them). Those are public code, so the gaps exercise the metrics and test nothing held
out, and the report says so, as it does whenever a held-out set is built in or defined by a plugin
loaded from evalgen's own `tests/` folder, whatever nonce the config draws (a copy of such a plugin
elsewhere is not recognised).
Without the new options nothing changes: the 0.2.0 comparison above regenerates byte for byte, which
`tests/test_regression_020.py` checks.
[`docs/DESIGN.md`](docs/DESIGN.md) describes the threat model, the domains and genres, the new
metrics and their limitations.

## What's in the repository

- `evalgen/`: the generator, its domains and genres, model adapters, scorer and report.
- `cli.py`: the command line (`generate`, `run`, `score`, `report`, `resolve`, `demo`,
  and `verify-config`, which checks a revealed config against the commitment published before the evaluation
  (`--commitment HEX`) or against a run).
- `configs/private.example.json`: the format of a private run config (an example only: real
  held-out sets must stay private).
- `tests/`: 1328 tests.
- `runs/compare-100/` and `runs/compare-100-v2/`: each comparison's report (`report.md` and
  `report.json`), miss analysis and `manifest.json` (seed, settings, code fingerprint). The raw
  prompts and replies are not committed. The 0.1.0 run cannot be regenerated with the current code,
  because the rules text is part of every prompt.
- `runs/demo-synthetic-seed42/`, `runs/usgs-live-seed5/`: smaller demo runs (reports only).
- `runs/live-smoke-2026-11-15/`: the manifest of the 0.3.0 run at the final commit (b5a8d59) that read the
  live NWS, JPL and USGS sources; `runs/live-smoke-2026-11-12/` is the earlier run at 8ef8657 (see the design
  doc's limitations).
- [`docs/DESIGN.md`](docs/DESIGN.md): how evalgen works, including the threat model, the question
  grammar, traps, scoring, reproducibility and the method's limitations, and the domains, genres,
  mixes and held-out sets of 0.3.0. [`docs/PLAN-0.3.md`](docs/PLAN-0.3.md) is the plan of 0.3.0.

## License

MIT. See [LICENSE](LICENSE).
