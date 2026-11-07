# evalgen: live, fine-tuning-resistant LLM evaluation

A working demo of an evaluation method that is **fully public** (code, templates, reading rules
and scoring are all open) and still **cannot be fine-tuned against**. Every test instance is
generated fresh at evaluation time from a seeded RNG, its answer key is computed by code, and no
item is ever labelled by hand.

The domain is earthquake reports. A model receives a messy, news-style digest of earthquake
reports *inside the prompt* and answers 3–5 composed questions about it in strict JSON. It never
fetches anything, so scraping skill is irrelevant: what is tested is careful reading,
rule-following and arithmetic under realistic noise.

```bash
python3 -m pip install -r requirements.txt     # requests + pytest (Python 3.10+)
python3 cli.py demo --seed 42 --instances 20  # fully offline, no API keys, ~1 second
python3 -m pytest -q                           # 202 tests, ~25 seconds
```

The demo writes `runs/demo-synthetic-seed42/report.md` (and `report.json`).

---

## Threat model: public method, private instances

| The adversary knows | The adversary does not know |
|---|---|
| all code, the operation grammar, the public templates, the reading rules, the trap types, the scoring rules; they can generate unlimited practice instances themselves | the **seed** used at evaluation time, hence the concrete instances (thresholds, radii, units, time windows, points, regions, which quakes, which traps, which wording) |

Why fine-tuning on the benchmark buys nothing beyond the real skill:

1. **Fresh instances.** Every question has at least one RNG-drawn metric (magnitude threshold,
   depth limit, radius and its unit, point, time window, region pair, rank). The generator enforces
   that no concrete question is used twice in a run, and it costs nothing to make more, because
   code computes every key.
2. **Scraping is irrelevant.** The data is in the prompt. A model trained on scraping workflows
   has nothing to scrape. Even with `--source usgs` the live feed is fetched once by the harness,
   frozen and hashed, then rendered into text.
3. **Traps with known handling.** Re-reports by other networks, retracted phantom reports and
   conflicting magnitudes are planted so that a careless reader gets specific, predictable wrong
   answers. Planted lines copy the style of real ones: place notes that agree with their
   coordinates, the network's magnitude precision, and IDs that fit the network's own ID sequence.
   The way to catch a trap is to read the notices and apply the rules.
4. **Metamorphic consistency.** The same snapshot is re-rendered with different units, line order,
   formats and question wording. Correct answers cannot change, so any change in a model's answer
   is an inconsistency. This needs no ground truth and catches surface-pattern memorisation.
5. **Seen vs new templates.** Public templates are assumed to be in training data. Each run
   composes a "twin" of every public template, with the same final operation and the same number
   of filters but a different filter combination, and each twin pair is used equally often. The
   gap is then compared like for like: within strata of equal code-computed difficulty (how many
   traps can change the answer, and whether the filters matter). A large adjusted gap therefore
   flags overfitting to the published skeletons, not a difference in trap exposure.
6. **Delayed-resolution forecasts** (`--source usgs`): "Will an M4.5+ quake occur within R km of L in
   the next 24 h?" The answer does not exist yet, so it cannot be memorised. It is graded later with
   `resolve` (Brier score).
7. **Generator audit.** The generator checks itself (a text-only oracle must reproduce every key),
   and a ~2% sample goes to humans to estimate the generator's own error rate.

**Keep the seed secret until the run is over.** `manifest.json` (a private file) holds the seed, a
random 128-bit nonce, and the commitment `sha256("evalgen-seed:<seed>:<nonce>")`. Publish only the
commitment before the evaluation; the nonce makes it impossible to brute-force even for a small
seed such as 42. Revealing the seed and nonce afterwards lets anyone verify the commitment,
regenerate the exact instances and audit them.

What this design does *not* claim: a model that genuinely learns to read noisy reports carefully
will score well, and that is the intended skill. The template grammar is finite, so the
seen-vs-new gap is a flag, not proof (see Limitations).

---

## Workflow (one cycle)

```
  seed (secret until the run is over; a nonce-salted commitment can be published beforehand)
    │  every component gets its own SHA-256-derived child seed, all logged in manifest.json
    ▼
┌─────────────┐  synthetic world (default, offline)    or   USGS GeoJSON feed
│ 1 SNAPSHOT  │  ──────────────────────────────────────────────────────────────────►  snapshot/
└──────┬──────┘  canonical events + SHA-256 hash + seed   (feed frozen & hashed; one event per quake;
       │                                                    synthetic fallback when offline)
       │  per instance: sample 20–28 quakes, mostly from 3–4 focus regions
       ▼
┌─────────────┐  templates: public P01–P12  +  their new-this-run twins N01–N12
│ 3 QUESTIONS │  RNG draws every metric → margin guards → never the same question twice in a run
└──────┬──────┘  each question = an operation chain, e.g. dedupe → mag ≥ 3.1 → in_region → count
       ▼
┌─────────────┐
│ 4 KEY       │  execute(chain, snapshot events)   ← plain code; no LLM, no text parsing
└──────┬──────┘
       ▼
┌─────────────┐  one REVIEWED report per quake + traps aimed at the questions:
│ 2 DIGEST    │  duplicates · retraction · conflicting magnitude   (invariant: rules(digest) = snapshot)
│   + TRAPS   │  rendered 3×: v0 original, v1/v2 metamorphic (units, order, formats, wording)
└──────┬──────┘
       ▼
┌─────────────┐      prompt = reading rules + <digest> + questions + strict JSON format
│ 5 EVALUATE  │ ───► oracle · naive · mock · anthropic[:model] · openai[:model]  → responses/
└──────┬──────┘
       ▼
┌─────────────┐  6 score (exact / tolerance)   7 consistency across renderings
│   SCORE &   │  traps caught vs fallen        10 public-vs-new gap
│   REPORT    │  9 audit sample (~2%)           8 forecasts → pending → `resolve` later
└──────┬──────┘
       ▼
  report.md + report.json
```

(Questions are composed *before* the digest because traps are aimed at them. Each trap's placement
is rejection-sampled so that mishandling it changes at least one answer. When no placement does,
the trap is still planted, as plain noise, and the report counts it as not load-bearing.)

---

## Commands

```bash
# full cycle, offline, with the reference solvers and the simulated model
python3 cli.py demo --seed 42 --instances 20       # overwrites runs/demo-synthetic-seed42 each time

# step by step
python3 cli.py generate --seed 7 --instances 20            # -> runs/synthetic-seed7/
python3 cli.py run runs/synthetic-seed7 --models mock,oracle,naive
python3 cli.py score runs/synthetic-seed7
python3 cli.py report runs/synthetic-seed7

# real models (keys from the environment). Replies are cached; a re-run asks again only for
# prompts whose API call failed or whose reply hit the output-token limit, so re-running is
# free and safe.
export ANTHROPIC_API_KEY=...   OPENAI_API_KEY=...
python3 cli.py run runs/synthetic-seed7 --models anthropic,openai:gpt-4.1 --workers 4
python3 cli.py score runs/synthetic-seed7 && python3 cli.py report runs/synthetic-seed7

# live USGS data: fetched once, frozen + SHA-256 hashed; falls back to synthetic if offline
python3 cli.py generate --source usgs --feed day --seed 7
python3 cli.py generate --source usgs --snapshot-file runs/usgs-seed7/snapshot/raw_feed.geojson --seed 7 --out runs/replay
python3 cli.py resolve runs/usgs-seed7      # after the 24 h window (+2 h grace): Brier scores
```

`--force` overwrites an existing run directory, but only one that evalgen created (it carries
evalgen's marker file or its own manifest); any other folder is left untouched.

`--max-tokens` (default 16000) is the output budget per reply for API models. Models that think
before answering need room: at 4,096 tokens, Claude Haiku 5.5 hit the limit on two thirds of its
prompts, almost always before writing any answer. A reply that stops at the limit is scored on
whatever text it returned (usually none), flagged in the report, and asked again by the next `run`.

| model spec | what it is |
|---|---|
| `oracle` | rule-based solver that reads **only the digest text**, applies the reading rules and runs each question's operation chain; must score 100% |
| `naive` | the same parser, but every line counts as a separate quake and notices are ignored; it loses points only where traps (alone or combined) change the answer, apart from rare display-rounding effects |
| `mock` | a **simulated** contaminated model, not a language model. It answers public-template questions correctly 92% of the time (80% when paraphrased) and new-template questions 62% of the time; its errors are seeded by the prompt text and usually fall into the traps. It exists to exercise every metric without API keys |
| `anthropic[:model]` | Anthropic Messages API (`ANTHROPIC_API_KEY`; default model `claude-opus-5-5`, override with `:model` or `EVALGEN_ANTHROPIC_MODEL`) |
| `openai[:model]` | OpenAI Chat Completions (`OPENAI_API_KEY`; default `gpt-4.1`, override with `:model` or `EVALGEN_OPENAI_MODEL`; `OPENAI_BASE_URL` points it at any compatible server) |

API models receive only the prompt text. The oracle and naive solvers also receive each question's
operation chain (its machine-readable meaning), because the goal is to verify the *digest*, not to
build a parser for English questions.

---

## The pieces

### 1. Snapshot (`evalgen/sources.py`)
* **synthetic** (default): 360 events across 10 regions (Northern California, Southern Alaska,
  Tonga, …). Magnitudes follow Gutenberg–Richter, depths fit each region, IDs are USGS-style, and
  the time window is a seed-derived day in 2026 (never the wall clock).
* **usgs**: `all_day` or `all_hour` GeoJSON. Earthquakes only (quarry blasts and explosions are
  dropped). The raw bytes are saved and SHA-256 hashed. Coarse regions come from the place
  strings (`"6 km NE of Delta, B.C., MX"` → `Mexico`, `"Southern Alaska"` → `Alaska`,
  `"off the coast of Central America"` → `Central America`; proper names such as *West Virginia*
  are kept). Real feeds sometimes list one quake twice, for example an
  automatic and a reviewed solution, or a regional and a global one. Solutions within 3 s and
  15 km of each other are therefore collapsed to the reviewed one, and the report notes how many.
  Network errors fall back to synthetic data, and the fallback is logged in the report.
* In the digest, **REVIEWED** marks each quake's authoritative report: the snapshot's own row. For
  USGS data this is the feed's preferred solution, whatever its USGS review status.

### 2. Digest and traps (`evalgen/digest.py`, `evalgen/traps.py`)
The digest mixes four line styles (bulletin, prose, wire, log), depths in km, miles or both,
coordinates as decimal-hemisphere, signed or degrees-and-minutes, and six time formats including
local time with an explicit UTC offset. It can be shuffled, grouped by network or sorted by time,
and adds chatter lines. Every value is reversible by a careful reader: a round-trip test parses
every rendering back to the exact reports.

| trap | forms | correct handling (reading rules) | a careless reader… |
|---|---|---|---|
| duplicate | same event ID from another network, **or** the network's own reference + "same event as `<id>`"; slightly different values, PRELIMINARY | count once, use the REVIEWED values (rules 1–2) | counts it twice |
| retraction | a phantom report + a later RETRACTION notice (quarry blast, false trigger, …) | ignore it entirely (rule 4) | counts the phantom |
| conflicting magnitude | the reviewed report shows a wrong value and a CORRECTION fixes it, **or** a PRELIMINARY report beside the REVIEWED one | use the corrected / reviewed magnitude (rules 2–3) | uses the wrong magnitude, or counts the preliminary line as another quake |

Each trap records its careless readings, and every question is re-run under them:
* **affected**: questions whose answer changes when only this trap is misread. These are its
  load-bearing questions.
* **fallen**: every wrong answer a reader reaches by misreading this trap, alone or together with
  other traps.

The report counts an answer as *fell for the trap* when it equals one of those fallen answers.

Placement prefers traps whose wrong answers sit clearly outside the scoring tolerance, so that
display rounding cannot move a careless reader across the edge.

Planted lines fit the feed they sit in:
* **IDs:** in USGS mode a planted ID is taken from the network's own sequence. A numeric ID gets a
  free value on the network's step between its time neighbours; a time-coded ID keeps its
  neighbour's prefix.
* **Place notes:** computed from the town position implied by the real notes, so they agree with
  the line's coordinates to the format's own precision (whole km, 16-point bearings).

On today's live feed, a planted ID is no more often out of range or out of prefix than a held-out
real one.

### 3. Questions (`evalgen/questions.py`, `evalgen/ops.py`)
Operations: filters `mag_at_least`, `mag_below`, `depth_shallower_than`, `depth_at_least`,
`within_distance` (point + radius in km or miles), `time_between`, `in_region`; the `dedupe` step;
terminals `count`, `argmax_mag`, `rank_mag(k)`, `max_mag`, `max_depth`, `min_depth`,
`nearest_to`, `nearest_distance`, `compare_regions(a, b)`, `time_of_largest`; post-steps
`convert` (unit) and `round`.

**One defensible answer per item.** Margin guards make the key robust to the digest's rounding and
to every reasonable reading of the wording:
* **Thresholds:** no quake may lie within 0.6 km (or 3%) of a depth threshold, within 3 km (or 6%)
  of a radius, or within 2 minutes of a time-window edge.
* **Rankings:** argmax and rank answers need a gap of at least 0.1 magnitude between every
  neighbour down to the asked rank, so "the third-largest" means the same thing under competition
  and dense ranking.
* **Nearest questions:** the nearest quake must beat the runner-up by 3 km (or 8%) and lie within
  500 km, which keeps the 3% distance tolerance tight.
* **Filters that cannot matter are banned:** filter/terminal pairs where the filter can never
  change the answer (e.g. `mag_at_least` → `max_mag`) are never generated.
* **Station labels** are unique within an instance.

Each question's English wording is generated from its chain. Metamorphic variants paraphrase it
and may restate thresholds in the other unit, which the guards make safe.

**Templates.** Public `P01–P12` cover all ten final operations (assume they have been trained on).
New `N01–N12` are composed each run, one twin per public template: the same terminal and number of
filters, a different filter set, and never a public skeleton. Questions are balanced between the
two pools, and a template never runs more than one question ahead of its twin. That matters when
run-wide uniqueness exhausts a template with few possible parameters: a region-only template
cannot run out while its twin keeps going.

**No repeats.** A question's identity ignores station labels and the order of the two regions in a
comparison, so "Utah or Nevada?" and "Nevada or Utah?" count as the same question.

### 4. Answer key
`answer_key[qid] = execute(chain, snapshot_events)`: plain code, no LLM. The generator also
asserts that applying the reading rules to each digest recovers the snapshot exactly, and that the
text-only oracle reproduces every key on every rendering. The counts are logged as the "automated
self-check".

### 5–6. Evaluate and score (`evalgen/adapters.py`, `evalgen/answers.py`, `evalgen/scoring.py`)
Replies must be one JSON object, `{"answers": {"Q1": 7, "Q2": "nc75012345", ...}}`. Parsing
records compliance:
* **strict:** the whole reply is the object.
* **lenient:** the answers were recovered from code fences, reasoning around the JSON, comments,
  trailing commas, single quotes, Python literals, or `Q1: 7` lines.
  * With several JSON objects, the one covering the most question IDs wins (ties: the one wrapped
    in `answers`, then the later one).
  * If no single object covers them all, for example one object per question, they are merged.
  * When scraping lines, the last mention of each question wins; a final JSON cut off mid-reply
    overrides only the questions it reached.
* **failed:** nothing usable.

API failures and replies cut off at the output-token limit are recorded separately from parse
failures, and the report warns about both.

| answer type | rule |
|---|---|
| count, event ID, region / `tie` | exact (IDs and regions case-insensitive) |
| magnitude | within 0.05 |
| depth (asked unit) | within max(0.3, 2%) |
| distance (asked unit) | within max(1.5, 3%) |
| UTC time `HH:MM` | within 1 minute |
| forecast probability | Brier score after resolution (an unanswered forecast counts as 0.5) |

A string answer is read only when it is unambiguous: `"7 events"` is 7 and `"5,088.8 km"` is
5088.8, but `"7 or 8"`, `"3-4"` or a string naming two event IDs counts as no answer.

**Head to head.** When a run has two or more API models, the report compares each pair question
by question (questions are the independent units). It shows how many questions only one of them got
right, the paired difference with an Agresti–Min 95% interval, and an exact McNemar test, twice:
once on the original renderings, and once counting a question as right only if every rendering of
it was answered correctly. The second row also rewards robustness to surface changes, and it can
separate models that tie on the originals. The report also lists the tokens each API reported for
the scored replies (output tokens include any thinking) and the median latency.

### 7. Metamorphic consistency
Each instance has `v0` plus 2 variants with different presets for units, time formats,
coordinates, line styles, layout and wording. Consistency = the share of (question, variant)
answer pairs equal to the model's `v0` answer, compared with the same tolerances as scoring.
Questions the model never answered are left out.

### 8. Delayed-resolution forecasts (`evalgen/forecast.py`, USGS mode)
Each USGS instance adds `F1`: the probability of an M4.5+ quake within R km of a point in the next
24 h. It is stored in `forecasts.json` as pending, with its exact rule. `resolve` queries the USGS
FDSN event service once the window plus 2 h has passed, and the report shows Brier scores.

### 9. Generator audit (`evalgen/audit.py`)
`ceil(2% × items)` questions are sampled (seeded) into `audit/audit_sample.jsonl`, with a
readable checklist in `audit_sample.md`: question, chain, key, evidence events with canonical
values, the traps that affect it, and the digest. Auditors set `verdict` to `ok` or `error`, and
`report` then estimates the generator error rate (Wilson interval; rule of three when there are
no errors). The aim is to measure the generator, not to label data.

### 10. Seen vs new gap
Accuracy on public vs new templates is reported in two forms:
* **raw gap:** public minus new accuracy, with a 95% interval;
* **adjusted gap:** the same difference compared like for like, within strata of equal
  code-computed difficulty (how many traps change the answer: 0, 1 or 2+; and whether the filters
  matter). Each stratum is weighted by its share of questions.

Each question's difficulty features are stored with it. The flag uses the adjusted gap: above 10
points with an interval that excludes 0.

Validation over 90 synthetic runs of 100 instances:
* the pool-agnostic `naive` baseline was flagged 0 times (the raw gap would have flagged it 10
  times);
* a simulated reader that misses each trap with probability 0.3 was flagged 0 times (raw: 7);
* the simulated contaminated `mock` was flagged in all 90.

---

## Reproducibility

* One master seed. Child seeds are `SHA-256("<seed>/<path>")` (never Python's salted `hash()`),
  and all of them are listed in `manifest.json`.
* The same seed gives byte-identical instances, prompts, snapshot, audit sample and metrics. This
  is tested. The code targets Python 3.10+ and also runs on 3.9; outputs were byte-identical
  between Python 3.9 and 3.14.
* Synthetic runs never read the clock. USGS runs are reproducible from the frozen
  `raw_feed.geojson` with `--snapshot-file`. The replay verifies the recorded SHA-256 and warns
  when the source run was generated with different code.
* `manifest.json` records the evalgen version and a SHA-256 fingerprint of the evalgen source.
  `report` prints the version and warns when the code has changed since the run was generated.
  The reading rules are part of every prompt, so a run is reproducible only with the evalgen
  version that generated it. Version 0.2.0 reworded the rules (see Limitations), so runs made
  with 0.1.0 do not regenerate byte for byte under 0.2.0.
* Model replies are cached in `responses/`, so `score` and `report` can be re-run without new API
  calls.

## Repository layout

```
cli.py                  commands: generate, run, score, report, resolve, demo
evalgen/
  seeds.py   util.py    seeding (SHA-256 child seeds, salted commitment) and small helpers
  sources.py            synthetic world, USGS feed (duplicate solutions collapsed), frozen + hashed snapshots
  rules.py              the reading rules (English for prompts, code for the key) + trap mishandlings
  ops.py                operation chains and the answer-key executor
  questions.py          templates (public + twins), RNG parameter drawing, margin guards, wording
  traps.py              trap planting (load-bearing), fallen answers, resolution invariant
  digest.py             messy rendering: styles, units, time formats, layouts
  solver.py             text-only oracle and naive solvers (digest parser)
  prompt.py answers.py  prompt construction; robust JSON reply parsing
  adapters.py           oracle, naive, mock, Anthropic, OpenAI
  scoring.py            normalisation, tolerances, consistency, intervals
  forecast.py           delayed-resolution questions and resolution (USGS FDSN)
  audit.py report.py    audit sample; report.md / report.json
  pipeline.py           one cycle end to end
tests/                  202 pytest tests: hand-built key cases, determinism, traps and oracle, guards,
                        digest parsing, reply parsing and scoring, USGS and forecasts, adapters, report,
                        head to head, reading-rules coverage, CLI
runs/<run>/             manifest.json, snapshot/, instances/, prompts/, responses/, scores/, audit/,
                        forecasts.json (usgs), report.md, report.json
```

## Limitations and next steps

* **One domain, finite grammar.** Ten terminals and seven filters. New twins are new *combinations*
  of known operations, so the seen-vs-new gap detects skeleton-level overfitting, not every kind.
  Growing the grammar, or adding new domains with the same machinery, widens it. The adjusted gap
  removes differences in trap exposure and filter relevance, but not every possible difference
  between filter types (a radius filter needs more arithmetic than a region filter).
* **Run size versus uniqueness.** Templates with few possible parameters (one per region) run out
  on very large runs. Their twins then stop too, so the pools stay matched, but those final
  operations become rarer.
* **The oracle shares the reading-rules code with the generator.** The self-check proves the text
  encodes the snapshot. Whether the rules are worded unambiguously for humans is what the 2% human
  audit is for. An LLM-run blind audit of 39 items during development found the ambiguity that the
  stricter rank guard now removes, and no wrong keys. An adversarial review of all 24 wrong
  answers from Claude Opus 5.5 and Haiku 5.5 on a 100-instance run confirmed every key. Its wording
  notes, and a second review of the reworded rules (a completeness audit, a skeptic and two blind
  test-takers), led to the 0.2.0 rules:
  * They name every notice wording (including "UPDATE ... withdrawn" and "Revised ...") and a
    quake's PRELIMINARY reports from its own network.
  * Depths are signed, and magnitudes on different scales are compared as plain numbers once
    rules 2 and 3 have settled them.
  * Regions are matched by their printed label, and lines that are neither reports nor notices
    change nothing.
  * Time-window ends, report versus notice times, and coordinate signs are stated.

  Coordinates are stored at the printed precision, so a reader working from the digest gets every
  distance key exactly. Tests check that the rules name every notice wording the renderer can
  print, and that the oracle reproduces the distance keys exactly.
* **Synthetic world.** Plausible but simplified: no aftershock sequences, and synthetic regions are
  coarse.
* **The mock is a simulation.** Its numbers show that the metrics work; they say nothing about any
  real model. Run `anthropic` / `openai` for real numbers.
* **Live runs so far are Anthropic only.** Claude Opus 5.5 and Claude Haiku 5.5 have been run
  against the live Anthropic API (20 instances on seed 42, and a 100-instance head-to-head). The
  OpenAI adapter has only been tested offline, against recorded request and response shapes.
* **Small samples.** Twenty instances give about 80 questions, so confidence intervals are wide.
  Use `--instances 100` or more for real comparisons.
