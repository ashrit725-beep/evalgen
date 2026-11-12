# evalgen: live, fine-tuning-resistant LLM evaluation

A working demo of an evaluation method that is **fully public** (code, templates, reading rules
and scoring are all open) and still **cannot be fine-tuned against**. Every test instance is
generated fresh at evaluation time from a seeded RNG, its answer key is computed by code, and no
item is ever labelled by hand.

The 0.2.0 domain is earthquake reports. A model receives a messy, news-style digest of earthquake
reports *inside the prompt* and answers 3–5 composed questions about it in strict JSON. It never
fetches anything, so scraping skill is irrelevant: what is tested is careful reading,
rule-following and arithmetic under realistic noise.

Version 0.3.0 makes the source and the kind of document a seeded draw too. It adds three domains
(weather observations, close approaches of near-Earth objects, and an invented domain whose names,
labels and units change with every instance) and six genres (news articles, email threads, chat
logs, tables, forms and radio transcripts). Mixes draw both per instance, and the evaluator can hold domains and genres
out in private plugins (see [0.3.0: several sources and genres](#030-several-sources-and-genres)).
Without the new options, 0.3.0 writes the 0.2.0 run byte for byte.

```bash
python3 -m pip install -r requirements.txt     # requests + pytest (Python 3.10+)
python3 cli.py demo --seed 42 --instances 20  # fully offline, no API keys, ~1 second
python3 cli.py demo --mix random               # 0.3.0: every domain and genre, offline, ~3 seconds
python3 -m pytest -q                           # 697 tests, ~2 minutes
```

The demo writes `runs/demo-synthetic-seed42/report.md` (and `report.json`); with `--mix random` it
writes `runs/demo-random-synthetic-seed42/`.

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
   of filters but a different filter combination, and the composer keeps the two pools in balance:
   each instance asks about as many public as new questions, and a template never runs
   more than one question ahead of its twin once the twin has been used (off the legacy path, from
   the first question; see [Questions](#3-questions-evalgenquestionspy-evalgenopspy)). The gap is
   then compared like for like: within strata of equal code-computed difficulty (how many traps can
   change the answer, and whether the filters matter). A large adjusted gap therefore flags
   overfitting to the published skeletons, not a difference in trap exposure.
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

### 0.3.0: public sources and genres, private held-out sets

Prof. Kishor Datta Gupta's review of 0.2.0 pointed out a risk in this model. Every instance came
from one source, the USGS feed (or its synthetic stand-in), written in one family of digest styles.
An adversary who knows the public code can tune a model to that source and practise on unlimited
earthquake digests in exactly the style the evaluation uses. Fresh instances do not stop that while
the source and the style never change. 0.3.0 widens both and holds some of each back:

| The adversary knows | The adversary does not know |
|---|---|
| everything in the 0.2.0 table above; the public domains (earthquakes, weather, neo, invented) and genres (classic, news, email, chat, table, form, radio), with their generators, live sources, trap words, reading rules and glossaries; the mix grammar and the format of the private run config. They can generate unlimited practice in every public domain and genre | the **seed**, as before; the **held-out domains and genres**: the evaluator's private plugins, and which domains and genres the private run config holds out; the config itself (its nonce and its lists) until it is revealed |

What this adds to the list above:

1. **An unpredictable source and genre.** Under `--mix random` (or a list), each instance draws its
   domain, and each of its renderings draws its genre, from the seed. A model cannot know which kind
   of record, or which kind of document, it will read.
2. **Held-out domains and genres.** A domain or genre defined in a private plugin is new to every
   model at evaluation time. The report's *source gap* and *genre gap* compare public with held-out
   sets like for like. A large adjusted gap flags a model tuned to the public domains or genres, as
   the template gap flags one tuned to the public skeletons.
3. **Cross-genre consistency.** A variant is rendered in another genre than its original, so the
   same question asked of an email thread and of a radio transcript must get the same answer. Like
   metamorphic consistency, this needs no ground truth.

**The private run config is committed like the seed.** The evaluator keeps it private until the
evaluation is over. It looks like this (`configs/private.example.json` is a published example):

```json
{"nonce": "<at least 32 random hexadecimal digits: secrets.token_hex(16)>",
 "domains": {"public": ["earthquakes", "weather"], "held_out": ["tides"]},
 "genres": {"public": ["classic", "news"], "held_out": ["fax"]},
 "plugins": [{"path": "plugins/tides.py", "sha256": "<the SHA-256 of the file's bytes>"},
             {"path": "plugins/fax.py", "sha256": "<...>"}],
 "note": "optional"}
```

A plugin is given by its path (relative to the config) and the SHA-256 of its bytes, or by its path
alone, as a plain string.

`generate --private-config` checks the config whole before it writes anything. It needs the three
required keys, distinct names that each appear in one list only, and names the registries know
once the plugins are loaded. The config is read strictly: a file that gives a key twice in one JSON
object is malformed (JSON readers keep one of the two values, so such a file could show one config
and verify as another), and so is a plugin whose registry refuses its entry, for example one that
takes a built-in name such as `earthquakes` or `classic`. Both stop `generate` with `error: ...`
(exit status 2) before anything is written. The manifest then records:

- the **commitment**, `sha256(canonical_json(config))` (sorted keys, no whitespace). It covers the
  whole config (the nonce, both lists, the plugin entries with their SHA-256 and the note), and it
  can be published before the evaluation, like the seed commitment;
- each **plugin**, by its path as the config names it and the SHA-256 of its bytes, and the config
  itself by its file name alone. A run names no folder of the evaluator's machine that the revealed
  config does not.

The random nonce stops anyone from recovering a config of a few names from its hash, so `generate`
refuses a nonce that looks typed: fewer than 32 hexadecimal digits, fewer than 8 different digits,
or one run of digits repeated. No check can prove a nonce random. A nonce that contains `EXAMPLE`,
as the published example's does, is accepted with a loud warning that its commitment can be
guessed, so that the held-out metrics can be demonstrated offline.

**What the evaluator publishes, and when.** Before the evaluation: the two commitments that
`generate` records in the manifest, `seed_commitment` and `private_config_commitment`, and nothing
else (not the manifest, which holds the seed, nor the config or its plugins). After it: the config,
its plugin files, and the seed with its nonce.

**Verifying a revealed config.** `python3 cli.py verify-config RUN_DIR CONFIG` prints `ok` (exit
status 0) when the config hashes to the run's commitment, and `MISMATCH` (1) otherwise, including
for a file that is not JSON. It exits with `error: ...` (2) for a run generated without a config,
which has no commitment to check, and for a config that gives a key twice in one object, which reads
as two configs. The hash is over canonical JSON, so re-indenting the file or reordering its keys
changes nothing, while any change to a name, to the order of a list, the nonce, a plugin entry or the
note breaks the match.

**What the commitment pins of a plugin.** A plugin given with its SHA-256 is pinned by the
commitment: `generate` and every later step check its bytes before any of it runs and refuse a file
whose bytes differ, so a held-out domain or genre cannot change once the commitment is published,
and what runs is exactly the bytes that were checked (compiled from them, with no bytecode cache read
or written beside the plugin). A plugin given by its path alone is committed to by that path only:
the manifest records its SHA-256, a later step loads a changed file with a warning and lists it in
`report.json` (`plugins_changed`), and only a regeneration of the revealed run shows what it was.
Regenerating the run with the same evalgen code, the revealed seed, mix and config, and its plugins
at the same relative paths, rebuilds the same instances and prompts byte for byte.

**Plugins never come from a run directory.** A plugin is a Python file that defines `DOMAIN`,
`GENRE` or both. `generate` loads the config's plugins before it looks up any name. Every later
step in a new process (`run`, `score`, `report`, `resolve`) loads them only from the config that
the evaluator passes with `--private-config`, after checking that config against the run's
commitment; within one process, a later step registers again the plugins that the process already
ran for that config, without running any of their code again. evalgen never executes a file that a
run directory names by itself, so reporting on a run received from someone else runs none of its
code.

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

Off the legacy path (0.3.0) the cycle is the same, step for step, in every domain:

- Step 1 builds one snapshot per domain of the mix: synthetic, or the domain's live source fetched
  once, frozen and hashed.
- Before step 3, each instance draws its domain and the genre of each rendering (seed path
  `instance/<i>/mix`) and, in a domain that draws them, its surface words (`instance/<i>/surface`).
- Step 2 renders each rendering in its genre.
- The prompt prints the domain's glossary between the reading rules and the digest.
- The oracle of step 5 reads each prompt with the parser of the genre that recognises its digest.

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

0.3.0 adds several domains and genres, and held-out sets
([0.3.0: several sources and genres](#030-several-sources-and-genres)):

```bash
python3 cli.py demo --mix random                                    # every built-in domain and genre, all public
python3 cli.py demo --mix random --private-config configs/private.example.json   # held-out metrics, offline
python3 cli.py generate --seed 7 --mix weather,neo:email,invented:table         # an explicit mix
python3 cli.py generate --seed 7 --mix random --private-config private.json     # the evaluator's held-out sets
python3 cli.py run runs/random-synthetic-seed7 --models mock,mock:held_out=0.3 --private-config private.json
python3 cli.py score runs/random-synthetic-seed7 --private-config private.json
python3 cli.py report runs/random-synthetic-seed7 --private-config private.json
python3 cli.py verify-config runs/random-synthetic-seed7 private.json           # once the config is revealed
python3 cli.py generate --seed 7 --mix random --source live                      # USGS, NWS and JPL, frozen + hashed
python3 cli.py generate --seed 7 --mix random --source live --snapshot-dir runs/random-live-seed7/snapshot --out runs/replay
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
| `naive` | the same parser, but every report counts as a separate quake and notices are ignored; it loses points only where traps (alone or combined) change the answer, apart from rare display-rounding effects. Off the legacy path it reads each digest with its genre's parser, as the oracle does, which has already set aside the quoted and read-back repeats and the message times that the rules say are no reports. So it never meets a genre's noise, and its consistency across renderings and genres is high by construction |
| `mock` | a **simulated** contaminated model, not a language model. It answers public-template questions correctly 92% of the time (80% when paraphrased) and new-template questions 62% of the time; its errors are seeded by the prompt text and usually fall into the traps. It exists to exercise every metric without API keys. `mock:held_out=P` (0.3.0) also lowers each of those chances by P on an instance from a held-out domain and by P on a rendering in a held-out genre, as a model tuned to the public ones would be (`held_out_domains=P` or `held_out_genres=P` for one kind, joined with `+` for different penalties). A chance stops at 0, so where the penalties that apply add up to more than 0.62, the chance on a new template, they lower accuracy by less than they say, and a gap they inject is smaller than the penalty; `run` prints a note for such a mock |
| `anthropic[:model]` | Anthropic Messages API (`ANTHROPIC_API_KEY`; default model `claude-opus-5-5`, override with `:model` or `EVALGEN_ANTHROPIC_MODEL`) |
| `openai[:model]` | OpenAI Chat Completions (`OPENAI_API_KEY`; default `gpt-4.1`, override with `:model` or `EVALGEN_OPENAI_MODEL`; `OPENAI_BASE_URL` points it at any compatible server) |

API models receive only the prompt text. The oracle and naive solvers also receive each question's
operation chain (its machine-readable meaning), because the goal is to verify the *digest*, not to
build a parser for English questions.

---

## The pieces

These are the pieces of the 0.2.0 cycle, which legacy runs still follow exactly. In 0.3.0 every
piece runs through the domain and genre interfaces
([0.3.0: several sources and genres](#030-several-sources-and-genres)).

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
two pools: each instance asks as many public as new questions, give or take one (an odd number of
questions draws the pool of its last one at random), or rarely two, when no template of one pool fits
an instance's records (2 instances in 300). And once a template's twin has been used, the
template never runs more than one question ahead of it. That matters when run-wide uniqueness
exhausts a template with few possible parameters: a region-only template cannot run out while its
twin keeps going. Under the 0.2.0 rule, which legacy runs keep, a template can run ahead freely until
its twin is first used, so a legacy run can end with a pair several questions apart (the legacy
demo, `cli.py demo`, asks P12 four times and N12 once). Off the legacy path the rule holds from a
run's first question, so every twin pair of every domain ends at most one question apart.

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
Questions the model never answered are left out. Off the legacy path (0.3.0), two answers that are
both scored right also count as equal: scoring accepts an answer within its tolerance on either side
of the key, so two right answers can be up to twice the tolerance apart (4.96 and 5.04 for a
magnitude of 5.0), and the tolerance alone would count them as an inconsistency. Legacy reports
compare as 0.2.0 did (see Limitations).

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

## 0.3.0: several sources and genres

0.3.0 runs everything above through two interfaces. A **domain** says what the records are: their
fields and units, groups, sources, IDs, traps and reading rules. A **genre** says what kind of
document a digest is. Operations, templates, guards, traps, scoring and the report are written once
against both. Both registries discover their built-in entries from their packages
(`evalgen/domains/*.py`, `evalgen/genres/*.py`, in sorted order), and private plugins add entries
under names that no built-in entry has taken.

### The domain interface (`evalgen/domains/base.py`)

A domain module exports `DOMAIN`, an instance of a `Domain` subclass. The registry checks it
(`Domain.validate`) when it is loaded and registers its answer types with the scorer.

| part | what it declares |
|---|---|
| vocabulary | `noun`, `plural`, `id_label`, `source_label`, `group_label`: "weather observation", "observation ID", "feed", "region" |
| `fields` | one `FieldSpec` per numeric field: its key and label; its canonical unit, other units and stored decimals; its answer type and tolerance (abs, rel); its guard margin (abs, rel) and rank gap; how thresholds are drawn (quantiles, jitter, steps); the filter and terminal kinds that apply to it; a qualifier stored with the value (an earthquake's magnitude type); whether it is signed and whether notices can correct it |
| `roles` | which field plays the 0.2.0 *primary* role (magnitude) and which the *secondary* one (depth), so that P01–P12 bind to every domain |
| `spatial` | radii, distance guards and the precision of positions, for records with a location. A domain without one drops the spatial templates (P02, P07, P09, P12) and their twins as a pair |
| groups and sources | every record belongs to a group (a region, a body, a depot), which `in_group` and `compare_groups` ask about and digests focus on (3–4 per digest), and has a source that filed it (a network, a feed, a catalogue, an agency) |
| `synthetic(rng, n, compiled_at, window_hours)` | the seeded generator, which never reads the clock |
| `live` | an optional live source. `fetch` downloads it once, and the raw bytes are frozen in the run and SHA-256 hashed. `parse` reads them without the clock or the network, so a replay rebuilds the snapshot exactly. A source that cannot be reached, or that gives too few usable records, falls back to synthetic data, with a note |
| IDs | `id_pattern` (the scorer also reads ID answers with it), `new_id` and `mimic_id`, so that planted reports carry IDs shaped like real ones |
| `statuses` | the status of the report that settles a record's values, then the one that does not ("reviewed", "preliminary") |
| `trap_vocab` | re-reporting sources, duplicate notes, retraction reasons, and the phrases of retraction and correction notices |
| `chatter`, `lexicon` | lines without data, and the wording of questions (`standard_lexicon` words any domain from its own words) |
| `rules_text()`, `glossary(surface)` | the reading rules and the glossary its prompts print |
| `draw_surface(rng)`, `bind(surface)` | per-instance surface words, for a domain that draws them |
| `resolve`, `to_generic`, `from_generic` | the reading rules in code, and the map between a domain's own record types and the generic ones that genres write |

Earthquakes keep their 0.2.0 record types (`Event`, `Report`), which gained read-only generic names
(`values`, `group`, `source`). Every other domain uses the generic `Record` (the ground truth) and
`RecordReport` (one report in a digest). Operations are generic kinds on a field
(`{"op": "at_least", "field": "temp", "value": 20.5, "unit": "°C"}`). The 0.2.0 operations are
aliases of these kinds, so earthquake chains keep their 0.2.0 form exactly.

**Units and the exact codec** (`evalgen/units.py`). A `Unit` maps a printed value to the field's
canonical unit: canonical = value × scale + offset, which is affine for °F and K. Values are stored
in the canonical unit to the field's decimals.

- `encode` writes a stored value in any of its field's units with the fewest decimals that read
  back to it, and `decode` reads it back. Both work in exact decimal arithmetic, with the scale and
  offset exactly as the glossary prints them, never in binary floating point.
- `encode` never writes a number whose conversion falls exactly halfway between two stored values.
  It writes 0.7 °C as 273.85 K, not 273.9 K, which converts to 0.75 °C exactly. A careful reader
  therefore recovers every stored value, whichever way they round a half.
- A unit can carry its own answer places, scoring tolerance and threshold step (inches of mercury
  beside hectopascals). `validate` refuses a domain whose guard margin is smaller than the rounding
  of a threshold restated in another unit, or whose tolerance would mean different things in
  different units.

**Traps.** The three 0.2.0 traps, made generic:

- a *duplicate*: another source re-reports a record, under its ID or under its own reference with a
  duplicate note, with slightly different values and the unsettled status;
- a *retraction*: a phantom report, plausible for its group, that a later notice retracts;
- a *conflicting value* of a field (a correctable one, when the domain has one), settled by a
  correction notice or by the settling report beside an unsettled one (`conflicting_value`;
  earthquakes keep `conflicting_magnitude`).

Values move by a share of their field's spread in the digest, so the traps scale to every field.
Off the legacy path, placement is also *guarded*. A careless reading counts only where the misread
digest passes the key's own guards: margins that hold with the planted values, rank and nearest
gaps, time margins. A placement whose every reading passes is preferred. A careless reader's wrong
answer is then the one the printed text gives, not one decided by a last digit or a tie.

**Rules text.** `record_rules` writes the reading rules of any domain in its own words. They are
genre-neutral and say:

- which report settles a record, and that unsettled values are superseded;
- how re-reports, corrections and retractions read, quoting every phrase of the domain's trap
  vocabulary;
- that values are converted as the glossary says;
- that times are given in UTC or with their offset, and that the time a message was sent,
  forwarded or relayed is never a record's time;
- that a quoted or read-back repeat of a report or notice is not a new one, while a forwarded or
  relayed one counts, once;
- that group names are matched as printed;
- that lines which are neither reports nor notices change nothing.

Weather, neo and invented add rules of their own. Earthquake prompts off the legacy path print the
0.2.0 rules 1–8 word for word, followed by three rules: rule 9, which now also names the section
titles, greetings and signatures that the other genres print among the lines without data, and the
rules on message times and repeats (rules 10 and 11). They are headed `Reading rules:`, as every
domain's are, since the 0.2.0 heading ("identical for every digest") is not true of a run whose
domains print different rules. Legacy prompts print the 0.2.0 rules alone, heading included.

**Question wording.** Off the legacy path, a radius is worded "within 40 km of …" or "at a distance
of no more than 40 km from …", in every domain with a location. The 0.2.0 wording "no more than 40 km
from …", joined to a depth threshold, reads as a range of depths ("with a depth of 16 km or more and
no more than 40 km from station US-28"). Earthquake questions off the legacy path keep every other
0.2.0 wording and draw exactly as before (`Domain.prompt_lexicon`); legacy prompts keep the 0.2.0
wording (see Limitations).

**The glossary.** Every prompt off the legacy path prints its domain's glossary between the reading
rules and the digest, in `<glossary>` tags. It gives:

- what the records, IDs, sources and groups are called, and the ID pattern;
- the two statuses, and the precision of positions;
- each field's key, label, stored unit and decimals, with every unit and its exact conversion.

Words are quoted as JSON strings. `render_glossary` refuses a glossary that would not read back
exactly, and `parse_glossary` reads it from the prompt text. A glossary never names the domain's
registry name, which a held-out plugin keeps private. The ID pattern describes IDs lower-cased, and
digests print them as their sources file them (`KJFK-48213`, `KX/40512`), so the glossary says that
the pattern ignores letter case. One instance of the invented domain printed (seed 7, mix
`invented:classic`, `inst-000`):

```
<glossary>
Glossary: what the records in the digest are, and how their values are written.
- Records: each is one "crate" (plural "crates"), identified by its "tally number"; IDs match the pattern [a-z]{2,3}[-/.]\d{5}, ignoring letter case.
- Sources: each report is filed by one "carrier".
- Groups: each record belongs to one "warehouse".
- Statuses: a "verified" report settles a record's values; a "provisional" report does not.
- Field "f1": "tal bulk", stored in "lg" to 1 decimal place.
  - "lg" ("ligs"): the stored unit.
  - "bl" ("bals"): a value in "lg" is the value in "bl" × 50.
- Field "f2": "pik stretch", stored in "bb" to 1 decimal place.
  - "bb" ("brubs"): the stored unit.
  - "zk" ("zisks"): a value in "bb" is the value in "zk" × 60.
</glossary>
```

**Per-instance surfaces.** A domain with `has_surface` draws the words of each instance from a seed
path of its own (`instance/<i>/surface`, which only runs off the legacy path create). The words are
the record noun and the labels of IDs, sources and groups; the field labels and the units with
their conversions; the ID prefix; the names of groups and sources; the trap words and the chatter.
`bind` returns the domain as that instance words it, and checks it again (margins, tolerances,
distinct names). `dress` renames the snapshot's records, so one snapshot serves every instance. The
instance stores its surface, so that scoring and the audit can read it back.

### The built-in domains

| domain | records | fields (canonical unit first) | groups; sources | location | live source |
|---|---|---|---|---|---|
| `earthquakes` | the 0.2.0 domain | magnitude (with its type); depth (km, mi) | regions; seismic networks | yes | USGS GeoJSON feed |
| `weather` | observations at 24 US airport stations | temperature (°C, °F, K); wind speed (km/h, mph, kt, m/s); pressure (hPa, inHg) | six US regions; station feeds and relay feeds | yes | NWS `api.weather.gov` station observations |
| `neo` | close approaches of near-Earth objects | miss distance (km, au, LD); relative velocity (km/s, km/h, mph); absolute magnitude H | the body approached; catalogues | no | JPL Close Approach Data, `ssd-api.jpl.nasa.gov/cad.api` |
| `invented` | consignments at depots of a world that does not exist | two invented quantities, in invented units | depots; agencies (both renamed per instance) | no | none |

- **earthquakes** is 0.2.0. Its field specs carry the 0.2.0 threshold draws and margins, its op
  names are the 0.2.0 operations, its lexicon holds every 0.2.0 wording and its trap words are the
  0.2.0 ones. A legacy run reaches USGS through the 0.2.0 code (the hour or the day feed). Off the
  legacy path its live source is the day feed (`--source live`), and its USGS instances still ask
  the forecast question. Genres other than classic write and read its 0.2.0 reports and magnitude
  corrections in the generic form of every other domain.
- **weather**: temperature, a signed field, plays the primary role (thresholds, the warmest
  observation, its rank and its time) and wind speed the secondary one. Pressure appears in the
  twins' filters.
  - Every conversion is exact as defined (1 kt = 1.852 km/h, 0 °C = 32 °F = 273.15 K,
    1 inHg = 33.86389 hPa). Every value is stored to a tenth, and answers in K and inHg keep two
    places.
  - IDs are `<feed>-<5 digits>` (`KJFK-48213`, `MADIS-30571`). The digits are an opaque serial, so
    a planted report's ID never encodes a time. The statuses are "verified" and "preliminary" (the
    NWS quality-control codes V and Z).
  - Synthetic observations follow each station's climate normals, a daily cycle and a weather
    system per region.
  - The live source asks api.weather.gov for the last 24 hours of each station's observations,
    with the User-Agent the API requires, and freezes every response as it was served, its HTTP
    status and the exact text of its body, in one JSON bundle (`raw_feed.json`, format
    `evalgen-nws-bundle/2`). The snapshot's `raw_sha256` therefore commits to what NWS served;
    bundles of the earlier format 1 can no longer be replayed. Reading the bundle back drops every
    observation that lacks a value, or holds one that NWS quality control rejected or questioned, so
    no value is ever invented, and keeps one observation per station and UTC hour (the hourly METAR
    when there is one).
  - A station that fails on its own (an error status, a timeout, a dropped connection after the
    first station answered, a body that is not UTF-8) is recorded as failed, and the others are
    kept. Only a first request that cannot connect, a fetch in which no station answers with
    observations, or fewer than 40 usable observations fall back to synthetic data, with a note
    naming the failures.
- **neo**: relative velocity plays the primary role and miss distance the secondary one; H appears
  in thresholds only.
  - 1 au is 149,597,870.7 km, and 1 LD is 384,400 km as this domain defines it.
  - Records have no position, so the spatial templates and their twins drop out: eight public
    templates instead of twelve.
  - The groups are the bodies approached (Earth, Moon, Venus, Mars, Mercury), and the sources are
    catalogues.
  - Synthetic snapshots cover 30 days, so time windows lie on 3- or 6-hour marks, and a time
    question asks for the UTC time of day.
  - An approach ID is the object's designation without spaces, then `-ca` and the number of its
    approach. `2026vk2-ca1` and `2026vk2-ca2` are one object passing Earth and then the Moon: two
    records.
  - The live source is every approach within 0.05 au of any body that CAD lists for the next 60
    days (`raw_cad.json`). CAD times are TDB; records keep UTC.
- **invented** has no live source, by design: nothing outside evalgen describes its records.
  - Each instance draws its record noun, ID label and prefix, source and group labels, a coined
    name for every depot and agency, its field labels, its units (a stored unit and one or two
    others with exact stated conversions such as × 12 or × 0.25), its trap words and its chatter.
  - Coined words never repeat within an instance. No coined name, label, unit name or unit symbol
    is a real unit, a word the digests print around the data, a common English word of two or three
    letters (a symbol follows every value, so "a heft of 15.3 so" must not read as prose) or a word
    with an offensive part; unit symbols are checked like names. Some seeds therefore draw other
    surfaces than they did before this check (`tests/fixtures/invented/pinned.json` was pinned again;
    the synthetic world is unchanged).
  - Values come from fixed log-normal distributions per canonical depot, so margins and steps keep
    their meaning under every surface. Tolerances are relative, so they hold in every drawn unit.
  - An extra rule says that the glossary is the only definition of its words.

### Genres (`evalgen/genres/`)

A genre is how one kind of document presents an instance's fixed content: its reports, notices and
chatter lines. It draws a style (`choose_style`; a variant in the same genre must differ from the
original's), renders the text, recognises its own text (`sniff`) and parses it back (`parse`, the
exact inverse of `render`).

| genre | a digest is | notices appear as | noise that carries no data |
|---|---|---|---|
| `classic` | the 0.2.0 bulletin, prose, wire and log lines, and their version for every other domain | NOTICE, RETRACTION, UPDATE, CORRECTION and Revised lines | header, section heads, chatter lines |
| `news` | an article: headline, byline and dateline, then one paragraph per report, in five sentence forms | "Update", "Correction", "Retraction notice" and "Correction notice" paragraphs, at the top, among the paragraphs or at the end | quotes from officials, a background paragraph, subheads, a sign-off, editor's notes |
| `email` | a thread of messages with From, To, Date and Subject headers; reports as bullets, sentences or pasted blocks | a forwarded message, every one | greetings, sign-offs, signatures, remarks; replies that quote earlier messages with ">" |
| `chat` | a channel log of `[HH:MM] user: message` lines; reports in four voices | messages, pinned messages, or the pinned board above the log; the casual ones open with their keyword and time ("heads up: RETRACTION issued …: …") | joins, greetings, acknowledgements, off-topic remarks; quotes and read-backs of earlier messages |
| `table` | CSV or markdown, one row per report, columns named in the glossary's words | numbered footnotes | a caption, comment lines |
| `form` | one field or incident report form per report, in four frames (boxed, ruled, numbered, dotted) | amendment forms | form codes and serial numbers, sent and relayed times, preparers, blank checker and signature fields (a blank field that shares a boxed row with another shows a line, `________`, so the row cannot be misread) |
| `radio` | the transcript of a reporting net: stations call in and read out their reports | dispatch broadcasts to all stations | call signs, procedure words, radio checks, acknowledgements; Control's read-backs |

- **The rules settle every line.** A notice names itself with words every domain's rules name:
  RETRACTION, CORRECTION, "revised", or a phrase of the domain's trap vocabulary, which its rules
  quote. Every notice is worded as the notice itself, never as a pointer to another notice, and none
  is headed as a kind of line that the rules say carries no data (an editor's note, a reminder). A
  test per genre checks every wording that genre can print.
  - Quotes (email, chat) repeat an earlier report or notice word for word, and read-backs (chat,
    radio) repeat a report's ID and values; the rules say that a repeat is not a new report or
    notice. Forwarded and relayed messages count, once: every email notice is forwarded, and chat
    users and bots relay what each source files. In every genre, a reader who strikes out every
    repeat still reads each notice exactly once.
  - A message's own time (an email's Date header, a chat time stamp, a transmission time, a form's
    sent time) is never a record's time, and every report states its record's time in full, to the
    second.
  - Lines that are neither reports nor notices carry no data, and the parsers read nothing from
    them. In a news paragraph every free word (a source, a group, a place) lies between typed slots
    (an ID, a status, a time, a position, a value in a unit) or long literal text, so it cannot be
    taken for a delimiter, and form codes and serial numbers never match the ID pattern.
- **Exact round trip.** Every genre writes values through the exact codec, times to the second (in
  UTC or with the UTC offset stated) and positions as stored, in the shared formats of
  `genres.base`. Every genre parses them back through the prompt's glossary alone, and
  `check_round_trip` refuses, at render time, any rendering that does not read back to exactly its
  reports and notices. The one exception is deliberate: earthquakes in the classic genre are the
  0.2.0 digest, at 0.2.0 precision (see Limitations).
- **The oracle stays text-only.** It reads the glossary from the prompt (`parse_glossary`), picks
  the parser by sniffing the digest (`detect_genre`: the first genre other than classic that
  recognises it, else classic, so a legacy digest is read as 0.2.0 read it) and parses it. Besides
  the text it gets each question's chain, as in 0.2.0, and the name of the instance's domain, which
  picks the reading rules in code and the record types. Sniffing is exclusive: each genre's digests
  are claimed by that genre alone, in every domain. The generator's self-check confirms that every
  rendering is recognised as the genre it was written in.
- **Every domain in every genre.** `tests/test_matrix.py` runs all 28 built-in cells (domain ×
  genre). In each cell, every rendering reads back exactly from its prompt, the oracle scores 100%,
  no question repeats, every instance plants all three kinds of trap, every recorded careless
  reading is reproduced from the text of every rendering, and the naive reader errs only where a
  trap explains it.

### Mixes (`--mix`, `evalgen/mix.py`)

| mix | instances come from |
|---|---|
| `legacy` (the default) | earthquakes in the classic genre, on the 0.2.0 seed paths: the 0.2.0 run. It takes no private config |
| `random` | every domain, in every genre. Without a private config, that is every registered domain and genre (built-ins in module order, then plugins), all public; with one, exactly the domains and genres it lists, with their splits |
| a list, e.g. `weather,neo:email,invented:table` | the entries named: `domain:genre`, or a bare domain, in any genre (with a config, any genre it lists). With a config, the list may name only what the config lists |

A mix that leaves a choice draws each instance from `seedlog.rng("instance", i, "mix")`, a seed path
that no legacy run creates. It draws the instance's domain (every domain of the mix alike), then the
genre of its original rendering (every genre allowed for that domain alike), then the genre of each
variant: one other than the original's whenever the mix allows one, every other genre in turn
before any of them twice. A mix of one domain and one genre draws nothing.

Each domain keeps its own snapshot (`snapshot/<domain>/` in a run of several), its own twin
templates (seed path `templates/new/<domain>`) and its own balance of the template pools, so the
public and new pools stay matched within every domain. No concrete question repeats in the run.
Instances record their domain, its split and any surface words, and renderings record their genre.
The manifest records the mix, the inventory, the config's commitment and the plugins, and, for a
run of several domains, one snapshot per domain; legacy runs record none of these. A run off the
legacy path is named after its mix, as in `runs/random-synthetic-seed7`.

### Command-line options

| option | commands | what it does |
|---|---|---|
| `--mix MIX` | generate, demo | which domains and genres instances come from (above) |
| `--private-config PATH` | generate, demo | the private run config: the held-out domains and genres and the plugins that define them, checked whole and committed in the manifest. A legacy run refuses one |
| `--private-config PATH` | run, score, report, resolve | loads a run's plugins from its config, after checking the config against the run's commitment. A run with plugins needs it in every new process |
| `--source live` | generate, demo | off the legacy path, each domain's own live source: USGS, NWS or JPL, and synthetic data, with a note, for a domain without one. `--source usgs` still names the USGS feed (off the legacy path, only for a mix of earthquakes alone), and `--feed hour` is for legacy runs only |
| `--snapshot-file PATH` | generate | replays one frozen live feed: the USGS GeoJSON of a legacy run, as in 0.2.0, or the frozen feed of the one domain of a mix |
| `--snapshot-dir DIR` | generate | replays an earlier run's snapshot folder, off the legacy path. Each domain is rebuilt as that run read it: from its frozen bytes, or from the seed where that run drew synthetic records. A live run of several domains thus regenerates byte for byte, offline |
| `verify-config RUN CONFIG` | its own command | checks a revealed config against a run's commitment: `ok` (exit status 0), `MISMATCH` (1, also for a file that is not JSON), or `error:` (2: no commitment to check, or a config that gives a key twice) |

Frozen bytes are parsed and checked before the run directory is created, on the legacy path too: a
`--snapshot-file`, and every frozen feed under `--snapshot-dir`. A domain without a live source,
another source's bytes (even bytes that parse to no records) or a missing file stops `generate` with
`error: ...` and exit status 2, as does a private config that is malformed or names a plugin that its
registry refuses.
| `mock:held_out=P` | run `--models` | the simulated model with a penalty for held-out sets (see the model table above). Without a penalty it draws exactly as before |

### Report additions

Reports of runs off the legacy path add, for each model:

- **accuracy by domain** (on the original renderings, by the instance's domain) and **by genre** (on
  every rendering, by its own genre), each marked public or held out;
- the **source gap**: accuracy on instances from public domains minus accuracy on instances from
  held-out ones, on the original renderings;
- the **genre gap**: accuracy on renderings in public genres minus accuracy on renderings in
  held-out ones, over every rendering;
- **cross-genre consistency**: agreement between every pair of renderings of a question in two
  different genres, compared like metamorphic consistency off the legacy path (two answers agree when
  they are equal within the scoring tolerance, or when both are scored right), and the share of
  questions answered alike in all their genres. Like metamorphic consistency, it needs no ground
  truth beyond the scores.

Each gap is given raw, with a 95% normal-approximation interval, and *adjusted*, like the template
gap: compared within strata of equal code-computed difficulty, each stratum weighted by its share
of the rows, with a 95% interval from the stratified variance. The report also gives its *coverage*,
the share of rows in strata that both sides have.

- **The strata of both gaps** are the answer's family (a count, an ID, a field's value, a distance,
  a group or a time), the trap exposure (0, 1 or 2+) and whether the filters matter; the template
  gap uses the last two. The genre gap's strata also tell the original rendering from a reworded
  variant. Neither gap needs a stratum for the template pool, because every instance asks about as
  many public questions as new ones (see Templates), and within each domain a twin pair never drifts
  more than one question apart.
- **Each gap is also stratified by the other kind, by name.** The source gap's strata hold the genre
  of the original rendering, and the genre gap's strata the instance's domain. A list mix can tie a
  domain to genres: `earthquakes,toy:toy-telex` renders held-out toy in held-out toy-telex alone, and
  `weather,neo:email,invented:table`, with email held out, renders public neo in held-out email alone.
  Compared within each domain, a model weak on neo shows no genre gap, whereas within the split of
  the domain (every domain of that mix is public) neo's errors read as a gap of the email genre, and
  are flagged. A domain that is never rendered in genres of both splits drops out of the genre gap, and a
  genre never given to originals of domains of both splits out of the source gap; where none is left,
  the adjusted gap is n/a. Comparing within names costs a random mix, which draws every genre alike
  for every domain, some power (see Limitations).
- **The intervals count instances, not rows.** An instance's questions are read in one digest, and
  each of its renderings asks them again, so a model's answers within an instance go together: one
  misreading costs several rows, and the genre gap counts a question once per rendering. Each interval
  is widened by the design effect of its rows (Kish): the cluster-robust variance of the difference
  between the two sides, with the instance as the independent unit, over its variance with every row
  independent, when that ratio is above 1. The adjusted gap takes the design effect of the rows that it
  compares (its strata are too small for a cluster-robust variance of their own). Where answers do not
  cluster, the design effect is close to 1 and the intervals hardly change; the simulations below
  measure both cases. The template gap keeps its 0.2.0 interval: every instance asks both pools.
- **The flag** is the template gap's: an adjusted gap above 10 points whose 95% interval excludes
  0. A gap needs rows on both sides, so it exists only when the private config holds something out.

The run section adds the mix, the inventory (each domain with its instances, each genre with its
renderings), each domain's snapshot and the config's commitment. The console summary shows each
model's adjusted source and genre gaps and its cross-genre consistency. Besides:

- **Held-out sets that are public code.** `report.json` lists the held-out domains and genres that
  are built into evalgen (`run.held_out_built_in`), and the held-out section then warns that anyone
  can practise on them. It warns, too, when the run's config keeps the nonce of the published example
  (the manifest's `private_config_example`), whose held-out sets are public test code. So a demo is
  not taken for a test of anything held out.
- **What the models read.** Each example instance prints its glossary before its digest, exactly as
  the prompts print it: an invented domain's digest cannot be read without it. The answer types that
  0.2.0 did not have (`weather_temperature`, `invented_f1`, ...) are explained under their table.
- **Templates by domain.** `by_template` in `report.json` is keyed `<domain>/<template>`
  (`weather/N01`), since every domain has a P01 of its own and twins of its own; legacy reports keep
  the bare template IDs.
- **Accurate wording.** The naive reader and the balance of the template pools are described as
  they are off the legacy path (see the model table and Templates above).
- **No local paths.** The report names the run by its ID, never by its folder (in the
  `verify-config` hint and the forecast hint), and a changed plugin by its path as the config names
  it.

Legacy reports are unchanged, word for word, including two sentences of the 0.2.0 text that say more
than holds (see Limitations).

Validation with the simulated model, on synthetic data, one run per row. Each run was answered by
the oracle, the naive reader, the mock, and the mock with a penalty of 0.2 or 0.3 on held-out
domains or on held-out genres. "CI" is the half-width of the adjusted gap's 95% interval, in
points:

| mix; held-out domains; held-out genres | seed | instances | source gap: CI, coverage | genre gap: CI, coverage | the penalties |
|---|---|---|---|---|---|
| random; neo, invented; table, form, radio | 42 | 100 | ±13–15, 64% | ±6, 92% | each flagged by its own gap only, but the domain penalty of 0.2 was missed (+12 ± 13) |
| the same | 43 | 300 | ±6, 84% | ±3, 98% | each flagged by its own gap only |
| random; invented; radio | 44 | 300 | ±7, 82% | ±5, 88% | each flagged by its own gap only |
| random; toy; toy-telex (the example config) | 45 | 300 | ±6–7, 89% | ±3, 99% | each flagged by its own gap only |
| the same (the demo, 0.3 penalties only) | 42 | 20 | ±35–40, 41% | ±17–19, 60% | the genre penalty flagged; the domain penalty missed |
| `earthquakes,toy:toy-telex`; toy; toy-telex | 46 | 100 | ±14–16, 60% | ±8, 49% | each flagged by its own gap only |
| `earthquakes:classic,toy:toy-telex`; toy; toy-telex | 47 | 60 | n/a | n/a | neither: the raw gaps move together |

(The half-widths are those of the penalised mocks, measured with the released code; a reworded
prompt reseeds the mock's errors, so a gap's value moves a little with the code, its width hardly.
Before each gap was compared within the other kind's names and its intervals counted instances, the
first row read ±10, 87% and ±6, 97%, and every penalty was flagged by its own gap. The random mixes
of several domains and genres lost coverage to the finer strata. In the mixes of the example config,
whose names and splits coincide, only the design effect changed anything: the source intervals of
the mocks penalised on genres widened by up to a seventh, since a genre penalty costs all of an
original's questions at once, so their rows go together.) In every one of these runs, the oracle,
the naive reader and the mock without a penalty were flagged by neither gap. In the partly tied mix,
a penalty on held-out domains also raised the raw genre gap (by 12 to 18 points), and a penalty on
held-out genres the raw source gap (by 11 to 14); the adjusted gaps kept each penalty on its own gap.

Simulated readers on the same kind of design (random mixes of 40 or 100 instances, neo and invented
held out, and table, form and radio), 600 to 2,000 runs per cell, measured the intervals and the
flag, now and (in brackets) before those two changes. A reader with no gap at all (a chance of 0.77
of a right answer, or of 0.8 with a random effect) should have its gap of 0 covered in 95% of runs
and be flagged in at most 5%:

| reader with no gap | 40 instances: source gap covered, flagged | genre gap | 100 instances: source gap | genre gap |
|---|---|---|---|---|
| answers independent | 96%, 2% (95%, 3%) | 96%, 2% (94%, 1%) | 95%, 2% (95%, 2%) | 94%, 0% (94%, 0%) |
| a random effect per rendering (SD 2, logit scale) | 93%, 4% (83%, 9%) | 93%, 3% (81%, 7%) | 95%, 3% (85%, 8%) | 93%, 1% (80%, 2%) |
| a random effect per instance (SD 2) | 93%, 3% (84%, 8%) | 94%, 3% (89%, 4%) | 95%, 2% (82%, 9%) | 96%, 0% (88%, 0%) |
| right 65% of the time on neo, 85% elsewhere, in `weather,neo:email,invented:table` with only email held out | | 98%, 2% (4%, 94%) | | |
| right 65% of the time in table, 85% in classic, in `earthquakes:classic,invented:classic,weather:table` with only weather held out | n/a in every run (29%, 71%) | | | |

The template gap needs neither change: every instance asks both pools, so an instance's own
difficulty cancels in it (with a random effect per rendering or per instance of SD 2, its interval
covered the gap of 0 in 98% of 1,000 runs, and flagged at most 1%).

### The 0.2.0 regression guarantee

- `cli.py generate --seed 16222556419173536875 --instances 100`, with no new options, writes the
  same 405 files as 0.2.0, byte for byte. Its manifest differs only in fields that differ between
  two identical runs anyway (the run ID, the seed nonce and its commitment, the generation time) or
  that describe the code (the version and the code fingerprint). `tests/test_regression_020.py`,
  whose hashes were taken from 0.2.0 before the refactor, pins all of them.
- The legacy path calls the 0.2.0 functions with the same arguments and the same random streams.
  Every new random choice has a seed path of its own (`instance/<i>/mix`, `instance/<i>/surface`,
  `snapshot/<domain>/…`, `templates/new/<domain>`), which only the new options create, so no legacy
  draw moves.
- Legacy manifests, instances, scores and reports carry none of the new keys. A legacy report is
  pinned to its hash from before the new metrics existed (`tests/test_mix.py`), apart from what
  differs between any two runs or names the code: the seed commitment, the version and the code
  fingerprint. The mock without a penalty draws exactly as before.
- Legacy runs are byte-identical under Python 3.9 and 3.14 (see Reproducibility).

So the 0.2.0 study in the README regenerates byte for byte under 0.3.0.

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
  with 0.1.0 do not regenerate byte for byte under 0.2.0. Version 0.3.0 leaves the legacy path as
  it was, so runs made with 0.2.0 regenerate byte for byte under 0.3.0.
* Model replies are cached in `responses/`, so `score` and `report` can be re-run without new API
  calls.
* Runs off the legacy path (0.3.0) regenerate exactly from the seed, the mix and the private config
  with its plugins. Live sources are frozen per domain, and `--snapshot-dir` replays them. Runs off
  the legacy path made by development versions of 0.3.0 do not regenerate byte for byte under the
  release. Before commit `f8a0835` (their manifests still say 0.2.0) traps, twins, the codec and the
  rules changed. The last commits before the release changed what those prompts print (the
  glossary's ID line, the earthquake rules' heading and rule 9, the wording of radii, some news,
  chat and form wordings, and the invented domain's surfaces) and the format of the frozen NWS
  bundle. Legacy runs from every one of those versions do regenerate.
* Python 3.9 for 0.3.0 was checked by script, because pytest is not installed for the system
  Python 3.9 of the machine that built it. `tests/test_python39.py` scans the code for what 3.9
  cannot run. Under Python 3.9.6, the 0.2.0 regression run matches the hashes its test pins, and
  three runs off the legacy path match the same runs under 3.14 file for file: a random mix that
  renders every built-in domain in every genre, a random mix with the example config's plugins, and
  an explicit list.

## Repository layout

```
cli.py                  commands: generate, run, score, report, resolve, demo, verify-config
configs/                private.example.json: the format of a private run config (EXAMPLE only)
docs/                   DESIGN.md (this file), PLAN-0.3.md (the plan of 0.3.0)
evalgen/
  seeds.py   util.py    seeding (SHA-256 child seeds, salted commitment) and small helpers
  sources.py            synthetic world, USGS feed (duplicate solutions collapsed), frozen + hashed snapshots;
                        the snapshots of every domain, live or synthetic, and their replay
  domains/              the Domain interface (base.py) and the domains: earthquakes, weather, neo, invented
  genres/               the Genre interface, glossary and shared formats (base.py) and the genres:
                        classic, news, email, chat, table, form, radio
  registry.py           discovery of domains and genres, and private plugins
  mix.py                mixes, the private run config, its commitment, plugins, held-out inventories
  units.py              units and the exact codec of every printed number
  rules.py              the reading rules (English for prompts, code for the key) + trap mishandlings
  ops.py                operation chains (generic kinds, 0.2.0 aliases) and the answer-key executor
  questions.py          templates (public + twins), RNG parameter drawing, margin guards, wording
  traps.py              trap planting (load-bearing, guarded off the legacy path), fallen answers, resolution invariant
  digest.py             messy rendering: styles, units, time formats, layouts (the 0.2.0 classic digest)
  solver.py             text-only oracle and naive solvers (digest parser; genre detection off the legacy path)
  prompt.py answers.py  prompt construction (with the glossary off the legacy path); robust JSON reply parsing
  adapters.py           oracle, naive, mock (with held-out penalties), Anthropic, OpenAI
  scoring.py            normalisation, tolerances, consistency, intervals
  forecast.py           delayed-resolution questions and resolution (USGS FDSN)
  audit.py report.py    audit sample; report.md / report.json
  pipeline.py           one cycle end to end
tests/                  697 pytest tests: hand-built key cases, determinism, traps and oracle, guards,
                        digest parsing, reply parsing and scoring, USGS and forecasts, adapters, report,
                        head to head, reading-rules coverage, CLI; each domain and genre, every domain in
                        every genre (test_matrix.py), mixes and held-out sets, the 0.2.0 regression test;
                        test-only plugins (toy_domain.py, toy_genre.py); fixtures/ holds recorded NWS and
                        JPL responses and pinned invented instances
runs/<run>/             manifest.json, snapshot/ (snapshot/<domain>/ for several domains), instances/,
                        prompts/, responses/, scores/, audit/, forecasts.json (usgs), report.md, report.json
```

## Limitations and next steps

* **Finite grammar.** Ten terminals and seven filters in the earthquake domain. New twins are new
  *combinations* of known operations, so the seen-vs-new gap detects skeleton-level overfitting,
  not every kind. 0.3.0 applies the same operations to the fields of every domain, so it widens the
  sources and genres a model must generalise over, not the grammar. The adjusted gap removes
  differences in trap exposure and filter relevance, but not every possible difference between
  filter types (a radius filter needs more arithmetic than a region filter).
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

### 0.3.0

**Held-out sets**

* **Held-out sets protect only while they stay private.** Once a config and its plugins are
  revealed, so that the run can be verified, its held-out domains and genres are public, and the
  next evaluation needs new ones. They also wear out with use: every prompt shows a model the
  glossary of its domain and the look of its genre, so a held-out set is entirely new only the first
  time its prompts reach a model, or an API provider that may keep them.
* **Built-in held-out examples are public.** `configs/private.example.json` holds out the test-only
  toy domain and toy-telex genre from `tests/`, and a config may mark built-in domains and genres
  held out. Both exercise the machinery and the metrics, but they hide nothing from an adversary
  who reads the repository (a report says so, for a built-in held-out set and for a config that keeps
  the example's nonce), and the example's nonce is public, so its commitment can be guessed.
  Real held-out sets come from private plugins, with a nonce of their own.
* **What the commitment binds.** It binds the config's content: the names, the nonce and each
  plugin's entry. A plugin given with its SHA-256 is bound by its bytes too, and evalgen refuses to
  run one whose bytes differ. A plugin given by its path alone is bound by that path only: its bytes
  could change after the commitment is published, which the manifest's record and a regeneration of
  the revealed run show only afterwards. Give every real plugin its SHA-256. The nonce check refuses
  words, constants and patterns, but it cannot prove a nonce random.
* **What a run's manifest names.** It names the private config by its file name and each plugin by
  its path as the config gives it, so a run names no folder that the revealed config does not. The
  0.2.0 fields that name files are as they were: the frozen feed a run replays (`snapshot_file`, as
  given on the command line), the run it came from (`replay.source_run`, an absolute path), and
  `--snapshot-dir` as given. A legacy report also names the run's folder, as given, in its forecast
  hint. Check those before publishing a run directory.
* **Plugins are code.** They run in the evaluator's process, and a run that uses them needs
  `--private-config` again for `run`, `score`, `report` and `resolve` in a new process.

**The source and genre gaps**

* **Power.** The validation table above gives the scale. About 100 instances with two held-out
  domains and three held-out genres put the adjusted source gap within about ±13 points and the
  genre gap within ±6; 300 instances within ±6 and ±3. With 20 instances, a 30-point penalty on
  held-out domains went unflagged. Fewer rows on the held-out side widen the interval: one held-out
  genre out of seven gave ±5 at 300 instances, where three gave ±3. In simulated 100-instance runs of
  that design, with answers independent, a reader 20 points worse on the held-out domains was flagged
  in 93% of runs, 15 points worse in 72% and 10 points in 38%; one 15 points worse in the held-out
  genres in 95% and 10 points in 50%.
* **Comparing within names costs a random mix power.** Each gap compares within the names of the
  other kind (see Report additions), which a list mix needs and a random mix, which draws every genre
  alike for every domain, does not; there the finer strata compare fewer rows. In the first row of
  the validation table the source gap's coverage fell from 87% to 64% and its interval widened from
  ±10 to ±13 points, and in simulation the share of 100-instance runs that flag a reader 15 points
  worse on held-out domains fell from 83% to 72% (and of 40-instance runs from 42% to 25%); the genre
  gap, compared within four domains rather than seven genres, lost almost nothing (95% either way).
* **Tied splits.** Both adjusted gaps compare within the other kind's names. When a mix ties domains
  to genres, the adjusted gap has low coverage (49–60% in the partly tied mix measured; 17% in a
  40-instance run of `weather,neo:email,invented:table` with email held out, in which only weather is
  rendered in genres of both splits) or is n/a, and the raw gaps mix the two kinds (in the fully tied
  mix they are the same for both).
* **The gaps' intervals count instances, not domains or genres.** With one or two held-out domains,
  the gap compares those particular domains with the public ones. A held-out domain or genre that is
  harder in itself (an invented domain read through its glossary, a genre with more noise) shows a
  gap for an honest model too, and the strata do not remove a domain's or a genre's own difficulty.
  The simulated penalties above do not model this. A gap says something about domains or genres in
  general only with several held-out ones on each axis, so that their own differences average out.
  It is a flag, not proof.
* **The other intervals count questions.** The accuracy intervals, the template gap's, the
  consistency intervals (over answer pairs) and the head-to-head comparison treat questions as
  independent, as 0.2.0 did, so they are too narrow for a model whose errors cluster within
  instances. Of these only the template gap's feeds a flag, and its pools are asked alike in every
  instance (see the simulation above). Legacy reports keep all of them exactly.
* **The design effect is an approximation.** The adjusted gap's interval is widened by the design
  effect of the rows it compares taken as one raw difference, because its strata are too small for a
  cluster-robust variance of their own. In simulation its intervals covered 93–96% of runs with
  strong clustering, and at most 4% of the readers with no gap were flagged; with few held-out
  instances (a demo of 20) every interval is rough.
* **Legacy consistency compares as 0.2.0 did.** A legacy report's metamorphic consistency compares
  two answers by the scoring tolerance alone, so two right answers on either side of the key (4.96
  and 5.04 for a magnitude of 5.0) count as an inconsistency there. Off the legacy path both
  consistency metrics count two answers that are both scored right as agreeing.

**Domains and genres**

* **The 0.2.0 text that legacy output keeps.** Legacy prompts and reports are the 0.2.0 ones, byte
  for byte, so three flaws that 0.3.0 fixes off the legacy path remain in them. A radius can be
  worded "no more than 40 km from …" beside a depth threshold, which reads as a range of depths
  (rarely: none of the 1,191 question texts of the 0.2.0 regression run; 2 of 471 in a 40-instance
  random mix before 0.3.0 reworded it there). The report says that "each twin pair is used
  equally often", which the 0.2.0 composer does not do (see Templates). And it describes the naive
  reader as treating "every line" as a separate quake, which is true of the classic digest only.
* **Invented words.** Coined names of four or five letters, and their unit words, can still be English
  words (a unit called "halls"), and the list of offensive parts that no coined word may contain is a
  list: it is not a filter of every word that gives offence.
* **Live sources: USGS, NWS and JPL only.** The invented domain has none, by design. The NWS and
  JPL CAD adapters have been exercised only against recorded API responses and fake transports,
  not yet against the live APIs.
* **A possible tell in live weather digests.** Real NWS observations are stamped to the minute,
  with winds in whole knots and pressures in hundredths of an inch of mercury. The generic trap
  builders take no hint from the domain, so planted lines can carry seconds and continuous values,
  and a reader could spot one by its form rather than by its note or notice. Synthetic data has no
  such tell, and no answer key depends on it. Fixing it needs a domain hook in `traps.py`.
* **Weather stations.** Beyond the five stations sampled from the API for the recorded fixture
  (KJFK, KORD, KDEN, KSEA, KLAX), the names and positions of the 24 stations are a hand-made table
  in the NWS style. In a live run an observation takes its position from the API whenever it gives
  one, and its station's name always from the table.
* **neo has five bodies.** Templates whose only parameter is the body (P03 and P08, and their
  twins) run out after about five questions per run. In a 60-instance neo run, P03 and P08 asked
  five questions each and their twins six.
* **The invented domain's realism.** It is a synthetic world of fixed shape: eight depots, five
  agencies, two log-normal quantities per depot. Its surface words are drawn per instance, but its
  status words ("verified", "provisional") and field keys ("f1", "f2"), which every glossary prints,
  are fixed for the domain, as are its ID shape and its distributions. An adversary can practise on
  it like any public domain. It tests reading through a glossary, not the reading of a real source.
* **Earthquakes in the classic genre off the legacy path are the 0.2.0 digest.** It rounds depths to
  0.1 km or mi, positions to 0.001° or a hundredth of an arc-minute and some times to the minute,
  and the 0.2.0 parser does not read back place notes, retraction reasons or the magnitude that a
  correction replaces. The margin guards keep the oracle at 100%, and the cross-matrix test reads
  that cell back at 0.2.0 precision.
* **Duplicated adapter code.** Five genres (news, email, chat, table, form) keep private copies of
  the map between the earthquake `Report` and `Notice` and the generic form; radio uses
  `Domain.to_generic` and `Domain.from_generic`. Consolidating them is a follow-up.

**Mixes**

* **Random draws depend on the registry.** Without a private config, `--mix random` draws from every
  registered domain and genre in registry order, so adding a built-in domain or genre changes what a
  seed draws in random mode. Legacy runs are unaffected, and a config fixes the inventory: a new
  built-in that it does not list changes nothing.
