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
python3 -m pytest -q                           # 866 tests, ~4 minutes
```

The demo writes `runs/demo-synthetic-seed42/report.md` (and `report.json`), as 0.2.0 did, which replaces
the report committed there, of a run that Claude Opus 5.5 answered through the API (`git checkout
runs/demo-synthetic-seed42` restores it; give `--out` to write elsewhere). With `--mix random` it writes
`runs/demo-random-synthetic-seed42/`, and with `--private-config` too
`runs/demo-heldout-random-synthetic-seed42/`. Each demo replaces its own folder every time.

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
   answers. Off the legacy path a planted report stands in for a record drawn with the digest but
   not kept, so its time, place, source, values and ID are a real record's: only the signals the
   rules name tell it apart (a shared ID or a note naming one, its status, the notices), and the way
   to catch a trap is to read them and apply the rules (see Traps under 0.3.0). The legacy path
   keeps the 0.2.0 placement, whose lines copy the style of real ones (place notes that agree with
   their coordinates, the network's magnitude precision, IDs that fit the network's own ID
   sequence) but not where real lines lie, which a reader can use instead (see Limitations).
4. **Metamorphic consistency.** The same snapshot is re-rendered with different units, line order,
   formats and question wording. Correct answers cannot change, so any change in a model's answer
   is an inconsistency. This needs no ground truth on the legacy path, and none beyond the scores off
   it, where two answers that are both scored right agree; it catches surface-pattern memorisation.
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
commitment before the evaluation, all 64 hexadecimal digits of it, as `generate` prints it and the
manifest records it (`seed_commitment`); the nonce makes it impossible to brute-force even for a small
seed such as 42. Revealing the seed and nonce afterwards lets anyone verify the commitment,
regenerate the exact instances and audit them.

**Use a seed that nobody can guess.** The nonce protects only the commitment. The instances are a
function of the seed and the public code, so anyone can generate those of a small seed in advance, and
any one prompt of a run gives a small seed back by brute force in seconds: its compile time and record
IDs pin it down. For an evaluation, leave `--seed` out: `generate` draws a fresh 64-bit seed and logs
it. Small seeds such as 42 or 7, as in the examples here, are for demos, tests and reproductions, and
`generate` warns (on the console only; the run's files do not change) about an explicit seed of fewer
than 48 bits.

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
   the template gap flags one tuned to the public skeletons: above 10 points, with a 95% interval
   that excludes 0 and, unlike the template gap, compared rows from two instances or more on each
   side (see Report additions).
3. **Cross-genre consistency.** A variant is rendered in another genre than its original, so the
   same question asked of an email thread and of a radio transcript must get the same answer. Like
   metamorphic consistency, this needs no ground truth beyond the scores (two answers agree when they
   are equal within the tolerance or both scored right).

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
required keys, distinct names that each appear in one list only, and names that evalgen or one of the
config's own plugins defines once they are loaded (a name that the process knows only because a
caller, or another config's plugin, registered it could not regenerate the run in another process).
The config is read strictly: a file that gives a key twice in one JSON
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
or one run of digits repeated. No check can prove a nonce random. The one nonce exempt from the check
is the published example's own (`EXAMPLE-ONLY-draw-your-own-with-secrets.token_hex`, `mix.EXAMPLE_NONCE`):
it is accepted with a loud warning that its commitment can be guessed, so that the held-out metrics can
be demonstrated offline. Any other nonce is checked, one that merely contains `EXAMPLE` included.

**What the evaluator publishes, and when.** Before the evaluation: the two commitments that
`generate` records in the manifest, `seed_commitment` and `private_config_commitment`, and nothing
else (not the manifest, which holds the seed, nor the config or its plugins). Publish each in full, all
64 hexadecimal digits, as `generate` prints them: a prefix of 16 digits could be matched by another
config or seed in about 2^32 tries. After it: the config, its plugin files, and the seed with its nonce.

**Verifying a revealed config.** `python3 cli.py verify-config RUN_DIR CONFIG` prints `ok` (exit
status 0) when the config hashes to the run's commitment, and `MISMATCH` (1) otherwise, including
for a file that is not JSON. It exits with `error: ...` (2) for a run generated without a config,
which has no commitment to check, for a config that gives a key twice in one object, which reads
as two configs, and for arguments that are not a run and a config file (given the wrong way round,
say), so that status 1 always means MISMATCH. The hash is over canonical JSON, so re-indenting the file or reordering its keys
changes nothing, while any change to a name, to the order of a list, the nonce, a plugin entry or the
note breaks the match.

**What the commitment pins of a plugin.** A plugin given with its SHA-256 is pinned by the
commitment: `generate` and every later step check its bytes before any of it runs and refuse a file
whose bytes differ, so a held-out domain or genre cannot change once the commitment is published,
and what runs is exactly the bytes that were checked (compiled from them, with no bytecode cache read
or written beside the plugin). A plugin given by its path alone is committed to by that path only:
the manifest records the SHA-256 of the bytes that ran, a later step loads a changed file with a
warning and lists it in `report.json` (`plugins_changed`), and only a regeneration of the revealed
run shows what it was. A process runs one version of each plugin: once it has run a plugin file, a
run or a step that would load that file with other bytes (it was edited since) stops with
`error: ...`, pinned or not, rather than run the old code under the new bytes' name, and a new
process loads the new bytes.
Regenerating the run with the same evalgen code, the revealed seed, mix and config, and its plugins
at the same relative paths, rebuilds the same instances and prompts byte for byte.

**Plugins never come from a run directory.** A plugin is a Python file that defines `DOMAIN`,
`GENRE` or both. `generate` loads the config's plugins before it looks up any name. Every later
step in a new process (`run`, `score`, `report`, `resolve`) loads them only from the config that
the evaluator passes with `--private-config`, after checking that config against the run's
commitment; within one process, a later step registers again the plugins that the process already
ran for that config, as they ran, without running any of their code again. evalgen never executes a
file that a run directory names by itself, so reporting on a run received from someone else runs
none of its code.

**A received run cannot reach outside its folder.** Every step after `generate` reads a run strictly
before it reads or writes anything else. It refuses, with `error: ...` (exit status 2):

- a run directory that holds a symbolic link anywhere, which could point at any file (the folder
  itself may be reached through a link: the evaluator names it), or an entry that is neither a file
  nor a folder (a named pipe in a file's place would make a step wait for ever);
- a folder without a `manifest.json` that evalgen wrote, so that a mistyped path is an error, never a
  run with nothing to do, and a manifest whose seed, instances, plugins, inventory or config are not
  of the shape evalgen writes;
- an instance file whose instance ID is not its file name (`instances/inst-007.json` holds
  `inst-007`), or whose renderings are not `v0`, `v1`, ... each once. These IDs name the files that
  `run` writes and `score` reads (`responses/<model>/<instance>__<variant>.json`), so an ID such as
  `../../x` would otherwise send them outside the run (in ASCII digits: `\d` would take any
  script's);
- an instance file without an answer key for each of its questions, a scores file that is not an
  object with its rows and replies, and a `forecasts.json` that is not a list of forecasts;
- instance files that the manifest does not list, or listed ones that are missing.

Every file evalgen writes into a run directory goes through one writer (`util.write_in`). It refuses a
path that leaves the folder or passes through a link, and it writes a new file beside the old one that
then takes its name, so a link or a hard link planted in a file's place is replaced, never written
through, and no reader sees half a file.

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
  It draws its digest's trap plan, how many traps of each kind its digest holds (`instance/<i>/plan`),
  then its records, one fewer for each report the plan adds, together with 30 more that it does not
  keep, its slots, in whose places its traps are planted (`instance/<i>/slots`; see Traps). Once its
  traps are planted, a question about the record nearest to a place is asked about a place near a
  report of the digest, a real or a planted one (`instance/<i>/anchors`; see Traps).
- Step 2 renders each rendering in its genre.
- The prompt prints the domain's glossary between the reading rules and the digest.
- In step 5 the oracle, the naive reader and the mock are given each rendering's genre and its
  instance's domain by name, and read the prompt with that genre's parser. The generator's self-check
  instead sniffs the genre that recognises each digest, and requires it to be the genre the rendering
  was written in, so the two readings pick the same parser.

---

## Commands

```bash
# full cycle, offline, with the reference solvers and the simulated model
python3 cli.py demo --seed 42 --instances 20       # overwrites runs/demo-synthetic-seed42 each time, a
                                                   # committed run's report included (--out writes elsewhere)

# step by step
python3 cli.py generate --seed 7 --instances 20            # -> runs/synthetic-seed7/
python3 cli.py run runs/synthetic-seed7 --models mock,oracle,naive
python3 cli.py score runs/synthetic-seed7
python3 cli.py report runs/synthetic-seed7

# real models (keys from the environment, read without echoing them, which keeps them out of your
# shell history, as the README shows; in zsh: read -s "ANTHROPIC_API_KEY?Anthropic API key: ").
# Replies are cached; a re-run asks again only for prompts whose API call failed or whose reply hit
# the output-token limit, so re-running is free and safe.
read -rs -p "Anthropic API key: " ANTHROPIC_API_KEY && export ANTHROPIC_API_KEY
read -rs -p "OpenAI API key: " OPENAI_API_KEY && export OPENAI_API_KEY
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
                                                    # (-> runs/demo-heldout-random-synthetic-seed42)
python3 cli.py generate --seed 7 --mix weather,neo:email,invented:table         # an explicit mix
python3 cli.py generate --seed 7 --mix random --private-config private.json     # the evaluator's held-out sets
# (an evaluation leaves --seed out, and a fresh 64-bit seed is drawn: the small seeds here are for examples)
python3 cli.py run runs/random-synthetic-seed7 --models mock,mock:held_out=0.3 --private-config private.json
python3 cli.py score runs/random-synthetic-seed7 --private-config private.json
python3 cli.py report runs/random-synthetic-seed7 --private-config private.json
python3 cli.py verify-config runs/random-synthetic-seed7 private.json           # once the config is revealed
python3 cli.py generate --seed 7 --mix random --source live                      # USGS, NWS and JPL, frozen + hashed
python3 cli.py generate --seed 7 --mix random --snapshot-dir runs/random-live-seed7/snapshot   # -> runs/random-replay-seed7
```

`--force` overwrites an existing run directory, but only one that evalgen created (it carries
evalgen's marker file or its own manifest); any other folder is left untouched. Nor does it replace a
folder that holds what the new run reads: the frozen feed or snapshots it replays (`--snapshot-file`,
`--snapshot-dir`), its private config or a plugin, however the path names it: as written, as it
resolves, or with another letter case or Unicode form, which the file systems of macOS and Windows
take for the same name (the file system says which folder each is). A run folder can hold the only
copy of a live feed, so `generate` stops with `error: ...` (exit status 2) before deleting anything
and asks for another `--out`. A frozen snapshot whose `snapshot.json` is not the object evalgen
writes, and a plugin whose own code fails (a syntax error, an import it lacks, an exception it raises),
stop it before the run folder is made. `--instances` is at least 1 and `--variants` at least 0.

Each step needs what the steps before it wrote, and says so with `error: ...` (exit status 2) instead
of writing less. `report` refuses a run whose instance files are missing, as in a committed run (git
keeps only a run's manifest and report: regenerate it from the manifest's seed and config first), and
a run that has no scores yet, so that it never replaces a report with an empty one. `resolve` refuses
a path that is not a run, and a run whose instance files are missing: a committed run keeps no
`forecasts.json` either, so it cannot tell whether the run asked any forecast. `run` reads every model
spec before it asks any model: a spec that names no model, or options its model does not take
(`oracle:x`, a mock penalty given twice), is refused; an API model whose key is not set is skipped with
a note, and a `run` with no model left to ask is an error.

`--max-tokens` (default 16000) is the output budget per reply for API models. Models that think
before answering need room: at 4,096 tokens, Claude Haiku 5.5 hit the limit on two thirds of its
prompts, almost always before writing any answer. A reply that stops at the limit is scored on
whatever text it returned (usually none), flagged in the report, and asked again by the next `run`.

| model spec | what it is |
|---|---|
| `oracle` | rule-based solver that reads **only the digest text**, applies the reading rules and runs each question's operation chain; must score 100%. Off the legacy path `run` also gives it the rendering's genre and the instance's domain by name (see the oracle under Genres) |
| `naive` | the same parser, but every report counts as a separate quake and notices are ignored; it loses points only where traps (alone or combined) change the answer, apart from rare display-rounding effects. Off the legacy path it reads each digest with its genre's parser, as the oracle does, which has already set aside the quoted and read-back repeats and the message times that the rules say are no reports. So it never meets a genre's noise. Its answers depend on which reports a digest holds and on what it prints of them; every rendering prints the same reports, so its consistency across renderings and genres is high by construction, though a time or value that one rendering prints more coarsely than another can break a tie differently (the classic digest of earthquakes prints some times to the minute) |
| `mock` | a **simulated** contaminated model, not a language model. It answers public-template questions correctly 92% of the time (80% when paraphrased) and new-template questions 62% of the time; its errors are seeded by the prompt text and usually fall into the traps. It exists to exercise every metric without API keys. `mock:held_out=P` (0.3.0) also lowers each of those chances by P on an instance from a held-out domain and by P on a rendering in a held-out genre, as a model tuned to the public ones would be (`held_out_domains=P` or `held_out_genres=P` for one kind, joined with `+` for different penalties). A chance stops at 0, so where the penalties that apply add up to more than 0.62, the chance on a new template, they lower accuracy by less than they say, and a gap they inject is smaller than the penalty; `run` prints a note for such a mock |
| `anthropic[:model]` | Anthropic Messages API (`ANTHROPIC_API_KEY`; default model `claude-opus-5-5`, override with `:model` or `EVALGEN_ANTHROPIC_MODEL`) |
| `openai[:model]` | OpenAI Chat Completions (`OPENAI_API_KEY`; default `gpt-4.1`, override with `:model` or `EVALGEN_OPENAI_MODEL`; `OPENAI_BASE_URL` points it at any compatible server) |

API models receive only the prompt text. The oracle and naive solvers also receive each question's
operation chain (its machine-readable meaning), because the goal is to verify the *digest*, not to
build a parser for English questions, and off the legacy path the names of the instance's domain and
of the rendering's genre.

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

Where these lines lie is another matter, which legacy runs keep byte for byte. A re-report lies 3 to
40 s and 1 to 8 km from its quake and comes from one of the networks that re-report others' events
(us, at, pt, ew), of which only us files events of its own; a second report sits at its quake's very
time and place; a
conflicting magnitude lies within 0.2 of an asked threshold. None of this is in the rules, and a
reader can use it instead of them (see Limitations). Off the legacy path no trap is placed this way
(see Traps under 0.3.0).

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
questions draws the pool of its last one at random), or rarely more, when no template of one pool fits
an instance's records: two apart in 2 instances of 300 in the built-in domains, and three apart in 3 of
the 300 instances of a random mix with the example config, all three of the toy plugin domain, whose
records fewer templates fit. Twin pairs stay within one question in every run. And once a template's
twin has been used, the
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
failures, and the report warns about both. A prompt that has no stored reply that can be read (a
`run` cut short, a reply removed or broken since) counts as unanswered too, with the status
`missing`, so accuracy and the gaps are always over every question of the run, while metamorphic and
cross-genre consistency and the trap counts, which compare a model's own answers, leave those prompts
out, as before: a missing reply is a gap in the run, not an answer. The report and the console summary
say how many replies are missing (`format.missing` in `report.json`, only when there are any), and the
next `run` asks only those prompts.

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
FDSN event service once the window plus 2 h has passed, and the report shows Brier scores. A USGS
instance is an earthquake instance read from the USGS feed: `--source usgs`, `--source live` with
earthquakes in the mix (0.3.0), or a frozen USGS feed replayed with `--snapshot-file` or
`--snapshot-dir`.

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

Validation of 0.2.0, over 90 synthetic runs of 100 instances (the 0.2.0 code):
* the pool-agnostic `naive` baseline was flagged 0 times (the raw gap would have flagged it 10
  times);
* a simulated reader that misses each trap with probability 0.3 was flagged 0 times (raw: 7);
* the simulated contaminated `mock` was flagged in all 90.

The 0.3.0 calibration at the final commit (120 random-mix runs of 100 instances, see Report
additions) gives the same picture off the legacy path: the naive baseline's template gap was never
flagged (0 of 120), and the mock's 30-point contamination was flagged in all 600 runs of the mock and
its four penalised variants.

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
| `fields` | one `FieldSpec` per numeric field: its key and label; its canonical unit, other units and stored decimals; its answer type and tolerance (abs, rel); its guard margin (abs, rel) and rank gap; how thresholds are drawn (quantiles, jitter, steps); the filter and terminal kinds that apply to it (exactly the terminals that P01–P12 end in on its role: `validate` refuses one that no question would end in); a qualifier stored with the value (an earthquake's magnitude type); whether it is signed and whether notices can correct it |
| `roles` | which field plays the 0.2.0 *primary* role (magnitude) and which the *secondary* one (depth), so that P01–P12 bind to every domain |
| `spatial` | radii, distance guards and the precision of positions, for records with a location. A domain without one drops the spatial templates (P02, P07, P09, P12) and their twins as a pair |
| groups and sources | every record belongs to a group (a region, a body, a depot), which `in_group` and `compare_groups` ask about and digests focus on (3–4 per digest), and has a source that filed it (a network, a feed, a catalogue, an agency) |
| `synthetic(rng, n, compiled_at, window_hours)` | the seeded generator, which never reads the clock |
| `live` | an optional live source. `fetch` downloads it once, and the raw bytes are frozen in the run and SHA-256 hashed. `parse` reads them without the clock or the network, so a replay rebuilds the snapshot exactly. A source that cannot be reached, that answers with bytes its parser cannot read, or that gives too few usable records (`min_records`), falls back to synthetic data, with a note. The minimum applies to a fetch only: a replay cannot fall back, and reads its frozen bytes whatever number of records they give (none is refused), so a frozen feed smaller than the minimum, which a live run never freezes but a hand-made one can be, gives digests of fewer records than usual (the 4 observations of the NWS fixture `tests/fixtures/weather/raw_feed.json` give digests of 1 or 2 records). The note names the failure without what changes from one moment to the next (a URL's query, such as the start time of an NWS request, and an object's address), so two offline runs of one seed write the same files |
| IDs | `id_pattern` (the scorer also reads ID answers with it), `new_id` and `mimic_id`, so that a planted report that stands in for no record (the 0.2.0 placement, or when no slot is left) carries an ID shaped like real ones; `id_place`, for IDs that number their records in time order (a close approach's `-ca1`, `-ca2`), and `ids_name_place`, for IDs that name their record's place, so that a report filed under a record's ID never contradicts what the ID says (see Traps) |
| `statuses` | the status of the report that settles a record's values, then the one that does not ("reviewed", "preliminary") |
| `trap_vocab` | re-reporting sources (those of the 0.2.0 placement; off the legacy path a re-report may come from any source), duplicate notes, retraction reasons, and the phrases of retraction and correction notices |
| near twins, planted values | `twin_seconds`, `twin_share` and `twin_key`: which real records lie as close together as a re-report and its record, and how often a digest draws them together; `twin_fields`: the fields whose values a record shares, give or take, with its twins (a station's temperature and pressure, an object's distance, speed and H), so that a corrected value agrees with them as a real one does (`validate` refuses a field it does not have, or twin fields without twins); `planted_value` and `rereport_values`: how a value planted on a real record's report is written, and what a report filed in a slot's place gives for a record (a close approach's own H, under its ID: see Traps) |
| `chatter`, `lexicon` | lines without data, which never say what a digest holds (a status line says when a station was offline, with its date, not that its reports were left out, which the digest's own reports could contradict; off the legacy path its window never holds a report of the station's source, and genres that post lines at times of their own post it after the window: `digest.timed_chatter`), and the wording of questions (`standard_lexicon` words any domain from its own words) |
| `rules_text()`, `report_time()`, `glossary(surface)` | the reading rules (in which `report_time` says what a report's time is) and the glossary its prompts print |
| `draw_surface(rng)`, `bind(surface)`, `surface_group_labels` | per-instance surface words, for a domain that draws them, and every group label they can draw: the scorer, which reads answers by the types registered when the domain was loaded, lets a group answer end in any of them ("Nerm Halt berth" is "Nerm Halt"), and `bind` refuses a surface that calls its groups anything else, names a group with one, or names one that the scorer would not read alike with its label after it (with "area" and "sea area" both group words, "Kessel Sea area" reads as "kessel") |
| `resolve`, `to_generic`, `from_generic` | the reading rules in code, and the map between a domain's own record types and the generic ones that genres write and read (every genre but classic, through `genres.base.generic_items` and `domain_items`) |
| `group_preposition`, `utc_offset` | how prose places a record in its group ("in" a region, "near" a body approached), and the UTC offset of the local time a located record is written in (the whole hour nearest the solar time of its longitude, unless the domain knows its records' real zones, as weather does) |

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

Off the legacy path these traps are planted by one set of builders for every domain, earthquakes and
plugin domains included, and a planted report stands in for a *slot*: a record drawn with the digest,
exactly as its records are, but not kept (`pipeline.sample_with_slots`, seed path
`instance/<i>/slots`). A phantom is a slot filed as it is. A re-report is a slot filed under its own
ID with a note naming the real record of its group nearest to it in time, or under the ID of a real
record of its group that is not one of the slot's own neighbours (not its nearest record, nor a twin of
it: `traps._record_for`), with the unsettled status and the slot's values (`Domain.rereport_values`:
under the record's own ID a close approach gives its object's H as another catalogue fits it). A second
report is the same, under the record's ID. A correction shows a slot's value on the record's settling
report, written to that record's places (`Domain.planted_value`), or, for a field tied to the record's
near twins, a value that agrees with them (see *Tied values*). So a planted report has the
neighbours, in time, space and value, that a real record has, even once the reports that share its ID
are set aside; its source files real records as often as any; and its values, their precision and
range, and its ID are a real record's. Only the signals the rules name set it apart: a shared ID or a
note naming one, the status, and the notices. Nine more things make that hold:

- *Near twins.* Real records are sometimes as close to each other as a re-report can be to its record:
  an aftershock seconds after its mainshock, a station's next observation, an object's approaches to
  Earth and the Moon, an agency's second lot at a depot within a minute and a half. A digest draws the
  twin of a record it draws for a share of them (`Domain.twin_seconds`, `twin_share`, `twin_key`).
- *Sources.* Every source that re-reports others' records also files records of its own (USGS's `us`,
  `at` and `pt`, the NWS relay feeds, the Minor Planet Center, the invented clearing agencies), and a
  re-report may come from any source, the record's own included.
- *IDs.* A report filed under a record's ID agrees with whatever the ID says. A weather observation's
  ID is an opaque serial (`WX-482137`), so a re-report under it stands where its slot stands, at a
  station of the observation's region. (When the ID began with the station's identifier, the re-report
  had to stand at that station, and so was never the only report there, as a real observation often is;
  `Domain.ids_name_place` declares IDs that name their record's place, and no built-in domain's do.)
  An approach ID names its object and numbers its approaches in time order, and every approach of an
  object prints the object's one H. Beside another approach of the object, a re-report under the ID
  would have to give that H, its record's very H, a copy that no two real reports of a body share, or
  another H, which no real approach gives. So a report is filed under an approach's own ID only where
  its object makes no other approach in the digest or its slots (`Domain.id_place`), and gives the
  object's H as another catalogue fits it.
- *Scopes.* Placing traps where a careless reading changes an answer puts planted reports in more of
  the questions' scopes (the questions all of whose filters they pass) than real records lie in, above
  all in counting questions' scopes: a planted report changes a count whenever it lies in its scope,
  and a largest or smallest value only when it holds it. So each trap draws the scopes of a real record
  of the digest, how many of the counting questions' scopes and how many of the others' it lies in, and
  prefers a placement whose readings hold and whose planted report lies in exactly as many counting
  questions' scopes, and in as many of the others' give or take one, to any placement that changes more
  answers (`TrapConfig.match_scope`).
- *Leads.* A planted report that takes the record a question names, or its value (the largest, the
  smallest, the k-th), from a real one must pass it, but need pass it only by the question's gap: placed
  where it changes answers, it led the report it passed by half as much as a real record that decides a
  question leads the next one (a station's next observation is the likeliest to pass it by a degree).
  So each trap also draws, for each question, how far a real record decides *another* question by (the
  key's own leads, of the same kind of question where the digest asks one, `traps._Leads`), and
  prefers, before its scopes, a placement in which its planted report, and every report planted
  before, leads within a factor of two of the lead drawn for the question it decides
  (`traps.LEAD_FACTOR`); a report that decides no question leads alike, and so does one that decides
  a question for which no other question, decided by another record, lends a lead. The lead lent is
  never the question's own, nor that of a question its decider decides: a planted report that takes the
  record a question names passes the key's decider, whose lead over the next report is then the planted
  report's runner-up's lead. Matched to that very lead (as the third round matched it, wherever the
  digest asked one question of a kind), a planted decider led its runner-up by about what the runner-up
  led the next report by: within a factor of two for 58-72% of planted deciders against 31-36% of real
  ones (an AUC of 0.28 to 0.36 for |log2| of the ratio of the two leads, on every sample, held-out toy
  and CAD replay included), though no reader gained by it.
- *Places.* A question about the record nearest to a place, or how far it lies, was asked about a
  place drawn near a real record, which a planted report could take only by lying nearer still, so
  real reports decided those questions, by tens or hundreds of km, and planted ones almost never. Once
  the traps are planted, such a question is asked about a place drawn near a report it asks about, real
  or planted, that then lies nearest to it, clear of the next, with the next clear of the one after it
  (`questions.reanchor_nearest`, seed path `instance/<i>/anchors`). (A place near a planted report keeps
  the key's nearest record clear of the next record, and that record is most often the report that
  comes next; when only those places kept the clearance, the next report stood clear of the one after it
  more often where a planted report decided, with AUCs of 0.57 to 0.60, and 0.50 in earthquakes after
  the review's change, measured on its code; see *Reviewed* below.)
  Of the reports that can, one is drawn with the weight of its kind (a record's
  own, a re-report, a phantom, a conflict's line): how many reports of that kind the digest's questions
  about a place ask about, over how many of them can decide one, so that each kind decides these
  questions about as often as its share of the reports they ask about. The place gets a label of its
  own, drawn as for a place drawn near a record (a number that names no source), and its key's nearest
  record may lie any distance
  away: near a planted report the key's answer is a record further away, and the bound a place drawn
  near a record keeps (`Spatial.nearest_max_km`) let a planted report that decides such a question lead
  the next by no more than it, where a real one, its own key's answer, leads by any distance. Its answer
  and every trap's readings are computed again.
- *The plan.* A digest's traps are planned before its records (`traps.draw_plan`, seed path
  `instance/<i>/plan`), each kind on its own: none, one or two retractions (`RETRACTION_WEIGHTS`, none
  the likeliest, as the status rule counts every phantom in a question's scope, so that its count is the
  likeliest right count); none in 45% of digests, else one to three re-reports under IDs of their own
  (`CROSS_REF_WEIGHTS`); none in 30%, else one to eight reports under their records' own IDs
  (`SAME_ID_WEIGHTS`), a fifth of them second reports and the rest re-reports; and none to three
  corrections (`CORRECTION_WEIGHTS`). With two or three re-reports, a conflict and a retraction or more
  in every digest (and one more re-report or conflict wherever one changed no answer), planted reports
  were a steady share of a digest, about a sixth, and of any question's scope, and a reader who took
  that share off every count (round(0.84 x the naive count)) closed a tenth of the gap between the naive
  reader and the oracle without reading a status, an ID, a note or a notice, as much on held-out domains
  and genres. Now a digest holds none to thirteen planted reports (5 on average, none in one digest in
  fifteen), and for any share of a digest's records that a counting question's scope holds, the planted
  reports in it are none about as often as any other number (for the naive reader, within 5 points at
  most; exactly for the status rule), so neither a share nor a number taken off a count does much better
  than the count. The digest keeps a record less for each report its plan adds, so that it prints 20 to
  28 reports however many are planted (from a pool of that many records at least: see `live` in the
  domain table for a replayed feed of fewer), and the traps that add a report are planted first (a
  re-report shares its record with another re-report where no record is left free of traps), so that it
  prints every one. Corrections, which change no count, make up the traps that change answers (two a
  digest).
- *Tied values.* A record's values are sometimes tied to its near twins': a station's temperature to its
  observations an hour away, an object's speed and distance past the Moon to its speed and distance
  past Earth (`Domain.twin_fields`). A correction's wrong value of such a field, where the record has a
  twin among the digest's reports or free slots, is that twin's value moved as one of the real twin
  pairs of the digest's records and slots moved over about as long a time (`traps._twin_values`), so it
  agrees with the twin as the record's own value does; without a twin it is a slot value of the group
  from about the record's time. (A slot value of the group, often another station's, jumped a median
  4.5 °C from its station's nearest report, where real ones jump 1 °C.)
- *Notices.* A retraction is issued 92 minutes to 3 hours after its phantom, and a correction as long
  after its record unless the compilation comes first (`traps.NOTICE_DELAY`): later than any genre sends
  a report (a form is sent up to 90 minutes and 59 seconds after its record's time), so that no genre
  sends a planted report sooner because a notice about it follows. A phantom therefore lies at least
  93 minutes before the compilation.

*Readings.* Each trap records its careless readings (`Trap.mishandles`): a re-report counted as a record
of its own, a phantom kept (its retraction ignored), a correction ignored, and a second report's values
taken for its record's or the report counted as a record of its own. A re-report under its record's own
ID reads exactly as a second report does (the same ID, the unsettled status, a slot's values, time and
place; no genre prints a report's role), so it records both of a second report's readings. It recorded
the first alone, and the answers that taking its values changed (in 18.5% of such re-reports in
earthquakes, neo and weather) were missing from its affected questions, from their trap exposure and
from the attribution of wrong answers to traps. "Its values" names one report only while the record has
no other unsettled report, so where two re-reports share a record (only where no record is free of
traps) neither records that reading (`traps._unambiguous`).

A careless reader's joint misreadings (`Trap.fallen`) are enumerated, off the legacy path, per question
and over only the traps that change the rows it reads, which gives exactly what every combination of
every trap gives at a cost that no longer doubles with every trap a digest holds. A trap acts on records
of its own, and every terminal's answer follows from a summary of its rows to which each trap's rows add
on their own (how many there are, how many in each group compared, or the best of them in the terminal's
order), so the combinations are built trap by trap from those summaries, each kept once with every trap
misread in reaching it (`traps._reduced`), rather than read one by one: with two readings for a re-report
under its record's ID, one question of a digest with eight such re-reports and five other traps that
change its rows had 3^8 x 2^5 = 209,952 combinations, which took 16 of the 20 seconds of a 20-instance
demo. Read one by one instead, they give the
same answers (`tests/test_trap_tells.py`).

When no slot is left, as can happen with a live pool little larger than a digest, a builder makes one
beside a real record (`traps._virtual_slot`), with values drawn from its group's and folded into
their range; none of the runs measured below needed one.

The measurements in the paragraphs that follow (*Measured*, *Measured again*, *Measured a third time*,
*Reviewed* and *Measured at the final commit*), and the simulated readers of the Limitations, were made
with red-team and simulation scripts that are not part of the repository, so their figures cannot be
reproduced from it alone. The first four are a history: each reports the figures of its round's code,
before and after that round's changes, and every later change of placement (the last: a re-report under
its record's ID records a second report's readings, see *Readings* above) draws other instances, so
their "after" figures are not the final code's. *Measured at the final commit* gives the final code's.
`tests/test_trap_tells.py` checks the same properties, from the prompt text, on a small sample of its
own: four instances of each built-in domain in each genre, with the toy plugin, where it holds an AUC to
the looser band [0.3, 0.7] that a sample of that size needs.

*Measured* (the first red-team round, on its code). A red team read every report of fresh runs from the
prompt text alone (each built-in domain in each genre, 1,008 instances a sample; the toy plugin domain
in classic and in the toy-telex genre, both held out by the example config, 252 instances; neo replayed
from the recorded CAD response, 252 instances) and tried readers that merge or drop reports by a feature
the rules never mention, without reading a status, an ID, a note or a notice. Before this placement, the
best of them (merge reports within 40 s and 8.5 km, keep the one not from a re-reporting source, or the
commoner source's) closed 39% of the gap between the naive reader and the oracle (36 to 41% in every
domain, 36 to 43% in every genre), and 42% both for the held-out toy plugin and on the CAD replay. After
it, the best closed 0.7% (at most 5.2% in any domain or genre), 0.4% for the toy plugin and 2.4% on the
CAD replay, while the status rule alone still closed 40%. Of reports that share an ID, keeping the first
printed, the earliest, the latest or the commoner source's closed 6% whichever it kept, as a coin would,
and keeping the settling one 17%: only the status says which is real (before, keeping the commoner
source's closed 17%, against 20% for the status). Of the 2,450 cells (feature, kind of planted report,
all reports or one domain) that measure a feature the rules never mention, 535 had had an AUC outside
[0.35, 0.65] with a 95% interval excluding 1/2; after the change none had, on two independent samples.
The cells of single genres held about 150 instances each, and about 3% of them crossed the band by
chance and by the weak effects of the Limitations (before, 20%); what an instance holds is drawn alike
in every genre. `tests/test_trap_tells.py` checks this on its small sample (above).

*Measured again* (the second round, on its code). A second red team found what that placement left, and
the IDs, scopes and retractions above remove it. On two more independent samples of each kind (seeds
61001–61084 and 62001–62084), measured before and after that round's changes on the same seeds:

- A close approach re-reported under its own ID gave its object an H that the object's other approach
  did not print (an AUC of 0 for its sibling's H), and could lie out of the order of the approaches'
  numbers (0.74 to 0.81). No report is filed under an approach's ID beside a sibling since, so neither
  can differ.
- Planted reports decided more questions than real ones: 0.59 to 0.66 for the number of questions whose
  naive answer a report alone changes, and 0.58 to 0.64 for the counting questions' share of the scopes
  it lies in. After the changes both lay between 0.45 and 0.59.
- A reader who took an offset learned per template off the status rule's count closed 6.6% of the gap
  beyond the status rule (5.6 to 8.2% in each domain, 3.6% on the CAD replay). After the changes it
  gained nothing in any domain or on the CAD replay, and an offset off the naive count closed 3.0%
  (before, 6.6%).
- A logistic model of every unintended feature, fitted on one sample and scored on the other, told
  removable reports from real ones with an AUC of 0.68 to 0.74 (0.62 for the toy plugin, 0.66 on the
  CAD replay; 0.64 to 0.69 from the questions' features alone); after the changes 0.57 to 0.62 (0.58,
  0.62; 0.55 to 0.60). Dropping the reports it ranked likeliest planted closed at most 1.9% (before,
  7.2% on the CAD replay).
- A weather re-report under its record's ID stood at the record's station (0.37 to 0.66 for the number
  of reports at its station and the gap to the nearest there; merging a station's observations within
  an hour closed up to 6.5%); after the changes 0.46 to 0.56, and at most 2.5%, with intervals that
  include 0.

After the second round's changes, on its samples, the first round's readers closed at most 1.0% of the
gap overall, 2.4% in any domain, 3.9% in any genre, 0.7% for the toy plugin and 1.0% on the CAD replay
(6.9% in one of its genres, where 10.1% did before), and the status rule alone 54% (before, 41%: a
digest with no retraction had become the likeliest; at the final commit the status rule closes 38.2%,
see *Measured at the final commit*). Of reports that shared an ID, keeping the first printed or the one
whose ID names its source did within 1.2 points of a coin overall (3 to 4 points better in earthquakes:
see Limitations), and keeping the settling one 17 to 21 points better. With the second red team's
features, 17 and 15 of the 3,300 cells of all reports or one domain failed the target on the two
samples before the changes, 12 of them on both, and 6 and 5 of the CAD replay's 605; none failed after
them on any sample, nor any of the toy plugin's. Of the 7,900 cells of single genres 21 and 31 failed
(before, 66 and 62), one of them on both samples on the same side (see Limitations). On those runs the
oracle scored 100% (36,072 questions), every careless reading a trap records was reproduced from the
text (32,793), and every rendering read back to its reports and notices (the 0.2.0 digest at the
precision it prints).

*Measured a third time* (the third round, on its code). A third red team, with question-aware readers
and features of their own, found what the second round left, and *The plan*, *Leads*, *Places*, *Tied
values* and *Notices* above remove most of it. On three independent samples (seeds 3001–3084,
4001–4084 and 6001–6084, each with the toy plugin and the CAD replay), before and after that round's
changes:

- Planted reports were a steady share of every digest, so one factor off every naive count
  (round(0.84 x the naive count)) closed about a tenth of the gap between the naive reader and the
  oracle (up to 12.6% in a domain and 13.8% in a genre; 11 to 13% for the toy plugin, 7.7 to 10.1% on
  the CAD replay). After the changes, fitted on one sample, a count factor closed 2.9% [1.2, 4.9] and
  4.0% [1.8, 5.9] on the others, 0.2 to 1.0% for the toy plugin and 1.7 to 4.0% on the CAD replay;
  every question-aware rule at once (count factors and offsets per domain, rank and top shifts, the
  margin of a comparison, the places' labels and bands) closed 2.1 to 5.1% overall (0.5 to 1.7% for the
  toy plugin, 2.6% on the CAD replay) and at most 6.7% in a domain. In one genre of one sample it closed
  11.8% [6.8, 17.8] (form, fitted on A and scored on C; 4.5 and -0.4% the other ways), the most any such
  reader closed in any cell.
- A corrected temperature jumped from its station's nearest observation (AUCs of 0.84 to 0.90 for the
  jump per hour), and a corrected speed past the Moon from the object's speed past Earth (0.75 to 0.81,
  0.66 to 0.85 on the CAD replay). After the changes, 0.50 to 0.55 for the jump, 0.51 to 0.56 for its
  median and the local hour, and 0.47 to 0.49 (0.42 to 0.55) for the speed.
- A planted report that decided a question led the next by half as much as a real one (0.27 to 0.28
  in weather, 0.32 to 0.35 for earthquake phantoms, 0.64 to 0.73 for phantoms on the CAD replay), and in
  weather decided almost none of the questions about a place (a fifth as many as its share of the
  reports asked about, once a place could lie near any report). After the changes, 0.41 to 0.47 in
  weather (0.39 to 0.43 for re-reports), 0.48 to 0.56 for earthquake phantoms and 0.47 to 0.56 on the
  CAD replay; planted reports decided questions about a place half as often as their share in weather
  and four fifths as often in earthquakes, and the report that decides one lies in the band of a drawn
  place, whether planted or real (see Limitations); no label names a source any more (see Question
  wording).
- A phantom's form was sent sooner than a real one (0.29; 0.36 for a correction). After the changes,
  0.55 [0.51, 0.58] and 0.51.

After the changes, of the cells of all reports or one domain (about 3,830 a sample, 1,350 of the CAD
replay's and 1,076 of the toy plugin's), 1, 1 and 2 failed on the three samples and none on two (a
decider's lead in weather, see Limitations, and a correction's in the invented domain, at 0.653), none
of the CAD replay's, and one of the toy plugin's on one sample (0.653, of 10 reports); of the cells of
single genres (about 8,900 a sample and 4,550 of the CAD replay's), 32 to 37 and 17 to 22 a sample
(0.4%), three of them on two samples on the same side. A model of every unintended feature told
removable reports from real ones with an AUC of 0.58 to 0.61. The oracle scored 100% (45,420
questions, and 9,084 of the toy plugin's), every careless reading a trap records was reproduced from
the text (44,502 and 7,269), and every rendering read back to its reports and notices (11,340 and
2,268).

*Reviewed* (the review after the third round, on its code). A red team on fresh samples (seeds
8201–8284, 9301–9384 and 11401–11484, with the toy plugin and the CAD replay) confirmed the third
round's targets, and found one tell of its own making:
the leads were matched to the key's own lead of the question decided, which is the lead of the
report a planted decider passes, so a planted decider led its runner-up by about what the runner-up
led the next report by (|log2| of the ratio of the two leads: AUCs of 0.28 to 0.36 for phantoms and
cross-referenced re-reports in every domain, on every sample, the held-out toy plugin and the CAD
replay; 0.47 with the matching switched off). *Leads* now lends each question a lead of another
question decided by another record. Measured on 1,344 instances in memory (seeds 107–118, 28 a
domain a seed, read from the prompts) and 840 of the CAD replay (seeds 201–230), removable deciders
against real ones, before and after:

| value questions | lead | ratio of the two leads | within a factor of two (removable, real) |
|---|---|---|---|
| built-in domains | 0.508 → 0.514 | 0.332 → 0.477 | 0.61, 0.31 → 0.35, 0.33 |
| CAD replay | 0.489 → 0.507 | 0.360 → 0.485 | 0.50, 0.25 → 0.32, 0.28 |

Without any matching the ratio was 0.48 but the lead 0.45 (weather 0.41). (Recomputed then from the
same seeds with the helpers of `tests/test_trap_tells.py`, the ratio's AUC and the shares within a
factor of two came out the same, and the decider's lead 0.533 rather than 0.514: the review's script,
which is not published, read the leads its own way.) Of the questions about a place, after the review's
change the next report stood clear of the one after it as often whoever decided (*Places*; the four
seeds of `tests/test_trap_tells.py` on that code: 1.00 of 25 planted deciders and 0.88 of 128 real
ones, against 0.93 and 0.65), and how far the decider led kept an AUC of 0.50 in earthquakes and 0.47
in weather (16 seeds). The final checks then changed placement once more (see *Readings*), so these
counts are not the final code's. What the review left is under Limitations.

*Measured at the final commit* (4179c0c, whose generator is that of the final checks). The scripts and
seeds of a shortcut measurement made at 8e63aee (5001–5084, three instances and three renderings a seed)
measured the final code on four samples: every built-in domain with the genre rotating with the seed
(1,008 instances, G), the same instances in classic (C), `--mix random` (252 instances, R) and the
legacy path (252, L; see Limitations). Shares are of the gap between the naive reader and the oracle,
with 95% intervals over instances; the 27 rule-free readers (13 single rules and 14 combinations, none
reading a status, an ID, a note or a notice) are picked after the fact in each slice, which favours
them:

- Off the legacy path the best of them closes 0.0% [−0.7, +0.8] on G and 0.0% on R and C, where the "best"
  is the precision rule that never fires and every reader that changes a digest loses (the 40 s and
  8.5 km merge −0.9%, −1.2% and −0.8%; dropping the relay sources −38% to −41%). The status rule closes
  38.2% [35.5, 40.8] on G, 37.5% [31.9, 43.2] on R and 38.2% on C, and no shortcut adds to it (with the
  40 s merge, 36.5% on G).
- The largest best reader of a domain is weather's: +1.4% [−1.1, +3.8] on G, +2.0% [+0.0, +5.2] on R and
  +1.3% on C. Of G's 28 domain-by-genre cells (36 instances each) the best is weather/radio, +7.6%
  [−4.7, +17.7], and no cell's interval lies above 0.
- AUC cells of features the rules never mention (planted against real reports, 1,000 cluster-bootstrap
  resamples over instances) are counted as failing by the interval rule (the 95% interval excludes 1/2
  and reaches outside [0.35, 0.65]) / by the point rule (the estimate lies outside [0.35, 0.65]) / out
  of all cells. Over all reports: 2 / 0 / 1,314 on G and on C, 19 / 2 / 1,290 on R. Per domain: 13 / 4 /
  2,515 on G, 14 / 5 / 2,515 on C, 44 / 3 / 2,468 on R. Per genre: 185 / 22 / 8,919 on G, 261 / 93 /
  6,771 on R. Per domain-by-genre cell of G: 589 / 77 / 17,115. The interval rule fails 2.1% of G's
  genre cells and 3.4% of its domain-by-genre cells, fewer than the 5% that intervals excluding 1/2
  reach by chance. G's four point-rule failures by domain are the decider's lead (how far a report
  that decides a question leads the runner-up) of weather's re-reports (0.339 [0.277, 0.400], 72
  planted; 0.333 for those under the record's ID, 49) and of earthquake phantoms (0.348 [0.252, 0.451],
  15), and the |z| of an earthquake second report's value among its group's reports within two hours
  (0.694 [0.559, 0.810], 19); C, the same instances in classic, repeats them. R's two point-rule
  failures over all reports (a correction's |z| in its group, 0.691, 22 planted; a cross-referenced
  re-report's change of wind at its station, 0.318, 31) are 0.560 and 0.500 on G, with four times the
  instances. On the legacy path, 182 of 600 cells fail the interval rule and 147 the point rule.
- The oracle answered every question right in every sample.

Off the legacy path, placement is also *guarded*. A careless reading counts only where the misread
digest passes the key's own guards: margins that hold with the planted values, rank and nearest
gaps, time margins. A placement whose every reading passes is preferred. A careless reader's wrong
answer is then the one the printed text gives, not one decided by a last digit or a tie.

**Rules text.** `record_rules` writes the reading rules of any domain in its own words. They are
genre-neutral and say:

- which report settles a record, and that an unsettled report's time, position (for records that have
  one) and values are superseded, not its values alone: a re-report stands where its slot stands, at a
  time and a place of its own, which can lie inside a question's window when its record's does not;
- what a notice is (a message about a report, not a report), and that it applies wherever in the
  digest it is printed, before or after that report: most genres print some notices first;
- how re-reports, corrections and retractions read, quoting every phrase of the domain's trap
  vocabulary that a digest prints: the notes of re-reports, each followed by the ID it names, and the
  phrases of corrections and retractions (a re-report carries a note naming the ID it *duplicates*:
  the rules keep "repeat" for quotes and read-backs);
- that values are converted as the glossary says, and, for records with a location, that distances
  are great-circle distances given in km or in miles (1 mile = 1.609344 km);
- that times are given in UTC or with their offset, what a report's time is (when its record was
  recorded; for a close approach, the moment the object passed closest: `Domain.report_time`), and
  that the time a message was sent, forwarded or relayed is never a record's time, while a WIRE line
  is a report whose bracketed time is its own;
- that a quoted or read-back repeat of a report or notice is not a new one, while a forwarded or
  relayed one counts, once, as the report or notice it carries, and rule 1 still says which record
  that report is;
- that group names are matched as printed;
- that lines which are neither reports nor notices change nothing.

Every article fits the word it stands before in the words of every built-in domain and of every
surface the invented domain draws ("an outpost"), in the rules and in the glossary's line on
statuses. A plugin's words get theirs from `domains.base.indefinite`, which goes by the first letter,
with a short list of exceptions ("an hour", "a unit"), and can miss others (it gives "a" before
"MRI"). No duplicate note borrows a word that the rules give a meaning of their own: weather's notes
are "same observation as", "duplicate of" and "mirror of", since a re-report noted "relay of" an ID
would read as the relayed report that rule 7 counts on its own, and the invented domain no longer
draws "repeat filing of", which named what rule 7 calls a repeat (`tests/test_reading_rules.py`
checks the notes of every domain, of the toy plugin and of 200 drawn surfaces against those words).

The glossary's line on statuses says the same as rule 2, in every domain ('a "verified" report
settles a record's time, position and values', or 'time and values' for records without a position),
so it says it in earthquake prompts too, whose rules keep the 0.2.0 rule 2 ("use the values from its
REVIEWED report").

Weather, neo and invented add rules of their own. Earthquake prompts off the legacy path print the 0.2.0
rules 1–8 word for word (whose rule 1 names a re-report's note by what it does, "a note naming the event
ID it duplicates", and quotes none of the three: "same event as", "same earthquake as", "Dup of"),
followed by three rules: rule 9, which now opens with what a notice is, in the words of every domain's
rules, and also names the section titles, greetings and signatures that the other genres print among the
lines without data, and the rules on message times (with the WIRE line's time) and repeats (rules 10 and
11). They are headed `Reading rules:`, as every domain's are, since the 0.2.0 heading ("identical for
every digest") is not true of a run whose domains print different rules. Legacy prompts print the 0.2.0
rules alone, heading included.

**Question wording.** Off the legacy path, a radius is worded "within 40 km of …" or "at a distance
of no more than 40 km from …", in every domain with a location. The 0.2.0 wording "no more than 40 km
from …", joined to a depth threshold, reads as a range of depths ("with a depth of 16 km or more and
no more than 40 km from station US-28"). Earthquake questions off the legacy path also never say that
an earthquake was "recorded" at a time ("that occurred from … up to …", "When … did the strongest …
occur?"), since the rules say when it *occurred* and a reader could take "recorded" for the time a
message was sent. They keep every other 0.2.0 wording and draw exactly as before
(`Domain.prompt_lexicon`); legacy prompts keep the 0.2.0 wording (see Limitations). A question about
the time of a record's largest value names its records first ("Among the …, when (UTC, HH:MM) was …
recorded?"), so that the verb never trails a long filter, and every prompt off the legacy path says
that an HH:MM answer drops the seconds rather than rounding them, as the key does. Off the legacy
path, too:

- a quantity of exactly 1 takes its unit's singular ("colder than 1 degree Celsius", "1 mile";
  `Unit.one`, which an invented unit takes from its coined name), where 0.2.0 wrote "1 degrees". A
  radio transcript, which may speak a report's units by their words, speaks a value of exactly 1 with
  the unit's symbol ("wind speed 1 kt", not "1 knots"), as the glossary it is read through names no
  singular;
- the two groups of a comparison are quoted ('which region has more of them: "Honshu, Japan" or
  "Southern Alaska"?'), so that a name with a comma reads as one choice, as an answer must give it;
- a question's point is named by its number alone, its position right after it ("station 34 at
  (17.210, -66.510)", "COOP site 51 (40.466°N, 80.093°W)"). The point lies 5 to 60 km from a record
  drawn at random, which is almost always the nearest record, so a label that named that record's
  source (0.2.0's "station PR-34", weather's "COOP site 51 near KPIT") named the answer's: in 39 of 43
  nearest questions of three 20-instance weather runs, where 12% of the candidate observations came
  from that station, and in 38 of 38 earthquake questions, against 39%. `Domain.validate` refuses a
  point label that names a source, however it is written (`{source!s}`, `{source.upper}`), or that has
  no `{number}`, which keeps an instance's points apart.

**One answer, from one record, however a tie is rounded.** Off the legacy path a question whose answer
is a value of one record (the largest or smallest value of a field, the time of the largest, the
distance of the nearest record) is redrawn when the runner-up record's answer, computed as the key is,
would be scored right: a reader who takes the wrong record must not be counted right. The nearest gap
already keeps a nearest distance clear of its tolerance; a max, a min or a time had no such guard, and
12 of 333 such questions in twelve runs of 15 instances (three per domain) accepted the runner-up's
answer (a shallowest depth of 26.4 km beside one of 26.8 km, within the 2% depth tolerance; the closest
approaches of the fastest two objects a minute apart). The time of the largest value checks every other
record's time, not the runner-up's alone: a time does not follow its value, so a smaller record half a
minute from the largest one answered its minute too. A question about the largest or smallest value,
asked in a unit, is also redrawn when that value, converted exactly into the unit (a printed value
stands for the stored one), lies exactly halfway between two answers of the asked places; when the
number that genres print for it does (82.8 km is printed 51.45 mi, which a reader asked for one decimal
place may round to 51.4 or 51.5, though it stands for 51.4495…); or when the key, computed in floating
point, is not the exact value rounded (`questions.strict_answer_ok`). In the invented domain, a stored
68.3 in a unit of 4 is exactly 17.075: binary floating point rounds it to 17.07, and half up or half
even to 17.08. The legacy path keeps the 0.2.0 guards (see Limitations).

**The glossary.** Every prompt off the legacy path prints its domain's glossary between the reading
rules and the digest, in `<glossary>` tags. It gives:

- what the records, IDs, sources and groups are called, and the ID pattern;
- the two statuses (the first settles a record's time, its position if it has one, and its values), and
  the precision of positions;
- each field's key, label, stored unit and decimals, with every unit and its exact conversion, and the
  qualifier given with each value, named in words too where digests print it without its key (an
  earthquake's "mag_type" ("magnitude type"), which the 0.2.0 lines print as "(md)", "MD 3.2" or
  "type=md").

Words are quoted as JSON strings. `render_glossary` refuses a glossary that would not read back
exactly, and `parse_glossary` reads it from the prompt text. A glossary never names the domain's
registry name, which a held-out plugin keeps private. The ID pattern describes IDs lower-cased, and
digests print them as their sources file them (`WX-482137`, `KX/40512`), so the glossary says that
the pattern ignores letter case. One instance of the invented domain printed (seed 7, mix
`invented:classic`, `inst-000`):

```
<glossary>
Glossary: what the records in the digest are, and how their values are written.
- Records: each is one "crate" (plural "crates"), identified by its "tally number"; IDs match the pattern [a-z]{2,3}[-/.]\d{5}, ignoring letter case.
- Sources: each report is filed by one "carrier".
- Groups: each record belongs to one "warehouse".
- Statuses: a "verified" report settles a record's time and values; a "provisional" report does not.
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
  corrections in the generic form of every other domain, all through one map, `to_generic` and
  `from_generic` (`genres.base.generic_items` and `domain_items`): a line that is no earthquake report
  (no magnitude type, say) is no report in any genre, and the reading rules refuse an earthquake
  correction of another field than the magnitude rather than ignore it. Before, five genres kept
  copies of the map of their own, which read those inputs in five ways. Off the legacy path its
  synthetic world (`earthquakes.synthetic_world`) is the 0.2.0 world with aftershocks (15% of events,
  seconds to hours after a larger event of their network, a few km away, never within 3 s and 15 km
  of another event: no two are one quake's solutions), and with a share of the events filed by the
  networks that re-report others' events, as the USGS feed shows them doing: the tsunami warning
  centres (at, pt) in Alaska, Hawaii and Tonga, the NEIC (us) for larger events in the US regions.
- **weather**: temperature, a signed field, plays the primary role (thresholds, the warmest
  observation, its rank and its time) and wind speed the secondary one. Pressure appears in the
  twins' filters.
  - Every conversion is exact as defined (1 kt = 1.852 km/h, 0 °C = 32 °F = 273.15 K,
    1 inHg = 33.86389 hPa). Every value is stored to a tenth, and answers in K and inHg keep two
    places.
  - IDs are `WX-` and six digits (`WX-482137`), an opaque serial that names neither the station nor
    the feed nor the time: a re-report filed under an observation's ID stands where its slot stands, at
    a station of the observation's region, at a time of its own, which an ID that named them would
    contradict. (When IDs began with the station's identifier, a re-report under one had to stand at
    its observation's station, so it was never the only report there, as a real observation often is.)
    A live observation's serial is a hash of its station and time, so a replay gives the same IDs. The
    statuses are "verified" and "preliminary" (the NWS quality-control codes V and Z).
  - Synthetic observations follow each station's climate normals, a daily cycle and a weather
    system per region. They are timed as NWS stations take them, each at its station's own minute
    of the hour or, a quarter of them, on a five-minute mark (a five-minute observation, in whole
    degrees), and report winds in whole knots and pressures in hundredths of an inch of mercury,
    stored in km/h and hPa. The relay feeds (MADIS, NOAAPORT, AWOSNET) file a fifth of them.
  - An observation's local time is its station's own (`Weather.utc_offset`): the standard offset of
    the station's zone, an hour more while US daylight saving time is in force (from 2:00 on the
    second Sunday of March to 2:00 on the first Sunday of November; Phoenix keeps none), computed
    from the calendar rather than the machine's time zone database. The whole hour nearest the solar
    time of a longitude, which other located records keep, put three stations an hour off in winter
    (Detroit at UTC-06:00), and 23 of the 24 an hour or two off in summer.
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
    catalogues. A body is no place a record is in, so the prose of every genre records an approach
    *near* its body ("recorded near Moon", subheads "Near Venus"), where it records an observation
    *in* its region (`Domain.group_preposition`); it read "recorded … in Moon" before.
  - Synthetic snapshots cover 30 days, so time windows lie on 3- or 6-hour marks, and a time
    question asks for the UTC time of day.
  - An approach ID is the object's designation without spaces, then `-ca` and the number of its
    approach. `2026vk2-ca1` and `2026vk2-ca2` are one object passing Earth and then the Moon: two
    records, with one H, the object's. A report is filed under an approach's ID only where its object
    makes no other approach in the digest, which it would have to agree with in H and in order, and it
    gives the object's H as another catalogue fits it (see Traps).
  - A synthetic object is found around its close approach (up to three weeks before it, or days
    after), never after the compilation, and its provisional designation is of the half-month it was
    found in, so a December approach in a digest compiled on 1 January carries the old year's
    designation. A recently found object's H has three decimals, an older one's one or two, and a
    numbered object's number lies between 3000 and 649,999. A Moon approach's distance is drawn
    within its range (folded back at its least, 0.0003 au, never clamped there), so no distance is
    shared by more approaches than chance gives. All four catalogues, the Minor Planet Center's
    included, list approaches of their own.
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
    surfaces than they did before this check (`tests/fixtures/invented/pinned.json` was pinned again).
  - Two clearing agencies file lots at every depot (15% of lots), and they and one depot agency are
    the agencies that re-report; 12% of filings are followed by a second lot of the same agency at the
    same depot 5 to 90 s later. The world and the surfaces were pinned again for this.
  - Values come from fixed log-normal distributions per canonical depot, so margins and steps keep
    their meaning under every surface. Tolerances are relative, so they hold in every drawn unit,
    and the group type strips every group label a surface can draw, so a group answer may end in
    the instance's own ("Nerm Halt berth"), as an earthquake answer may end in "region".
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
| `email` | a thread of messages with From, To, Date and Subject headers; reports as bullets, sentences or pasted blocks | a forwarded message, every one | greetings, sign-offs, signatures, remarks; replies that quote the message before them with ">": a reply that only acknowledges always does, from someone other than its sender and under its subject ("Re: Fwd: CORRECTION: ..." after a forward), and a message of reports, titled after the thread, never quotes a forward |
| `chat` | a channel log of `[HH:MM] user: message` lines; reports in four voices | messages, pinned messages, or the pinned board above the log; the casual ones open with their keyword and time ("heads up: RETRACTION issued …: …") | joins, greetings, acknowledgements, off-topic remarks; quotes and read-backs of earlier messages |
| `table` | CSV or markdown, one row per report, columns named in the glossary's words | numbered footnotes | a caption, comment lines |
| `form` | one field or incident report form per report, in four frames (boxed, ruled, numbered, dotted) | amendment forms, which cite the number of the form they amend as its "Original form no." in every frame (in the boxed frame, beside "Amendment issued", the amendment's own time) | form codes and serial numbers (in filing order, so an amendment's number is always above the original's, wherever the batch prints it), sent and relayed times, preparers, blank checker and signature fields (a blank field that shares a boxed row with another shows a line, `________`, so the row cannot be misread) |
| `radio` | the transcript of a reporting net: stations call in and read out their reports; the session line says which window the reports are about (every transmission is sent after it) | broadcasts by net control to all stations, in words that name no station ("Net broadcast"), whatever the control's call sign | call signs, procedure words, radio checks, acknowledgements; Control's read-backs |

- **The rules settle every line.** A notice names itself with words every domain's rules name:
  RETRACTION, CORRECTION, "revised", or a phrase of the domain's trap vocabulary, which its rules
  quote, and every correction gives its new value after such a phrase (a classic Revised line gave it
  after "now", which only the rules' clause on a value that "was revised" covered). Every notice is
  worded as the notice itself, never as a pointer to another notice, and none
  is headed as a kind of line that the rules say carries no data (an editor's note, a reminder). A
  test per genre checks every wording that genre can print.
  - Quotes (email, chat) repeat an earlier report or notice word for word, and read-backs (chat,
    radio) repeat a report's ID and values; the rules say that a repeat is not a new report or
    notice. Forwarded and relayed messages count, once: every email notice is forwarded, and chat
    users and bots relay what each source files. In every genre, a reader who strikes out every
    repeat still reads each notice exactly once.
  - A message's own time (an email's Date header, a chat time stamp, a transmission time, a form's sent
    time) is never a record's time, and every report states its record's time in full, to the second. A
    chatter line that tells of a window ("Sensor KPHL.PDF was offline for maintenance from 00:00 UTC on
    4 March 2026 until 00:45 UTC") is posted after the window, in a chat log, and is a remark of a
    message sent after it, in an email thread (of a reply that only acknowledges, sent last, when no
    other is); chat posted it anywhere in the log, before the window it told of in 22 of 36 such lines
    of 120 weather renderings. And its window never holds a report of the station's source, which the
    line would set a reader to doubt, a distractor that no trap records. A classic wire line prints it
    in brackets after its source ("KPHL WIRE [05:55:26 (UTC-05:00) 18 Feb 2026] ..."), where a chat log
    prints a message's stamp, so the rules say that a WIRE line is a report and its bracketed time the
    report's. A classic prose line of a domain other than earthquakes says that its record was recorded
    at its time, and names its source last ("06:12:19 UTC on 12 April 2026 — reading by-12345 (reviewed)
    was recorded in North Sea, at (56.512, 3.104): … Listed by ukmo."), where its earlier form ("… —
    ukmo lists reading by-12345 …") read as the time of the listing.
  - A form's sent time is drawn alike for every form of a batch, up to one shared limit below the
    shortest time from any record to a notice about it, so that no form is sent after a notice about
    its record, and no form's delay tells whether a notice names its record. Capping only the forms
    that notices name sent a planted report's form sooner than a real one of its batch in 70-75% of
    pairs.
  - A notice applies wherever it is printed: the pinned board of a chat log heads the log, and
    most genres print some notices before the report they name, so the rules define a notice by
    what it says, never as a message about an "earlier" report.
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
- **The oracle stays text-only.** It reads the glossary from the prompt (`parse_glossary`) and parses
  the digest with its genre's parser. Besides the text it gets each question's chain, as in 0.2.0,
  and, in `run`, the names of the instance's domain, which picks the reading rules in code and the
  record types, and of the rendering's genre, which picks the parser; the naive reader and the mock,
  which starts from the oracle's answers, get the same. The generator's self-check is given no genre:
  it picks the parser by sniffing the digest (`detect_genre`: the first genre other than classic that
  recognises it, else classic, so a legacy digest is read as 0.2.0 read it). Sniffing is exclusive:
  each genre's digests are claimed by that genre alone, in every domain, and the self-check confirms
  that every rendering is recognised as the genre it was written in, so the genre a reader is given
  and the one the text names are the same. It sniffs among the run's own genres (the
  built-in ones, then those of its config's plugins, and any other genre its mix names), so a genre
  that the process loaded for an earlier run, such as another config's held-out plugin, never changes
  a later run's self-check or appears in it.
- **Every domain in every genre.** `tests/test_matrix.py` runs all 28 built-in cells (domain ×
  genre). In each cell, every rendering reads back exactly from its prompt, the oracle scores 100%,
  no question repeats, the cell plants re-reports and conflicts and each instance the traps its plan
  draws (none or more of each kind: see *The plan* under Traps), every recorded careless reading is
  reproduced from the text of every rendering, and the naive reader errs only where a trap explains
  it.

### Mixes (`--mix`, `evalgen/mix.py`)

| mix | instances come from |
|---|---|
| `legacy` (the default) | earthquakes in the classic genre, on the 0.2.0 seed paths: the 0.2.0 run. It takes no private config |
| `random` | every domain, in every genre. Without a private config, that is every built-in domain and genre, all public, and never a plugin's, even one that the process has loaded; with one, exactly the domains and genres it lists, with their splits |
| a list, e.g. `weather,neo:email,invented:table` | the entries named: `domain:genre`, or a bare domain, in any built-in genre (with a config, any genre it lists). With a config, the list may name only what the config lists. Without one, it may also name an entry that a caller of the Python API registered itself, but not one that a private run config loaded into the process |

A mix that leaves a choice draws each instance from `seedlog.rng("instance", i, "mix")`, a seed path
that no legacy run creates. It draws the instance's domain (every domain of the mix alike), then the
genre of its original rendering (every genre allowed for that domain alike), then the genre of each
variant: one other than the original's whenever the mix allows one, every other genre in turn
before any of them twice. A mix of one domain and one genre draws nothing.

Before anything is drawn, what a mix allows is put in a canonical order: the built-in domains and
genres in module order, then those of the config's plugins in the config's plugin order (and any
other entry that a list names, by name). So `weather,neo` and `neo,weather` are one mix and draw one
run, and a list that allows a domain in a genre twice (`weather,weather:email`: a bare domain is in
every genre) is refused. Nothing that a process loaded for an earlier run changes what a later run
may draw, or in which order, not even a config that was refused halfway through loading its plugins:
each entry a config's plugin defines is marked as the config's as soon as it is registered. (Before
this rule, a list drew in the order it named its entries, so a run of a list written out of that
order by an earlier version draws differently now.)

Each domain keeps its own snapshot (`snapshot/<domain>/` in a run of several), its own twin
templates (seed path `templates/new/<domain>`) and its own balance of the template pools, so the
public and new pools stay matched within every domain. No concrete question repeats in the run.
Instances record their domain, its split and any surface words, and renderings record their genre.
The manifest records the mix in its canonical form, the inventory, the config's commitment and the
plugins, and, for a run of several domains, one snapshot per domain; legacy runs record none of these
(nor does `--mix " legacy "`, which is the legacy run). A run off the
legacy path is named after its mix, as in `runs/random-synthetic-seed7`. The default folder, and so
the run ID, keep the mix as it was typed: `--mix weather,neo` and `--mix neo,weather` draw one run,
recorded as `neo,weather`, but in the folders `weather_neo-synthetic-seed7` and
`neo_weather-synthetic-seed7`.

### Command-line options

| option | commands | what it does |
|---|---|---|
| `--mix MIX` | generate, demo | which domains and genres instances come from (above) |
| `--private-config PATH` | generate, demo | the private run config: the held-out domains and genres and the plugins that define them, checked whole and committed in the manifest. A legacy run refuses one |
| `--private-config PATH` | run, score, report, resolve | loads a run's plugins from its config, after checking the config against the run's commitment. A run with plugins needs it in every new process |
| `--source live` | generate, demo | off the legacy path, each domain's own live source: USGS, NWS or JPL, and synthetic data, with a note, for a domain without one. `--source usgs` still names the USGS feed (off the legacy path, only for a mix of earthquakes alone), and `--feed hour` is for legacy runs only |
| `--snapshot-file PATH` | generate | replays one frozen live feed: the USGS GeoJSON of a legacy run, as in 0.2.0, or the frozen feed of the one domain of a mix, whatever number of records it gives (the live source's minimum applies to a fetch only) |
| `--snapshot-dir DIR` | generate | replays an earlier run's snapshot folder, off the legacy path. A domain that run read live is rebuilt from its frozen bytes, whatever the seed (and keeps the time they were fetched); a domain it drew synthetic records for, by choice or after a fallback, is drawn again from `--seed`, so it is that run's only with that run's seed (the manifest's `replay.same_seed` says which, and `generate` warns when the seeds differ). A live run of several domains thus regenerates byte for byte, offline. A replay reads its snapshots whatever `--source` says, and its default folder is named for a replay (`runs/random-replay-seed7`) |
| `mock:held_out=P` | run `--models` | the simulated model with a penalty for held-out sets: `held_out=P` (domains and genres), `held_out_domains=P` or `held_out_genres=P`, joined with `+` for two penalties (see the model table above). Without a penalty it draws exactly as before |
| `verify-config RUN CONFIG` | its own command | checks a revealed config against a run's commitment: `ok` (exit status 0), `MISMATCH` (1, also for a file that is not JSON), or `error:` (2: no commitment to check, a config that gives a key twice, or arguments that are not a run and a config) |

Frozen bytes are parsed and checked before the run directory is created, on the legacy path too: a
`--snapshot-file`, and every frozen feed under `--snapshot-dir`. A domain without a live source,
another source's bytes (even bytes that parse to no records) or a missing file stops `generate` with
`error: ...` and exit status 2, as does a private config that is malformed or names a plugin that its
registry refuses.

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
  variant. The template pool is not a stratum. The pools are balanced within each domain over a run
  (every instance asks as many public questions as new ones, give or take one, or rarely more where
  no template of one pool fits its records, see Templates; and within each domain a twin pair never
  drifts more than one question apart), but not within each difficulty stratum, where a domain's
  share of public questions differs by chance and by which templates fit its records. For a model
  with a large template gap, a run's adjusted source gap therefore moves by a few points either way,
  with a bias over designs that is small but not zero. With the mock's 30-point template gap and no
  held-out effect, the expected adjusted source gap of the 120 designs of the final calibration (100
  instances, one held-out domain of four; see the validation below) averaged −0.30 points [−0.59,
  −0.03] (SD 1.6, from −3.9 to +3.3), against intervals of about ±14; over the first 40 instances of
  40 of those designs it ranged from about −7 to +10 (SD 4.4), against intervals of about ±29. The genre
  gap's bias is smaller still (SD 0.4 at 100 instances). A pool stratum would cost more coverage than
  it removes.
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
  measure both cases. The template gap keeps its 0.2.0 interval: almost every instance asks both
  pools.
- **The flag** is the template gap's: an adjusted gap above 10 points whose 95% interval excludes
  0, and, for the source and genre gaps, compared rows from two instances or more on each side (see
  Limitations: one instance leaves its side's interval unwidened). A gap needs rows on both sides, so
  it exists only when the private config holds something out.

The run section adds the mix, the inventory (each domain with its instances, each genre with its
renderings), each domain's snapshot and the config's commitment. The console summary shows each
model's adjusted source and genre gaps and its cross-genre consistency. Besides:

- **Held-out sets that are public code.** `report.json` lists the held-out domains and genres that
  are built into evalgen (`run.held_out_built_in`), and those that plugin files of evalgen's own
  `tests/` folder define (`run.held_out_published`: the test-only toy domain and genre, which the
  example config holds out), and the held-out section then warns that anyone can practise on them,
  whatever nonce the config draws. It warns, too, when the run's config keeps the nonce of the
  published example (the manifest's `private_config_example`), whose held-out sets are public test
  code. So a demo is not taken for a test of anything held out.
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

**Calibration at the final commit.** The metrics were calibrated with the simulated model on
synthetic data at the final commit of 0.3.0 (4179c0c): 120 random-mix seeds of 100 instances, three
renderings each, with the built-in weather domain held out (one domain of four) and the email and
radio genres (two genres of seven), every run answered by the oracle, the naive reader, the mock (its
built-in 30-point template contamination and no held-out effect) and four penalised mocks, which take
0.15 or 0.30 off the mock's chance of a right answer on instances of the held-out domain or on
renderings in the held-out genres; and 15 seeds of a confounded mix. The held-out sets are built in
and public, so this measures the metrics, not a model. Flagged runs out of 120 (95% Wilson
intervals):

| model | template gap | source gap | genre gap |
|---|---|---|---|
| oracle | 0 | 0 | 0 |
| naive reader | 0 | 1 (0.8% [0.1, 4.6]) | 0 |
| mock | 120 (its contamination) | 1 (0.8% [0.1, 4.6]) | 1 (0.8% [0.1, 4.6]) |
| mock, 0.15 off on the held-out domain | 120 | **65 (54.2% [45.3, 62.8])** | 1 |
| mock, 0.30 off on the held-out domain | 120 | **118 (98.3% [94.1, 99.5])** | 0 |
| mock, 0.15 off in the held-out genres | 120 | 2 | **108 (90.0% [83.3, 94.2])** |
| mock, 0.30 off in the held-out genres | 120 | 4 | **120 (100% [96.9, 100])** |

- **False flags.** The models with no held-out effect raised 3 flags in 720 checks of the source and
  genre gaps (0.4% [0.0, 1.0]). The genre gap's one lay 3.6 standard errors from the gap that the
  run's design leads one to expect; with the two of the same seeds' calibration at 8e63aee, that makes 3
  null genre flags in 240 runs where a normal interval of its width predicts 0.3. Its coverage is
  nominal (below); its tail beyond three standard errors is a little heavy.
- **Cross-flags.** A penalty on one kind flagged the other kind's gap in 7 of 480 runs, 2 of them on
  seeds where the unpenalised mock flagged that gap too.
- **Recovery and coverage.** The adjusted gaps recovered 99–101% of the injected penalties. The 95%
  intervals of the targeted gaps contained the injected gap as realised in 95.0% [92.1, 97.5] of the
  480 penalised runs, and the gap's expectation over the mock's draws, computed exactly from each
  design with the report's own code, in 94.8% [92.3, 96.9]; over all three gaps of the five mocks, in
  95.9% [94.4, 97.2] of 1,800. The exact spread of a targeted gap over the mock's draws is 1.01 to 1.03
  times the spread that its interval implies (at most 1.10 in a design): the intervals of a penalised
  gap run a little narrow, those of the unpenalised mock's source gap a little wide (0.91).
- **Width and comparability.** Mean half-widths over the five mocks: template gap ±8.4 points, source
  gap ±14.0, genre gap ±6.5. The source gap compares 55% of rows like for like (37–72% in a run), the
  genre gap 88% and the template gap 100%. The oracle's source interval, ±12.1 at 100% accuracy, is the
  floor that the smoothing of small strata sets: at 100 instances with one held-out domain, the source
  gap cannot resolve a difference much under 12 points, and a 15-point one is flagged in about half the
  runs. At 40 instances the source interval is about ±29 points.
- **Design bias.** The unpenalised mock's expected adjusted source gap averages −0.30 points over the
  120 designs (SD 1.6; see the strata above).
- **The spread condition** (compared rows from two instances or more on each side) decided no flag.
- **The confounded mix** (`neo:email,weather:radio,earthquakes:classic,invented:news`, with neo,
  weather, email and radio held out; 15 seeds, the mock with and without a 0.30 penalty on held-out
  domains): both adjusted held-out gaps were n/a in 30 of 30 runs and never flagged, while the domain
  penalty raised both raw gaps (source +30, genre +29 points on average).

**Earlier measurements.** What follows was measured on earlier code (the merged threat, registry,
rules and stats fixes), but for the demo's row, measured again at the final commit. It records how the
strata and the intervals were chosen, and its figures are those of that code. Validation with the
simulated model, on synthetic data, one run per row. Each run was answered by the oracle, the naive
reader, the mock, and the mock with a penalty of 0.2 or 0.3 on held-out domains or on held-out genres.
"CI" is the half-width of the adjusted gap's 95% interval, in points:

| mix; held-out domains; held-out genres | seed | instances | source gap: CI, coverage | genre gap: CI, coverage | the penalties |
|---|---|---|---|---|---|
| random; neo, invented; table, form, radio | 42 | 100 | ±12, 64% | ±6, 92% | each flagged by its own gap only; the mock without a penalty was flagged too, by the source gap (+12 ± 12) |
| the same | 43 | 300 | ±6–7, 84% | ±3, 98% | each flagged by its own gap only |
| random; invented; radio | 44 | 300 | ±7, 82% | ±5, 88% | each flagged by its own gap only |
| random; toy; toy-telex (the example config) | 45 | 300 | ±6–7, 89% | ±3, 99% | each flagged by its own gap only |
| the same (the demo, 0.3 penalties only; measured again at the final commit) | 42 | 20 | ±47–48, 29% | ±17–19, 50% | the genre penalty flagged (+26 ± 17); the domain penalty missed (+20 ± 48). With the code just before, the reverse (the domain penalty flagged at +67 ± 44, the genre penalty missed at +14 ± 24): at 20 instances either is chance |
| `earthquakes,toy:toy-telex`; toy; toy-telex | 46 | 100 | ±14–16, 60% | ±8–9, 49% | each flagged by its own gap only |
| `earthquakes:classic,toy:toy-telex`; toy; toy-telex | 47 | 60 | n/a | n/a | neither: the raw gaps move together |

(The half-widths are those of the penalised mocks, measured with the code of the merged threat,
registry, rules and stats fixes, but for the demo's row, measured at the final commit. A reworded prompt
reseeds the mock's errors, so each row is a new draw of the same experiment: rewording the reading rules
moved the first row's source intervals from ±13–15 to ±12, caught the domain penalty of 0.2 that the
earlier draw missed (+12 ± 13), and flagged the mock without a penalty (below). Before each gap was
compared within the other kind's names and its intervals counted instances, the first row read ±10, 87%
and ±6, 97%, and every penalty was flagged by its own gap. The random mixes of several domains and
genres lost coverage to the finer strata. In the mixes of the example config, whose names and splits
coincide, only the design effect changed anything: the source intervals of the mocks penalised on genres
widened by up to a seventh, since a genre penalty costs all of an original's questions at once, so their
rows go together. The later rewordings of the rules, the chatter and the questions, by the fixes of
misleading text and of what blind readers found, make a run of today another such draw.) The oracle and
the naive reader were flagged by neither gap in any of these runs, and the mock without a penalty in all
but the first, where its source gap read +12 ± 12: a chance excess of about two standard errors, which
the simulated readers below showed in 2–3% of 100-instance runs per gap, and the final calibration
above in 1 of 120 per gap. In the partly tied mix, a penalty
on held-out domains also raised the raw genre gap (by 12 to 18 points), and a penalty on held-out genres
the raw source gap (by 10 to 14); the adjusted gaps kept each penalty on its own gap.

Simulated readers on the same kind of design (random mixes of 40 or 100 instances, neo and invented
held out, and table, form and radio), 600 to 2,000 runs per cell, measured on the code of that time
the intervals and the flag, after and (in brackets) before those two changes. A reader with no gap at
all (a chance of 0.77
of a right answer, or of 0.8 with a random effect) should have its gap of 0 covered in 95% of runs
and be flagged in at most 5%:

| reader with no gap | 40 instances: source gap covered, flagged | genre gap | 100 instances: source gap | genre gap |
|---|---|---|---|---|
| answers independent | 96%, 2% (95%, 3%) | 96%, 2% (94%, 1%) | 95%, 2% (95%, 2%) | 94%, 0% (94%, 0%) |
| a random effect per rendering (SD 2, logit scale) | 93%, 4% (83%, 9%) | 93%, 3% (81%, 7%) | 95%, 3% (85%, 8%) | 93%, 1% (80%, 2%) |
| a random effect per instance (SD 2) | 93%, 3% (84%, 8%) | 94%, 3% (89%, 4%) | 95%, 2% (82%, 9%) | 96%, 0% (88%, 0%) |
| right 65% of the time on neo, 85% elsewhere, in `weather,neo:email,invented:table` with only email held out | | 98%, 2% (4%, 94%) | | |
| right 65% of the time in table, 85% in classic, in `earthquakes:classic,invented:classic,weather:table` with only weather held out | n/a in every run (29%, 71%) | | | |

The template gap needs neither change: almost every instance asks both pools, so an instance's own
difficulty cancels in it (in that simulation, with a random effect per rendering or per instance of SD
2, its interval covered the gap of 0 in 97-98% of 1,000 runs, and flagged 1-2% at 40 instances; in the
final calibration it covered the mocks' expected template gap in 96.7% of 600 runs).

### The 0.2.0 regression guarantee

- `cli.py generate --seed 16222556419173536875 --instances 100`, with no new options, writes the
  same 405 files as 0.2.0, byte for byte. Its manifest differs only in fields that differ between
  two identical runs anyway (the run ID, the seed nonce and its commitment, the generation time) or
  that describe the code (the version and the code fingerprint). `tests/test_regression_020.py`,
  whose hashes were taken from 0.2.0 before the refactor, pins all of them.
- The legacy path calls the 0.2.0 functions with the same arguments and the same random streams.
  Every new random choice has a seed path of its own (`instance/<i>/mix`, `instance/<i>/surface`,
  `instance/<i>/slots`, `instance/<i>/plan`, `instance/<i>/anchors`, `snapshot/<domain>/…`,
  `templates/new/<domain>`),
  which only the new options
  create, so no legacy draw moves.
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
  is tested. The code targets Python 3.10+ and also runs on 3.9; outputs, reports included, are
  byte-identical between Python 3.9 and 3.14. (The design effect that widens the held-out gaps' intervals
  sums its terms exactly, with `math.fsum`: Python 3.12 changed how the built-in `sum()` adds floats, and
  the last digits of those intervals in `report.json` differed between the two.)
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
  rules changed. The last commits before the release changed what those prompts print (the glossary's
  ID line and an earthquake's qualifier, the earthquake rules' heading and rule 9, every domain's
  rules on notices and times, the wording of radii, of a quantity of 1, of compared groups and of a
  question's point, which questions the guards keep, some news, chat and form wordings, how prose
  places a body approached and times a classic prose line, weather's duplicate notes and local times,
  the forms' sent times, and the invented domain's surfaces) and the format of the frozen NWS bundle.
  The trap fixes then changed where traps are planted and what each instance draws (its slots, on a
  seed path of its own), and the synthetic worlds of every domain (aftershocks, relays that file
  records, station timing, neo designations, invented batches), so runs off the legacy path from
  before them draw other instances. Their second round changed them again: which record a re-report
  under its record's ID reports, the scopes each trap draws and how many retractions a digest plans,
  the weather IDs (opaque serials) and the H that a close approach's re-report gives. Their third
  round changed them again: every digest's trap plan (a seed path of its own, `instance/<i>/plan`, in
  place of `instance/<i>/retractions`), how far a planted report that decides a question leads, a
  corrected value tied to its record's twins, the places and labels of the questions about the record
  nearest to one (`instance/<i>/anchors`) and when a notice is issued. The review that followed
  changed them again: the leads that traps are matched to (another question's), which places the
  questions about the nearest record are asked about (the next report clear of the one after it),
  which questions the guard on the time of the largest value keeps, and a form header's wording. The
  final checks changed them once more: the reading rules (rule 1 quotes the duplicate notes, rule 2
  names the time and position) and the glossary's line on statuses, the invented domain's duplicate
  notes (and so its surfaces), the classic Revised notices, the readings that a re-report under its
  record's ID records (and so where traps are placed, and the trap exposure of questions), the trap
  descriptions of the audit sample, which window a chatter line names and when chat and email post
  it (instances record that time, `chatter_after`), email acknowledgements and their subjects, the
  boxed amendment's issue label and how radio speaks a value of 1. Legacy runs from every one of
  those versions do regenerate.
* Python 3.9 for 0.3.0 was checked by script, because pytest is not installed for the system
  Python 3.9 of the machine that built it. `tests/test_python39.py` scans the code for what 3.9
  cannot run. Under Python 3.9.6, the 0.2.0 regression run matches the hashes its test pins, and
  three runs off the legacy path match the same runs under 3.14 file for file: a random mix that
  renders every built-in domain in every genre, a random mix with the example config's plugins
  answered by the oracle, the naive reader, the mock and a penalised mock, scored and reported (its
  `report.json`, with the held-out gaps, included), and an explicit list.

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
  traps.py              trap planting (planned per digest, load-bearing, guarded and in slots' places off the
                        legacy path), fallen answers, resolution invariant
  digest.py             messy rendering: styles, units, time formats, layouts (the 0.2.0 classic digest)
  solver.py             text-only oracle and naive solvers (digest parser; genre detection off the legacy path)
  prompt.py answers.py  prompt construction (with the glossary off the legacy path); robust JSON reply parsing
  adapters.py           oracle, naive, mock (with held-out penalties), Anthropic, OpenAI
  scoring.py            normalisation, tolerances, consistency, intervals
  forecast.py           delayed-resolution questions and resolution (USGS FDSN)
  audit.py report.py    audit sample; report.md / report.json
  pipeline.py           one cycle end to end
tests/                  866 pytest tests: hand-built key cases, determinism, traps and oracle, guards,
                        digest parsing, reply parsing and scoring, USGS and forecasts, adapters, report,
                        gap intervals, head to head, reading-rules coverage, CLI and its errors, received
                        runs that try to reach outside their folder; each domain and genre, every domain in
                        every genre (test_matrix.py), mixes and held-out sets, what tells planted reports
                        apart (test_trap_tells.py), the 0.2.0 regression test;
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
* **Synthetic world.** Plausible but simplified: synthetic regions are coarse, and the 0.2.0 world
  has no aftershock sequences (off the legacy path a share of events are aftershocks, see
  earthquakes under 0.3.0).
* **The legacy traps can be told apart without the rules.** Legacy runs keep the 0.2.0 placement byte
  for byte (see Digest and traps): its re-reports lie seconds and a few km from their quake, mostly
  from networks that file no event of their own. Readers that read no status, note or notice close
  much of the gap between the naive reader and the oracle there. Measured over every rendering of 252
  fresh synthetic legacy instances (seeds 5001–5084), at 8e63aee and again at the final commit
  (4179c0c), whose instances, prompts and figures are the same (legacy output is the same at every
  0.3.0 commit), with 95% intervals over instances: merging reports within 40 s and 8.5 km, keeping the
  commoner network's, closes 24.7% [21.4, 28.2] (minute-rounded times blunt the window); the same merge
  within a region over 10 minutes or an hour, 35.6% and 37.8%; dropping every report from at, pt and ew,
  the re-reporting networks that file no event of their own in the synthetic world, 28.5% [24.9, 32.4];
  and the hour's merge with that drop, the best of 27 readers picked after the fact, 38.9% [35.3, 42.9],
  against 41.1% [37.3, 44.9] for the status rule that the rules intend (a paired difference of 2.2
  points [−0.6, +4.9], a ratio of 0.95, so the two cannot be told apart). 182 of the 600 AUC cells of
  features the rules never mention fail there by the interval rule and 147 by the point rule (see
  *Measured at the final commit*). On the legacy path, therefore, how often a model handles the traps
  is not evidence that it read the rules, and that holds for the trap rates of the README's study too.
  Off the legacy path no such reader closes a measurable share: the best of the 27 closes 0.0%
  [−0.7, +0.8] (see *Measured at the final commit* under Traps).
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
* **Neither commitment binds the generation settings.** The seed commitment binds the seed and the
  config commitment the config's content. Neither binds the mix, the number of instances or variants,
  or the other settings that the manifest records under `config` (source, pool size, audit rate). An
  evaluator could choose them after the commitments were published, and the revealed run would still
  verify and regenerate. Announce them with the commitments when that matters, minding that a list
  mix names its domains and genres, held-out ones included.
* **What a run's manifest names.** It names the private config by its file name and each plugin by
  its path as the config gives it, so a run names no folder that the revealed config does not. The
  0.2.0 fields that name files are as they were: the frozen feed a run replays (`snapshot_file`, as
  given on the command line), the run it came from (`replay.source_run`, an absolute path), and
  `--snapshot-dir` as given. A legacy report also names the run's folder, as given, in its forecast
  hint. Check those before publishing a run directory.
* **Plugins are code.** They run in the evaluator's process, and a run that uses them needs
  `--private-config` again for `run`, `score`, `report` and `resolve` in a new process.

**The source and genre gaps**

* **Power.** The calibration at the final commit (see Report additions) gives the scale. With 100
  instances, one held-out domain of four and two held-out genres of seven, the adjusted source gap's
  interval is about ±14 points (55% of rows compared like for like) and the genre gap's ±6.5 (88%); a
  mock 15 points worse on the held-out domain was flagged in 54% [45, 63] of runs and 30 points worse
  in 98% [94, 99.5], one 15 points worse in the held-out genres in 90% [83, 94] and 30 points worse in
  100% [97, 100]. At 40 instances the source gap's interval is about ±29 points. With 20 instances,
  which 30-point penalty is flagged is a matter of the draw: at the final commit the demo of the
  example config flags the penalty on held-out genres and misses the one on held-out domains, whose
  adjusted interval is about ±48 points at 29% coverage, and the code just before the final checks did
  the reverse. Measured on earlier code (the validation table's): two held-out domains and three
  held-out genres put the source gap within about ±12–15 points and the genre gap within ±6 at 100
  instances, and within ±6–7 and ±3 at 300; one held-out genre out of seven gave ±5 at 300 instances,
  where three gave ±3; and in simulated 100-instance runs of that design, with answers independent, a
  reader 20 points worse on the held-out domains was flagged in 93% of runs, 15 points worse in 72% and
  10 points in 38%, one 15 points worse in the held-out genres in 95% and 10 points in 50%.
* **Comparing within names costs a random mix power.** Each gap compares within the names of the
  other kind (see Report additions), which a list mix needs and a random mix, which draws every genre
  alike for every domain, does not; there the finer strata compare fewer rows. When this was
  introduced (on the code of that time), the first row of the earlier validation table lost
  source-gap coverage, from 87% to 64%, and its interval widened from ±10 to ±13 points, and in
  simulation the share of 100-instance runs that flag a reader 15 points worse on held-out domains fell
  from 83% to 72% (and of 40-instance runs from 42% to 25%); the genre gap, compared within four
  domains rather than seven genres, lost almost nothing (95% either way). At the final commit, with one
  held-out domain, the source gap compares 55% of rows and a 15-point domain penalty is flagged in 54%
  of 100-instance runs.
* **Tied splits.** Both adjusted gaps compare within the other kind's names. When a mix ties domains
  to genres, the adjusted gap has low coverage (measured on earlier code: 49–60% in the partly tied mix
  of the validation table; 17% in a 40-instance run of `weather,neo:email,invented:table` with email
  held out, in which only weather is rendered in genres of both splits) or is n/a (in all 30 runs of
  the fully tied mix of the final calibration), and the raw gaps mix the two kinds (in a fully tied mix
  they move together: a 30-point penalty on held-out domains raised the raw source and genre gaps by 30
  and 29 points on average).
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
  instances. Of these only the template gap's feeds a flag, and its pools are asked alike in almost
  every instance (see the simulation above). Legacy reports keep all of them exactly.
* **The design effect is an approximation.** The adjusted gap's interval is widened by the design
  effect of the rows it compares taken as one raw difference, because its strata are too small for a
  cluster-robust variance of their own. In simulation on earlier code its intervals covered 93–96% of
  runs with strong clustering, and at most 4% of the readers with no gap were flagged; in the final
  calibration, whose mock answers each question independently, they covered the expected gap in 95.9%
  of 1,800 mock gaps, and the exact spread of a penalised gap was 1.01 to 1.03 times the spread its
  interval implies. With few held-out instances (a demo of 20) every interval is rough. A side whose
  compared rows all come from one
  instance shows no spread between instances, so nothing widens its interval: one misread digest of a
  held-out domain beside thirty public instances made an adjusted gap of +80 points (+49 to +111). A
  gap is therefore flagged only when the compared rows of each side come from two instances or more.
* **Legacy consistency compares as 0.2.0 did.** A legacy report's metamorphic consistency compares
  two answers by the scoring tolerance alone, so two right answers on either side of the key (4.96
  and 5.04 for a magnitude of 5.0) count as an inconsistency there. Off the legacy path both
  consistency metrics count two answers that are both scored right as agreeing.

**Domains and genres**

* **The 0.2.0 text that legacy output keeps.** Legacy prompts and reports are the 0.2.0 ones, byte
  for byte, so the flaws that 0.3.0 fixes off the legacy path remain in them. A radius can be
  worded "no more than 40 km from …" beside a depth threshold, which reads as a range of depths
  (rarely: none of the 1,191 question texts of the 0.2.0 regression run; 2 of 471 in a 40-instance
  random mix before 0.3.0 reworded it there). The report says that "each twin pair is used
  equally often", which the 0.2.0 composer does not do (see Templates). And it describes the naive
  reader as treating "every line" as a separate quake, which is true of the classic digest only.
  Legacy questions also keep the 0.2.0 wording and guards (see Question wording and One answer):
  "1 miles"; the two regions of a comparison unquoted ("Honshu, Japan or Southern Alaska"); a point
  named after the network of the quake it was drawn near ("station NC-42"), which is almost always
  the nearest quake's; and a largest or smallest value whose runner-up the scorer accepts too (in the
  legacy run of seed 4242, 40 instances, a smallest depth of 8.0 km beside one of 8.3 km).
* **Invented words.** Coined names of four or five letters, and their unit words, can still be English
  words (a unit called "halls"), and the list of offensive parts that no coined word may contain is a
  list: it is not a filter of every word that gives offence.
* **Live sources: USGS, NWS and JPL only.** The invented domain has none, by design. The NWS and JPL
  CAD adapters have one run against the real APIs behind them, besides the recorded API responses
  and fake transports that the tests use. On 2026-11-12 (at 8ef8657), `generate --mix
  "weather:classic,neo:table,earthquakes:news" --source live --seed 8435451682288510821 --instances
  6 --variants 1` fetched 532 observations from the 24 NWS stations (a 24-hour window), 69 close
  approaches from the JPL CAD API (1,344 hours) and 240 events from the USGS day feed, none falling
  back to synthetic data; the oracle's self-check reproduced the key in 56 of 56 checks (28
  questions, each in its original rendering and one variant). That run's manifest, which records
  these counts, each feed's SHA-256 and the self-check, is
  `runs/live-smoke-2026-11-12/manifest.json` (its run ID, `run`, is the scratch folder it was
  written to); as for every committed run, the frozen feeds, instances and prompts are not
  committed. Replayed with `--snapshot-dir` while every request went to an unusable proxy, each
  frozen feed's SHA-256 checked, and every instance, prompt and audit file came out byte-identical;
  only the snapshots' metadata differed (the time of the fetch, which a replay has kept since, and a
  note that says it is a replay), with the manifest's volatile and replay keys. One run is not a
  record of how the APIs behave over time: their formats or limits may still change, which the
  frozen bytes and the fallback to synthetic data are there for.
* **What is left of the trap tells off the legacy path.** A planted report is a record drawn with the
  digest (see Traps), so it is told apart only by what the rules name, and by what follows from them or
  from placing traps where they change answers. At the final commit no rule-free reader closes a
  measurable share of the gap (the best of 27, picked after the fact: 0.0% [−0.7, +0.8] overall, at
  most +2.0% [+0.0, +5.2] in a domain), but AUC cells of unintended features still fail. In the sample
  of every domain and genre (G, 1,008 instances), over all reports 2 of 1,314 cells fail by the
  interval rule (the 95% interval excludes 1/2 and reaches outside [0.35, 0.65]) and none by the point
  rule (the estimate lies outside [0.35, 0.65]); per domain 13 and 4 of 2,515 (see *Measured at the
  final commit* for every sample). What follows says where; figures not marked as the final commit's
  were measured in the earlier rounds, on their code:
  - A report filed under a record's ID lies in that record's group, which then holds two reports at
    least (earlier rounds: an AUC of about 0.6 for the size of a second report's group on the CAD
    replay), and an earthquake ID names the network that issued it, which a re-report from another
    network does not share (earlier rounds: 0.39 to 0.41; of the reports that share an ID, keeping the
    one whose ID names its network did 3 to 4 points better than a coin, and the status 16 to 19). The
    shared ID says as much. At the final commit an earthquake second report's value stands out a little
    from its group's other reports within two hours (|z|: 0.694 [0.559, 0.810] on 19 planted reports,
    failing both rules in G's earthquake slice; 0.599 [0.514, 0.681] over all reports, the interval rule
    only).
  - A trap is placed where its careless readings change an answer and pass the key's guards, and its
    slot is chosen by the scopes its report lies in, so a planted report is a little likelier than a real
    one to lie clear of an asked threshold or to rank high in its group (earlier rounds: AUCs up to
    about 0.6, 0.53 to 0.60 for a close approach's second report's distance from the nearest asked
    distance threshold; its cells of single genres, of 10 to 21 planted reports, reached 0.61 to 0.68 in
    the news genre on every sample, and up to 0.78 in another genre on one, though no trap is placed by
    its genre). A correction shows a slot's value, a little farther in value from its group's other
    reports than a real one is (earlier rounds: 0.53 to 0.64). In the earlier rounds no reader that
    dropped or merged reports by such a feature closed more than 2.5% of the gap in any domain, and a
    model of all of them at most 1.9%. At the final commit 185 of G's 8,919 single-genre cells (2.1%)
    fail the interval rule and 22 the point rule, fewer than the 5% that intervals excluding 1/2 give by
    chance.
  - The weather residual. In weather a planted report decides a question about a place half as often as
    its share of the reports the question asks about (earlier rounds; four fifths as often in
    earthquakes). A place near it has a record elsewhere for its key's nearest, which must stand clear
    of the next, and weather's stations lie hundreds of km apart, several observations to a station; the
    weights of `questions.reanchor_nearest` make up for that within a digest, but where no planted
    report of a digest can decide such a question, none does. As a lead in km dwarfs one in shares of a
    field's spread, how far a report that decides a question leads the runner-up (the least of its
    leads) tells weather's planted reports from real ones. In the earlier rounds it gave AUCs of 0.34 to
    0.49 for weather's re-reports, phantoms and second reports (0.39 to 0.43 for the re-reports
    together), outside [0.35, 0.65] in a cell or two of a sample. At the final commit it gives three of
    the four point-rule failures of G's domains, and the two with the most planted reports: weather's
    re-reports, 0.339 [0.277, 0.400] on 72 planted reports (0.333 [0.253, 0.406] on the 49 under the
    record's ID, both failing both rules; 0.353 on the 23 cross-referenced ones, the interval rule only),
    and earthquake phantoms, 0.348 [0.252, 0.451] on 15. The same instances in classic give 0.343, 0.345
    and 0.338 for the three kinds of weather re-report, all failing both rules, and 0.327 for earthquake
    phantoms. No reader gains by it: weather is the domain whose best rule-free reader closes the most,
    +1.4% [−1.1, +3.8] on G and +2.0% [+0.0, +5.2] on the `--mix random` sample, and its best
    domain-by-genre cell, weather/radio on G, +7.6% [−4.7, +17.7]; in the earlier rounds a reader who
    answered such a question from the
    reports in the band of its place or named by its label closed none of the gap.
  - A form's sent time no longer follows its digest's notices. Every form of a batch shares one limit
    below the shortest time from a record to a notice about it (see Genres), and the more notices a
    digest held, the lower the limit tended to be, while every phantom comes with its retraction's: pooled
    over renderings, a phantom's form was sent sooner than a real one (an AUC of 0.42 [0.39, 0.45] on the
    first samples; on the second round's, 0.29 for a phantom and 0.36 for a correction); within its own
    rendering, all a reader can compare, alike. A notice now follows its report by 92 minutes at least
    (`traps.NOTICE_DELAY`), later than any form is sent, so the limit binds only in a rendering that
    holds a correction of a record that close to the compilation (issued just before it), and then
    for every form of the rendering alike (measured after the third round's change: 0.547 [0.512,
    0.584] for a phantom and 0.512 for a correction, pooled, and 0.535 within a rendering).
  - The leads, the questions about a place and the count, as the review measured them on its code. In
    weather, a planted report that decided the time of the warmest observation, or the k-th warmest,
    still led by less than a real one (AUCs of 0.32 and 0.40 on 21 and 19 planted deciders; 0.46 for
    weather's value questions together): the weather residual above. Where no report of a digest can be
    the clear nearest of a place (a weather station with two observations in the scope ties), the
    question keeps the place drawn near a record, whose next report need not stand clear of the one
    after it, so in weather a planted decider's next report still stood clear more often (an AUC of 0.62
    on 23 planted deciders). A count factor fitted on one sample closed 2 to 5% of the gap on another
    (6.6% at most in a domain, on the CAD replay); single genres of single samples reached 10 to 16%,
    none of them again on another sample, so the plan's weights were not spread further (which would
    plant fewer traps that change an answer) before they are measured on more seeds. A model of the
    unintended features of a pair of reports that share an ID (their gaps in time to their group's other
    reports, a weather temperature's jump) broke the pair's tie 3.6 to 6.0% of the gap better than a
    coin (6.4 to 9.3% on the CAD replay, where the status broke it 24%); this predates the third round,
    and the slot choice behind it was not found.
* **Fewer placements change an answer.** A re-report used to copy its record's values give or take a
  little, so a re-report of the largest record often changed the largest answer; a slot's values are
  its own. The figures that follow were measured in the red-team rounds, each on its code. With slots,
  a digest of a built-in synthetic domain kept as many traps that change answers as before, with the
  extra trap that follows one that draws no scope, but where scopes are narrow or slots are few the
  number of traps that change a question's answer fell: from 0.82 a question to 0.76 for the toy plugin
  domain, and from 0.83 to 0.68 for neo replayed from the 40 approaches of the recorded CAD response,
  which leave 12 to 20 slots a digest (the built-in synthetic domains kept 0.75 to 0.83). Choosing a
  placement by its scopes before the answers it changes, and a digest that holds no retraction in 40%
  of digests, took a little more: in the second round's samples the share of traps that change an
  answer fell from 0.67–0.75 to 0.57–0.65 in the built-in synthetic domains (from 0.57 to 0.54 for the
  toy plugin, from 0.63 to 0.48 on the CAD replay), and the number of traps that change a question's
  answer from 0.74–0.81 to 0.73–0.80 (0.75 to 0.73, 0.66 to 0.64). The third round's plan, which plants
  no trap of a kind in a share of digests and no report at all in one in fifteen, took more: 0.62 to
  0.80 traps changed a question's answer in the built-in synthetic domains (from 0.72–0.81; weather
  0.62, earthquakes 0.65, neo 0.70, invented 0.80), 0.62 for the toy plugin (from 0.73) and 0.69 on the
  CAD replay (from 0.64); 3.1 to 3.9 traps of a digest changed an answer (2.7 for the toy plugin, 3.2
  on the CAD replay). The final checks then made a re-report under its record's ID record a second
  report's readings too, which makes more traps change an answer: in the final 40-instance demo, 201 of
  306 planted traps change one (175 at 8e63aee, on the same seed).
* **A digest may hold no trap of a kind.** Its plan draws each kind on its own, and none of it in a
  share of digests (see *The plan* under Traps: no retraction in half of them, no conflict or correction
  in some), so a run of a few instances may plant none of a kind, and its report then has no row for it.
* **Weather stations.** Beyond the five stations sampled from the API for the recorded fixture
  (KJFK, KORD, KDEN, KSEA, KLAX), the names and positions of the 24 stations are a hand-made table
  in the NWS style. In a live run an observation takes its position from the API whenever it gives
  one, and its station's name always from the table.
* **A re-report stands where its slot stands.** A planted report keeps its slot's time, place and
  source (see Traps), so a re-report or a second report can lie hours and hundreds of km from the
  record it names, and seconds and a fraction of a km from another printed record (its slot's
  aftershock, or its twin), and at a fixed site it stands at a station of the record's region, not
  the record's own (a Phoenix report noted as a duplicate of a Denver observation). Only its ID or
  its note says which record it is, as the rules say. That is deliberate: placed beside its record,
  or kept clear of every printed one, it would stand where no real record does, a tell that the red
  teams measured (an AUC of about 0.6 for the gap to the nearest report of its group, and up to 0.66
  for the reports at a weather re-report's station, before slots). Blind readers flagged such reports
  as odd, and every one of them answered by the rules; reading one by its place instead breaks rule 1.
* **neo's renderings say a catalogue "recorded" an approach.** Every genre prints a close approach's
  time beside the words of every record domain ("the catalogue jpl recorded close approach
  2026dk15-ca1 near Venus at ...", "Recorded:"), which a reader could take for when a catalogue
  logged it. Rule 6 says what that time is (the moment the object passed closest to the body), each
  report prints one time only, and the rule on message times says that no message's time is a
  record's, so no answer depends on the wording; but the words were not made the domain's own (a
  phrase such as "passed closest" in each of the four genres that write "recorded" and in their
  parsers).
* **Nominal local times of earthquakes.** Every genre writes a located record's local time with its
  offset stated. A weather observation takes its station's real zone, but an earthquake (like any
  located record of a domain that knows no zones) takes the whole hour nearest the solar time of its
  longitude (`Domain.utc_offset`, `geo.utc_offset_hours`), which is no real time zone: a September
  earthquake in Southern California reads UTC-08:00, where the place keeps UTC-07:00. The 0.2.0
  digest, which legacy runs print, does the same. The stated offset and rule 6 fix every instant, so
  no answer changes, but a reader who knows the zones may hesitate.
* **neo has five bodies.** Templates whose only parameter is the body (P03 and P08, and their
  twins) run out after about five questions per run. In a 60-instance neo run, P03 and P08 asked
  five questions each and their twins six.
* **The invented domain's realism.** It is a synthetic world of fixed shape: eight depots, seven
  agencies (five depot agencies and two clearing agencies), two log-normal quantities per depot. Its
  surface words are drawn per instance, but its status words ("verified", "provisional") and field keys
  ("f1", "f2"), which every glossary prints, are fixed for the domain, as are its ID shape and its
  distributions. An adversary can practise on it like any public domain. It tests reading through a
  glossary, not the reading of a real source.
* **Earthquakes in the classic genre off the legacy path are the 0.2.0 digest.** It rounds depths to
  0.1 km or mi, positions to 0.001° or a hundredth of an arc-minute and some times to the minute,
  and the 0.2.0 parser does not read back place notes, retraction reasons or the magnitude that a
  correction replaces. The margin guards keep the oracle at 100%, and the cross-matrix test reads
  that cell back at 0.2.0 precision.

**Mixes**

* **Random draws depend on the built-in set.** Without a private config, `--mix random` draws from
  every built-in domain and genre in module order, and a bare domain from every built-in genre, so
  adding a built-in domain or genre changes what a seed draws in those mixes. Legacy runs are
  unaffected, and a config fixes the inventory: a new built-in that it does not list changes nothing.
