# evalgen 0.3.0: multiple sources and unpredictable genres

## Why

Prof. Kishor Datta Gupta's review: *"Relying solely on the USGS feed creates a risk, as an adversary
could fine-tune their approach specifically for that source. We should broaden our data sources and
make the text source genres unpredictable."* 0.2.0 has one domain (earthquakes, synthetic or USGS)
and one family of renderings (bulletin, wire, log and prose lines). An adversary who knows the public
code can practise on unlimited earthquake digests in that one style. 0.3.0 makes both the *domain*
and the *genre* of each test instance a seeded draw, and holds some of each back.

## Hard constraints

1. **0.2.0 stays byte-identical.** `cli.py generate --seed 16222556419173536875 --instances 100`
   with no new flags writes the same 405 files. `tests/test_regression_020.py` (committed before
   the refactor) pins them and must pass after every step.
2. **Keys are computed by code.** No LLM or human in the key.
3. **Every new random choice goes through SeedLog**, so a run regenerates exactly from its seed.
   New seed paths are created only when a new option is used, so the default path's draws never
   shift.
4. **The text-only oracle parses every domain × genre rendering back to the exact reports and
   notices.** Round-trip tests cover the whole matrix.
5. **Every protection keeps working in every domain**: margin guards, no repeated questions, traps
   with recorded careless readings, metamorphic variants, and the seen-vs-new template gap.
6. **Tests never touch the network.** Live sources are fetched once, frozen and SHA-256 hashed, with
   a synthetic fallback when offline, exactly like USGS.
7. **No paid model calls.** End-to-end checks use the simulated model and `cli.py demo`.
8. **Branch `multi-source`**, one commit or more per phase, nothing pushed.

## Design

### Records and domains

- The earthquake dataclasses (`Event`, `Report`, `Notice`) keep their fields and field order, so
  their JSON is unchanged. They gain read-only properties (`values`, `group`, `source`) that map
  generic names onto their attributes (`values = {"mag": mag, "depth": depth_km}`,
  `group = region`, `source = network`). Properties do not appear in `asdict`.
- Other domains use generic dataclasses in `evalgen/domains/base.py`:
  - `Record(id, time, lat, lon, group, source, values: dict, qualifiers: dict, place)`, the
    ground truth;
  - `RecordReport(ref, source, status, time, lat, lon, group, values, qualifiers, place, same_as,
    role, trap_id)`, one line in a digest.
- `Notice` gains optional `field`, `new_value`, `old_value` and `qualifier`. They are omitted from
  JSON when `None`, so earthquake notices serialize as before.
- `Instance` gains optional `domain`, `split` and `surface`, and `Rendering` gains optional `genre`.
  All are omitted from JSON when unset (legacy), and `from_dict` tolerates their absence.

A **`Domain`** (`evalgen/domains/base.py`) declares:

| part | meaning |
|---|---|
| `name`, `noun`, `plural`, `id_label`, `source_label`, `group_label` | vocabulary |
| `fields: tuple[FieldSpec, ...]` | numeric fields; `FieldSpec(key, label, canonical_unit, units, decimals, answer_type, tolerance, margin, rank_gap, quantiles, threshold_step, filters, terminals, qualifier, signed, correctable)` |
| `roles` | which field plays the 0.2.0 *primary* role (earthquake magnitude) and *secondary* role (depth), so the role-based public templates P01–P12 map onto every domain |
| `has_location` | whether `within_distance`, `nearest_to` and `nearest_distance` apply |
| `units` | `Unit(symbol, words, scale, offset)`: canonical = value × scale + offset (affine for °F) |
| `synthetic(rng, n, compiled_at, window_hours)` | the seeded generator |
| `live` (optional) | `LiveAdapter(fetch, parse, raw_filename, user_agent)`: fetched once, frozen, hashed, replayable, synthetic fallback |
| `ids` | `new_id`, `mimic_id`, `id_pattern` (also used by the scorer's ID matching) |
| `trap_vocab` | wording of duplicate notes, retraction reasons and correction notices |
| `rules_text()` | the domain's reading rules, describing notices genre-neutrally |
| `glossary(surface)` | the data dictionary printed in non-legacy prompts (labels, units, conversions) |
| `resolve(reports, notices, mishandle)` | the reading rules in code (generic for all non-earthquake domains) |

Domains are discovered from `evalgen/domains/*.py` (sorted by name), plus private plugins named in
the private run config.

**Earthquakes** become `evalgen/domains/earthquakes.py`. On the legacy path it calls the existing
functions unchanged (`sources.build_snapshot`, `questions.compose_questions`, `traps.inject_traps`,
`digest.render_digest`, `solver.parse_digest`, `rules.READING_RULES`).

### Field-generic operations

`ops.py` gets generic kinds, and the earthquake ops become aliases:

| generic | earthquake alias |
|---|---|
| `at_least(field)` | `mag_at_least` (mag), `depth_at_least` (depth, with unit) |
| `below(field)` | `mag_below` (mag), `depth_shallower_than` (depth, with unit) |
| `in_group` | `in_region` |
| `argmax(field)`, `rank(field, k)`, `time_of_max(field)` | `argmax_mag`, `rank_mag`, `time_of_largest` |
| `max(field)`, `min(field)` | `max_mag`, `max_depth`, `min_depth` |
| `compare_groups` | `compare_regions` |
| `within_distance`, `time_between`, `dedupe`, `count`, `nearest_to`, `nearest_distance`, `convert`, `round` | unchanged |

A generic step looks like `{"op": "at_least", "field": "temp", "value": 20.5, "unit": "°C"}`.
Earthquake chains keep the 0.2.0 dicts exactly. Answer types come from a registry: the earthquake
types and their tolerances stay, and each `FieldSpec` registers its own type.

Templates are role-based. P01–P12 keep their IDs and skeletons, with the *primary* and *secondary*
fields substituted per domain. Domains without a location drop the spatial templates and their
twins as a pair, so the public and new pools stay matched. New twin templates are composed per
domain from `seedlog.rng("templates", "new", domain)`, used only for non-earthquake domains.
Thresholds come from quantiles of each field's values and are rounded to the field's
`threshold_step`. Guards use each field's `margin` (abs, rel) and `rank_gap`, so they scale to every
field's distribution.

### Genres: content separated from presentation

A **`Genre`** (`evalgen/genres/base.py`) provides:

```python
class Genre:
    name: str
    def choose_style(self, rng, domain, avoid=None) -> dict
    def render(self, ctx: RenderContext) -> str      # reports, notices, chatter, glossary, times, style, rng
    def sniff(self, digest_text: str) -> bool        # recognise its own output
    def parse(self, digest_text: str, glossary: Glossary) -> DigestData   # exact inverse of render
```

| genre | presentation | notices appear as | realistic noise |
|---|---|---|---|
| `classic` | bulletin, wire, log and prose lines (0.2.0; generic version for other domains) | NOTICE, RETRACTION, UPDATE and CORRECTION lines | headers, section heads, chatter lines |
| `news` | headline, dateline, paragraphs | an "Update" or "Correction" paragraph | quotes, background paragraphs |
| `email` | thread with headers, quoted replies, a forwarded notice, signatures | a forwarded message | greetings, signatures, quoted earlier text |
| `chat` | `[HH:MM] user: message` log | a pinned or posted message | acknowledgements, off-topic lines |
| `table` | CSV or markdown table (style draw) | footnote rows below the table | comment lines, a caption |
| `form` | field or incident report forms | an amendment form | checkboxes, form numbers, blank optional fields |
| `radio` | dispatch transcript | a dispatch broadcast | call signs, "copy" and "over", readbacks |

- **Every genre must stay unambiguous to a careful reader.** The reading rules say, genre-neutrally:
  - what counts as a notice;
  - that a quoted or read-back repeat of an earlier message is not a new report;
  - that a message's own timestamp is not the record's time;
  - that lines which are neither reports nor notices change nothing.
- **Exact round trip.** Each value is printed in a unit drawn by the style, with the smallest number
  of decimals that parses back to the stored canonical value. `render` checks this per value, so
  the oracle recovers every report exactly.
- **The glossary.** Non-legacy prompts include the domain glossary, a data dictionary of field
  labels, units and conversions. It is the only way to read an invented domain. The oracle parses
  it from the prompt text (`parse_glossary`), so it stays text-only.
- **Legacy stays as it is.** The legacy path renders with `classic`, the 0.2.0 code, with no glossary
  and the 0.2.0 rules text. The oracle tries each genre's `sniff` and falls back to `classic`.
- **Variants change genre.** Under a non-legacy mix, each metamorphic variant uses a genre different
  from the original's. That makes **cross-genre consistency** a new robustness check.

### Mixes, the private run config and held-out sets

- **`--mix legacy`** (the default) is 0.2.0 behaviour: earthquakes in the classic genre.
- **`--mix random`** draws the domain and the genre per instance from
  `seedlog.rng("instance", i, "mix")`, used only in this mode.
- **An explicit list** such as `--mix weather,neo:email,invented:table` restricts the draw. A bare
  domain name means any genre.
- **The private run config** is a JSON file the evaluator keeps private (`--private-config PATH`):
  `{"nonce", "domains": {"public", "held_out"}, "genres": {"public", "held_out"}, "plugins"}`.
  - Like the seed, it is committed by hash: the manifest records
    `sha256(canonical_json(config))`, and the random nonce stops anyone guessing a small config
    from its hash.
  - Held-out domains and genres can come from **plugins**: Python files outside the public repo,
    loaded at evaluation time. The built-in domains and genres can also be marked held out, for
    demos and tests.
  - Without a config, every built-in domain and genre is public.
- Each instance records its domain and its split (public or held-out), and each rendering records
  its genre. These keys are written only for non-legacy runs.

### Report additions (non-legacy runs only)

- accuracy by domain (original renderings), and by genre (all renderings, by each rendering's genre);
- a **source gap**: public minus held-out domains, stratified by question difficulty like the
  template gap (trap exposure, whether the filters matter, answer family), with a 95% CI;
- a **genre gap**: public minus held-out genres over renderings, stratified the same way;
- **cross-genre consistency**: agreement between renderings of the same question in different
  genres;
- the domain and genre inventory, and the private-config commitment.

Legacy reports are unchanged.

## Phases

Each phase ends with the full test suite, the 0.2.0 regression test, an independent review, and a
commit on `multi-source`.

### Phase 1: Domain abstraction

- **1a. Earthquakes onto the interface.**
  - Domain registry and base classes; earthquake properties.
  - Generic ops with aliases; an answer-type registry.
  - Role-based templates; FieldSpec-driven draws and guards.
  - A domain-aware `resolve`; a domain-aware `build_instance` and `generate` (defaults keep the
    legacy calls); and `classic` as a Genre wrapping the 0.2.0 renderer and parser.
  - Fix the code fingerprint to cover subpackages.
  - **Acceptance:** the regression test passes. Tests show that alias chains and their generic
    equivalents give identical results, that the derived earthquake guard sets equal the 0.2.0
    literals, and that the earthquake domain object reproduces the legacy pipeline.
- **1b. The generic record path.**
  - `Record` and `RecordReport`; a generic snapshot, save and load.
  - Generic question composition and phrasing via a domain lexicon.
  - Generic traps: duplicate, retraction and correction, with recorded careless readings.
  - The generic `classic` renderer and parser for any schema; the glossary renderer and parser.
  - Prompt assembly, and oracle dispatch by `sniff`.
  - **Acceptance:** a small test-only domain runs generate, oracle, naive, mock, score and report.
    The oracle reaches 100%, round trips are exact, and the regression test passes.

### Phase 2: More sources

| domain | records | fields | group | location | live source |
|---|---|---|---|---|---|
| `weather` | station observations | temperature (°C, °F), wind speed (km/h, mph, kt, m/s), pressure (hPa, inHg) | US region | yes | NWS `api.weather.gov` station observations (keyless; User-Agent required) |
| `neo` | close approaches of near-Earth objects | miss distance (au, lunar distances, km), relative velocity (km/s, km/h, mph), absolute magnitude H | body approached | no | JPL `ssd-api.jpl.nasa.gov/cad.api` (keyless) |
| `invented` | invented records, e.g. consignments at depots | two invented quantities in invented units with stated conversions | invented group names | no | none (synthetic by definition) |

- **Invented surfaces are seeded words.** `invented` draws its record noun, ID prefix, group names,
  source names, field labels and unit vocabulary from seeded word lists per instance (from
  `seedlog.rng("instance", i, "surface")`). Even the surface form of the domain is unpredictable,
  and the glossary in each prompt defines it.
- **Each domain module provides:**
  - field specs, with margins and threshold steps scaled to that field's distribution;
  - trap vocabulary, IDs, a lexicon and rules text;
  - a synthetic generator;
  - a live adapter where one exists.
- **Each domain is tested with:**
  - recorded fixtures for its live adapter, plus an offline fallback test;
  - determinism;
  - a round trip through `classic`;
  - the guards;
  - the oracle at 100% on a small run.

### Phase 3: Unpredictable genres

- **Add `news`, `email`, `chat`, `table`, `form` and `radio`**, each with render, sniff and parse,
  and with style draws (such as CSV versus markdown, or address and signature pools).
- **Round-trip tests cover every domain × genre.** For several seeded instances each, the parsed
  reports and notices must equal the originals exactly, and the oracle must reach 100%.
- **Genre-neutral reading rules.** One more test shows that every notice wording in every genre is
  named by the rules.

### Phase 4: Held-out sources and genres

- **CLI:** `--mix` and `--private-config` for `generate` and `demo`.
- **Mix draws:** the per-instance domain and genre draws, with variants that change genre, and
  plugin loading.
- **Manifest:** the private-config commitment, written only when a config is used.
- **Report:** accuracy by domain and genre, the source and genre gaps, and cross-genre consistency.
- **Mock model:** its pool lookup no longer crashes on held-out pools.
- **Tests:** determinism of random mixes; commitment verification; the gap metrics on a fixture run;
  and that legacy reports are unchanged.

### Phase 5: Docs, tests, demo

- **Docs:**
  - `docs/DESIGN.md`: the threat model (the adversary knows the public domains and genres, but not
    the held-out ones or the seed), the domain interface, genres, the new gaps, and limitations.
  - A short README section, "0.3.0: multiple sources and genres".
- **Release checks:** bump the version to 0.3.0, run the full suite, and run
  `cli.py demo --mix random` offline with the report shown.

## Execution

- Phase 1 runs in two sequential implementation passes (1a, then 1b). Each is followed by a review
  and a commit.
- Phases 2 and 3 run as parallel agents in separate git worktrees: one per domain, then one per
  genre. Each adds only its own module, tests and fixtures, and commits on its own branch.
  Discovery-based registries mean there are no shared edits. The branches are merged into
  `multi-source`, then cross-matrix tests and a review run.
- Phase 4 is one pass, and Phase 5 is done directly.

## Risks

| risk | mitigation |
|---|---|
| A refactor shifts a random draw on the default path | The default path calls the 0.2.0 functions; new seed paths exist only in non-default modes; the regression test runs after every change |
| Unit rounding breaks exact round trips | Per-value smallest-exact-decimals formatting, checked at render time; round-trip tests over every domain × genre |
| Genre noise creates real ambiguity (quoted repeats, readbacks, message times) | Genre-neutral rules name each case; the parsers apply exactly those rules; a review looks for ambiguity |
| Live APIs change or rate-limit | Frozen raw responses, SHA-256, replay, synthetic fallback; tests use fixtures only |
| Held-out code ends up published | Held-out sets can come from private plugins; demos use built-ins marked held out, and the docs say so |
