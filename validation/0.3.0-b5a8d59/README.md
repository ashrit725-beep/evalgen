# Validation summary at b5a8d59

This folder records the validation of evalgen 0.3.0 at its final code commit, `b5a8d59`. The commits after it (up to
`1b0d3d7`) change only the README, the design document and one live manifest. Every calibration run here has code
fingerprint `11d06dc7...`, the fingerprint of the `evalgen/` package at `b5a8d59`.

Start with [paper-numbers.md](paper-numbers.md). Each figure there names the file in this folder that it comes from.

## Contents

- `paper-numbers.md`: the validation figures, with their sources and methods, and the stale or unresolved items
  (section 14).
- `numbers.json`: the same figures as data.
- `calibration/`: 870 calibration result lines, one per seed and model: 120 random-mix seeds with 7 models each, and
  15 confounded-mix seeds with 2 models each. Each slice's `checks.json` holds its self-checks, an independent
  recomputation of every gap from the score rows, and the penalty-isolation checks. `pool_check.py` pools the results
  into the flag, power, coverage and reply tables, and `pool_check.txt` is its output.
- `shortcuts/`: trap-shortcut tables (`tables.md`) and summary (`summary.json`) for the legacy sample (L) and the 0.3.0
  samples (G, R, C), seeds 5001 to 5084.
- `blind-readers.md`: counts from the careful-reader tests, for both rounds.
- `identity-result.txt`: the 0.2.0 identity check at `b5a8d59`.

## Not included

- **Calibration configs.** They hold each slice's nonce and private settings, which stay private until the reveal. The
  held-out domains and genres they use are already named in `docs/DESIGN.md`.
- **Run folders, frozen snapshots, prompts and model replies.** They are large and regenerate from a seed and a config.
  The committed live-run manifest is `runs/live-smoke-2026-11-15/manifest.json`.
- **Synthesis and shortcut scripts.** They refer to local folders. `pool_check.py` is the independent pooling script.
- **Careful-reader prompts and transcripts.** They are test material; the counts are in `blind-readers.md`.
- **The demo report.** It needs the demo's private config, which is not committed. The public command is
  `python3 cli.py demo --mix random --private-config configs/private.example.json`.

## Reproducing the calibration tables

    python3 validation/0.3.0-b5a8d59/calibration/pool_check.py validation/0.3.0-b5a8d59/calibration

This prints the tables that `calibration/pool_check.txt` holds.

## Caveats

- The careful-reader rounds ran at `4179c0c`, not at `b5a8d59`. Rule wording and chatter changed afterwards, so no reader
  has read the final prompts.
- Some audit figures in the docs are labelled as not recomputed at `b5a8d59`.
