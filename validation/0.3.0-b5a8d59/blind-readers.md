# Careful-reader tests at 4179c0c

Two rounds of blind reading, both on the code at `4179c0c` (the final code before the audit fixes). A reader
answered each prompt from its text and the reading rules alone, and a checker scored every answer with the
scorer and judged every mismatch. Each round covers 14 domain x genre cells: the two domains of each genre.
Only counts are here: no prompt, digest, answer or key.

Round A is the first final round (six genres checked, radio re-scored after its first checker failed).
Round B covers the other 14 cells. The 'wording notes' are the checkers' remarks about rule text and
realism; their text is not included.

| round | genre | questions | matched | slips | ambiguities | bugs | wording notes |
|---|---|---|---|---|---|---|---|
| A | classic | 9 | 9 | 0 | 0 | 0 | 1 |
| A | news | 10 | 10 | 0 | 0 | 0 | 1 |
| A | email | 9 | 9 | 0 | 0 | 0 | 3 |
| A | chat | 9 | 9 | 0 | 0 | 0 | 2 |
| A | table | 9 | 9 | 0 | 0 | 0 | 3 |
| A | form | 9 | 9 | 0 | 0 | 0 | 5 |
| A | radio | 8 | 8 | 0 | 0 | 0 | 0 |
| A | **total** | 63 | 63 | 0 | 0 | 0 | 15 |
| B | classic | 8 | 8 | 0 | 0 | 0 | 0 |
| B | news | 9 | 9 | 0 | 0 | 0 | 0 |
| B | email | 8 | 8 | 0 | 0 | 0 | 5 |
| B | chat | 7 | 7 | 0 | 0 | 0 | 2 |
| B | table | 9 | 9 | 0 | 0 | 0 | 0 |
| B | form | 8 | 8 | 0 | 0 | 0 | 0 |
| B | radio | 6 | 6 | 0 | 0 | 0 | 2 |
| B | **total** | 55 | 55 | 0 | 0 | 0 | 9 |
| all | **total** | 118 | 118 | 0 | 0 | 0 | 24 |

All 118 questions across the 28 cells were matched: 118 of 118 (100.0%; Wilson 95% interval 96.8 to 100.0).
Round A's radio cell was re-scored (`radio-r2`) after its first checker stopped without a result.
