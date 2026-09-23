# Canonical ABCD v2.2 dataset (not redistributed here)

The canonical dataset is **not redistributed in this public repository**.
Authorized collaborators may receive it separately and place it at the
expected local path.

## Expected local file

    data/canonical_abcd_v2_2/ABCD_clean_v2_2_OFFICIAL_DIJA_FINAL.jsonl

The path is git-ignored. Do not force-add it.

## Expected contents

- SHA-256: `4392afca6d8f6b788a848d0417c92807ce01aeb077a67d6424bf12d69138eaaf`
- A = 500, B = 500, C = 500, D = 500 (total 2000)
- health = 1600, non-health = 400
- 500 groups, each with exactly one A, B, C and D row

| Bucket | Meaning |
|---|---|
| A | harmful, clean |
| B | harmful + DIJA |
| C | benign + DIJA |
| D | benign clean rewrite |

Schema (12 fields per row): `id`, `bucket`, `label`, `variant`, `injection_type`, `domain`, `prompt`,
`paired_A_id`, `paired_D_id`, `clean_prompt`, `source_dataset`, `needs_manual_audit`.

## Verify

    python data/canonical_abcd_v2_2/verify_dataset.py
    # or, for a file elsewhere:
    python data/canonical_abcd_v2_2/verify_dataset.py /path/to/ABCD_clean_v2_2_OFFICIAL_DIJA_FINAL.jsonl

Prints `DATASET_VERIFICATION=PASS` and exits 0, or prints the failures and exits non-zero.
The verifier is read-only. See `PROVENANCE.md` for sources and why the file is shared privately.
