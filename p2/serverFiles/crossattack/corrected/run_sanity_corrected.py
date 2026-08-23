#!/usr/bin/env python
"""crossattack/corrected/run_sanity_corrected.py

Re-run the 6-check leakage battery on the CORRECTED states, REUSING
crossattack/sanity/sanity_battery.py UNCHANGED (imported). We only override the
state dirs (via the shared transfer module) + the battery's output paths, then
call sanity_battery.main().
"""
import os, sys

ROOT = "/home/ore99/serverFiles"
sys.path.insert(0, os.path.join(ROOT, "crossattack", "probe"))
sys.path.insert(0, os.path.join(ROOT, "crossattack", "sanity"))

import transfer as T   # noqa: E402  (chdir's to ROOT on import)
T.DIJA_DIR = "outputs"
T.A2_DIR = "crossattack/corrected/transfer_attack2"
T.A2_PROMPTS = "crossattack/prompts/attack2"

import sanity_battery as S   # noqa: E402  (imports the same, already-overridden, transfer module)
S.OUT_JSON = "crossattack/corrected/sanity_results.json"
S.OUT_MD = "crossattack/corrected/SANITY_corrected.md"

if __name__ == "__main__":
    S.main()
