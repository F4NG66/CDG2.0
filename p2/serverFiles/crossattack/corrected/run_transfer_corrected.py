#!/usr/bin/env python
"""crossattack/corrected/run_transfer_corrected.py

Re-run the frozen-probe cross-attack transfer on the CORRECTED states, REUSING
crossattack/probe/transfer.py UNCHANGED (imported). We only override the module
constants that point at the state dirs + outputs, then call transfer.main():

  DIJA_DIR   = "outputs"                              (existing DIJA states, 84% valence)
  A2_DIR     = "crossattack/corrected/transfer_attack2"  (reused A2 clean + NEW B2 injected)
  A2_PROMPTS = "crossattack/prompts/attack2"          (TF-IDF floor; behaviors byte-identical)
  OUT_JSON   = "crossattack/corrected/transfer_results.json"
  OUT_MD     = "crossattack/corrected/SUMMARY_corrected.md"

Everything else (probe geometry: harm scope, frac 0.05, layers {11,16,26},
hidden+SAE, StandardScaler->LogReg C=1.0, GroupKFold ceiling, freeze-and-apply
both directions, never refit on held-out family) is transfer.py verbatim.
"""
import os, sys

ROOT = "/home/ore99/serverFiles"
sys.path.insert(0, os.path.join(ROOT, "crossattack", "probe"))
import transfer as T   # noqa: E402  (chdir's to ROOT on import)

T.DIJA_DIR = "outputs"
T.A2_DIR = "crossattack/corrected/transfer_attack2"
T.A2_PROMPTS = "crossattack/prompts/attack2"
T.OUT_JSON = "crossattack/corrected/transfer_results.json"
T.OUT_MD = "crossattack/corrected/SUMMARY_corrected.md"

if __name__ == "__main__":
    T.main()
