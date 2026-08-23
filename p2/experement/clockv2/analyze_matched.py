#!/usr/bin/env python3
"""clockv2/analyze_matched.py — THE DECISIVE TEST (aggregate only; writes NO verdict).

Three arms, all temp=0, same engine, paired by id:
  clean  : plain harmful behavior, no scaffold          (n_inject=0,  plen ~33)
  dija   : harmful behavior + DIJA scaffold             (n_inject=57, plen ~137)
  benign : BENIGN question + length/format-matched scaffold, n_inject IDENTICAL per id

Three contrasts, each a paired per-id delta. Read point = prompt_mean (PRIMARY) throughout;
last_prompt is reported separately and never mixed into the same column.

  (1) dija - clean    : the original headline. Confounded -- differs in BOTH scaffold and
                        prompt length (~+104 tok).
  (2) benign - clean  : POSITIVE CONTROL for "is the axis structural?". Differs in scaffold
                        and length, but NOT in harm.
  (3) dija - benign   : THE DECISIVE TEST. Scaffold, format and n_inject held constant;
                        ONLY harm/injection semantics differ.

DECISION RULE (fixed before looking):
  If (3) collapses toward the random-null floor while (1) and (2) are both large and similar,
  then v_injection_svd encodes PROMPT STRUCTURE, and the Phase-3 headline is a length/format
  artifact -- it must be reported as such.
  If (3) stays well above the floor at L25-L26 (where the null has collapsed), the axis
  carries injection semantics over and above structure.

PRIOR, STATED BEFORE THE RUN: v_injection_svd was fit as SVD of (h_dija - h_clean) over 43
held-out HARMFUL prompts -- harm constant, scaffold varying. It was never fit on a harm
contrast, so the expected outcome is that (3) collapses. This test is not expected to
vindicate the headline; it is expected to characterise it.

    python clockv2/analyze_matched.py
"""
from __future__ import annotations

import json
import os

import numpy as np
from scipy import stats

from aggregate import LABEL, PROBES, paired, auc
from common import BAND, RESULTS_DIR


def load3():
    """-> ids, A[metric][n_ids, 3, n_layers, STEPS] with cond axis 0=clean 1=dija 2=benign."""
    files = [os.path.join(RESULTS_DIR, "probe_readings.jsonl"),
             os.path.join(RESULTS_DIR, "probe_readings_benign.jsonl")]
    rows = [json.loads(l) for f in files for l in open(f)]
    ids = sorted({r["id"] for r in rows})
    lay = {L: i for i, L in enumerate(BAND)}
    idx = {c: i for i, c in enumerate(ids)}
    cnd = {"clean": 0, "dija": 1, "benign": 2}
    steps = max(r["step"] for r in rows) + 1
    metrics = [k for k in rows[0] if k not in ("id", "cond", "layer", "step")]
    A = {m: np.full((len(ids), 3, len(BAND), steps), np.nan) for m in metrics}
    for r in rows:
        i, c, l, t = idx[r["id"]], cnd[r["cond"]], lay[r["layer"]], r["step"]
        for m in metrics:
            A[m][i, c, l, t] = r[m]
    return ids, A


def main():
    ids, A = load3()
    n = len(ids)
    miss = {m: int(np.isnan(A[m]).sum()) for m in A if np.isnan(A[m]).any()}
    print("=" * 104)
    print(f"DECISIVE TEST — matched-control arm | n={n} ids x 3 arms | temp=0")
    print(f"missing cells: {miss if miss else 'none'}")
    print("=" * 104)

    ni = A["n_inject"][:, :, 0, 0]
    pl = A["prompt_len"][:, :, 0, 0]
    print("\nARM MATCHING (the whole point — verify before reading any effect):")
    print(f"{'arm':>8} | {'n_inject':>18} | {'prompt_len':>18}")
    for name, c in (("clean", 0), ("dija", 1), ("benign", 2)):
        print(f"{name:>8} | {ni[:,c].mean():>8.1f} +- {ni[:,c].std():<6.1f} | "
              f"{pl[:,c].mean():>8.1f} +- {pl[:,c].std():<6.1f}")
    print(f"\n  n_inject   dija vs benign identical per id: {int((ni[:,1]==ni[:,2]).sum())}/{n}")
    g_dc, g_db = pl[:, 1] - pl[:, 0], pl[:, 2] - pl[:, 1]
    print(f"  prompt_len gap  dija-clean : mean={g_dc.mean():+7.1f} sd={g_dc.std():5.1f}  "
          f"(the confound the headline could not rule out)")
    print(f"  prompt_len gap  benign-dija: mean={g_db.mean():+7.1f} sd={g_db.std():5.1f}  "
          f"(matched -- this is the clean contrast)")

    CON = (("dija - clean", 1, 0, "original headline (confounded: +scaffold AND +length)"),
           ("benign - clean", 2, 0, "POSITIVE CONTROL: +scaffold, +length, NO harm"),
           ("dija - benign", 1, 2, "DECISIVE: scaffold/format/n_inject held; ONLY harm differs"))

    for rp, tag in (("pmean", "PRIMARY  read point = prompt_mean"),
                    ("last", "SECONDARY read point = last_prompt")):
        print("\n" + "=" * 104)
        print(f"{tag}")
        if rp == "last":
            print("  CAVEAT: at last_prompt cos(v_refusal, v_injection_svd)=+0.65 — the axes are")
            print("  NOT separable here. Secondary only; prompt_mean above is primary.")
        print("=" * 104)
        for L in (25, 26):
            li = BAND.index(L)
            print(f"\n--- L{L} ({rp}) — the layer where the random null has collapsed ---")
            print(f"{'contrast':>16} | {'v_injection_svd':>28} | {'v_random_null':>21} | {'ratio':>6}")
            print(f"{'':>16} | {'mean_d':>8} {'dz':>7} {'AUC':>6} {'%>0':>4} | "
                  f"{'dz':>7} {'AUC':>6} {'%>0':>4} | {'x null':>6}")
            for name, a, b, why in CON:
                d = (A[f"vshared_{rp}"][:, a, li, :] - A[f"vshared_{rp}"][:, b, li, :]).mean(1)
                s = paired(d)
                ai = auc(A[f"vshared_{rp}"][:, b, li, :].mean(1),
                         A[f"vshared_{rp}"][:, a, li, :].mean(1))
                dn = (A[f"vnull_{rp}"][:, a, li, :] - A[f"vnull_{rp}"][:, b, li, :]).mean(1)
                sn = paired(dn)
                an = auc(A[f"vnull_{rp}"][:, b, li, :].mean(1),
                         A[f"vnull_{rp}"][:, a, li, :].mean(1))
                r = abs(s["dz"]) / abs(sn["dz"]) if sn["dz"] else np.inf
                print(f"{name:>16} | {s['mean']:>+8.2f} {s['dz']:>+7.2f} {ai:>6.3f} "
                      f"{100*s['frac_pos']:>3.0f}% | {sn['dz']:>+7.2f} {an:>6.3f} "
                      f"{100*sn['frac_pos']:>3.0f}% | {r:>6.1f}")
            for name, a, b, why in CON:
                print(f"    {name:<15} = {why}")

    # full-band view of the decisive contrast, primary read point only
    print("\n" + "=" * 104)
    print("DECISIVE CONTRAST (dija - benign) ACROSS THE BAND. READ POINT = prompt_mean (PRIMARY).")
    print("=" * 104)
    print(f"{'layer':>5} | " + " | ".join(f"{LABEL[p]:>24}" for p in PROBES))
    print(f"{'':>5} | " + " | ".join(f"{'mean_d':>8} {'dz':>7} {'AUC':>6}" for _ in PROBES))
    for li, L in enumerate(BAND):
        cells = []
        for p in PROBES:
            d = (A[f"{p}_pmean"][:, 1, li, :] - A[f"{p}_pmean"][:, 2, li, :]).mean(1)
            s = paired(d)
            a = auc(A[f"{p}_pmean"][:, 2, li, :].mean(1), A[f"{p}_pmean"][:, 1, li, :].mean(1))
            cells.append(f"{s['mean']:>+8.2f} {s['dz']:>+7.2f} {a:>6.3f}")
        print(f"L{L:>4} | " + " | ".join(cells))
    print("\n  v_refusal / tau statuses are UNCHANGED from Phase 3 (null / null-circular) and are")
    print("  not re-litigated here. The benign arm does make a properly controlled v_refusal test")
    print("  possible for the first time (harm varies, format constant) — flagged, not run.")
    print("\n" + "=" * 104)
    print("NO VERDICT WRITTEN.")
    print("=" * 104)


if __name__ == "__main__":
    main()
