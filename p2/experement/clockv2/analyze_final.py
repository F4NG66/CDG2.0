#!/usr/bin/env python3
"""clockv2/analyze_final.py — FINAL TEST across four arms (aggregate only; NO verdict).

  clean     : plain harmful behavior, no scaffold        n_inject=0   plen~33   fp 87%
  dija      : harmful behavior + DIJA scaffold           n_inject=57  plen~137  fp 87%
  benign    : arm 2 -- INFORMATIONAL benign + matched scaffold        fp  0%   <- stance MISmatched
  benign_op : arm 3 -- OPERATIONAL first-person benign + matched scaffold  fp 100%  <- stance matched

Arm 3 exists because arm 2 could not separate harm from stance: its pool was 99/99
"What are the ..." / 0% first-person against a DIJA arm that is 82% "How can I" / 87%
first-person. Arm 3 matches register, leaving harm as the contrast.

Read point = prompt_mean (PRIMARY) for every headline number. last_prompt is reported
separately and never mixed into a column (the axes are +0.65 aligned there).

DECISION RULE (fixed before looking):
  dija - benign_op holds up at L25-L26 against the true-zero null -> the content component
    is harm/injection-semantic; the stance confound is resolved.
  dija - benign_op collapses -> the ~1/3 content effect found in arm 2 was operational-vs-
    informational STANCE, and must be reported as such.

v_refusal stays NULL per instruction. NOTE it is reported in the band table only for
completeness: arm 2's v_refusal reading was IN-SAMPLE (arm 2's questions ARE v_refusal's
99-item harmless fitting pool), and v_refusal's two fitting pools are 100% separable by
question form, so it is inseparable from a form axis. Arm 3 is neither in-sample nor
form-confounded -- flagged for the record, not promoted here.
tau stays NULL/circular and is not re-litigated.

    python clockv2/analyze_final.py
"""
from __future__ import annotations

import json
import os
import re

import numpy as np

from aggregate import LABEL, PROBES, paired, auc
from common import BAND, CASES_A, RESULTS_DIR

ARMS = ["clean", "dija", "benign", "benign_op"]


def loadN():
    files = ["probe_readings.jsonl", "probe_readings_benign.jsonl",
             "probe_readings_benign_op.jsonl"]
    rows = [json.loads(l) for f in files for l in open(os.path.join(RESULTS_DIR, f))]
    ids = sorted({r["id"] for r in rows})
    lay = {L: i for i, L in enumerate(BAND)}
    idx = {c: i for i, c in enumerate(ids)}
    cnd = {c: i for i, c in enumerate(ARMS)}
    steps = max(r["step"] for r in rows) + 1
    metrics = [k for k in rows[0] if k not in ("id", "cond", "layer", "step")]
    A = {m: np.full((len(ids), len(ARMS), len(BAND), steps), np.nan) for m in metrics}
    for r in rows:
        i, c, l, t = idx[r["id"]], cnd[r["cond"]], lay[r["layer"]], r["step"]
        for m in metrics:
            A[m][i, c, l, t] = r[m]
    return ids, A


def contrast(A, probe, rp, a, b, li):
    d = (A[f"{probe}_{rp}"][:, a, li, :] - A[f"{probe}_{rp}"][:, b, li, :]).mean(1)
    s = paired(d)
    s["auc"] = auc(A[f"{probe}_{rp}"][:, b, li, :].mean(1), A[f"{probe}_{rp}"][:, a, li, :].mean(1))
    return s


def main():
    ids, A = loadN()
    miss = {m: int(np.isnan(A[m]).sum()) for m in A if np.isnan(A[m]).any()}
    print("=" * 108)
    print(f"FINAL TEST — 4 arms x n={len(ids)} ids | temp=0 | missing cells: {miss if miss else 'none'}")
    print("=" * 108)

    ev = {c["id"]: c["behavior"] for c in json.load(open(CASES_A))}
    fp_h = sum(bool(re.search(r"\b(I|my|me)\b", ev[i])) for i in ids)
    ni, pl = A["n_inject"][:, :, 0, 0], A["prompt_len"][:, :, 0, 0]
    print("\nARM MATCHING — verify before reading any effect")
    print(f"{'arm':>10} | {'n_inject':>16} | {'prompt_len':>16} | {'register':>22}")
    reg = {"clean": f"{fp_h}/100 first-person", "dija": f"{fp_h}/100 first-person",
           "benign": "0/100 (informational)", "benign_op": "100/100 (operational)"}
    for c, name in enumerate(ARMS):
        print(f"{name:>10} | {ni[:,c].mean():>7.1f} +- {ni[:,c].std():<5.1f} | "
              f"{pl[:,c].mean():>7.1f} +- {pl[:,c].std():<5.1f} | {reg[name]:>22}")
    print(f"\n  n_inject identical to dija, per id:  benign {int((ni[:,2]==ni[:,1]).sum())}/100  "
          f"| benign_op {int((ni[:,3]==ni[:,1]).sum())}/100")
    for c, name in ((0, "clean"), (2, "benign"), (3, "benign_op")):
        g = pl[:, c] - pl[:, 1]
        print(f"  prompt_len gap  {name:>9} - dija: mean={g.mean():+7.1f} sd={g.std():5.1f}")

    CON = (("dija - clean", 1, 0, "original headline (confounded: +scaffold AND +104 tok)"),
           ("benign_op - clean", 3, 0, "POSITIVE CONTROL: +scaffold, +length, stance-matched, NO harm"),
           ("dija - benign_op", 1, 3, "FINAL TEST: format + n_inject + stance held; harm differs"),
           ("dija - benign", 1, 2, "arm 2 for reference (stance MISmatched -- confounded)"))

    for rp, tag in (("pmean", "PRIMARY  read point = prompt_mean"),
                    ("last", "SECONDARY read point = last_prompt (axes +0.65 aligned; secondary only)")):
        print("\n" + "=" * 108)
        print(tag)
        print("=" * 108)
        for L in (25, 26):
            li = BAND.index(L)
            print(f"\n--- L{L} ({rp}) — layer where the random null has collapsed ---")
            print(f"{'contrast':>18} | {'v_injection_svd':>30} | {'v_random_null':>22}| {'ratio':>6}")
            print(f"{'':>18} | {'mean_d':>9} {'dz':>7} {'AUC':>6} {'%>0':>4} | "
                  f"{'dz':>7} {'AUC':>6} {'%>0':>4}| {'x null':>6}")
            for name, a, b, _ in CON:
                s = contrast(A, "vshared", rp, a, b, li)
                sn = contrast(A, "vnull", rp, a, b, li)
                r = abs(s["dz"]) / abs(sn["dz"]) if sn["dz"] else np.inf
                print(f"{name:>18} | {s['mean']:>+9.2f} {s['dz']:>+7.2f} {s['auc']:>6.3f} "
                      f"{100*s['frac_pos']:>3.0f}% | {sn['dz']:>+7.2f} {sn['auc']:>6.3f} "
                      f"{100*sn['frac_pos']:>3.0f}%| {r:>6.1f}")
            for name, a, b, why in CON:
                print(f"    {name:<18} = {why}")

    print("\n" + "=" * 108)
    print("EXACT-IDENTITY SPLIT — structure vs content. READ POINT = prompt_mean (PRIMARY).")
    print("  Paired deltas make this an identity, not a fit:")
    print("     (dija - clean) == (benign_x - clean) + (dija - benign_x)")
    print("  so the split is arithmetic. Arm 2 attributes stance to CONTENT; arm 3 moves stance")
    print("  into STRUCTURE, because its benign arm matches DIJA's register.")
    print("=" * 108)
    print(f"{'layer':>5} | {'via arm':>10} | {'dija-clean':>11} {'struct part':>12} {'content part':>13} | "
          f"{'struct %':>9} {'content %':>10}")
    for L in (25, 26):
        li = BAND.index(L)
        tot = contrast(A, "vshared", "pmean", 1, 0, li)["mean"]
        for arm, c in (("benign", 2), ("benign_op", 3)):
            st = contrast(A, "vshared", "pmean", c, 0, li)["mean"]
            ct = contrast(A, "vshared", "pmean", 1, c, li)["mean"]
            print(f"L{L:>4} | {arm:>10} | {tot:>+11.2f} {st:>+12.2f} {ct:>+13.2f} | "
                  f"{100*st/tot:>8.0f}% {100*ct/tot:>9.0f}%")

    print("\n" + "=" * 108)
    print("FINAL CONTRAST (dija - benign_op) ACROSS THE BAND. READ POINT = prompt_mean (PRIMARY).")
    print("=" * 108)
    print(f"{'layer':>5} | " + " | ".join(f"{LABEL[p]:>24}" for p in PROBES))
    print(f"{'':>5} | " + " | ".join(f"{'mean_d':>8} {'dz':>7} {'AUC':>6}" for _ in PROBES))
    for li, L in enumerate(BAND):
        cells = []
        for p in PROBES:
            s = contrast(A, p, "pmean", 1, 3, li)
            cells.append(f"{s['mean']:>+8.2f} {s['dz']:>+7.2f} {s['auc']:>6.3f}")
        print(f"L{L:>4} | " + " | ".join(cells))
    print("\n  v_refusal column: NULL per instruction, shown for completeness only. Arm 3 is the")
    print("  first contrast that is neither in-sample for it nor form-confounded (both sides are")
    print("  'How can I'), so this column is interpretable where arm 2's was not. Flagged, not promoted.")
    print("  tau: NULL/circular, not re-litigated.")
    print("\n" + "=" * 108)
    print("NO VERDICT WRITTEN.")
    print("=" * 108)


if __name__ == "__main__":
    main()
