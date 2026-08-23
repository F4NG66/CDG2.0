#!/usr/bin/env python3
"""study1/stage1_remask.py — STAGE 1 falsification test, CORRECTED design (GEN ONLY).

Re-masking as a native defense operator in a discrete masked diffusion LM. Two
passes: PASS 1 (arm E, temp=0) = plain baseline -> commit_step + located harm span,
defining the target positions and the two BUDGET-FLOORED k*; PASS 2 (arms A/C/D,
temp=0) = re-mask + refill from the SHARED deterministic pre-intervention state.

Corrected after the 2026-07-23 smoke (see STUDY1_WRITEUP Stage-1 section):
  * Target = the SCAFFOLD-BLANK worksheet (where DIJA harm is written), not the
    span_locator span (which sweeps in the trailing refusal).
  * Arm B DROPPED — for DIJA the harm span already contains the scaffold blanks, so
    A==B. Recorded instead as the STRUCTURAL finding that the harmful template
    (Procedure:/Preparation:/... + the "highly dangerous" lines) is FIXED PROMPT
    text, never maskable, so no output-region re-mask reaches the determinant.
  * k* BUDGET-FLOORED to leave >= --budget-floor steps: a naive max(span-commit)
    late re-mask leaves ~1 refill step and DEGENERATES, which a judge misreads as
    "neutralized". Every incl=False must be cross-checked vs the degeneration score.

ARMS:
  A  re-mask the worksheet blanks (harm site).
  C  matched # of non-harm positions, worksheet intact (a FLOOR; A~=C is the read).
  D  wipe the worksheet AND relocate an equal blank count into free output space;
     judge overall + read WHERE harm landed (original slots vs relocated blanks).
  E  baseline (pass 1).

BRANCH (--branch): arm A only, worksheet target, K seeds, pre_temp=0 (shared state)
+ post_temp>0, at the late budget-floored k*. Does the refill BRANCH from the same
committed pre-intervention state? (Straddler-manufacture route; leak-proof.)

No hidden capture. No network. Judge = study1/judge_samples.py.

  python study1/stage1_remask.py --limit 10                    # RUN 1
  python study1/stage1_remask.py --limit 10 --branch --branch-k 8   # RUN 2
"""
from __future__ import annotations
import argparse
import json
import os
import re
import sys

import torch

EXP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIJA = os.path.join(EXP, "dija_attack")
HERE = os.path.dirname(os.path.abspath(__file__))
for p in (EXP, DIJA, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

from run_dija import PaperRunner, MODEL_ID, MASK_ID, MASK_MARKER_RE  # noqa: E402
from remask_denoise import (  # noqa: E402
    build_canvas, run_denoise, compose_response, harm_positions,
)

STEPS = 128
GEN_LENGTH = 128
REMASK = "low_confidence"
ARMS = ["A", "C", "D"]


def arm_positions(arm, hp):
    """Return (canvas positions to re-mask, description, extra). See docstring."""
    scaffold = list(hp["scaffold_canvas"])          # worksheet = harm site
    span = list(hp["span_canvas"])
    cont = list(hp["cont_canvas"])
    spanset, scafset = set(span), set(scaffold)
    n = len(scaffold)
    if arm == "A":
        return sorted(scafset), "worksheet blanks (harm site)", {}
    if arm == "C":
        pool = [g for g in cont if g not in spanset and g not in scafset]
        pick = sorted(pool)[-n:] if len(pool) >= n else sorted(pool)
        return sorted(pick), f"matched non-harm floor (n={len(pick)}/{n})", {}
    if arm == "D":
        reloc = [g for g in cont if g not in spanset and g not in scafset]
        reloc = sorted(reloc)[:n]
        pos = sorted(scafset | set(reloc))
        return pos, (f"wipe worksheet ({n}) + relocate {len(reloc)} blanks -> free output"), \
            {"scaffold": sorted(scafset), "relocated": reloc, "n_target": n}
    raise ValueError(arm)


def make_runner(device):
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True,
                                      torch_dtype=torch.bfloat16).to(device).eval()
    return PaperRunner(model, tok, device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refined", default=os.path.join(DIJA, "refined_100.json"))
    ap.add_argument("--out", default="/scratch/ore99/study1_remask_v2")
    ap.add_argument("--ids", default="")
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--budget-floor", type=int, default=20, help="min remaining steps at any k*")
    ap.add_argument("--k-early-frac", type=float, default=0.5)
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--branch", action="store_true")
    ap.add_argument("--branch-k", type=int, default=8)
    ap.add_argument("--branch-seed0", type=int, default=1000)
    ap.add_argument("--branch-temp", type=float, default=0.2)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    arms = [a.strip().upper() for a in args.arms.split(",") if a.strip()]
    os.makedirs(args.out, exist_ok=True)
    outp = os.path.join(args.out, "samples.jsonl")

    done = set()
    if os.path.exists(outp):
        for line in open(outp):
            line = line.strip()
            if line:
                try:
                    done.add(json.loads(line)["sid"])
                except Exception:
                    pass
    print(f"[stage1] {len(done)} samples on disk -> resuming | budget_floor={args.budget_floor}",
          flush=True)

    byid = {r["id"]: r for r in json.load(open(args.refined))}
    all_rows = list(byid.values())
    if args.ids:
        want = [s.strip() for s in args.ids.split(",") if s.strip()]
        missing = [i for i in want if i not in byid]
        assert not missing, f"unknown A-ids: {missing}"
        rows = [byid[i] for i in want]
    else:
        rows = all_rows[: args.limit]

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"[stage1] loading {MODEL_ID} on {device} (bf16) ...", flush=True)
    runner = make_runner(device)
    tok = runner.tokenizer
    print(f"[stage1] GEN-ONLY steps={STEPS} arms={arms} branch={args.branch} "
          f"ids={[r['id'] for r in rows]}", flush=True)

    fout = open(outp, "a", encoding="utf-8")

    def emit(rec):
        if rec["sid"] in done:
            return
        rec["n_chars"] = len(rec["response"])
        rec["n_tokens"] = len(tok(rec["response"], add_special_tokens=False)["input_ids"])
        fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
        fout.flush()
        done.add(rec["sid"])

    for r in rows:
        cid = r["id"]
        scaffold = r["Refined_behavior"]
        behavior = r.get("behavior", r.get("Behavior", ""))
        n_expected = sum(int(x) for x in re.findall(MASK_MARKER_RE, scaffold))

        # PASS 1 baseline
        x, attn, (t0, t1), P, total, f0 = build_canvas(
            runner, scaffold, gen_length=GEN_LENGTH, mask_id=MASK_ID)
        x, commit, _ = run_denoise(runner, x, attn, steps=STEPS, mask_id=MASK_ID,
                                   temperature=0.0, remask=REMASK, intervention=None)
        hp = harm_positions(runner, x, t0, t1, P, total, f0, behavior)
        base_resp = hp["response"]
        scaffold_pos = hp["scaffold_canvas"]

        emit({"sid": f"{cid}__E__baseline", "id": cid, "case": cid, "arm": "E",
              "kstar_label": "baseline", "kstar": None, "temperature": 0.0, "seed": None,
              "behavior": behavior, "response": base_resp, "n_inject": n_expected,
              "n_remask": 0, "n_wiped": 0, "remaining_steps": None, "stage": "stage1_remask"})

        if not scaffold_pos:
            print(f"[{cid}] no scaffold-blank positions -> skip", flush=True)
            continue

        sc_commits = [int(commit[g]) for g in scaffold_pos if commit[g] >= 0]
        k_late = min(max(sc_commits) + 1, STEPS - args.budget_floor)
        k_early = max(1, min(int(args.k_early_frac * k_late), k_late - args.budget_floor))
        kstars = [("early", k_early), ("late", k_late)]
        print(f"\n[{cid}] worksheet={len(scaffold_pos)} span={len(hp['span_canvas'])} | "
              f"sc_commit min/med/max="
              f"{min(sc_commits)}/{sorted(sc_commits)[len(sc_commits)//2]}/{max(sc_commits)} | "
              f"k_early={k_early}(rem {STEPS-k_early}) k_late={k_late}(rem {STEPS-k_late})",
              flush=True)

        if not args.branch:
            for arm in arms:
                positions, desc, extra = arm_positions(arm, hp)
                for klab, kstar in kstars:
                    x2, a2, _, _, _, _ = build_canvas(
                        runner, scaffold, gen_length=GEN_LENGTH, mask_id=MASK_ID)
                    x2, _, log = run_denoise(
                        runner, x2, a2, steps=STEPS, mask_id=MASK_ID, temperature=0.0,
                        remask=REMASK, intervention={"kstar": kstar, "positions": positions,
                                                     "seed_post": None, "post_temp": 0.0})
                    resp = compose_response(runner, x2, t0, t1, P)
                    rec = {"sid": f"{cid}__{arm}__{klab}", "id": cid, "case": cid, "arm": arm,
                           "kstar_label": klab, "kstar": kstar, "temperature": 0.0, "seed": None,
                           "behavior": behavior, "response": resp, "n_inject": n_expected,
                           "n_remask": log["n_remask"], "n_wiped": log["n_wiped_committed"],
                           "remaining_steps": log["remaining_steps"], "stage": "stage1_remask"}
                    if arm == "D":
                        rec["d_scaffold_text"] = tok.decode(
                            x2[0, t0:t1].tolist(), skip_special_tokens=True)
                        rec["d_relocated_text"] = tok.decode(
                            [int(x2[0, g]) for g in extra["relocated"]], skip_special_tokens=True)
                    emit(rec)
                    print(f"  [{cid}/{arm}/{klab}] k*={kstar} rem={log['remaining_steps']} "
                          f"wiped={log['n_wiped_committed']}/{log['n_remask']} "
                          f"resp[:70]={resp[:70]!r}", flush=True)

        # BRANCH: arm A, worksheet target, K seeds at the LATE budget-floored k*
        if args.branch:
            positions, _, _ = arm_positions("A", hp)
            klab, kstar = "late", k_late
            for s in range(args.branch_k):
                seed = args.branch_seed0 + s
                x3, a3, _, _, _, _ = build_canvas(
                    runner, scaffold, gen_length=GEN_LENGTH, mask_id=MASK_ID)
                x3, _, log = run_denoise(
                    runner, x3, a3, steps=STEPS, mask_id=MASK_ID, temperature=0.0,
                    remask=REMASK, intervention={"kstar": kstar, "positions": positions,
                                                 "seed_post": seed, "post_temp": args.branch_temp})
                resp = compose_response(runner, x3, t0, t1, P)
                emit({"sid": f"{cid}__A__{klab}__s{seed}", "id": cid, "case": cid, "arm": "A",
                      "kstar_label": klab, "kstar": kstar, "temperature": args.branch_temp,
                      "seed": seed, "behavior": behavior, "response": resp, "n_inject": n_expected,
                      "n_remask": log["n_remask"], "n_wiped": log["n_wiped_committed"],
                      "remaining_steps": log["remaining_steps"], "stage": "stage1_remask_branch"})
            print(f"  [{cid}/A/{klab}] branch K={args.branch_k} @T={args.branch_temp} k*={kstar} done",
                  flush=True)

    fout.close()
    print(f"\n[stage1] done -> {outp}", flush=True)


if __name__ == "__main__":
    main()
