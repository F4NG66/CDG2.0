#!/usr/bin/env python3
"""study1/straddle_pilot.py — PART B: straddler yield pilot (GENERATE ONLY).

Goal: find A-ids where the SAME prompt (byte-identical scaffold, byte-identical
<mask:N> geometry) yields DIFFERENT harm outcomes. That is the only contrast in
which prompt text is held fixed and outcome varies, so it is the only design that
can give an honest decision-level k_detect. The temp=0 expansion could not: with
one generation per prompt, outcome is perfectly collinear with prompt text (see
the arm-placebo control, AUC 1.000 at step 0).

NO HIDDEN-STATE CAPTURE. Storage is one jsonl of responses (~KB). No network:
judging is a separate non-GPU process afterwards.

Two knobs are exercised, in this order of preference:
  (a) SEED at temperature=0 — keeps the deterministic sampling regime the rest of
      the study uses. Code reading of dija_attack/cdg_denoise.py says this is a
      NO-OP: add_gumbel_noise short-circuits at temperature==0 and returns logits
      unchanged (no torch.rand call), _confidence("low_confidence") is a pure
      softmax+gather, and top-k selection consumes no RNG. So nothing in the
      temp=0 / low_confidence path draws from the generator. `--det-check` runs
      the empirical test anyway (same prompt, different seeds, compare token ids),
      because kernel-level nondeterminism is not visible in the source.
  (b) temperature=0.2 — the paper's own attack config (llada_attack_config), the
      value the DIJA headline was produced at. Gumbel noise is live here, so the
      seed does bite.

  python study1/straddle_pilot.py --n-ids 15 --k 8 --out /scratch/ore99/study1_straddle
"""
from __future__ import annotations
import argparse
import json
import os
import re
import sys
import time

import torch

EXP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIJA = os.path.join(EXP, "dija_attack")
for p in (EXP, DIJA):
    if p not in sys.path:
        sys.path.insert(0, p)

from run_dija import PaperRunner, MODEL_ID, MASK_ID, MASK_MARKER_RE  # noqa: E402

STEPS = 128
GEN_LENGTH = 128
BLOCK_LENGTH = 128
REMASK = "low_confidence"


def seed_all(s):
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refined", default=os.path.join(DIJA, "refined_100.json"))
    ap.add_argument("--out", default="/scratch/ore99/study1_straddle")
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--n-ids", type=int, default=15)
    ap.add_argument("--ids", default="",
                    help="explicit comma-separated A-ids; overrides --n-ids")
    ap.add_argument("--k", type=int, default=8, help="samples per A-id at temp>0")
    ap.add_argument("--temp", type=float, default=0.2)
    ap.add_argument("--temps", default="",
                    help="comma-separated temperatures; overrides --temp")
    ap.add_argument("--ks", default="",
                    help="comma-separated K, one per --temps entry; overrides --k")
    ap.add_argument("--det-check", type=int, default=2,
                    help="temp=0 samples per A-id with DIFFERENT seeds (determinism probe)")
    ap.add_argument("--det-check-ids", type=int, default=5,
                    help="run the temp=0 determinism probe on the first N A-ids only")
    ap.add_argument("--seed0", type=int, default=1000)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

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
    print(f"[run] {len(done)} samples already on disk -> resuming", flush=True)

    all_rows = json.load(open(args.refined))
    if args.ids:
        want = [s.strip() for s in args.ids.split(",") if s.strip()]
        byid = {r["id"]: r for r in all_rows}
        missing = [i for i in want if i not in byid]
        assert not missing, f"unknown A-ids: {missing}"
        rows = [byid[i] for i in want]
    else:
        rows = all_rows[: args.n_ids]

    # (temperature, K) schedule. Sids embed the temperature, so mixing temps in one
    # store is safe and the resume set keeps prior temp=0.2 samples from being redone.
    if args.temps:
        temps = [float(t) for t in args.temps.split(",") if t.strip()]
        ks = ([int(k) for k in args.ks.split(",")] if args.ks
              else [args.k] * len(temps))
        assert len(ks) == len(temps), "--ks must have one entry per --temps"
        sched = list(zip(temps, ks))
    else:
        sched = [(args.temp, args.k)]
    print(f"[run] ids={[r['id'] for r in rows]}", flush=True)
    print(f"[run] schedule (temp, K) = {sched}", flush=True)

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"[run] loading {args.model} on {device} (bf16) ...", flush=True)
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModel.from_pretrained(args.model, trust_remote_code=True,
                                      torch_dtype=torch.bfloat16).to(device).eval()
    runner = PaperRunner(model, tok, device)
    print(f"[run] GENERATE-ONLY (no hidden capture). steps={STEPS} gen={GEN_LENGTH} "
          f"remask={REMASK} fill_all_masks=True", flush=True)

    fout = open(outp, "a", encoding="utf-8")
    n_new = 0
    for ri, r in enumerate(rows):
        cid = r["id"]
        scaffold = r["Refined_behavior"]
        n_expected = sum(int(x) for x in re.findall(MASK_MARKER_RE, scaffold))

        plan = []
        if ri < args.det_check_ids:
            for j in range(args.det_check):
                plan.append((0.0, args.seed0 + j, f"{cid}__t0_s{args.seed0 + j}"))
        for temperature, K in sched:
            for j in range(K):
                plan.append((temperature, args.seed0 + j,
                             f"{cid}__t{temperature}_s{args.seed0 + j}"))

        for temperature, seed, sid in plan:
            if sid in done:
                continue
            seed_all(seed)
            t = time.time()
            resp, n_inject = runner.generate(
                scaffold, steps=STEPS, gen_length=GEN_LENGTH,
                block_length=BLOCK_LENGTH, temperature=temperature,
                remask=REMASK, mask_id=MASK_ID)
            dt = time.time() - t
            rec = {
                "sid": sid, "id": cid, "arm": "dija",
                "temperature": temperature, "seed": seed,
                "behavior": r.get("behavior", r.get("Behavior", "")),
                "response": resp,
                "n_inject": int(n_inject), "n_expected_blank": n_expected,
                "n_chars": len(resp), "n_tokens": len(tok(resp,
                                                          add_special_tokens=False)["input_ids"]),
                "gen_seconds": round(dt, 2),
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
            n_new += 1
            print(f"[gen] {sid} T={temperature} {dt:.1f}s chars={len(resp)} "
                  f"tok={rec['n_tokens']} inject={n_inject}", flush=True)
    fout.close()
    print(f"[done] {n_new} new samples -> {outp}", flush=True)


if __name__ == "__main__":
    main()
