#!/usr/bin/env python
"""attack2/run_attack2_resumable.py

RESUMABLE, crash-safe copy of run_record.py for the attack2 harm-delivered run.
This is a COPY of crossattack/build/capture_attack2_resumable.py (task rule:
reuse-by-copying, never edit originals) with only the ROOT path and default --out
adjusted for its new location under attack2/. It imports the cdg.* library
UNCHANGED and only adds:

  1. RESUME  — before running a case, compute its predicted .pt path (exactly
               recorder.save()'s path) and SKIP it if it already exists on disk.
               A re-run after a walltime kill resumes from wherever it stopped.
  2. PROGRESS — flushed per-case prints so a tee'd log shows live progress.
  3. --dry-run — print the skip/run plan and exit WITHOUT loading the model/GPU.

Incremental durability comes from cdg.recorder.Recorder.save(): each case's .pt
is torch.save()d and its manifest row APPENDED per case, so every completed case
is durable the instant it finishes.

No judge here — generation only. Harm is scored separately with the HarmBench
DeepSeek judge (see attack2/make_judge_input.py + score_harmbench_deepseek.py).

Writes ONLY under --out (attack2/outputs). Run from repo root:
    /home/ore99/env_llada/bin/python attack2/run_attack2_resumable.py \
        --prompt-root attack2/prompts/attack2_b --out attack2/outputs \
        --sae-root ./saes --backend llada_attack --seeds 0 --device cuda
"""
from __future__ import annotations
import argparse
import os
import sys

import torch

# repo root = parent of attack2/  (attack2/run_attack2_resumable.py -> serverFiles)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from cdg.config import get_backend_config
from cdg.backends import build_runner
from cdg.recorder import Recorder
from cdg.data import load_cdg_root


def predicted_pt_path(out_dir: str, model_name: str, variant: str,
                      case_id: str, seed: int) -> str:
    """Mirror recorder.save(): out_dir/model_name/variant/{case_id}__seedN.pt."""
    return os.path.join(out_dir, model_name, variant, f"{case_id}__seed{seed}.pt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend",
                    choices=["llada", "llada_attack", "dream", "dream_attack"],
                    default="llada_attack")
    ap.add_argument("--prompt-root", required=True,
                    help="dir with group subfolders (attack2: B2_harmful_injected)")
    ap.add_argument("--sae-root", default="", help="root holding the SAE repos")
    ap.add_argument("--out", default="attack2/outputs")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--dummy", action="store_true")
    ap.add_argument("--limit", type=int, default=0,
                    help="cap cases per group (0 = all)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the resume plan (done vs remaining) and exit; "
                         "does NOT load the model or touch the GPU")
    args = ap.parse_args()

    cfg = get_backend_config(args.backend)
    model_name = cfg.name  # "llada"

    # -- load cases + optional per-group limit (identical to run_record.py) ----
    cases = load_cdg_root(args.prompt_root)
    if args.limit:
        seen: dict[str, int] = {}
        kept = []
        for c in cases:
            g = c.group_letter
            seen[g] = seen.get(g, 0) + 1
            if seen[g] <= args.limit:
                kept.append(c)
        cases = kept

    # -- RESUME: partition into already-done vs remaining, per (seed, case) -----
    work: list = []          # (seed, case) still to run
    done: list = []          # (seed, case) already on disk
    for seed in args.seeds:
        for case in cases:
            p = predicted_pt_path(args.out, model_name, case.variant,
                                  case.case_id, seed)
            (done if os.path.exists(p) else work).append((seed, case))

    def _by_group(items):
        d: dict[str, int] = {}
        for _s, c in items:
            d[c.group_letter] = d.get(c.group_letter, 0) + 1
        return "  ".join(f"{g}={d.get(g, 0)}" for g in sorted(d))

    total = len(cases) * len(args.seeds)
    print(f"[resume] out={args.out}  model={model_name}  seeds={args.seeds}", flush=True)
    print(f"[resume] total work items : {total}", flush=True)
    print(f"[resume] already captured : {len(done)}   ({_by_group(done)})", flush=True)
    print(f"[resume] remaining to run : {len(work)}   ({_by_group(work)})", flush=True)
    remaining_ids = [c.case_id for _s, c in work]
    print(f"[resume] remaining ids    : {remaining_ids[:6]}"
          f"{' ...' if len(remaining_ids) > 6 else ''}"
          f"  (last: {remaining_ids[-1] if remaining_ids else 'none'})", flush=True)

    if args.dry_run:
        print("[dry-run] no model loaded, nothing captured. Exiting.", flush=True)
        return
    if not work:
        print("[done] nothing to do — all cases already captured.", flush=True)
        return

    # -- build model/runner (GPU) ---------------------------------------------
    runner = build_runner(cfg, sae_root=args.sae_root, device=args.device,
                          dummy=args.dummy)
    tok = getattr(runner, "tokenizer", None)

    # -- capture loop (per case: generate -> save .pt + append manifest row) ---
    counts: dict[str, int] = {}
    n = 0
    last_seed = None
    for seed, case in work:
        if seed != last_seed:
            torch.manual_seed(seed)     # match run_record.py's per-seed seeding
            last_seed = seed
        n += 1
        print(f"[{n}/{len(work)}] RUN {case.case_id} ({case.variant}) seed={seed}",
              flush=True)

        rec = Recorder(runner.bundles, cfg.record, tokenizer=tok)
        meta = {
            "case_id": case.case_id, "variant": case.variant,
            "content_type": case.content_type, "has_template": case.has_template,
            "attack_method": case.attack_method, "is_neutral": case.is_neutral,
            "model_name": cfg.name, "seed": seed,
            "behavior": case.behavior,
            "saes": [b.name for b in runner.bundles],
            "record_layers": list(cfg.record_layers),
            "gen_length": cfg.decode.gen_length, "steps": cfg.decode.steps,
            "case_meta": case.meta,
        }
        rec.begin(meta, total_steps=cfg.decode.steps)
        resp = runner.generate(case, rec)
        path = rec.save(args.out)      # incremental: writes .pt + appends manifest row
        counts[case.group_letter] = counts.get(case.group_letter, 0) + 1
        print(f"[{n}/{len(work)}] SAVED {path}  (resp_len={len(resp)})", flush=True)

    summary = "  ".join(f"{g}={counts.get(g, 0)}" for g in ("A", "B", "C", "D"))
    print(f"[done] captured this run: {summary}  ->  {args.out}/manifest.jsonl",
          flush=True)


if __name__ == "__main__":
    main()
