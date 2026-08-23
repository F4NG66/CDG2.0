#!/usr/bin/env python3
"""study1/remask_denoise.py — STAGE 1 intervention engine (GENERATE ONLY).

The forward (noising) process as a native defense operator in a discrete masked
diffusion LM: at a chosen denoising step k*, re-mask a set of already-committed
positions and let the model re-decide from the resulting partial state. This is
NOT a new denoiser — every RNG / schedule / confidence primitive is imported
UNEDITED from dija_attack/cdg_denoise.py; this module only (a) tracks per-position
commit steps, (b) injects a re-mask at k*, and (c) recomputes the transfer
schedule over the remaining steps so the re-masked positions actually refill.

Design invariants (leak-proof-by-construction, see STUDY1_WRITEUP.md):
  * PRE-INTERVENTION state is SHARED. Phase 1 (steps 0..k*-1) runs at pre_temp.
    With pre_temp=0 the trajectory is deterministic and bit-identical to the
    untouched baseline (arm E), so the state the intervention acts on is exactly
    the state the model reached on its own — no counterfactual smuggled in.
  * The seed is applied AT k*, so at pre_temp=0 it bites ONLY the post-k* refill.
    Different seeds => divergent refills FROM THE SAME pre-intervention state:
    that branch is the whole measurement.
  * get_num_transfer_tokens over the CURRENT mask set for the REMAINING steps
    guarantees every re-masked position is refilled by step `steps` (no leftover
    <mask> in the output). Fills-per-step is logged so step-budget starvation at a
    late k* is visible and never confused with determinism.

NO hidden-state capture. NO network. Storage is one jsonl of responses.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

EXP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIJA = os.path.join(EXP, "dija_attack")
for _p in (EXP, DIJA, os.path.dirname(os.path.abspath(__file__))):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# imported UNEDITED — these are the paper's own primitives.
from cdg_denoise import add_gumbel_noise, get_num_transfer_tokens, _confidence  # noqa: E402
from capture_union import _segment_char_offsets  # noqa: E402 (pure text/offset helper)
import span_locator  # noqa: E402


def seed_all(s):
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


@torch.no_grad()
def _one_step(runner, x, attn, k_transfer, temperature, mask_id, remask,
              commit_step, step_idx):
    """One denoising step, verbatim in structure with cdg_denoise.denoise's inner
    loop (fill_all_masks path), plus first-commit tracking. Mutates x in place."""
    mask_index = (x == mask_id)
    logits = runner.forward(x, attn)
    logits_noised = add_gumbel_noise(logits, temperature)
    x0 = torch.argmax(logits_noised, dim=-1)
    conf = _confidence(logits, x0, remask)
    x0 = torch.where(mask_index, x0, x)
    neg = torch.tensor(-np.inf, device=x.device, dtype=conf.dtype)
    confidence = torch.where(mask_index, conf, neg)

    transfer = torch.zeros_like(x0, dtype=torch.bool)
    if k_transfer > 0:
        _, sel = torch.topk(confidence[0], k=k_transfer)
        transfer[0, sel] = True

    newly = transfer[0] & (commit_step == -1)      # FIRST commit only
    commit_step[newly] = step_idx
    x[transfer] = x0[transfer]


@torch.no_grad()
def build_canvas(runner, scaffold, *, gen_length, mask_id):
    """Return (x, attn, (t0,t1), P, total, fillable0). Mirrors PaperRunner.generate."""
    ids, (t0, t1) = runner.build_inputs(scaffold)
    P = ids.shape[1]
    total = P + gen_length
    x = torch.full((1, total), mask_id, dtype=torch.long, device=ids.device)
    x[:, :P] = ids
    attn = torch.ones((1, total), dtype=torch.long, device=ids.device)
    fillable0 = (x == mask_id).clone()
    return x, attn, (t0, t1), P, total, fillable0


@torch.no_grad()
def run_denoise(runner, x, attn, *, steps, mask_id, temperature, remask,
                intervention=None):
    """Unified fill_all_masks denoise with optional re-mask intervention.

    intervention=None                         -> plain baseline (== run_dija), tracks commit.
    intervention=dict(kstar, positions, seed_post, post_temp)
        -> phase 1 (0..kstar-1) at `temperature`; at kstar re-mask `positions`,
           seed, recompute schedule over remaining steps, phase 2 at post_temp.

    Returns (x, commit_step[total] long, log dict). x is mutated in place.
    """
    total = x.shape[1]
    commit_step = torch.full((total,), -1, dtype=torch.long, device=x.device)
    fillable0 = (x == mask_id)
    ntt = get_num_transfer_tokens(fillable0, steps)          # baseline schedule

    if intervention is None:
        for i in range(steps):
            _one_step(runner, x, attn, int(ntt[0, i]), temperature, mask_id,
                      remask, commit_step, i)
        log = {"kstar": None, "masked_final": int((x == mask_id).sum())}
        return x, commit_step, log

    kstar = int(intervention["kstar"])
    positions = list(intervention["positions"])
    post_temp = float(intervention.get("post_temp", temperature))
    seed_post = intervention.get("seed_post", None)
    assert 0 < kstar < steps, f"kstar {kstar} must be in (0,{steps})"

    # phase 1 — shared pre-intervention trajectory
    for i in range(kstar):
        _one_step(runner, x, attn, int(ntt[0, i]), temperature, mask_id,
                  remask, commit_step, i)

    log = {"kstar": kstar, "remaining_steps": steps - kstar,
           "masked_before_remask": int((x == mask_id).sum())}

    # intervention — re-mask
    if positions:
        pos = torch.tensor(positions, device=x.device, dtype=torch.long)
        wiped = int((x[0, pos] != mask_id).sum())            # committed -> wiped
        x[0, pos] = mask_id
    else:
        wiped = 0
    log.update(n_remask=len(positions), n_wiped_committed=wiped,
               masked_after_remask=int((x == mask_id).sum()))

    # recompute schedule over CURRENT masks for the remaining steps
    remaining = steps - kstar
    cur_mask = (x == mask_id)
    ntt2 = get_num_transfer_tokens(cur_mask, remaining)
    log["refill_fills_per_step"] = [int(v) for v in ntt2[0].tolist()]
    log["masks_to_refill"] = int(cur_mask.sum())

    # phase 2 — seed applied HERE so pre_temp=0 makes it bite only the refill
    if seed_post is not None:
        seed_all(int(seed_post))
    for j in range(remaining):
        _one_step(runner, x, attn, int(ntt2[0, j]), post_temp, mask_id,
                  remask, commit_step, kstar + j)

    log["masked_final"] = int((x == mask_id).sum())          # must be 0
    return x, commit_step, log


def compose_response(runner, x, t0, t1, P):
    """run_dija's exact response composition: decode(scaffold blanks) + continuation."""
    tok = runner.tokenizer
    cont = tok.decode(x[0, P:].tolist(), skip_special_tokens=True)
    if t0 is not None:
        tpl = tok.decode(x[0, t0:t1].tolist(), skip_special_tokens=True)
        return (tpl + "\n" + cont).strip()
    return cont.strip()


def harm_positions(runner, x, t0, t1, P, total, fillable0, behavior):
    """Locate the harmful span on the composed response and map it to CANVAS
    positions. Also returns the scaffold-blank and continuation position sets.

    Reuses capture_union._segment_char_offsets + span_locator.locate_harm_span; the
    char-interval -> position mapping mirrors capture_union.map_span_to_commit so
    the span positions here are the same ones the k_commit analysis used. All
    indices returned are CANVAS indices (into x[0, :]).
    """
    tok = runner.tokenizer
    gen_start = t0 if t0 is not None else P
    a0, a1 = (t0 - gen_start, t1 - gen_start) if t0 is not None else (0, 0)
    b0 = P - gen_start

    scaffold_ids = x[0, t0:t1].tolist() if t0 is not None else []
    cont_ids = x[0, P:total].tolist()
    sc_text, sc_spans, _ = _segment_char_offsets(tok, scaffold_ids) if scaffold_ids else ("", [], False)
    ct_text, ct_spans, _ = _segment_char_offsets(tok, cont_ids)

    joined = sc_text + "\n" + ct_text
    lstrip_n = len(joined) - len(joined.lstrip())
    response = joined.strip()
    cont_shift = len(sc_text) + 1

    # (char_start, char_end, canvas_index) for every generated token
    pos_intervals = []
    for k, (c0, c1) in enumerate(sc_spans):
        pos_intervals.append((c0 - lstrip_n, c1 - lstrip_n, gen_start + a0 + k))
    for k, (c0, c1) in enumerate(ct_spans):
        pos_intervals.append((c0 + cont_shift - lstrip_n, c1 + cont_shift - lstrip_n,
                              gen_start + b0 + k))

    res = span_locator.locate_harm_span(response, True, judge_row={"behavior": behavior})
    char_spans = res["char_spans"]
    span_canvas = []
    for (s, e) in char_spans:
        for (c0, c1, g) in pos_intervals:
            if c1 > s and c0 < e and bool(fillable0[0, g]):
                span_canvas.append(g)
    span_canvas = sorted(set(span_canvas))

    scaffold_canvas = [g for g in range(t0, t1) if bool(fillable0[0, g])] if t0 is not None else []
    cont_canvas = [g for g in range(P, total) if bool(fillable0[0, g])]

    span_text = response[char_spans[0][0]:char_spans[0][1]] if char_spans else ""
    return {
        "response": response, "span_text": span_text,
        "span_canvas": span_canvas,
        "scaffold_canvas": scaffold_canvas,
        "cont_canvas": cont_canvas,
        "char_span": char_spans[0] if char_spans else None,
    }
