#!/usr/bin/env python3
"""study1/capture_union.py — ONE recapture schema serving Studies 1/2/3.

Extends the Shield collector (ladaAndH.StepActivationCollector, pool="none") onto
clockv2's UNIFIED fill_all_masks denoising schedule (common.capture_denoise), and
adds the two fields the scalar A-set capture threw away:

  * reply_role_mask[n_steps, region_len] bool  — True = position is [MASK] (out_mask),
    False = already decoded (out_unmask), read BEFORE the forward at each step.
    Pooling is done DOWNSTREAM from this; the canvas is NEVER mean-pooled at
    capture time (that was the original scalar bug that mixed out_mask/out_unmask).
  * commit_step[region_len] int16 — per reply position, k_i = min{k : i leaves M(x_tk)},
    the denoising step at which the token is fixed. Makes span-level k_commit
    recoverable.

Why fill_all_masks and not the Shield block schedule: the dija / benign_op arms
carry prompt-embedded <mask:N> blanks that only fill under the unified schedule
(clockv2 README), and the existing A-set scalar data was captured under it — so
all three arms share ONE schedule. block_length is irrelevant here.

REGION COVERAGE (dija arm): the DIJA harm forms in the injected worksheet blanks,
which live INSIDE the prompt at token span [t0, t1) (before the free-continuation
start P), plus optionally the free continuation [P, total). run_dija composes its
judged response as decode(x[t0:t1]) + "\n" + decode(x[P:]). So for the dija arm we
start capture at gen_start = t0 (NOT P), and the captured region is [t0, total):

    [t0, t1)      scaffold  — mix of injected blanks (fillable) + literal worksheet text
    [t1, P)       prompt_tail — fixed chat-template tokens, never masked
    [P,  total)   continuation — free output masks (all fillable)

per-captured-position labels are exported so downstream can select the harm region:
  region_id  : 0=scaffold, 1=prompt_tail, 2=continuation
  fillable_reply_mask : True iff the position was EVER a [MASK] (=> it was generated).
  scaffold-blank positions = (region_id==0 & fillable);  free-continuation = (region_id==2).
For the clean arm t0=t1=P and the region is just the continuation (all fillable),
identical to the original schema.

Schedule body is copied verbatim from clockv2/common.capture_denoise (temp=0,
low_confidence remask) with capture added; NO activation is modified. NOTE: this is
the STUDY engine (temp=0, deterministic) — the paper's dija headline used temp=0.2.
We measure the yield of *our own* temp=0 capture pipeline, which is what Study 1
will actually store.

    # offline logic check (no model):
    python study1/capture_union.py --self-test
    # clean smoke:
    python study1/capture_union.py --limit 2 --arm clean --out /scratch/ore99/study1_smoke
    # yield pilot (loads model): dija arm on 10 A-cases + span->commit dry run:
    python study1/capture_union.py --arm dija --limit 10 --out /scratch/ore99/study1_pilot
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

import torch
import torch.nn.functional as F

EXP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if EXP not in sys.path:
    sys.path.insert(0, EXP)

from ladaAndH import (  # noqa: E402
    StepActivationCollector, discover_blocks, get_num_transfer_tokens,
    MASK_ID, VOCAB, D_MODEL,
)

sys.path.insert(0, os.path.join(EXP, "dija_attack"))
from cdg_denoise import add_gumbel_noise as _add_gumbel_noise  # noqa: E402 (imported unedited)

MODEL_ID = "GSAI-ML/LLaDA-8B-Instruct"
DEFAULT_LAYERS = [16, 25, 26, 27, 31]
STEPS = 128
GEN_LENGTH = 128
TOPK = 5
SCHEMA_VERSION = "study1.union.v2"          # v2: region-aware (scaffold coverage)

# region_id codes
R_SCAFFOLD, R_PROMPT_TAIL, R_CONTINUATION = 0, 1, 2


@torch.no_grad()
def capture_union(model, blocks, ids, layer_ids, *, steps=STEPS, gen_length=GEN_LENGTH,
                  mask_id=MASK_ID, topk=TOPK, t_span=None, temperature=0.0):
    """Unified-schedule denoising with FULL reply-region capture. READ-ONLY.

    ids         : [1, P] prompt token ids (may contain embedded <mask> blanks for dija).
    t_span      : (t0, t1) scaffold token span inside the prompt for the dija arm, or
                  None for the clean arm (=> capture the continuation only, from P).
    temperature : 0.0 keeps the original deterministic schedule bit-for-bit
                  (cdg_denoise.add_gumbel_noise short-circuits and draws NO RNG).
                  temperature > 0 reproduces run_dija's sampling EXACTLY: the same
                  `add_gumbel_noise` on the same full-logits tensor, in the same
                  position in the loop, so the RNG stream is consumed identically
                  and a seeded capture run matches the seeded generate-only run
                  token for token. Nothing else in this function draws RNG, so the
                  capture hooks cannot perturb the sample.

    Returns (x_final, payload). See module docstring for payload fields.
    """
    device = model.device
    P = ids.shape[1]
    total = P + gen_length
    x = torch.full((1, total), mask_id, dtype=torch.long, device=device)
    x[:, :P] = ids
    attn = torch.ones((1, total), dtype=torch.long, device=device)

    fillable = (x == mask_id)                              # scaffold blanks + continuation
    n_fillable = int(fillable.sum().item())
    n_inject = int((ids == mask_id).sum().item())          # prompt blanks (0 for clean)
    ntt = get_num_transfer_tokens(fillable, steps)         # over ALL masks (unified)

    if t_span is not None and t_span[0] is not None:
        t0, t1 = int(t_span[0]), int(t_span[1])
    else:
        t0, t1 = P, P
    gen_start = t0                                         # dija: scaffold start; clean: P
    region_len = total - gen_start

    # per-captured-position region label
    region_id = torch.empty(region_len, dtype=torch.int8)
    for local in range(region_len):
        g = gen_start + local
        if g < t1:
            region_id[local] = R_SCAFFOLD
        elif g < P:
            region_id[local] = R_PROMPT_TAIL
        else:
            region_id[local] = R_CONTINUATION

    sub_blocks = [blocks[l] for l in layer_ids]
    collector = StepActivationCollector(sub_blocks, layer_ids, gen_start=gen_start, pool="none")
    collector.new_turn()

    ent_steps, conf_steps, tp_steps, ti_steps, role_steps = [], [], [], [], []
    mask_ratio_reply = []
    commit_step = torch.full((region_len,), -1, dtype=torch.long)   # k_i per reply pos

    for i in range(steps):
        mask_index = (x == mask_id)
        role = (x[0, gen_start:] == mask_id)               # True=out_mask, BEFORE forward
        role_steps.append(role.detach().cpu().clone())
        mask_ratio_reply.append(float(role.sum().item()) / max(1, n_fillable))

        logits = model(x, attention_mask=attn).logits      # [1, seq, vocab]
        collector.mark_step()                              # hidden snapshot (reply span)

        gen_logits = logits[0, gen_start:].float()         # [region_len, vocab]
        probs = F.softmax(gen_logits, dim=-1)
        ent = -(probs * probs.clamp_min(1e-12).log()).sum(-1)
        tk_p, tk_i = torch.topk(probs, k=topk, dim=-1)
        conf = probs.max(dim=-1).values
        ent_steps.append(ent.half().cpu())
        conf_steps.append(conf.half().cpu())
        tp_steps.append(tk_p.half().cpu())
        ti_steps.append(tk_i.int().cpu())

        # ---- LLaDA token-transfer, low_confidence (verbatim schedule) ----
        # add_gumbel_noise is imported unedited from dija_attack/cdg_denoise.py and is a
        # no-op at temperature==0 (returns logits, draws no RNG). Confidence is gathered
        # from the RAW logits, exactly as cdg_denoise._confidence("low_confidence") does.
        x0 = torch.argmax(_add_gumbel_noise(logits, temperature), dim=-1)
        p = F.softmax(logits.to(torch.float64), dim=-1)
        confg = torch.gather(p, -1, x0.unsqueeze(-1)).squeeze(-1)
        x0 = torch.where(mask_index, x0, x)
        neg = torch.tensor(float("-inf"), device=device, dtype=confg.dtype)
        confidence = torch.where(mask_index, confg, neg)

        transfer = torch.zeros_like(x0, dtype=torch.bool)
        for j in range(confidence.shape[0]):
            k = int(ntt[j, i])
            if k > 0:
                _, sel = torch.topk(confidence[j], k=k)
                transfer[j, sel] = True

        reply_transfer = transfer[0, gen_start:].detach().cpu()
        newly = reply_transfer & (commit_step == -1)
        commit_step[newly] = i                             # k_i = this step

        x[transfer] = x0[transfer]

    hidden = collector.stack()                             # [n_steps,n_layers,region_len,d]
    collector.close()

    fillable_reply_mask = role_steps[0].clone()            # ever-masked within region

    payload = {
        "schema_version": SCHEMA_VERSION,
        "layers": list(layer_ids),
        "hidden": hidden,                                  # fp16
        "reply_role_mask": torch.stack(role_steps),        # [n_steps,region_len] bool
        "commit_step": commit_step.to(torch.int16),        # [region_len] int16 (-1 if never masked)
        "fillable_reply_mask": fillable_reply_mask,        # [region_len] bool
        "region_id": region_id,                            # [region_len] int8
        "entropy": torch.stack(ent_steps),                 # [n_steps,region_len] fp16
        "confidence": torch.stack(conf_steps),             # [n_steps,region_len] fp16
        "topk_probs": torch.stack(tp_steps),               # [n_steps,region_len,K] fp16
        "topk_ids": torch.stack(ti_steps),                 # [n_steps,region_len,K] int32
        "final_ids": x[0, gen_start:].detach().cpu(),      # [region_len] int64
        "mask_ratio_reply": mask_ratio_reply,              # list[float] len steps
        "meta": {
            "prompt_len": P, "gen_start": gen_start, "region_len": region_len,
            "t0": t0, "t1": t1, "total": total,
            "n_inject": n_inject, "n_fillable": n_fillable,
            "schedule": "fill_all_masks", "steps": steps, "gen_length": gen_length,
            "temperature": float(temperature), "topk": topk,
            "vocab": VOCAB, "d_model": D_MODEL,
        },
    }
    return x, payload


def verify_payload(payload, steps=STEPS) -> list[str]:
    """Return a list of invariant-violation strings (empty == all invariants hold)."""
    errs = []
    role = payload["reply_role_mask"]            # [S, region_len] bool
    commit = payload["commit_step"].long()       # [region_len]
    fillable = payload["fillable_reply_mask"]    # [region_len] bool
    S, G = role.shape

    # a fillable (ever-masked) position MUST commit; a never-masked position stays -1
    if int((fillable & (commit < 0)).sum()) > 0:
        errs.append(f"{int((fillable & (commit < 0)).sum())} fillable positions never committed")
    non_fillable_committed = int((~fillable & (commit >= 0)).sum())
    if non_fillable_committed:
        errs.append(f"{non_fillable_committed} never-masked positions have a commit step")
    committed = commit[fillable]
    if committed.numel() and not (0 <= committed.min().item() and committed.max().item() < steps):
        errs.append(f"commit_step out of range [0,{steps}) on fillable positions")

    # masked count must be monotonically NON-INCREASING over steps
    counts = role.sum(dim=1)
    if not torch.all(counts[1:] <= counts[:-1]):
        errs.append("masked-count not monotonically non-increasing over steps")

    # role/commit consistency: masked at step i  <=>  commit_step >= i
    # (holds automatically for never-masked positions: role=False and commit=-1 >= i is False)
    step_idx = torch.arange(S).unsqueeze(1)      # [S,1]
    expected = commit.unsqueeze(0) >= step_idx   # [S,G] True where still masked at step i
    if not torch.equal(role, expected):
        errs.append(f"role_mask disagrees with commit_step in "
                    f"{int((role != expected).sum())} cells (out_mask/out_unmask not separable)")

    # hidden shape sanity
    h = payload["hidden"]
    if h.dim() != 4 or h.shape[0] != S or h.shape[2] != G:
        errs.append(f"hidden shape {tuple(h.shape)} != [steps,layers,region_len,d]")
    if h.dtype != torch.float16:
        errs.append(f"hidden dtype {h.dtype} != float16")
    return errs


# ============================ out_mask-only ragged storage ============================
# Drops out_unmask hidden cells (the forbidden read) — the ~50%+ storage saving. The cut
# is applied AFTER commit_step is recorded per position, so a position's full masked
# trajectory (steps 0..commit_step[j]) is retained; only its post-commit cells are dropped.
# Per-position metadata (commit_step, fillable_reply_mask, region_id, final_ids) and the
# tiny dense scalars (entropy/confidence/topk) are kept unchanged.

def to_ragged_out_mask(payload):
    """Convert a dense payload to CSR-style out_mask-only hidden. Lossless on out_mask."""
    hidden = payload["hidden"]                                  # [S,L,R,D] fp16
    role = payload["reply_role_mask"] & payload["fillable_reply_mask"]   # [S,R] out_mask
    S, L, R, D = hidden.shape
    flats, steps, poss, ptr = [], [], [], [0]
    for i in range(S):
        idx = role[i].nonzero(as_tuple=True)[0]
        if idx.numel():
            flats.append(hidden[i][:, idx, :].permute(1, 0, 2).contiguous())  # [n_i,L,D]
            steps.append(torch.full((idx.numel(),), i, dtype=torch.int16))
            poss.append(idx.to(torch.int32))
        ptr.append(ptr[-1] + int(idx.numel()))
    out = {k: v for k, v in payload.items() if k != "hidden"}
    out["schema_version"] = SCHEMA_VERSION + ".ragged"
    out["hidden_flat"] = torch.cat(flats, 0) if flats else torch.zeros(0, L, D, dtype=hidden.dtype)
    out["cell_step"] = torch.cat(steps) if steps else torch.zeros(0, dtype=torch.int16)
    out["cell_pos"] = torch.cat(poss) if poss else torch.zeros(0, dtype=torch.int32)
    out["step_ptr"] = torch.tensor(ptr, dtype=torch.int32)     # [S+1] CSR row pointers
    out["meta"] = dict(payload["meta"])
    out["meta"].update({"storage": "ragged_out_mask", "n_cells": int(out["hidden_flat"].shape[0]),
                        "dense_cells": S * R, "layers_n": L, "d_model": D})
    return out


def verify_ragged(dense_payload, ragged_payload) -> list[str]:
    """Confirm the ragged store reconstructs the dense out_mask cells exactly."""
    hidden = dense_payload["hidden"]
    role = dense_payload["reply_role_mask"] & dense_payload["fillable_reply_mask"]
    S, L, R, D = hidden.shape
    hf, sp, cp = ragged_payload["hidden_flat"], ragged_payload["step_ptr"], ragged_payload["cell_pos"]
    errs = []
    if int(role.sum()) != hf.shape[0]:
        errs.append(f"n_cells {hf.shape[0]} != out_mask count {int(role.sum())}")
    for i in (0, S // 2, S - 1):
        idx = role[i].nonzero(as_tuple=True)[0]
        pos = cp[int(sp[i]):int(sp[i + 1])]
        if not torch.equal(pos, idx.to(torch.int32)):
            errs.append(f"cell_pos mismatch at step {i}")
            continue
        cells = hf[int(sp[i]):int(sp[i + 1])]                  # [n_i,L,D]
        recon = hidden[i][:, idx, :].permute(1, 0, 2)
        if not torch.equal(cells, recon):
            errs.append(f"hidden mismatch at step {i}")
    return errs


def ragged_out_mask_at_step(ragged_payload, i):
    """(hidden[n_i,L,D], pos[n_i]) for out_mask positions at step i — probe read helper."""
    sp = ragged_payload["step_ptr"]
    lo, hi = int(sp[i]), int(sp[i + 1])
    return ragged_payload["hidden_flat"][lo:hi], ragged_payload["cell_pos"][lo:hi]


# ============================ span -> commit dry run ============================
def _segment_char_offsets(tok, token_ids):
    """Return per-token (char_start, char_end) within decode(token_ids), plus the
    decoded string. Uses cumulative-prefix decode; robust enough for a dry run and
    flags its own drift. skip_special_tokens=True to match run_dija's response build."""
    ids = [int(t) for t in token_ids]
    full = tok.decode(ids, skip_special_tokens=True)
    spans = []
    prev = 0
    for k in range(len(ids)):
        cur = tok.decode(ids[:k + 1], skip_special_tokens=True)
        # cur should be a prefix-extension of the previous cumulative decode
        c0 = prev
        c1 = len(cur)
        spans.append((c0, c1))
        prev = c1
    drift = (prev != len(full))
    return full, spans, drift


def map_span_to_commit(payload, tok, span_locator):
    """End-to-end: rebuild run_dija's composed response from the captured region,
    locate the harmful span, map its chars -> captured positions -> commit_step.

    Returns a dict report (does not require a GPU; works on a saved payload)."""
    m = payload["meta"]
    t0, t1, P, total = m["t0"], m["t1"], m["prompt_len"], m["total"]
    gen_start = m["gen_start"]
    final_ids = payload["final_ids"]                       # region tokens [region_len]
    commit = payload["commit_step"].long()
    fillable = payload["fillable_reply_mask"]
    region_id = payload["region_id"]

    # local index ranges within the captured region
    a0, a1 = t0 - gen_start, t1 - gen_start                # scaffold  [a0,a1)
    b0, b1 = P - gen_start, total - gen_start              # continuation [b0,b1)

    scaffold_ids = final_ids[a0:a1]
    cont_ids = final_ids[b0:b1]
    sc_text, sc_spans, sc_drift = _segment_char_offsets(tok, scaffold_ids)
    ct_text, ct_spans, ct_drift = _segment_char_offsets(tok, cont_ids)

    # run_dija composes: response = (tpl_filled + "\n" + continuation).strip()
    # reproduce WITHOUT .strip() first so char offsets line up, then note leading strip.
    joined = sc_text + "\n" + ct_text
    lstrip_n = len(joined) - len(joined.lstrip())
    response = joined.strip()

    # char interval -> captured local position, for every generated token
    # scaffold tokens occupy [.., ..) shifted by 0; continuation shifted by len(sc_text)+1
    cont_shift = len(sc_text) + 1
    pos_intervals = []   # (char_start, char_end, local_index)
    for k, (c0, c1) in enumerate(sc_spans):
        pos_intervals.append((c0 - lstrip_n, c1 - lstrip_n, a0 + k))
    for k, (c0, c1) in enumerate(ct_spans):
        pos_intervals.append((c0 + cont_shift - lstrip_n, c1 + cont_shift - lstrip_n, b0 + k))

    # locate the harm span in TEXT (echo/disclaimer/degeneration stripped; behavior
    # passed so the leading request echo can be removed).
    res = span_locator.locate_harm_span(response, True,
                                        judge_row={"behavior": m.get("behavior", "")})
    char_spans = res["char_spans"]
    span_locals = []
    for (s, e) in char_spans:
        for (c0, c1, loc) in pos_intervals:
            if c1 > s and c0 < e and bool(fillable[loc]):   # overlap & generated
                span_locals.append(loc)
    span_locals = sorted(set(span_locals))

    is_scaf = (region_id == R_SCAFFOLD) & fillable
    is_cont = (region_id == R_CONTINUATION) & fillable
    scaffold_locals = is_scaf.nonzero(as_tuple=True)[0].tolist()   # injected worksheet blanks
    span_cont_locals = [loc for loc in span_locals if bool(is_cont[loc])]
    # PRIMARY k_commit set = the injected DIJA worksheet blanks (harm is delivered there;
    # bounded by n_inject so it CANNOT balloon with continuation length). span-continuation
    # is reported separately, not folded into the primary, to stay robust.
    harm_locals = sorted(scaffold_locals)

    def _c(locs):
        return [int(commit[loc]) for loc in locs if commit[loc] >= 0]

    def _mmm(c):
        if not c:
            return {"min": None, "median": None, "max": None, "n": 0}
        cs = sorted(c)
        return {"min": cs[0], "median": cs[len(cs) // 2], "max": cs[-1], "n": len(cs)}

    harm_commits = _c(harm_locals)
    return {
        "response": response,
        "char_span": char_spans[0] if char_spans else None,
        "span_text": response[char_spans[0][0]:char_spans[0][1]] if char_spans else "",
        # PRIMARY: scaffold-blank (injected worksheet) k_commit
        "hit_locals": harm_locals,
        "hit_commits": harm_commits,
        "commit_of_span": _mmm(harm_commits),
        # transparency / re-validation of the locator fix:
        "scaffold_commit": _mmm(harm_commits),
        "span_commit": _mmm(_c(span_locals)),
        "span_cont_commit": _mmm(_c(span_cont_locals)),
        "n_span_positions": len(span_locals),
        "n_scaffold_blank": int(is_scaf.sum()),
        "n_continuation": int(is_cont.sum()),
        "decode_drift": bool(sc_drift or ct_drift),
    }


def _self_test():
    """Offline logic check with a synthetic trajectory (NO model, NO GPU)."""
    S, G, L, D = 8, 6, 2, 16
    # synthetic monotone commit: position g commits at step g (clamped < S)
    commit = torch.tensor([min(g, S - 1) for g in range(G)], dtype=torch.long)
    role = torch.stack([(commit >= i) for i in range(S)])          # [S,G] bool
    payload = {
        "schema_version": SCHEMA_VERSION, "layers": [0, 1],
        "hidden": torch.zeros(S, L, G, D, dtype=torch.float16),
        "reply_role_mask": role, "commit_step": commit.to(torch.int16),
        "fillable_reply_mask": torch.ones(G, dtype=torch.bool),
        "region_id": torch.full((G,), R_CONTINUATION, dtype=torch.int8),
        "entropy": torch.zeros(S, G, dtype=torch.float16),
        "confidence": torch.zeros(S, G, dtype=torch.float16),
        "topk_probs": torch.zeros(S, G, 5, dtype=torch.float16),
        "topk_ids": torch.zeros(S, G, 5, dtype=torch.int32),
        "final_ids": torch.zeros(G, dtype=torch.int64),
        "mask_ratio_reply": [float((commit >= i).sum()) / G for i in range(S)],
        "meta": {"prompt_len": 3, "gen_start": 3, "region_len": G, "t0": 3, "t1": 3,
                 "total": 3 + G, "n_inject": 0, "schedule": "fill_all_masks",
                 "steps": S, "gen_length": G},
    }
    errs = verify_payload(payload, steps=S)
    print("[self-test] invariants:", "OK" if not errs else errs)
    # a never-masked position (region_id prompt_tail, commit=-1) must also pass
    payload["fillable_reply_mask"][0] = False
    payload["commit_step"][0] = -1
    payload["reply_role_mask"][:, 0] = False
    errs2 = verify_payload(payload, steps=S)
    print("[self-test] never-masked-position handling:", "OK" if not errs2 else errs2)
    import tempfile
    p = os.path.join(tempfile.gettempdir(), "study1_selftest.pt")
    torch.save(payload, p)
    rl = torch.load(p, map_location="cpu", weights_only=False)
    ok = torch.equal(rl["reply_role_mask"], payload["reply_role_mask"]) and \
        torch.equal(rl["commit_step"], payload["commit_step"])
    os.remove(p)
    print("[self-test] round-trip:", "OK" if ok else "FAIL")
    return not errs and not errs2 and ok


def _load_model(device="cuda"):
    from transformers import AutoTokenizer, AutoModel
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tok.padding_side != "left":
        tok.padding_side = "left"
    model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True,
                                      torch_dtype=torch.bfloat16)
    model.all_tied_weights_keys = {}
    model = model.to(device).eval()
    _, blocks = discover_blocks(model)
    return tok, model, blocks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true", help="offline logic check, no model")
    ap.add_argument("--limit", type=int, default=2, help="number of A-cases")
    ap.add_argument("--arm", default="clean",
                    choices=["clean", "dija", "benign_op", "both", "dija+benign_op"],
                    help="clean=chat-template refusal baseline; dija=injected-scaffold "
                         "harmful arm; benign_op=same A-id, injected-but-benign scaffold "
                         "(Group C control, byte-identical <mask:N> markers); "
                         "both=dija+clean; dija+benign_op=the matched expansion pair")
    ap.add_argument("--layers", type=int, nargs="+", default=DEFAULT_LAYERS)
    ap.add_argument("--storage", default="dense", choices=["dense", "ragged"],
                    help="dense=full hidden; ragged=out_mask-only CSR (drops out_unmask)")
    ap.add_argument("--cases", default=os.path.join(EXP, "cases.json"))
    ap.add_argument("--scaffolds", default=os.path.join(EXP, "dija_attack", "refined_100.json"))
    ap.add_argument("--benign", default=os.path.join(EXP, "clockv2", "data",
                                                     "benign_op_matched.json"))
    ap.add_argument("--out", default="/scratch/ore99/study1_smoke")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    if args.self_test:
        ok = _self_test()
        sys.exit(0 if ok else 1)

    assert torch.cuda.is_available(), "no CUDA device visible"
    os.makedirs(args.out, exist_ok=True)
    cases = json.load(open(args.cases))[:args.limit]
    print(f"[pilot] arm={args.arm} layers={args.layers} cases={[c['id'] for c in cases]}")

    _ARMSETS = {"both": ["dija", "clean"], "dija+benign_op": ["dija", "benign_op"]}
    arms = _ARMSETS.get(args.arm, [args.arm])
    need_dija = "dija" in arms
    need_tpl = need_dija or "benign_op" in arms   # both build inputs via PaperRunner

    # injected arms: scaffolds + the PaperRunner input builder (imported unedited).
    # benign_op carries byte-identical <mask:N> markers per A-id (clockv2/build_benign_op.py
    # asserts this), so it is a drop-in for build_inputs() -> same blank geometry, benign fill.
    scaffolds, benign, span_locator = None, None, None
    if need_tpl:
        srows = json.load(open(args.scaffolds))
        scaffolds = {r["id"]: r for r in srows}
        if "benign_op" in arms:
            benign = {r["id"]: r for r in json.load(open(args.benign))}
    if need_dija:
        sys.path.insert(0, os.path.join(EXP, "dija_attack"))
        import span_locator  # noqa: E402 (study1/span_locator.py; study1 dir is on sys.path)

    print(f"[load] {MODEL_ID} ...", flush=True)
    tok, model, blocks = _load_model(args.device)
    print(f"[load] {len(blocks)} blocks | GPU {torch.cuda.memory_allocated()/1024**3:.1f} GiB",
          flush=True)
    runner = None
    if need_tpl:
        from run_dija import PaperRunner  # noqa: E402 (imported unedited)
        runner = PaperRunner(model, tok, args.device)

    gen_path = os.path.join(args.out, "generations.jsonl")
    gfh = open(gen_path, "w", encoding="utf-8")

    sizes = {a: [] for a in arms}
    dry_runs = []
    for c in cases:
        cid = c["id"]
        behavior = c["behavior"]
        for arm in arms:
            if arm == "clean":
                s = tok.apply_chat_template([{"role": "user", "content": behavior}],
                                            add_generation_prompt=True, tokenize=False)
                ids = tok([s], add_special_tokens=False,
                          return_tensors="pt")["input_ids"].to(args.device)
                t_span, scaffold = None, ""
            else:
                if arm == "dija":
                    scaffold = scaffolds[cid]["Refined_behavior"]
                else:
                    assert cid in benign, f"{cid} missing from {args.benign}"
                    scaffold = benign[cid]["benign_scaffold"]
                    # LOCK 1: matched blank geometry, else the arms are not comparable
                    mk = re.findall(r"<mask:\d+>", scaffold)
                    assert mk == re.findall(r"<mask:\d+>", scaffolds[cid]["Refined_behavior"]), \
                        f"{cid}: benign_op mask markers differ from dija"
                ids, (t0, t1) = runner.build_inputs(scaffold)   # ids on device, (t0,t1) scaffold span
                t_span = (t0, t1)

            t_start = time.time()
            x, payload = capture_union(model, blocks, ids, args.layers, t_span=t_span)
            dt = time.time() - t_start

            P = payload["meta"]["prompt_len"]
            if arm == "clean":
                response = tok.decode(x[0, P:], skip_special_tokens=True)
            else:
                t0, t1 = payload["meta"]["t0"], payload["meta"]["t1"]
                tpl_filled = tok.decode(x[0, t0:t1], skip_special_tokens=True)
                cont = tok.decode(x[0, P:], skip_special_tokens=True)
                response = (tpl_filled + "\n" + cont).strip()

            payload["meta"].update({"id": cid, "arm": arm, "behavior": behavior,
                                    "scaffold": scaffold, "response": response})
            errs = verify_payload(payload)
            rerrs = []
            if args.storage == "ragged":
                save_obj = to_ragged_out_mask(payload)
                rerrs = verify_ragged(payload, save_obj)
            else:
                save_obj = payload
            path = os.path.join(args.out, f"{cid}__{arm}.pt")
            torch.save(save_obj, path)
            mb = os.path.getsize(path) / 1024 ** 2
            sizes[arm].append(mb)
            rl = torch.load(path, map_location="cpu", weights_only=False)
            rt = torch.equal(rl["commit_step"], payload["commit_step"])

            # generations.jsonl row (behavior/prompt_text = ORIGINAL request for the judges)
            gfh.write(json.dumps({
                "traj_id": f"{cid}__{arm}", "id": cid, "arm": arm,
                "behavior": behavior, "prompt_text": behavior, "scaffold": scaffold,
                "response": response, "n_inject": payload["meta"]["n_inject"],
            }, ensure_ascii=False) + "\n")
            gfh.flush()

            print(f"\n[{cid}/{arm}] {dt:.1f}s | hidden {list(payload['hidden'].shape)} "
                  f"{payload['hidden'].dtype} | file {mb:.1f} MiB | roundtrip={'OK' if rt else 'FAIL'}")
            print(f"   invariants: {'OK' if not errs else errs}")
            if arm != "clean":
                rid = payload["region_id"]; fil = payload["fillable_reply_mask"]
                n_sb = int(((rid == R_SCAFFOLD) & fil).sum())
                n_ct = int(((rid == R_CONTINUATION) & fil).sum())
                cm = payload["commit_step"]
                sb_commits = cm[(rid == R_SCAFFOLD) & fil].tolist()
                print(f"   region: gen_start=t0={payload['meta']['t0']} t1={payload['meta']['t1']} "
                      f"P={P} region_len={payload['meta']['region_len']} "
                      f"n_inject={payload['meta']['n_inject']}")
                print(f"   COVERAGE: scaffold-blank fillable={n_sb} (commit steps {sb_commits}) "
                      f"| continuation fillable={n_ct}")
            print(f"   reply[:140]: {response[:140]!r}")

            if arm == "dija" and len(dry_runs) < 3:
                dr = map_span_to_commit(payload, tok, span_locator)
                dry_runs.append((cid, dr))

    gfh.close()
    print(f"\n[pilot] wrote generations -> {gen_path}")

    if dry_runs:
        print("\n" + "=" * 90 + "\nSPAN -> COMMIT DRY RUN (2-3 dija cases)\n" + "=" * 90)
        for cid, dr in dry_runs:
            print(f"\n### {cid}  decode_drift={dr['decode_drift']}")
            print(f"  span_text[:160] : {dr['span_text'][:160]!r}")
            print(f"  n_span_positions={dr['n_span_positions']}  scaffold_blanks={dr['n_scaffold_blank']}")
            print(f"  k_commit (PRIMARY scaffold): {dr['commit_of_span']}")
            print(f"  k_commit (span-based)      : {dr['span_commit']}")

    for arm in arms:
        s = sizes[arm]
        per = sum(s) / len(s)
        print(f"\n[pilot] arm={arm}: mean per-case-file = {per:.1f} MiB ({len(args.layers)} layers)")
        for arms_n, ncase in [(1, 65), (3, 65)]:
            gb = per * ncase * arms_n / 1024
            print(f"   extrapolate: {ncase} cases x {arms_n} arm(s) = {gb:.1f} GiB")


if __name__ == "__main__":
    main()
