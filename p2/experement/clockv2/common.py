"""clockv2/common.py — shared read-only plumbing for the Plan B interaction study.

READ-ONLY CONTRACT (enforced, not just documented):
  * Every hook here returns None. A forward_hook returning None leaves the block
    output untouched — this is the only capture mechanism used anywhere in clockv2.
  * assert_readonly_hooks() below is a runtime guard: it drives a probe tensor
    through every registered hook and fails loudly if any hook returns non-None.
  * clock_attack.steerer.StepSteerer is NEVER imported. It edits activations
    (h[0, mrow, :] += vec) and is out of scope for this study by construction.

Nothing outside clockv2/ is modified. dija_attack/ and clock_attack/ are import-only.
"""
from __future__ import annotations

import json
import os
import re
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
if EXP not in sys.path:
    sys.path.insert(0, EXP)

# ---- constants (from ladaAndH.py; re-declared here only to avoid importing its
# module-level PROMPTS/OUT_DIR side effects — values are asserted against it below) ----
MODEL_ID = "GSAI-ML/LLaDA-8B-Instruct"
MASK_ID = 126336
EOS_ID = 126081
EOT_ID = 126348
D_MODEL = 4096

STEPS = 128
GEN_LENGTH = 128
TEMPERATURE = 0.0          # deterministic: see README "deviation from paper temp=0.2"

PROBE_DIR = os.path.join(HERE, "probes")
HELDOUT_DIR = os.path.join(HERE, "heldout")
RESULTS_DIR = os.path.join(HERE, "results")

# Mid-to-late band where the refusal axis is expected to live (Arditi et al.),
# plus the two tau layers the mu-bank was originally built for.
BAND = list(range(16, 32))          # L16..L31
TAU_LAYERS = [25, 29]

# ---- input paths (byte-identical reuse; never edited) ----
CASES_A = os.path.join(EXP, "cases.json")                       # the 100 EVAL cases
REFINED_100 = os.path.join(EXP, "dija_attack", "refined_100.json")
D_NEUTRAL = "/home/ore99/serverFiles/prompts/cdg_injection/D_neutral_clean/cases.json"
UNSAFE_MED = os.path.join(EXP, "unsafe.jsonl")                  # MedSafetyBench, 50


# ======================= leakage control =======================
def _toks(s: str) -> set:
    return set(re.findall(r"[a-z]{4,}", s.lower()))


def jaccard(a: str, b: str) -> float:
    ta, tb = _toks(a), _toks(b)
    return len(ta & tb) / max(1, len(ta | tb))


def max_jaccard_vs(text: str, corpus: list[str]) -> float:
    return max((jaccard(text, c) for c in corpus), default=0.0)


def eval_behaviors() -> list[str]:
    """The 100 A-set behaviors. Used ONLY as a leakage blocklist — never as probe input."""
    return [c["behavior"] for c in json.load(open(CASES_A))]


def build_heldout(j_thresh: float = 0.4, verbose: bool = True) -> dict:
    """Held-out contrastive sets, filtered against the 100 eval behaviors.

    harmful  <- unsafe.jsonl        (MedSafetyBench, medical-harmful)
    harmless <- D_neutral_clean     (medical-benign, D000..)

    Both halves are MEDICAL, so the harmful-minus-harmless difference isolates harm
    rather than domain. Anything within j_thresh Jaccard of any eval behavior is
    dropped: curated/ and neutral/ are excluded entirely (they ARE the eval set,
    99/100 exact match).
    """
    A = eval_behaviors()
    harmful_all = [json.loads(l)["question"] for l in open(UNSAFE_MED) if l.strip()]
    harmless_all = [d["behavior"] for d in json.load(open(D_NEUTRAL))]

    def keep(texts):
        kept, dropped = [], []
        for t in texts:
            j = max_jaccard_vs(t, A)
            (dropped if j > j_thresh else kept).append((t, j))
        return kept, dropped

    hk, hd = keep(harmful_all)
    lk, ld = keep(harmless_all)
    if verbose:
        print(f"[heldout] harmful : {len(hk)} kept / {len(hd)} dropped (J>{j_thresh}) "
              f"of {len(harmful_all)}")
        print(f"[heldout] harmless: {len(lk)} kept / {len(ld)} dropped (J>{j_thresh}) "
              f"of {len(harmless_all)}")
        for t, j in sorted(hd, key=lambda x: -x[1])[:3]:
            print(f"           dropped harmful  J={j:.2f} {t[:70]!r}")
        for t, j in sorted(ld, key=lambda x: -x[1])[:3]:
            print(f"           dropped harmless J={j:.2f} {t[:70]!r}")
    return {
        "harmful": [t for t, _ in hk],
        "harmless": [t for t, _ in lk],
        "j_thresh": j_thresh,
        "n_dropped": {"harmful": len(hd), "harmless": len(ld)},
        "max_j_kept": {"harmful": max([j for _, j in hk], default=0.0),
                       "harmless": max([j for _, j in lk], default=0.0)},
    }


# ======================= model =======================
def load_model(device="cuda", dtype=torch.bfloat16):
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True, torch_dtype=dtype)
    model.all_tied_weights_keys = {}          # transformers-compat patch (see generate.py)
    model = model.to(device).eval()
    from ladaAndH import discover_blocks, MASK_ID as _M
    assert _M == MASK_ID, f"mask_id drift: ladaAndH={_M} clockv2={MASK_ID}"
    _, blocks = discover_blocks(model)
    return tok, model, blocks


# ======================= read-only capture =======================
class PromptOnlyCapture:
    """Reads block outputs for a single forward pass. Returns None from every hook.

    Records, per layer:
      last  : hidden at the final prompt token   [d]   (Arditi Eq.1 extraction point)
      mean  : hidden mean-pooled over all prompt tokens [d]
    """

    def __init__(self, blocks):
        self.n = len(blocks)
        self.last, self.mean = {}, {}
        self._handles = [b.register_forward_hook(self._mk(i)) for i, b in enumerate(blocks)]

    def _mk(self, pos):
        def _hook(_m, _i, out):
            h = out[0] if isinstance(out, (tuple, list)) else out    # [1, seq, d]
            self.last[pos] = h[0, -1].detach().float().cpu()
            self.mean[pos] = h[0].mean(0).detach().float().cpu()
            return None                                             # READ-ONLY
        return _hook

    def clear(self):
        self.last, self.mean = {}, {}

    def close(self):
        for h in self._handles:
            h.remove()
        self._handles = []


def assert_readonly_hooks(blocks):
    """Runtime guard: fail loudly if ANY registered forward hook mutates the output.

    Drives a dummy tensor through each block's hook chain and checks the hook
    returns None (PyTorch treats a non-None return as a replacement output).
    """
    bad = []
    for i, b in enumerate(blocks):
        hooks = getattr(b, "_forward_hooks", {})
        for hid, fn in list(hooks.items()):
            dummy = torch.zeros(1, 4, D_MODEL)
            try:
                r = fn(b, (dummy,), (dummy,))
            except Exception:
                r = None                       # capture hooks may need real shapes; ignore
            if r is not None:
                bad.append((i, getattr(fn, "__qualname__", str(fn))))
    if bad:
        raise RuntimeError(
            "READ-ONLY VIOLATION: hook(s) returned a replacement output "
            f"(activation would be modified): {bad}")
    return True


@torch.no_grad()
def prompt_forward(model, tok, text_or_ids, cap: PromptOnlyCapture, device="cuda",
                   chat_template=True):
    """One prompt-only forward pass. No generation, no sampling, no canvas."""
    if isinstance(text_or_ids, torch.Tensor):
        ids = text_or_ids.to(device)
    else:
        s = text_or_ids
        if chat_template:
            s = tok.apply_chat_template([{"role": "user", "content": s}],
                                        add_generation_prompt=True, tokenize=False)
        ids = tok([s], add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)
    attn = torch.ones_like(ids)
    cap.clear()
    model(ids, attention_mask=attn)
    return ids.shape[1]


class StepCapture:
    """Per-denoising-step hidden capture. Every hook returns None (READ-ONLY).

    Mirrors ladaAndH.StepActivationCollector but fixes its footgun: that class hooks
    ALL blocks yet mark_step() reads self._cur[p] for p in range(len(layer_ids)), so it
    is only correct when layer_ids == range(n_layers). Here layers are keyed by block
    index directly, so any subset is safe.

    Captures THREE read points per step, so every probe can be read on the axis it was
    actually fit at (an unmatched read point is a silent methodological error):

      resp_mean   : mean over the RESPONSE canvas    -> matches tau_bank's mu_step
      prompt_mean : mean over the PROMPT span        -> matches v_refusal['v_mean'] and
                                                        v_injection_svd['v']['mean']
      last_prompt : the final prompt token           -> matches v_refusal['v_last'] and
                                                        v_injection_svd['v']['last']

    In a diffusion LM the prompt's hidden states evolve per step: attention is
    bidirectional, so the prompt span sees the filling canvas. prompt_mean/last_prompt are
    therefore genuine per-step signals, not constants. For the DIJA condition the prompt
    span also contains the injected blanks as they fill — which is exactly the span
    v_injection_svd was fit on.
    """

    def __init__(self, blocks, layers, gen_start):
        self.layers = list(layers)
        self.gen_start = gen_start
        self._cur = {}
        self.resp_mean = []          # list[step] -> {L: [d]}
        self.prompt_mean = []
        self.last_prompt = []
        self._want = set(self.layers)
        self._handles = []
        for i, b in enumerate(blocks):
            if i in self._want:
                self._handles.append(b.register_forward_hook(self._mk(i)))

    def _mk(self, pos):
        def _hook(_m, _i, out):
            h = out[0] if isinstance(out, (tuple, list)) else out      # [1, seq, d]
            self._cur[pos] = (
                h[0, self.gen_start:].mean(0).detach().float().cpu(),
                h[0, :self.gen_start].mean(0).detach().float().cpu(),
                h[0, self.gen_start - 1].detach().float().cpu(),
            )
            return None                                               # READ-ONLY
        return _hook

    def mark_step(self):
        if not self._cur:
            return
        self.resp_mean.append({L: self._cur[L][0] for L in self.layers})
        self.prompt_mean.append({L: self._cur[L][1] for L in self.layers})
        self.last_prompt.append({L: self._cur[L][2] for L in self.layers})
        self._cur = {}

    def close(self):
        for h in self._handles:
            h.remove()
        self._handles = []


@torch.no_grad()
def capture_denoise(model, ids, cap: StepCapture, *, steps=STEPS, gen_length=GEN_LENGTH,
                    temperature=TEMPERATURE, mask_id=MASK_ID, remask="low_confidence"):
    """Unified-schedule denoising with read-only capture + physical mask-ratio logging.

    This is cdg_denoise.denoise(fill_all_masks=True) (lines 57-83) with capture added:
    ONE schedule over every mask in the sequence (injected prompt blanks + response
    canvas). block_length is irrelevant under fill_all_masks -- the block loop is not
    entered at all. For a CLEAN prompt (no prompt blanks) this is exactly equivalent to
    ladaAndH.generate_with_capture at block_length=gen_length=128, so both conditions
    share one schedule and the mu-bank is valid for both.

    Returns (x, info) with per-step plain counts off (x == mask_id), read BEFORE the
    forward at step t. Never an intervention.

      info["mask_ratio"]        |masked RESPONSE positions| / gen_length   (as specified)
      info["mask_ratio_global"] |masked positions ANYWHERE| / n_fillable_at_entry
      info["n_inject"]          prompt-embedded blanks at entry (0 for clean)

    WHY BOTH: under fill_all_masks the 128-step budget is spread over ALL initial masks,
    i.e. gen_length + n_inject of them. DIJA adds n_inject prompt blanks, so some of each
    step's transfer budget lands on the prompt rather than the response canvas. The
    RESPONSE-only mask ratio is therefore mechanically depressed for DIJA relative to
    clean -- by arithmetic, before any representational effect exists. A raw clean-vs-DIJA
    comparison of disc_t = tau_read - (1 - mask_ratio) is confounded by n_inject.
    The global ratio is the deterministic ramp shared by both conditions and acts as the
    control: Phase 3 must regress disc_t against n_inject before claiming the discrepancy
    is representational rather than bookkeeping.
    """
    import torch.nn.functional as F
    device = model.device
    P = ids.shape[1]
    total = P + gen_length
    x = torch.full((1, total), mask_id, dtype=torch.long, device=device)
    x[:, :P] = ids
    attn = torch.ones((1, total), dtype=torch.long, device=device)

    fillable = (x == mask_id)
    n_fillable = int(fillable.sum().item())
    n_inject = int((ids == mask_id).sum().item())      # prompt blanks (0 for clean)
    from ladaAndH import get_num_transfer_tokens
    ntt = get_num_transfer_tokens(fillable, steps)

    mask_ratio, mask_ratio_global = [], []
    for i in range(steps):
        mask_index = (x == mask_id)
        # physical canvas clock: plain counts, read before the forward
        mask_ratio.append(float((x[0, P:] == mask_id).sum().item()) / gen_length)
        mask_ratio_global.append(float(mask_index.sum().item()) / max(n_fillable, 1))

        logits = model(x, attention_mask=attn).logits
        cap.mark_step()

        if temperature == 0:
            logits_n = logits
        else:
            lg = logits.to(torch.float64)
            noise = torch.rand_like(lg, dtype=torch.float64)
            logits_n = lg.exp() / ((-torch.log(noise)) ** temperature)
        x0 = torch.argmax(logits_n, dim=-1)
        p = F.softmax(logits.to(torch.float64), dim=-1)
        conf = torch.gather(p, -1, x0.unsqueeze(-1)).squeeze(-1)
        x0 = torch.where(mask_index, x0, x)
        neg = torch.tensor(float("-inf"), device=device, dtype=conf.dtype)
        confidence = torch.where(mask_index, conf, neg)

        transfer = torch.zeros_like(x0, dtype=torch.bool)
        for j in range(confidence.shape[0]):
            k = int(ntt[j, i])
            if k > 0:
                _, sel = torch.topk(confidence[j], k=k)
                transfer[j, sel] = True
        x[transfer] = x0[transfer]
    return x, {"mask_ratio": mask_ratio, "mask_ratio_global": mask_ratio_global,
               "n_inject": n_inject, "n_fillable": n_fillable}


def save_probe(name: str, payload: dict):
    os.makedirs(PROBE_DIR, exist_ok=True)
    p = os.path.join(PROBE_DIR, name)
    torch.save(payload, p)
    print(f"[save] {p}")
    return p
