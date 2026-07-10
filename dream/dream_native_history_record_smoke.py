import os
import torch

from cdg.config import get_backend_config
from cdg.backends import build_runner
from cdg.recorder import Recorder
from cdg.data import load_cdg_root

torch.manual_seed(0)

OUT = "outputs_dream_native_history_smoke"
os.makedirs(OUT, exist_ok=True)

cfg = get_backend_config("dream_attack")

runner = build_runner(
    cfg,
    sae_root="./saes",
    device="cuda",
    dummy=False,
)

tok = runner.tokenizer
dc = cfg.decode

print("model:", type(runner.model))
print("mask_id:", runner.mask_id)
print("layers:", cfg.record_layers)
print("steps:", dc.steps)
print("gen_length:", dc.gen_length)

cases = load_cdg_root("prompts/cdg_injection")

kept = []
seen = set()
for c in cases:
    if c.group_letter not in seen:
        kept.append(c)
        seen.add(c.group_letter)
    if seen == {"A", "B", "C", "D"}:
        break

for case in kept:
    print("\n==============================")
    print("case:", case.case_id, case.variant)

    rec = Recorder(runner.bundles, cfg.record, tokenizer=tok)

    # CRITICAL FIX:
    # Recorder needs the actual Dream mask token id.
    rec.mask_id = int(runner.mask_id)
    print("rec.mask_id:", rec.mask_id)

    meta = {
        "case_id": case.case_id,
        "variant": case.variant,
        "content_type": case.content_type,
        "has_template": case.has_template,
        "attack_method": case.attack_method,
        "is_neutral": case.is_neutral,
        "model_name": cfg.name,
        "seed": 0,
        "behavior": case.behavior,
        "saes": [b.name for b in runner.bundles],
        "record_layers": list(cfg.record_layers),
        "gen_length": dc.gen_length,
        "steps": dc.steps,
        "case_meta": case.meta,
        "dream_native_history": True,
    }

    rec.begin(meta, total_steps=dc.steps)

    x_prompt, regions = runner.build_inputs(case)
    x_prompt = x_prompt.to("cuda").long()
    P = x_prompt.shape[1]
    total = P + dc.gen_length

    regions = dict(regions)
    regions["output"] = (P, total)

    rec.rec.layout = {
        "prompt_len": P,
        "gen_length": dc.gen_length,
        "total": total,
        "native_history": True,
    }
    rec.rec.regions = regions

    attn_prompt = torch.ones_like(x_prompt, dtype=torch.float32, device="cuda")

    gc = runner.model.generation_config
    gc.eps = 0.001
    gc.steps = dc.steps
    gc.max_new_tokens = dc.gen_length
    gc.temperature = dc.temperature
    gc.top_p = 0.95
    gc.output_history = True
    gc.return_dict_in_generate = True
    gc.mask_token_id = int(runner.mask_id)

    with torch.no_grad():
        out = runner.model.diffusion_generate(
            x_prompt,
            attention_mask=attn_prompt,
            generation_config=gc,
        )

    seq = out.sequences.long()
    history = out.history

    print("seq shape:", tuple(seq.shape))
    print("history len:", len(history))
    print("history[0]:", tuple(history[0].shape))
    print("history[-1]:", tuple(history[-1].shape))

    for step in sorted(rec.step_to_frac.keys()):
        x_step = history[step - 1]

        if isinstance(x_step, (list, tuple)):
            x_step = x_step[0]

        if not torch.is_tensor(x_step):
            x_step = torch.tensor(x_step)

        x_step = x_step.to("cuda").long()

        if x_step.dim() == 1:
            x_step = x_step.unsqueeze(0)

        attn_full = torch.ones_like(x_step, dtype=torch.float32, device="cuda")

        runner.hooks.clear()
        with torch.no_grad():
            logits = runner.forward(x_step, attn_full)

        rec.record(step, x_step, logits, runner.hooks.buffers)

    out_ids = seq[0, P:]
    resp = tok.decode(out_ids.tolist(), skip_special_tokens=True)

    if regions.get("template"):
        t0, t1 = regions["template"]
        tpl_filled = tok.decode(seq[0, t0:t1].tolist(), skip_special_tokens=True)
        resp = (tpl_filled + "\n" + resp).strip()

    rec.finish(resp)
    path = rec.save(OUT)

    print("saved:", path)
    print("response preview:", resp[:300])

print("\nDONE:", OUT)
