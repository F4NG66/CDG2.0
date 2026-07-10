import re
import inspect
import torch

from transformers import AutoTokenizer, AutoModel

from cdg.data import load_cdg_root
from cdg.config import get_backend_config

torch.manual_seed(0)

model_path = "/scratch/nmmd294/diffusion_model/diffusion_models/Dream-v0-Instruct-7B"

print("===== loading Dream =====")
tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModel.from_pretrained(
    model_path,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
).to("cuda").eval()

print("tokenizer:", type(tok))
print("model:", type(model))
print("mask_token:", getattr(tok, "mask_token", None))
print("mask_token_id tokenizer:", getattr(tok, "mask_token_id", None))
print("mask_token_id config:", getattr(model.config, "mask_token_id", None))
print("hidden_size:", getattr(model.config, "hidden_size", None))
print("num_hidden_layers:", getattr(model.config, "num_hidden_layers", None))

print("\n===== generation methods =====")
methods = [m for m in dir(model) if "generate" in m.lower()]
print(methods)

for name in methods:
    obj = getattr(model, name)
    if callable(obj):
        try:
            print(f"\n{name} signature:")
            print(inspect.signature(obj))
        except Exception as e:
            print(f"{name} signature error:", repr(e))

cfg = get_backend_config("dream_attack")

cases = load_cdg_root("prompts/cdg_injection")
case = next(c for c in cases if c.case_id == "B000")

content = case.user_content.replace(cfg.template.behavior_placeholder, case.behavior)
content = re.sub(
    cfg.template.mask_marker_re,
    lambda m: cfg.mask_token * int(m.group(1)),
    content,
)

prompt = tok.apply_chat_template(
    [{"role": "user", "content": content}],
    add_generation_prompt=True,
    tokenize=False,
)

print("\n===== prompt preview =====")
print(prompt[:1200])

inputs = tok(prompt, return_tensors="pt").to("cuda")
input_ids = inputs["input_ids"]
attention_mask = inputs.get("attention_mask", None)

print("\ninput shape:", tuple(input_ids.shape))
if attention_mask is not None:
    print("attention_mask dtype before:", attention_mask.dtype)
    attention_mask = attention_mask.bool()
    print("attention_mask dtype after:", attention_mask.dtype)

def decode_output(out, label):
    print(f"\n===== {label} raw type =====")
    print(type(out))

    seq = None

    if hasattr(out, "sequences"):
        seq = out.sequences
    elif isinstance(out, torch.Tensor):
        seq = out
    elif isinstance(out, (list, tuple)) and len(out) > 0 and isinstance(out[0], torch.Tensor):
        seq = out[0]

    if seq is None:
        print("Could not find sequence tensor.")
        print(out)
        return

    print("sequence shape:", tuple(seq.shape))

    full = tok.decode(seq[0].detach().cpu().tolist(), skip_special_tokens=True)
    new = tok.decode(seq[0, input_ids.shape[1]:].detach().cpu().tolist(), skip_special_tokens=True)

    print("\n--- FULL DECODE FIRST 1200 ---")
    print(full[:1200])

    print("\n--- NEW TOKENS FIRST 1200 ---")
    print(new[:1200])

print("\n===== try native diffusion_generate =====")

if hasattr(model, "diffusion_generate"):
    fn = model.diffusion_generate

    trials = [
        dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=128,
            steps=128,
            temperature=0.2,
            top_p=0.95,
            alg="entropy",
            alg_temp=0.0,
            output_history=True,
            return_dict_in_generate=True,
        ),
        dict(
            inputs=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=128,
            steps=128,
            temperature=0.2,
            top_p=0.95,
            alg="entropy",
            alg_temp=0.0,
            output_history=True,
            return_dict_in_generate=True,
        ),
        dict(
            input_ids=input_ids,
            max_new_tokens=128,
            steps=128,
            temperature=0.2,
            top_p=0.95,
            alg="entropy",
            alg_temp=0.0,
            output_history=True,
            return_dict_in_generate=True,
        ),
    ]

    ok = False
    for i, kwargs in enumerate(trials, 1):
        try:
            print(f"\n--- trial {i} kwargs keys:", list(kwargs.keys()))
            with torch.no_grad():
                out = fn(**kwargs)
            decode_output(out, f"diffusion_generate trial {i}")
            ok = True
            break
        except Exception as e:
            print(f"trial {i} failed:", repr(e))

    if not ok:
        print("All diffusion_generate trials failed.")
else:
    print("No diffusion_generate method found.")

print("\n===== try model.generate fallback =====")
try:
    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=128,
            do_sample=True,
            temperature=0.2,
            top_p=0.95,
        )
    decode_output(out, "model.generate fallback")
except Exception as e:
    print("model.generate failed:", repr(e))
