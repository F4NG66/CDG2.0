import re
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

print("model:", type(model))
print("has diffusion_generate:", hasattr(model, "diffusion_generate"))

DG = model.diffusion_generate.__globals__.get("DreamGenerationConfig")
print("DreamGenerationConfig:", DG)

if DG is None:
    raise RuntimeError("Could not access DreamGenerationConfig from diffusion_generate globals")

print("\n===== build DreamGenerationConfig =====")
try:
    gen_cfg = DG.from_pretrained(model_path)
    print("loaded DreamGenerationConfig.from_pretrained")
except Exception as e:
    print("from_pretrained failed:", repr(e))
    gen_cfg = DG()
    print("created DreamGenerationConfig() manually")

# أهم شيء: eps لازم يكون موجود
if not hasattr(gen_cfg, "eps"):
    print("eps missing; setting manually")
    gen_cfg.eps = 1e-3

# match our Dream attack config
gen_cfg.max_new_tokens = 128
gen_cfg.steps = 128
gen_cfg.temperature = 0.2
gen_cfg.top_p = 0.95
gen_cfg.alg = "entropy"
gen_cfg.alg_temp = 0.0
gen_cfg.output_history = True
gen_cfg.return_dict_in_generate = True

# make ids explicit
gen_cfg.pad_token_id = getattr(tok, "pad_token_id", None) or getattr(tok, "eos_token_id", None)
gen_cfg.eos_token_id = getattr(tok, "eos_token_id", None)
gen_cfg.mask_token_id = getattr(tok, "mask_token_id", None) or getattr(model.config, "mask_token_id", None)

print("eps:", getattr(gen_cfg, "eps", None))
print("steps:", getattr(gen_cfg, "steps", None))
print("max_new_tokens:", getattr(gen_cfg, "max_new_tokens", None))
print("temperature:", getattr(gen_cfg, "temperature", None))
print("top_p:", getattr(gen_cfg, "top_p", None))
print("alg:", getattr(gen_cfg, "alg", None))
print("alg_temp:", getattr(gen_cfg, "alg_temp", None))
print("mask_token_id:", getattr(gen_cfg, "mask_token_id", None))

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
print(prompt[:1000])

inputs = tok(prompt, return_tensors="pt").to("cuda")
input_ids = inputs["input_ids"]
attention_mask = inputs.get("attention_mask", None)

if attention_mask is not None:
    attention_mask = attention_mask.bool()

print("input shape:", tuple(input_ids.shape))
print("attention dtype:", attention_mask.dtype if attention_mask is not None else None)

print("\n===== run diffusion_generate with DreamGenerationConfig =====")
with torch.no_grad():
    out = model.diffusion_generate(
        inputs=input_ids,
        attention_mask=attention_mask,
        generation_config=gen_cfg,
    )

print("out type:", type(out))

seq = None
if hasattr(out, "sequences"):
    seq = out.sequences
elif isinstance(out, torch.Tensor):
    seq = out
elif isinstance(out, (list, tuple)) and len(out) > 0 and isinstance(out[0], torch.Tensor):
    seq = out[0]

if seq is None:
    print("Could not find sequence tensor")
    print(out)
    raise SystemExit(1)

print("seq shape:", tuple(seq.shape))

full = tok.decode(seq[0].detach().cpu().tolist(), skip_special_tokens=True)
new = tok.decode(seq[0, input_ids.shape[1]:].detach().cpu().tolist(), skip_special_tokens=True)

print("\n===== FULL DECODE FIRST 1500 =====")
print(full[:1500])

print("\n===== NEW TOKENS FIRST 1500 =====")
print(new[:1500])
