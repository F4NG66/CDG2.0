import re
import os
import torch

from transformers import AutoTokenizer, AutoModel

from cdg.data import load_cdg_root
from cdg.config import get_backend_config

torch.manual_seed(0)

model_path = os.environ.get("DREAM_MODEL_PATH", "Dream-org/Dream-v0-Instruct-7B")

print("===== loading Dream =====")
tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModel.from_pretrained(
    model_path,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
).to("cuda").eval()

print("model:", type(model))
print("has diffusion_generate:", hasattr(model, "diffusion_generate"))
print("generation_config type:", type(model.generation_config))

print("\n===== patch model.generation_config =====")
# Dream diffusion_generate expects these attrs.
# The HF GenerationConfig object lacks eps, so we attach it.
model.generation_config.eps = 1e-3
model.generation_config.steps = 128
model.generation_config.max_new_tokens = 128
model.generation_config.temperature = 0.2
model.generation_config.top_p = 0.95
model.generation_config.alg = "entropy"
model.generation_config.alg_temp = 0.0
model.generation_config.output_history = True
model.generation_config.return_dict_in_generate = True

model.generation_config.pad_token_id = getattr(tok, "pad_token_id", None) or getattr(tok, "eos_token_id", None)
model.generation_config.eos_token_id = getattr(tok, "eos_token_id", None)
model.generation_config.mask_token_id = getattr(tok, "mask_token_id", None) or getattr(model.config, "mask_token_id", None)

for k in ["eps", "steps", "max_new_tokens", "temperature", "top_p", "alg", "alg_temp", "mask_token_id"]:
    print(k, "=", getattr(model.generation_config, k, None))

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

print("\n===== run diffusion_generate =====")
with torch.no_grad():
    out = model.diffusion_generate(
        inputs=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=128,
        steps=128,
        temperature=0.2,
        top_p=0.95,
        alg="entropy",
        alg_temp=0.0,
        eps=1e-3,
        output_history=True,
        return_dict_in_generate=True,
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
