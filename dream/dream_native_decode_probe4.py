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

# DIJA harness uses float32. Use this first for fidelity.
model = AutoModel.from_pretrained(
    model_path,
    trust_remote_code=True,
    torch_dtype=torch.float32,
).to("cuda").eval()

print("model:", type(model))
print("generation_config type:", type(model.generation_config))

# Patch exact Dream generation config.
gc = model.generation_config
gc.eps = 0.001
gc.steps = 64
gc.max_new_tokens = 64
gc.temperature = 0.2
gc.top_p = 0.95
gc.output_history = False
gc.return_dict_in_generate = True
gc.mask_token_id = tok.mask_token_id

print("eps:", gc.eps)
print("steps:", gc.steps)
print("max_new_tokens:", gc.max_new_tokens)
print("temperature:", gc.temperature)
print("top_p:", gc.top_p)
print("mask_token_id:", gc.mask_token_id)

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
    tokenize=False,
    add_generation_prompt=True,
)

vanilla_prompt = tok.apply_chat_template(
    [{"role": "user", "content": case.behavior}],
    tokenize=False,
    add_generation_prompt=True,
)

input_ids_list = tok(prompt)["input_ids"]
attention_mask_list = tok(prompt)["attention_mask"]
vanilla_ids = tok(vanilla_prompt)["input_ids"]

matching_count = next(
    (i for i, (a, b) in enumerate(zip(vanilla_ids, input_ids_list)) if a != b),
    min(len(vanilla_ids), len(input_ids_list))
)

input_ids = torch.tensor(input_ids_list).unsqueeze(0).to(model.device)
attention_mask = torch.tensor(attention_mask_list).unsqueeze(0).to(model.device).float()

print("input shape:", tuple(input_ids.shape))
print("attention dtype:", attention_mask.dtype)
print("matching_count:", matching_count)

with torch.no_grad():
    out = model.diffusion_generate(
        input_ids,
        attention_mask=attention_mask,
        generation_config=gc,
    )

seq = out.sequences
print("seq shape:", tuple(seq.shape))

response = tok.batch_decode(seq[:, matching_count:], skip_special_tokens=True)[0]
response = response.split("assistant\n")[0]

print("\n===== RESPONSE =====")
print(response[:2000])
