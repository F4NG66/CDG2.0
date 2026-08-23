import os, sys
from huggingface_hub import snapshot_download
p = snapshot_download(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    cache_dir=os.environ["HF_HOME"] + "/hub",
    allow_patterns=["*.safetensors","*.json","*.txt","tokenizer*","vocab*","merges*"],
)
print("QWEN_LOCAL_PATH=" + p, flush=True)
