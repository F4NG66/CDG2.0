# Model Dependency

The RRAE v2 experiments use the LLaDA-8B-Instruct base model.

## Expected local layout

    models/LLaDA-8B-Instruct/
    ├── config.json
    ├── configuration_llada.py
    ├── generation_config.json
    ├── modeling_llada.py
    ├── model.safetensors.index.json
    ├── model-00001-of-00006.safetensors
    ├── model-00002-of-00006.safetensors
    ├── model-00003-of-00006.safetensors
    ├── model-00004-of-00006.safetensors
    ├── model-00005-of-00006.safetensors
    ├── model-00006-of-00006.safetensors
    ├── tokenizer.json
    ├── tokenizer_config.json
    └── special_tokens_map.json

The model weight shards are not stored in this Git repository because their
combined size is approximately 15 GB.

Download the official LLaDA-8B-Instruct checkpoint separately and provide its
local path through `--model-path` or the corresponding environment variable.

The experiments do not modify the base LLaDA weight shards. Experiment-specific
RRAE checkpoints, steering vectors, and generated artifacts are handled
separately.
