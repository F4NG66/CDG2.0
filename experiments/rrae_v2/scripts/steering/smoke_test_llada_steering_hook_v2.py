#!/usr/bin/env python
import argparse
import torch
from transformers import AutoTokenizer, AutoModel


def get_block(model, layer):
    return model.model.transformer.blocks[layer]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="/path/to/LLaDA-8B-Instruct")
    ap.add_argument("--vector-path", required=True)
    ap.add_argument("--layer", type=int, default=11)
    ap.add_argument("--alpha", type=float, default=0.5)
    args = ap.parse_args()

    print("=" * 100, flush=True)
    print("[LOAD VECTOR]", args.vector_path, flush=True)
    vec_obj = torch.load(args.vector_path, map_location="cpu")
    v = vec_obj["v_injection_raw"].float()
    print("v shape:", tuple(v.shape), "norm:", float(v.norm()), "finite:", torch.isfinite(v).all().item(), flush=True)

    print("=" * 100, flush=True)
    print("[LOAD TOKENIZER]", args.model_path, flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        local_files_only=True,
    )

    print("[LOAD MODEL]", args.model_path, flush=True)
    model = AutoModel.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    ).to("cuda")
    model.eval()

    v = v.to("cuda", dtype=torch.bfloat16)

    messages = [
        {"role": "user", "content": "Reply with exactly: OK"}
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    print("input_ids shape:", tuple(inputs["input_ids"].shape), flush=True)

    block = get_block(model, args.layer)
    print("hook target:", f"model.model.transformer.blocks[{args.layer}]", type(block).__name__, flush=True)

    hook_stats = {
        "called": 0,
        "output_type": None,
        "output_shape": None,
        "delta_norm": None,
    }

    def steering_hook(module, module_input, module_output):
        hook_stats["called"] += 1
        hook_stats["output_type"] = str(type(module_output))

        # Most likely tensor: [batch, seq, hidden]
        if torch.is_tensor(module_output):
            h = module_output
            hook_stats["output_shape"] = tuple(h.shape)

            delta = args.alpha * v.view(1, 1, -1)
            hook_stats["delta_norm"] = float(delta.float().norm().detach().cpu())

            return h - delta

        # Some transformer blocks return tuple where first item is hidden states.
        if isinstance(module_output, tuple) and len(module_output) > 0 and torch.is_tensor(module_output[0]):
            h = module_output[0]
            hook_stats["output_shape"] = tuple(h.shape)

            delta = args.alpha * v.view(1, 1, -1)
            hook_stats["delta_norm"] = float(delta.float().norm().detach().cpu())

            new_h = h - delta
            return (new_h,) + module_output[1:]

        raise TypeError(f"Unsupported module_output type: {type(module_output)}")

    print("=" * 100, flush=True)
    print("[BASELINE FORWARD]", flush=True)
    with torch.no_grad():
        out0 = model(**inputs)
    print("baseline forward ok. output type:", type(out0), flush=True)

    print("=" * 100, flush=True)
    print("[STEERED FORWARD]", flush=True)
    handle = block.register_forward_hook(steering_hook)

    with torch.no_grad():
        out1 = model(**inputs)

    handle.remove()

    print("steered forward ok. output type:", type(out1), flush=True)
    print("hook_stats:", hook_stats, flush=True)

    if hook_stats["called"] <= 0:
        raise RuntimeError("Hook was not called.")

    print("=" * 100, flush=True)
    print("[DONE] hook smoke test passed.", flush=True)


if __name__ == "__main__":
    main()
