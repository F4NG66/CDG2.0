from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F


def add_gumbel_noise(logits, temperature):
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    num = torch.zeros(mask_num.size(0), steps, device=mask_index.device,
                      dtype=torch.int64) + base
    for i in range(mask_num.size(0)):
        num[i, :remainder[i]] += 1
    return num


@torch.no_grad()
def _confidence(logits, x0, strategy):
    if strategy == "random":
        return torch.rand(x0.shape, device=x0.device, dtype=torch.float64)
    p = F.softmax(logits.to(torch.float64), dim=-1)
    if strategy == "low_confidence":
        return torch.gather(p, -1, x0.unsqueeze(-1)).squeeze(-1)
    if strategy == "margin":
        top2 = torch.topk(p, 2, dim=-1).values
        return top2[..., 0] - top2[..., 1]
    if strategy == "entropy":                      # Dream alg='entropy'
        ent = -(p * torch.log(p + 1e-12)).sum(-1)
        return -ent
    raise NotImplementedError(strategy)


@torch.no_grad()
def denoise(runner, x, attention_mask, *, steps, gen_length, prompt_len,
            block_length=None, temperature=0.0, remask="low_confidence",
            mask_id, recorder=None, fill_all_masks=True):
    """Diffusion denoising.

    fill_all_masks=True  -> every mask_id in the WHOLE sequence is fillable
                            (injected-input blanks + appended output). Needed for
                            template-injection attacks where blanks live in the
                            prompt region.
    fill_all_masks=False -> original block-restricted behaviour (output only).
    """
    global_step = 0

    if fill_all_masks:
        # one unified schedule over all initial masks (input + output)
        fillable = (x == mask_id)                       # (B, T) fixed at entry
        ntt = get_num_transfer_tokens(fillable, steps)
        for i in range(steps):
            global_step += 1
            mask_index = (x == mask_id)                 # currently-masked
            runner.hooks.clear()
            logits = runner.forward(x, attention_mask)
            logits_noised = add_gumbel_noise(logits, temperature)
            x0 = torch.argmax(logits_noised, dim=-1)
            conf = _confidence(logits, x0, remask)
            x0 = torch.where(mask_index, x0, x)
            neg = torch.tensor(-np.inf, device=x.device, dtype=conf.dtype)
            confidence = torch.where(mask_index, conf, neg)

            if recorder is not None and recorder.should_record(global_step):
                recorder.record(global_step, x, logits, runner.hooks.buffers)

            transfer = torch.zeros_like(x0, dtype=torch.bool)
            for j in range(confidence.shape[0]):
                k = int(ntt[j, i])
                if k > 0:
                    _, sel = torch.topk(confidence[j], k=k)
                    transfer[j, sel] = True
            x[transfer] = x0[transfer]
        return x

    # ---- original output-only path (kept for clean/neutral baselines) ----
    block_length = block_length or gen_length
    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    assert steps % num_blocks == 0
    steps_per_block = steps // num_blocks
    for nb in range(num_blocks):
        b0 = prompt_len + nb * block_length
        b1 = prompt_len + (nb + 1) * block_length
        block_mask = (x[:, b0:b1] == mask_id)
        ntt = get_num_transfer_tokens(block_mask, steps_per_block)
        for i in range(steps_per_block):
            global_step += 1
            mask_index = (x == mask_id)
            runner.hooks.clear()
            logits = runner.forward(x, attention_mask)
            logits_noised = add_gumbel_noise(logits, temperature)
            x0 = torch.argmax(logits_noised, dim=-1)
            conf = _confidence(logits, x0, remask)
            x0 = torch.where(mask_index, x0, x)
            neg = torch.tensor(-np.inf, device=x.device, dtype=conf.dtype)
            confidence = torch.where(mask_index, conf, neg)
            confidence[:, :b0] = -np.inf
            confidence[:, b1:] = -np.inf
            if recorder is not None and recorder.should_record(global_step):
                recorder.record(global_step, x, logits, runner.hooks.buffers)
            transfer = torch.zeros_like(x0, dtype=torch.bool)
            for j in range(confidence.shape[0]):
                k = int(ntt[j, i])
                if k > 0:
                    _, sel = torch.topk(confidence[j], k=k)
                    transfer[j, sel] = True
            x[transfer] = x0[transfer]
    return x
