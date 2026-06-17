"""离线自检：不需要 GPU/真权重。验证 SAE 加载格式 + 整条记录流水线（mask+unmask 两个 SAE）。
跑: python selftest.py"""
import os, json, tempfile, torch, torch.nn as nn, torch.nn.functional as F


def test_sae_loader():
    from cdg.sae import load_sae, sae_ckpt_path
    d, n, k = 256, 1024, 80
    enc = nn.Linear(d, n); dec = nn.Linear(n, d, bias=False); b_dec = torch.randn(d)
    state = {"encoder.weight": enc.weight.data.clone(), "encoder.bias": enc.bias.data.clone(),
             "decoder.weight": dec.weight.data.clone(), "b_dec": b_dec.clone(),
             "k": torch.tensor(k, dtype=torch.int)}
    root = tempfile.mkdtemp()
    ddir = os.path.join(root, "resid_post_layer_16", "trainer_1"); os.makedirs(ddir)
    torch.save(state, os.path.join(ddir, "ae.pt"))
    json.dump({"trainer": {"k": k}, "buffer": {"dlm_mask_policy": "mask"}},
              open(os.path.join(ddir, "config.json"), "w"))
    ckpt, cfgp = sae_ckpt_path(root, 16, 1)
    sae = load_sae(ckpt, config_path=cfgp)
    x = torch.randn(4, d); z = sae.encode(x)
    pre = F.relu((x - b_dec) @ state["encoder.weight"].t() + state["encoder.bias"])
    tv, ti = torch.topk(pre, k, -1); z_ref = torch.zeros_like(pre).scatter_(-1, ti, tv)
    assert torch.allclose(z, z_ref, atol=1e-5), "encode mismatch"
    assert (z != 0).sum(-1).eq(k).all(), "k nonzeros wrong"
    print(f"[OK] SAE loader: d={sae.d_model} n={sae.n_features} k={sae.k}, encode/decode 与参考一致")


def test_pipeline():
    from cdg.config import get_backend_config
    from cdg.backends import build_runner
    from cdg.recorder import Recorder
    from cdg.data import load_prompt_root
    cfg = get_backend_config("llada")
    runner = build_runner(cfg, dummy=True)
    cases = load_prompt_root("./example_prompts/jb")
    case = cases[0]
    rec = Recorder(runner.bundles, cfg.record, tokenizer=runner.tokenizer)
    rec.begin({"case_id": case.case_id, "variant": case.variant, "is_neutral": case.is_neutral,
               "model_name": cfg.name, "seed": 0}, total_steps=cfg.decode.steps)
    runner.generate(case.messages, rec)
    bundles = [b.name for b in runner.bundles]
    assert bundles == ["llada_mask", "llada_unmask"], bundles
    assert set(rec.rec.sae) == {"llada_mask", "llada_unmask"}
    assert sorted(rec.rec.sae["llada_mask"]) == [0.1, 0.25, 0.5, 0.75, 1.0]
    assert list(rec.rec.sae["llada_mask"][0.5]) == [16, 26]
    assert rec.rec.entropy[0.5]["topk_entropy"].shape[0] == cfg.decode.gen_length
    print(f"[OK] pipeline: SAEs={bundles}, layers={list(cfg.record_layers)}, "
          f"fracs={sorted(rec.rec.sae['llada_mask'])}")


def test_judge_parse():
    from cdg.judge import _parse
    assert _parse('```json\n{"asr":1,"label":"comply"}\n```', "asr")["success"] == 1
    assert _parse('{"flip":0}', "flip")["success"] == 0
    print("[OK] judge JSON 解析")


if __name__ == "__main__":
    test_sae_loader()
    test_pipeline()
    test_judge_parse()
    print("\n全部通过 ✅")
