from __future__ import annotations
import torch
import torch.nn as nn

from ..config import BackendConfig
from ..hooks import HookManager
from ..sae import TopKSAE, SAEBundle
from .dlm_runner import DLMRunner


class _FakeTokenizer:
    """Whitespace word tokenizer; mask_token maps to a fixed mask_id."""

    def __init__(self, mask_token: str, mask_id: int):
        self.mask_token = mask_token
        self.mask_id = mask_id
        self.w2i: dict[str, int] = {mask_token: mask_id}
        self.i2w: dict[int, str] = {mask_id: mask_token}
        self._next = 1

    def apply_chat_template(self, messages, add_generation_prompt=True,
                            tokenize=False):
        body = " ".join(m["content"] for m in messages)
        return f"USER: {body} ASSISTANT:"

    def _split(self, text: str):
        for tok in (self.mask_token, "<<TPL>>", "<</TPL>>"):
            text = text.replace(tok, f" {tok} ")
        return [t for t in text.split() if t]

    def __call__(self, text, add_special_tokens=False, return_tensors=None):
        ids = []
        for w in self._split(text):
            if w not in self.w2i:
                while self._next == self.mask_id:
                    self._next += 1
                self.w2i[w] = self._next
                self.i2w[self._next] = w
                self._next += 1
            ids.append(self.w2i[w])
        # HF returns a FLAT list for a single string when return_tensors is None
        return {"input_ids": ids}

    def decode(self, ids, skip_special_tokens=True):
        out = []
        for i in ids:
            w = self.i2w.get(int(i), f"<u{int(i)}>")
            if skip_special_tokens and w == self.mask_token:
                continue
            out.append(w)
        return " ".join(out)


class _FakeBlock(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.lin = nn.Linear(d, d)

    def forward(self, h):
        return torch.tanh(self.lin(h)) + h


class _FakeConfig:
    def __init__(self, mask_id):
        self.mask_token_id = mask_id


class _FakeModel(nn.Module):
    def __init__(self, vocab, d, n_blocks, mask_id):
        super().__init__()
        self.config = _FakeConfig(mask_id)
        self.emb = nn.Embedding(vocab, d)
        self.transformer = nn.Module()
        self.transformer.blocks = nn.ModuleList([_FakeBlock(d) for _ in range(n_blocks)])
        self.head = nn.Linear(d, vocab)
        self.device = torch.device("cpu")

    def forward(self, x, attention_mask=None):
        h = self.emb(x)
        for blk in self.transformer.blocks:
            h = blk(h)
        logits = self.head(h)
        return _FakeOutput(logits)


class _FakeOutput:
    """Mimics a HF CausalLMOutput so DLMRunner.forward finds `.logits`."""
    def __init__(self, logits):
        self.logits = logits


class DummyRunner(DLMRunner):
    """Exercises the REAL build_inputs/generate/recorder path on CPU."""

    def __init__(self, cfg: BackendConfig, sae_root: str = "", device: str = "cpu"):
        # shrink config so a tiny fake model is enough
        cfg.record_layers = (1, 2, 3)
        cfg.layer_to_block_offset = 0
        cfg.decode.gen_length = 16
        cfg.decode.steps = 8
        cfg.decode.block_length = 16
        self.cfg = cfg
        self.device = "cpu"
        self._d = 32
        self._vocab = (cfg.mask_id or 126336) + 8
        self._load_model()
        self._resolve_mask_id()
        self._load_bundles(sae_root)
        self.hooks = HookManager(self.model, cfg.record_layers,
                                 offset=cfg.layer_to_block_offset)
        self._steer = None

    def _load_model(self):
        torch.manual_seed(0)
        self.tokenizer = _FakeTokenizer(self.cfg.mask_token, self.cfg.mask_id or 126336)
        self.model = _FakeModel(self._vocab, self._d, n_blocks=6,
                                mask_id=self.cfg.mask_id or 126336).eval()

    def _load_bundles(self, sae_root: str):
        torch.manual_seed(1)
        self.bundles = []
        for spec in self.cfg.saes:
            saes = {}
            for layer in self.cfg.record_layers:
                sae = TopKSAE(d_model=self._d, n_features=64, k=8)
                with torch.no_grad():
                    sae.W_enc.normal_(0, 0.1)
                    sae.W_dec.normal_(0, 0.1)
                saes[layer] = sae.eval()
            self.bundles.append(SAEBundle(name=spec.name, kind=spec.kind, saes=saes))


def build_runner(cfg: BackendConfig, sae_root: str = "", device: str = "cuda",
                 dummy: bool = False):
    if dummy:
        return DummyRunner(cfg, sae_root=sae_root, device="cpu")
    return DLMRunner(cfg, sae_root=sae_root, device=device)
