from __future__ import annotations

import json
import math
import os
import sys
from functools import partial
from typing import Optional

import numpy as np
import torch

# Make the parent CrAM repo importable (for the Qwen2 per-head-mask patch CrAM needs).
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from utils.re_weighting import Re_Weighting_Strategy

MODEL_NAME = "Qwen/Qwen1.5-0.5B-Chat"
DEFAULT_SELECTED_HEADS = os.path.join(
    REPO_ROOT, "datasets", "nq_0.5b", "qwen", "selected_heads.json"
)


def select_device(prefer: Optional[str] = None) -> str:
    if prefer:
        return prefer
    if torch.cuda.is_available():
        return "cuda"
    if (
        getattr(torch.backends, "mps", None) is not None
        and torch.backends.mps.is_available()
    ):
        return "mps"
    return "cpu"


def _np(t: torch.Tensor, decimals: int = 5):
    """Tensor -> rounded nested python list (strict-JSON-friendly: no inf/nan)."""
    a = t.detach().to("cpu", torch.float32)
    a = torch.nan_to_num(a, nan=0.0, posinf=1e9, neginf=-1e9)
    return np.round(a.numpy(), decimals).tolist()


# --------------------------------------------------------------------- RoPE
def rope_cos_sin(seq_len: int, head_dim: int, theta: float, device, dtype):
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim)
    )
    pos = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(pos, inv_freq)  # (T, head_dim/2)
    emb = torch.cat((freqs, freqs), dim=-1)  # (T, head_dim)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def rotate_half(x):
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x, cos, sin):
    # x: (heads, T, head_dim); cos/sin: (T, head_dim)
    return x * cos.unsqueeze(0) + rotate_half(x) * sin.unsqueeze(0)


class TraceCapturer:
    def __init__(self, model_name: str = MODEL_NAME, device: Optional[str] = None):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = select_device(device)
        self.dtype = torch.float32  # capture in fp32 for clean visualization numbers
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = (
            AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=self.dtype, attn_implementation="eager"
            )
            .to(self.device)
            .eval()
        )
        # Qwen2 eager attention rejects per-head additive masks; patch it so the
        # CrAM intervention (per-head credibility re-weighting) works.
        if getattr(self.model.config, "model_type", "") == "qwen2":
            from utils.qwen2_cram_patch import enable_per_head_attention_mask

            enable_per_head_attention_mask()
        cfg = self.model.config
        self.cfg = dict(
            model=model_name,
            num_layers=cfg.num_hidden_layers,
            num_heads=cfg.num_attention_heads,
            num_kv_heads=cfg.num_key_value_heads,
            hidden=cfg.hidden_size,
            head_dim=cfg.hidden_size // cfg.num_attention_heads,
            intermediate=cfg.intermediate_size,
            vocab=cfg.vocab_size,
            rope_theta=float(getattr(cfg, "rope_theta", 1e6)),
            rms_eps=float(getattr(cfg, "rms_norm_eps", 1e-6)),
        )
        # Required by the official CrAM mask editor. Reusing this loaded model
        # avoids constructing a second model through Re_Weighting_Strategy.
        self.model_num_attention_heads = cfg.num_attention_heads
        self.selected_heads_path = os.path.abspath(
            os.path.expanduser(
                os.environ.get("CRAM_SELECTED_HEADS", DEFAULT_SELECTED_HEADS)
            )
        )
        self.selected_heads = self._load_selected_heads(
            self.selected_heads_path,
            num_layers=self.cfg["num_layers"],
            num_heads=self.cfg["num_heads"],
        )
        self.cfg["cram_selected_heads"] = {
            str(layer): heads for layer, heads in self.selected_heads.items()
        }
        self.cfg["cram_selected_head_count"] = sum(
            len(heads) for heads in self.selected_heads.values()
        )

    @staticmethod
    def _load_selected_heads(path, num_layers, num_heads):
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"CrAM selected-head mapping not found: {path}. "
                "Generate selected_heads.json or set CRAM_SELECTED_HEADS."
            )
        with open(path, "r", encoding="utf-8") as file:
            raw = json.load(file)
        if not isinstance(raw, dict) or not raw:
            raise ValueError(f"CrAM selected-head mapping is empty or invalid: {path}")

        selected = {}
        for layer_raw, heads_raw in raw.items():
            layer = int(layer_raw)
            if not 0 <= layer < num_layers:
                raise ValueError(f"CrAM layer {layer} is outside [0, {num_layers})")
            if not isinstance(heads_raw, list):
                raise ValueError(f"CrAM heads for layer {layer} must be a list")
            heads = sorted({int(head) for head in heads_raw})
            if any(head < 0 or head >= num_heads for head in heads):
                raise ValueError(
                    f"CrAM layer {layer} contains a head outside [0, {num_heads})"
                )
            if heads:
                selected[layer] = heads

        if not selected:
            raise ValueError(f"CrAM selected-head mapping has no heads: {path}")
        return selected

    def resolve_cram_heads(self, cram):
        """Return {layer: [heads]} for this intervention.

        Explicit head_map/layers/heads values are useful for experiments. Normal
        explainer and audience requests use the generated influential-head map.
        """
        if cram is None:
            return {}
        if cram.get("head_map") is not None:
            raw_map = cram["head_map"]
            return {
                int(layer): sorted({int(head) for head in heads})
                for layer, heads in raw_map.items()
                if heads
            }
        if cram.get("layers") is not None or cram.get("heads") is not None:
            layers = (
                [int(layer) for layer in cram["layers"]]
                if cram.get("layers") is not None
                else list(range(self.cfg["num_layers"]))
            )
            heads = (
                [int(head) for head in cram["heads"]]
                if cram.get("heads") is not None
                else list(range(self.cfg["num_heads"]))
            )
            return {layer: heads for layer in layers}
        return self.selected_heads

    # --------------------------------------------------------------- hooks
    def _register(self, store):
        handles = []
        m = self.model.model  # Qwen2Model
        handles.append(
            m.embed_tokens.register_forward_hook(
                lambda mod, i, o: store.__setitem__("embed", o.detach())
            )
        )
        handles.append(
            m.norm.register_forward_hook(
                lambda mod, i, o: store.__setitem__("final_norm", o.detach())
            )
        )
        for li, layer in enumerate(m.layers):
            L = store.setdefault("layers", {}).setdefault(li, {})

            def grab(key, L=L):
                return lambda mod, i, o: L.__setitem__(key, (i[0].detach(), o.detach()))

            handles.append(layer.input_layernorm.register_forward_hook(grab("ln1")))
            handles.append(
                layer.post_attention_layernorm.register_forward_hook(grab("ln2"))
            )
            handles.append(layer.self_attn.q_proj.register_forward_hook(grab("q")))
            handles.append(layer.self_attn.k_proj.register_forward_hook(grab("k")))
            handles.append(layer.self_attn.v_proj.register_forward_hook(grab("v")))
            handles.append(layer.self_attn.o_proj.register_forward_hook(grab("o")))
            handles.append(layer.mlp.gate_proj.register_forward_hook(grab("gate")))
            handles.append(layer.mlp.up_proj.register_forward_hook(grab("up")))
            handles.append(layer.mlp.down_proj.register_forward_hook(grab("down")))
        return handles

    # ------------------------------------------------------------- capture
    def official_cram_hook(self, attention_weight, head_idx):
        """Bind the official repository CrAM mask editor to this model."""
        return partial(
            Re_Weighting_Strategy.edit_attention_mask,
            self,
            attention_weight=attention_weight,
            head_idx=head_idx,
        )

    @staticmethod
    def _attention_mask_observer(store, layer_idx):
        """Capture the mask after any official CrAM pre-hook has edited it."""

        def hook(module, args, kwargs):
            mask = kwargs.get("attention_mask")
            store.setdefault("attention_masks", {})[layer_idx] = (
                None if mask is None else mask.detach().clone()
            )
            return args, kwargs

        return hook

    @torch.no_grad()
    def capture(
        self,
        prompt,
        max_tokens=24,
        topk=12,
        use_chat_template=False,
        cram=None,
        compact=True,
        display_indices=None,
    ):
        """cram: None for vanilla, else {span:(start,end), credibility:c, layers:?, heads:?}.
        compact: store the heavy 1024/2816-d vectors only for the prediction token.
        display_indices: optional full-prompt token indices to serialize. The model
        still sees the complete prompt, but the browser receives a focused view."""
        if use_chat_template:
            messages = [{"role": "user", "content": prompt}]
            prompt = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
        enc = self.tokenizer(
            prompt, return_tensors="pt", add_special_tokens=not use_chat_template
        )
        input_ids = enc["input_ids"][:, :max_tokens].to(self.device)
        T = input_ids.shape[1]
        all_tokens = [self.tokenizer.decode([tid]) for tid in input_ids[0].tolist()]
        H, hd, theta, nL = (
            self.cfg["num_heads"],
            self.cfg["head_dim"],
            self.cfg["rope_theta"],
            self.cfg["num_layers"],
        )
        focus = T - 1
        if display_indices is None:
            shown = list(range(T))
        else:
            shown = sorted({int(i) for i in display_indices if 0 <= int(i) < T})
            if focus not in shown:
                shown.append(focus)
                shown.sort()
        shown_tensor = torch.tensor(shown, device=self.device, dtype=torch.long)
        shown_focus = shown.index(focus)

        # CrAM per-key credibility weight + log bias
        weight = torch.ones(T, dtype=self.dtype, device=self.device)
        cram_head_map = self.resolve_cram_heads(cram)
        if cram is not None:
            s, e = cram["span"]
            weight[s:e] = float(cram.get("credibility", 0.0))
        # Match run_RAG_with_attention_weighting in the official repository:
        # edit_attention_mask receives log credibility, including log(0)=-inf.
        official_attention_weight = torch.log(weight.unsqueeze(0))

        store = {}
        handles = self._register(store)
        for li, layer in enumerate(self.model.model.layers):
            heads = cram_head_map.get(li) if cram is not None else None
            if heads:
                handles.append(
                    layer.self_attn.register_forward_pre_hook(
                        self.official_cram_hook(official_attention_weight, heads),
                        with_kwargs=True,
                    )
                )
            # Registered second: this records precisely what Qwen consumes.
            handles.append(
                layer.self_attn.register_forward_pre_hook(
                    self._attention_mask_observer(store, li), with_kwargs=True
                )
            )
        try:
            out = self.model(input_ids, output_attentions=True, use_cache=False)
        finally:
            for h in handles:
                h.remove()

        cos, sin = rope_cos_sin(T, hd, theta, self.device, self.dtype)
        def vec(t2d):
            return (
                _np(t2d[focus]) if compact else _np(t2d.index_select(0, shown_tensor))
            )

        trace = {
            "config": self.cfg,
            "prompt": prompt,
            "tokens": [all_tokens[i] for i in shown],
            "token_ids": [input_ids[0, i].item() for i in shown],
            "source_indices": shown,
            "source_token_count": T,
            "compact": compact,
            "focus": shown_focus,
            "cram": None
            if cram is None
            else {
                "span": list(cram["span"]),
                "credibility": float(cram.get("credibility", 0.0)),
                "weight": _np(weight.index_select(0, shown_tensor)),
                "selected_heads": {
                    str(layer): heads for layer, heads in cram_head_map.items()
                },
                "implementation": "utils.re_weighting.Re_Weighting_Strategy.edit_attention_mask",
                "source": os.path.join(REPO_ROOT, "utils", "re_weighting.py"),
            },
            "embedding": _np(store["embed"][0].index_select(0, shown_tensor)),
            "final_norm": vec(store["final_norm"][0]),
            "layers": [],
        }

        for li in range(nL):
            Ld = store["layers"][li]
            q = Ld["q"][1][0].view(T, H, hd).transpose(0, 1)  # (H, T, hd)
            k = Ld["k"][1][0].view(T, H, hd).transpose(0, 1)
            v = Ld["v"][1][0].view(T, H, hd).transpose(0, 1)
            q_rope = apply_rope(q, cos, sin)
            k_rope = apply_rope(k, cos, sin)
            scores = torch.matmul(q_rope, k_rope.transpose(-1, -2))
            scaled = scores / math.sqrt(hd)
            captured_mask = store["attention_masks"][li]
            if captured_mask is None:
                raise RuntimeError(
                    f"Qwen layer {li} did not receive an attention mask; "
                    "a captured-mask visualization cannot be produced."
                )
            if captured_mask.shape[1] == 1:
                captured_mask = captured_mask.expand(-1, H, -1, -1)
            captured_mask = captured_mask[0, :, :T, :T]
            # These are returned by the real Qwen forward pass, after the
            # official repository hook edits its attention mask.
            softmax = out.attentions[li][0]
            attn_out = torch.matmul(softmax, v)

            def square(t2d):
                return t2d.index_select(0, shown_tensor).index_select(1, shown_tensor)

            heads = []
            for h in range(H):
                heads.append(
                    {
                        "q": _np(q[h].index_select(0, shown_tensor)),
                        "k": _np(k[h].index_select(0, shown_tensor)),
                        "v": _np(v[h].index_select(0, shown_tensor)),
                        "q_rope": _np(q_rope[h].index_select(0, shown_tensor)),
                        "k_rope": _np(k_rope[h].index_select(0, shown_tensor)),
                        "scores": _np(square(scores[h])),
                        "scaled": _np(square(scaled[h])),
                        "masked": _np(square(captured_mask[h])),
                        "softmax": _np(square(softmax[h])),
                        "attn_out": _np(attn_out[h].index_select(0, shown_tensor)),
                    }
                )
            gate = Ld["gate"][1][0]
            up = Ld["up"][1][0]
            act = torch.nn.functional.silu(gate) * up
            trace["layers"].append(
                {
                    "index": li,
                    "ln1_in": vec(Ld["ln1"][0][0]),
                    "ln1_out": vec(Ld["ln1"][1][0]),
                    "ln2_in": vec(Ld["ln2"][0][0]),
                    "ln2_out": vec(Ld["ln2"][1][0]),
                    "attention_source": "Qwen output_attentions",
                    "mask_source": (
                        "official CrAM edit_attention_mask"
                        if cram is not None and li in cram_head_map
                        else "Qwen causal attention mask"
                    ),
                    "heads": heads,
                    "o_proj": vec(Ld["o"][1][0]),
                    "mlp": {
                        "gate": vec(gate),
                        "up": vec(up),
                        "act": vec(act),
                        "down": vec(Ld["down"][1][0]),
                    },
                }
            )

        logits = out.logits[0, -1]
        probs = torch.softmax(logits, dim=-1)
        top = torch.topk(probs, topk)
        trace["next_token"] = [
            {
                "token": self.tokenizer.decode([int(i)]),
                "id": int(i),
                "prob": float(p) if math.isfinite(p) else 0.0,
            }
            for p, i in zip(top.values.tolist(), top.indices.tolist())
        ]
        return trace


if __name__ == "__main__":
    import argparse
    import json
    import time

    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--max-tokens", type=int, default=16)
    args = ap.parse_args()

    t0 = time.time()
    cap = TraceCapturer(device=args.device)
    print(f"loaded {MODEL_NAME} on {cap.device} in {time.time() - t0:.1f}s")
    tr = cap.capture(args.prompt, max_tokens=args.max_tokens)
    print(f"tokens={tr['tokens']}")
    print(f"layers={len(tr['layers'])} attentions captured from Qwen forward pass")
    print(
        "top next tokens:",
        [(t["token"], round(t["prob"], 3)) for t in tr["next_token"][:5]],
    )
    if args.out:
        with open(args.out, "w") as f:
            json.dump(tr, f)
        print(f"wrote {args.out} ({os.path.getsize(args.out) / 1e6:.1f} MB)")
