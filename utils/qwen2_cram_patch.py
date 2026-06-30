import inspect
import textwrap

import transformers.models.qwen2.modeling_qwen2 as _qwen2

# The exact boolean condition of the offending assertion in transformers 4.40.x.
_ASSERT_COND = "attention_mask.size() != (bsz, 1, q_len, kv_seq_len)"


def enable_per_head_attention_mask():
    """Idempotently patch ``Qwen2Attention.forward`` to allow per-head masks.

    Returns the patched class (or ``None`` if no patch was needed)."""
    cls = _qwen2.Qwen2Attention
    if getattr(cls.forward, "_cram_patched", False):
        return cls

    src = inspect.getsource(cls.forward)
    src = textwrap.dedent(src)
    # Drop any decorator lines so the recompiled source is a bare function def.
    lines = src.splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.lstrip().startswith("def forward"))
    src = "\n".join(lines[start:])

    if _ASSERT_COND not in src:
        # Newer transformers already relaxed this (slices instead of asserting);
        # nothing to do, just mark as handled.
        cls.forward._cram_patched = True
        return cls

    # `if attention_mask.size() != (...)` -> `if False:` : the raise never fires,
    # and the following per-head `attn_weights + attention_mask` add runs as-is.
    src = src.replace(_ASSERT_COND, "False")

    ns = {}
    # Compile against the module's globals so torch / repeat_kv / apply_rotary_pos_emb
    # / Cache / math / nn etc. resolve exactly as in the shipped implementation.
    exec(compile(src, "<cram_qwen2_patch>", "exec"), _qwen2.__dict__, ns)
    forward = ns["forward"]
    forward._cram_patched = True
    cls.forward = forward
    return cls
