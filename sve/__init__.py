"""Stateful Visual Encoder (SVE).

A minimal, self-contained implementation of the paper's winning **Cross+FFN**
design: cross-image conditioning injected inside a pretrained VLM's vision
encoder so the current image's features attend to the previous image's features
before reaching the language model.

Usage::

    from transformers import AutoModelForImageTextToText
    from sve import inject_sve

    model = AutoModelForImageTextToText.from_pretrained("...", trust_remote_code=True)
    inject_sve(model)                      # auto-detects family, applies the recipe
    # ... then finetune end-to-end (vision tower trainable).

Supported families: Qwen3.5, Qwen3-VL, GLM-4.6V, InternVL 3.5, Gemma-3.
"""

from .cross_ffn import CrossFFNBlock
from .inject import inject_sve

__all__ = ["inject_sve", "CrossFFNBlock"]
