"""Public entry point: inject the Stateful Visual Encoder into a VLM.

``inject_sve`` routes by ``model.config.model_type`` to the matching family
adapter and applies the paper's final **Cross+FFN** recipe in place: a
cross-attention + FFN block before every ViT block, with Q/K/V cloned from the
host block, output projections zero/small-initialized, predecessor K/V
stop-gradient'd, and a Z1 self-fallback for each sequence's first image.
"""

import logging

logger = logging.getLogger(__name__)

# model_type -> "import path:function". The five families validated in the paper.
_FAMILIES = {
    "qwen3_5":  ("sve.families.qwen3_5", "inject"),
    "qwen3_vl": ("sve.families.qwen3_vl", "inject"),
    "glm4v":    ("sve.families.glm4v", "inject"),
    "internvl": ("sve.families.internvl", "inject"),
    "gemma3":   ("sve.families.gemma3", "inject"),
}


def _resolve(model_type: str):
    key = model_type
    if key not in _FAMILIES:
        # tolerate variants like "internvl_chat", "gemma3_text"
        for k in _FAMILIES:
            if model_type.startswith(k):
                key = k
                break
    if key not in _FAMILIES:
        raise ValueError(
            f"SVE has no adapter for model_type={model_type!r}. "
            f"Supported: {sorted(_FAMILIES)}."
        )
    import importlib
    mod_path, fn = _FAMILIES[key]
    return getattr(importlib.import_module(mod_path), fn)


def inject_sve(model, proj_init_std: float = 0.0):
    """Add the Stateful Visual Encoder to a pretrained VLM, in place.

    Follows the paper's final recipe exactly: a Cross+FFN block before
    **every** ViT block, with the head count matching the host vision
    encoder. These are fixed by the method — the only choice is the output-
    projection init below.

    Args:
        model: a loaded HF VLM (one of the five supported families). The vision
            tower must be trainable — SVE trains end-to-end.
        proj_init_std: output-projection init. ``0.0`` (default) → zero-init, an
            exact no-op at start (synthetic-task recipe). ``1e-4`` → tiny-normal
            (real-world recipe). See paper Tab 16.

    Returns:
        Number of trainable SVE parameters added.
    """
    model_type = getattr(model.config, "model_type", "") or ""
    inject_fn = _resolve(model_type)
    n_params = inject_fn(model, proj_init_std=proj_init_std)
    logger.info(
        "[SVE] injected into %s (proj_init_std=%s); trainable SVE params: %s",
        model_type, proj_init_std, f"{n_params:,}",
    )
    return n_params
