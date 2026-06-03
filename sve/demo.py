"""Standalone SVE smoke test / demo.

Loads a VLM, runs a 2-image forward pass, injects the SVE with zero-init output
projections, and verifies the logits are unchanged (the SVE block is an
exact no-op at initialization). This validates that injection is structurally
correct for the given family and that the recipe preserves the pretrained model
at the start of finetuning.

    python -m sve.demo --model_path models/Qwen3.5-4B
    python -m sve.demo --model_path models/GLM-4.6V-Flash

Run from the repo root so ``import sve`` resolves.
"""

import argparse

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from sve import inject_sve


def _two_image_batch(processor):
    img_a = Image.new("RGB", (336, 336), (200, 60, 60))
    img_b = Image.new("RGB", (336, 336), (60, 60, 200))
    messages = [{"role": "user", "content": [
        {"type": "image"}, {"type": "image"},
        {"type": "text", "text": "What changed between the two images?"},
    ]}]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    return processor(text=[prompt], images=[img_a, img_b], return_tensors="pt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).to(args.device).eval()

    inputs = _two_image_batch(processor).to(args.device)

    with torch.no_grad():
        base = model(**inputs).logits.float()

    n_params = inject_sve(model, proj_init_std=0.0)

    with torch.no_grad():
        # image_seqlens_per_sample=[2]: one sample with two images (img1 -> img2).
        sve = model(**inputs, image_seqlens_per_sample=[2]).logits.float()
    noop_diff = (base - sve).abs().max().item()

    # Active-path check: perturb every SVE output projection and confirm the
    # logits now change — proves the SVE block is actually wired into the
    # forward (a no-op test alone would pass even if the path never ran).
    n_perturbed = 0
    with torch.no_grad():
        for m in model.modules():
            if type(m).__name__ == "CrossFFNBlock":
                m.proj.weight.add_(torch.randn_like(m.proj.weight) * 0.02)
                n_perturbed += 1
        active = model(**inputs, image_seqlens_per_sample=[2]).logits.float()
    active_diff = (base - active).abs().max().item()

    print(f"\nmodel_type      : {model.config.model_type}")
    print(f"SVE params : {n_params:,}  ({n_perturbed} SVE blocks)")
    print(f"no-op max|Δ|    : {noop_diff:.3e}  (expected ~0: zero-init no-op)")
    print(f"active max|Δ|   : {active_diff:.3e}  (expected >0: path is live)")
    ok = noop_diff < 1e-2 and active_diff > 1e-2
    print("RESULT          :", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
