"""Build the VisGym behavioral-cloning data from the public upstream dataset.

VisGym (`VisGym/visgym_data`) is public and self-contained: each task/split is a
set of `batch_*.jsonl` files, one episode per line, and every step's observation
image is embedded as **base64** in the trajectory. This script downloads the
requested tasks/splits, decodes the per-step images to disk, and writes a
**ShareGPT** JSONL whose image paths resolve against them — so VisGym is fully
reproducible from upstream without us hosting any images.

Mapping (upstream `history[step]` -> conversation):
  * human turn = ``"<image>" + step["prompt"]``   (step 0 = full task prompt;
    later steps = environment feedback + step counter)
  * gpt turn   = ``step["vlm_output"]``           (``<think>...</think><answer>(action)</answer>``)
  * image      = ``decode_base64(step["image"])`` -> one JPEG per step

VisGym uses the older ShareGPT schema (``conversations`` with ``{"from","value"}``)
and is trained with ``mask_history=False`` (every gpt turn is a real action).

Usage:
    # the four tasks from the paper, train split, cap 100k episodes each:
    python -m data_prep.build_visgym --split train --limit 100000 \
        --images_root data --out data/visgym_train.jsonl
    # eval (val split):
    python -m data_prep.build_visgym --split val \
        --images_root data --out data/visgym_eval.jsonl
"""

import argparse
import base64
import json
import os

REPO = "VisGym/visgym_data"
DEFAULT_TASKS = [
    "matchstick_rotation",
    "mental_rotation_3d_cube",
    "mental_rotation_3d_objaverse",
    "patch_reassembly",
]


def _episode_to_sample(task, split, epid, episode, images_root):
    """Decode one episode's steps to images and return a ShareGPT sample (or None)."""
    history = episode.get("history") or []
    if not history:
        return None
    rel_dir = os.path.join("visgym", task, split, epid)
    abs_dir = os.path.join(images_root, rel_dir)
    os.makedirs(abs_dir, exist_ok=True)

    conversations, images = [], []
    for step in history:
        b64 = step.get("image")
        if not b64:
            return None
        fn = f"step_{step['step']:03d}.jpg"
        with open(os.path.join(abs_dir, fn), "wb") as f:
            f.write(base64.b64decode(b64))
        images.append(os.path.join(rel_dir, fn))
        conversations.append({"from": "human", "value": "<image>" + step["prompt"]})
        conversations.append({"from": "gpt", "value": step["vlm_output"]})

    return {"episode": episode.get("episode"), "conversations": conversations, "images": images}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="*", default=DEFAULT_TASKS)
    ap.add_argument("--split", default="train", choices=["train", "val"])
    ap.add_argument("--images_root", default="data", help="root under which visgym/... images are written")
    ap.add_argument("--out", required=True, help="output ShareGPT JSONL")
    ap.add_argument("--limit", type=int, default=None, help="max episodes per task (default: all)")
    args = ap.parse_args()

    from huggingface_hub import hf_hub_download, list_repo_files

    all_files = list_repo_files(REPO, repo_type="dataset")
    n_total = 0
    with open(args.out, "w") as fout:
        for task in args.tasks:
            prefix = f"{task}/{args.split}/"
            batches = sorted(f for f in all_files if f.startswith(prefix) and f.endswith(".jsonl"))
            n_task = 0
            for b, batch_file in enumerate(batches):
                if args.limit is not None and n_task >= args.limit:
                    break
                local = hf_hub_download(REPO, batch_file, repo_type="dataset")
                for e, line in enumerate(open(local)):
                    if args.limit is not None and n_task >= args.limit:
                        break
                    epid = f"b{b:06d}_e{e:03d}"
                    sample = _episode_to_sample(task, args.split, epid, json.loads(line), args.images_root)
                    if sample is None:
                        continue
                    fout.write(json.dumps(sample) + "\n")
                    n_task += 1
                    n_total += 1
            print(f"{task}/{args.split}: {n_task} episodes")
    print(f"wrote {n_total} episodes -> {args.out}  (images under {args.images_root}/visgym/)")


if __name__ == "__main__":
    main()
