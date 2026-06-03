"""Build CLEVR-Multi-Change ShareGPT JSONL from rendered scene pairs.

Each scene pair is a before/after PNG plus a scene JSON whose ``change_record``
lists the 4 simultaneous edits (add / delete / move / replace) and the object
attributes involved. We turn each change into one sentence (randomly choosing one
of 5 equivalent templates per change type, seeded for reproducibility) and join
them into the target caption.

Expected layout (from the CLEVR-Multi-Change generator; see docs/DATA.md):
    <root>/images/CLEVR_new_<idx>.png          # before
    <root>/images/CLEVR_new_<idx>_after.png    # after
    <root>/scenes/CLEVR_new_<idx>.json         # change_record + object attrs

Usage:
    python -m data_prep.build_clevr_multichange --root data/clevr \\
        --out_train data/clevr_multichange_train.jsonl \\
        --out_test  data/clevr_multichange_test.jsonl
"""

import argparse
import json
import os
import random

CAPTION_TEMPLATES = {
    "add": ["A {s} {c} {t} {z} has been added.", "A {s} {c} {t} {z} shows up.",
            "There is a new {s} {c} {t} {z}.", "A new {s} {c} {t} {z} is visible.",
            "Someone added a {s} {c} {t} {z}."],
    "delete": ["The {s} {c} {t} {z} has disappeared.", "The {s} {c} {t} {z} is no longer there.",
               "The {s} {c} {t} {z} is missing.", "There is no longer a {s} {c} {t} {z}.",
               "Someone removed the {s} {c} {t} {z}."],
    "move": ["The {s} {c} {t} {z} changed its location.", "The {s} {c} {t} {z} is in a different location.",
             "The {s} {c} {t} {z} was moved from its original location.", "The {s} {c} {t} {z} has been moved.",
             "Someone changed location of the {s} {c} {t} {z}."],
    "replace": ["The {s} {c} {t} {z} was replaced by a {s1} {c1} {t1} {z1}.",
                "A {s1} {c1} {t1} {z1} replaced the {s} {c} {t} {z}.",
                "A {s1} {c1} {t1} {z1} is in the original position of {s} {c} {t} {z}.",
                "The {s} {c} {t} {z} gave up its position to a {s1} {c1} {t1} {z1}.",
                "Someone replaced the {s} {c} {t} {z} with a {s1} {c1} {t1} {z1}."],
}


def _instantiate(obj0, obj1, change_type):
    tmpl = random.choice(CAPTION_TEMPLATES[change_type])
    return tmpl.format(
        s=obj0["size"], c=obj0["color"], t=obj0["material"], z=obj0["shape"],
        s1=obj1["size"] if obj1 else "", c1=obj1["color"] if obj1 else "",
        t1=obj1["material"] if obj1 else "", z1=obj1["shape"] if obj1 else "",
    )


def _get_obj(change_type, order, scene):
    if change_type == "add":
        return scene["added_object"][order][0], None
    if change_type == "delete":
        return scene["dropped_object"][order], None
    if change_type == "move":
        return scene["moved_object"][order][0], None
    if change_type == "replace":
        return scene["replaced_object"][order], scene["new_object"][order][0]
    return None, None


def _caption_for(scene):
    counters = {"add": 0, "delete": 0, "move": 0, "replace": 0}
    captions = []
    for ct in scene.get("change_record", []):
        if ct not in counters:
            continue
        obj0, obj1 = _get_obj(ct, counters[ct], scene)
        counters[ct] += 1
        captions.append(_instantiate(obj0, obj1, ct))
    return " ".join(captions), len(captions)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="dir with images/ and scenes/")
    ap.add_argument("--out_train", required=True)
    ap.add_argument("--out_test", required=True)
    ap.add_argument("--n_train", type=int, default=100000)
    ap.add_argument("--n_test", type=int, default=1000)
    ap.add_argument("--min_changes", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    scenes_dir = os.path.join(args.root, "scenes")
    images_dir = os.path.join(args.root, "images")
    valid = []
    for f in sorted(os.listdir(scenes_dir)):
        if not f.endswith(".json"):
            continue
        idx = f.replace("CLEVR_new_", "").replace(".json", "")
        before = os.path.join(images_dir, f"CLEVR_new_{idx}.png")
        after = os.path.join(images_dir, f"CLEVR_new_{idx}_after.png")
        if os.path.exists(before) and os.path.exists(after):
            valid.append((before, after, os.path.join(scenes_dir, f)))

    random.seed(args.seed)
    random.shuffle(valid)
    test = valid[: args.n_test]
    train = valid[args.n_test: args.n_test + args.n_train]
    print(f"valid pairs={len(valid)}  train={len(train)}  test={len(test)}")

    for split, data, out in (("train", train, args.out_train), ("test", test, args.out_test)):
        n = 0
        with open(out, "w") as fo:
            for before, after, scene_path in data:
                scene = json.load(open(scene_path))
                if len(scene.get("change_record", [])) < args.min_changes:
                    continue
                caption, k = _caption_for(scene)
                if k < args.min_changes:
                    continue
                sample = {"messages": [
                    {"role": "user", "content": "<image>\nHere is an image of a scene with objects."},
                    {"role": "assistant", "content": "I see the scene. Please show me the next image."},
                    {"role": "user", "content": "<image>\nWhat changed between the two images?"},
                    {"role": "assistant", "content": caption},
                ], "images": [before, after]}
                fo.write(json.dumps(sample) + "\n")
                n += 1
        print(f"  {split}: {n} samples -> {out}")


if __name__ == "__main__":
    main()
