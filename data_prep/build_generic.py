"""Build a ShareGPT training JSONL from prepared records.

Input is a JSONL of records, one per line, that already pair the images with the
target answer — i.e. the dataset-specific extraction (DICOM->PNG, edit-instruction
lookup, caption selection, dot-geometry computation) has already been done. This
script only applies the correct conversation template (see ``formats.py``).

Record schema (one JSON object per line):
    {"images": ["a.png", "b.png"], "target": "the answer"}
    # distance_area also accepts an optional "n" = number of images (2..5):
    {"images": ["i1.png", ..., "i5.png"], "target": "0.2555", "n": 5}

Usage:
    python -m data_prep.build_generic --task levircc \\
        --records raw_levircc.jsonl --out data/levircc_train.jsonl

Supported tasks: clevr_multichange, diffvqa, imgedit, levircc, distance_area.
For CLEVR-Multi-Change starting from raw scene graphs, use
``data_prep/build_clevr_multichange.py`` instead (it derives the target captions).
"""

import argparse
import json

from .formats import build_sample


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True,
                    choices=["clevr_multichange", "diffvqa", "imgedit", "levircc", "distance_area"])
    ap.add_argument("--records", required=True, help="input JSONL of {images, target[, n]}")
    ap.add_argument("--out", required=True, help="output ShareGPT JSONL")
    args = ap.parse_args()

    n_in = n_out = 0
    with open(args.records) as fi, open(args.out, "w") as fo:
        for line in fi:
            line = line.strip()
            if not line:
                continue
            n_in += 1
            rec = json.loads(line)
            sample = build_sample(
                args.task, rec["images"], rec["target"], subtask_images=rec.get("n"),
            )
            fo.write(json.dumps(sample) + "\n")
            n_out += 1
    print(f"{args.task}: wrote {n_out}/{n_in} samples -> {args.out}")


if __name__ == "__main__":
    main()
