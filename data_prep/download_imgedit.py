"""Download the ImgEdit subset used by the SVE *Fine-grained Image Comparison* task.

We use 7 of ImgEdit's categories (add, adjust, background, content-memory,
content-understanding, remove, replace — style is excluded as shortcut-solvable),
which correspond to exactly **8 archives** on the public upstream dataset
``sysuyy/ImgEdit`` (https://huggingface.co/datasets/sysuyy/ImgEdit). Each archive is
stored as ``*.tar.split.NNN`` parts. This script downloads only those archives,
concatenates the parts, and extracts them so the image paths in
``imgedit_{train,test}.jsonl`` (``ImgEdit/extracted/results_*/...``) resolve.

The captions JSONL itself is at ``zwcolin/sve-data`` (``imgedit_{train,test}.jsonl``)
or build it from your own records with ``data_prep.build_generic --task imgedit``.

Usage:
    python -m data_prep.download_imgedit --out data        # -> data/ImgEdit/extracted/results_*/...
"""

import argparse
import os
import tarfile

REPO = "sysuyy/ImgEdit"

# The 8 upstream archives our subset draws from. `results_remove_part0` extracts to
# a top-level `results_remove/` directory (matching the JSONL paths).
SUBSET = [
    "Singleturn/results_add_laion_part0",
    "Singleturn/results_adjust_canny_laion_part0",
    "Singleturn/results_background_laion_part0",
    "Singleturn/results_remove_part0",
    "Singleturn/results_remove_laion_part1",
    "Singleturn/results_replace_laion_part4",
    "Multiturn/results_content_memory_part2",
    "Multiturn/results_content_understanding_part2",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data", help="root; images land under <out>/ImgEdit/extracted/")
    ap.add_argument("--keep_tar", action="store_true", help="keep the reassembled .tar files")
    args = ap.parse_args()

    from huggingface_hub import hf_hub_download, list_repo_files

    all_files = list_repo_files(REPO, repo_type="dataset")
    extract_root = os.path.join(args.out, "ImgEdit", "extracted")
    os.makedirs(extract_root, exist_ok=True)
    tmp_dir = os.path.join(args.out, "ImgEdit", "_tar")
    os.makedirs(tmp_dir, exist_ok=True)

    for archive in SUBSET:
        parts = sorted(f for f in all_files if f.startswith(archive + ".tar.split."))
        if not parts:
            print(f"WARNING: no split parts found upstream for {archive}")
            continue
        tar_path = os.path.join(tmp_dir, os.path.basename(archive) + ".tar")
        print(f"{archive}: {len(parts)} parts -> concatenating")
        with open(tar_path, "wb") as out:
            for p in parts:
                local = hf_hub_download(REPO, p, repo_type="dataset")
                with open(local, "rb") as fh:
                    while chunk := fh.read(1 << 24):
                        out.write(chunk)
        with tarfile.open(tar_path) as t:
            t.extractall(extract_root)
        if not args.keep_tar:
            os.remove(tar_path)
        print(f"  extracted -> {extract_root}/")
    if not args.keep_tar:
        try:
            os.rmdir(tmp_dir)
        except OSError:
            pass
    print(f"done. ImgEdit subset under {extract_root}/results_*/")


if __name__ == "__main__":
    main()
