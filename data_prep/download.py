"""Download the prepared ShareGPT JSONLs from the Hugging Face mirror.

Caption JSONLs for ImgEdit and LEVIR-CC are mirrored at ``zwcolin/sve-data``.
Image paths inside these JSONLs are relative to a ``data/`` root — fetch the raw
images from each dataset's upstream source (see docs/DATA.md) and place them so the
relative paths resolve.

**CLEVR-Multi-Change and Dot Distance/Area are hosted separately, with images**
(both generator-rendered, no public upstream):

    from huggingface_hub import snapshot_download
    snapshot_download("zwcolin/clevr-multichange", repo_type="dataset", local_dir="data/clevr")
    snapshot_download("zwcolin/dot-distance-area", repo_type="dataset", local_dir="data/dot_area")
    # then unzip the images_part_*.zip shards in each.

**VisGym** is reproduced from its public upstream with ``data_prep.build_visgym``
(downloads + decodes the embedded per-step images); no mirror needed.

Medical-Diff-VQA (MIMIC-CXR) is *not* mirrored; obtain it from PhysioNet and build
the JSONL locally with ``data_prep.build_generic --task diffvqa``.

Usage:
    python -m data_prep.download --out data            # all sve-data JSONLs
    python -m data_prep.download --out data --files levircc_train.jsonl
"""

import argparse

REPO = "zwcolin/sve-data"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data", help="destination dir")
    ap.add_argument("--files", nargs="*", default=None,
                    help="specific filenames to fetch (default: all .jsonl)")
    ap.add_argument("--repo", default=REPO)
    args = ap.parse_args()

    from huggingface_hub import hf_hub_download, list_repo_files

    files = args.files or [f for f in list_repo_files(args.repo, repo_type="dataset")
                           if f.endswith(".jsonl")]
    for f in files:
        path = hf_hub_download(args.repo, f, repo_type="dataset", local_dir=args.out)
        print(f"downloaded {f} -> {path}")
    print(f"done: {len(files)} files in {args.out}")


if __name__ == "__main__":
    main()
