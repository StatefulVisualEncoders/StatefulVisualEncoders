# Dataset preparation

All six task families are trained from **ShareGPT JSONL**. Each line is
`{"messages": [...], "images": [...]}` where the *k*-th `<image>` tag binds to
`images[k]`. The exact conversation templates (system prompts, filler turns) live
in [`data_prep/formats.py`](../data_prep/formats.py).

Two ways to get the data:

```bash
# 1) Download the prepared JSONLs for the redistributable datasets:
python -m data_prep.download --out data

# 2) Build from raw yourself (see per-dataset notes below):
python -m data_prep.build_clevr_multichange --root data/clevr \
    --out_train data/clevr_multichange_train.jsonl --out_test data/clevr_multichange_test.jsonl
python -m data_prep.build_generic --task levircc --records raw_levircc.jsonl \
    --out data/levircc_train.jsonl
```

Common conventions (paper Appendix B):
- `<image>` appears at the start of a user message, followed by a newline + text.
- Training uses `mask_history=True` for the single-shot tasks: only the final
  assistant turn (the answer) is supervised; the intermediate "filler" turns
  provide context but are masked. **VisGym** uses `mask_history=False` (every
  assistant turn is a real action).
- Seed **42** for all data preparation, so the sample sequence is identical across
  every model trained on a task (baseline and SVE consume byte-identical inputs).

| Task | `data_prep` `--task` | Train / Eval | Images/sample | Source license |
|------|----------------------|--------------|---------------|----------------|
| Spatial Aggregation (dot distance/area) | `distance_area` | 400k / 4k | 2–5 | AgentNet backgrounds (CC) — **images+captions hosted** at [`zwcolin/dot-distance-area`](https://huggingface.co/datasets/zwcolin/dot-distance-area) |
| Visual Differencing (CLEVR-Multi-Change) | `clevr_multichange` | 100k / 1k | 2 | CLEVR engine (BSD) — **images+captions hosted** at [`zwcolin/clevr-multichange`](https://huggingface.co/datasets/zwcolin/clevr-multichange) |
| Trajectory Cloning (VisGym) | `build_visgym` | 400k / 4k | per-step | [VisGym](https://huggingface.co/datasets/VisGym/visgym_data) — public; reproduce from upstream |
| Longitudinal Radiology (Medical-Diff-VQA) | `diffvqa` | 130k / 16k | 2 | **MIMIC-CXR — PhysioNet credentialed; not redistributable** |
| Image Comparison (ImgEdit) | `imgedit` | 301k / 1.4k | 2–3 | 8-archive subset of [`sysuyy/ImgEdit`](https://huggingface.co/datasets/sysuyy/ImgEdit) (see §5) |
| Remote Sensing (LEVIR-CC) | `levircc` | 34k / 1.9k | 2 | [`lcybuaa/LEVIR-CC`](https://huggingface.co/datasets/lcybuaa/LEVIR-CC) |

> **Licensing.** Medical-Diff-VQA derives from MIMIC-CXR, distributed via PhysioNet
> under a Data Use Agreement that prohibits redistribution. You must obtain it
> yourself (credentialed access). Check each other dataset's upstream license
> before redistributing.

The `data_prep.build_generic` builder takes records you've already prepared —
`{"images": [...], "target": "..."}` per line (the raw extraction: DICOM→PNG,
edit-instruction lookup, caption selection, dot-geometry computation) — and applies
the right conversation template. Below, what "raw" means for each task.

---

## 1. Cross-image Spatial Aggregation — Dot Distance / Area

Red dots overlaid on AgentNet screenshots (downsampled to **384×216**); the model
estimates a geometric quantity across 2–5 images. Four sub-tasks (distance over 2
images; triangle/convex-hull area over 3/4/5), 100k train + 1k eval each.

**This dataset (images + captions) is hosted in full** at
[`zwcolin/dot-distance-area`](https://huggingface.co/datasets/zwcolin/dot-distance-area)
(public). Download and unzip the shards:

```python
from huggingface_hub import snapshot_download
snapshot_download("zwcolin/dot-distance-area", repo_type="dataset", local_dir="data/dot_area")
```
```bash
cd data/dot_area && for f in images_part_*.zip; do unzip -q "$f"; done
```

To build the captions yourself, prepare records as `{"images": [...], "target":
"0.2555", "n": <2..5>}` where `target` is the precomputed normalized distance/area
(4 decimals) and `n` selects the system prompt, then `build_generic --task distance_area`.

## 2. Multi-object Visual Differencing — CLEVR-Multi-Change

Two-image change captioning over CLEVR scenes with **30–40 objects** and **4
simultaneous changes**, rendered at **768×768**.

**This dataset (images + captions) is hosted in full** — it is generator-rendered
and has no public upstream — at:

> `https://huggingface.co/datasets/zwcolin/clevr-multichange`  *(public)*

```python
from huggingface_hub import snapshot_download
root = snapshot_download("zwcolin/clevr-multichange", repo_type="dataset", local_dir="data/clevr")
```
```bash
cd data/clevr && for f in images_part_*.zip; do unzip -q "$f"; done   # -> images/CLEVR_new_*.png
```
The image paths in the JSONLs (`images/CLEVR_new_*.png`) then resolve under that dir.

To rebuild the captions yourself from raw scene graphs, regenerate the scenes with
the CLEVR-Multi-Change/Blender engine into `<root>/{images,scenes}/`, then run
`build_clevr_multichange` (it derives the captions, 5 templates per change type).

> CLEVR captions list changes whose **order is irrelevant**, and each change type
> has 5 valid lexical templates. Evaluate with permutation-invariant per-change
> scoring (bipartite matching over per-change sentence similarity).

## 3. Visual Trajectory Behavioral Cloning — VisGym

Episodic multi-turn imitation of oracle-solver demos across four tasks (matchstick
rotation, 3D mental rotation — cube and Objaverse, patch reassembly). Each `gpt`
turn is a real action (`mask_history=False`).

**Fully reproducible from upstream** — `VisGym/visgym_data` is public and
self-contained (each step's observation is embedded as base64 in the trajectory).
`build_visgym` downloads it, decodes the per-step images, and writes the ShareGPT
JSONL with matching paths:

```bash
python -m data_prep.build_visgym --split train --limit 100000 --images_root data --out data/visgym_train.jsonl
python -m data_prep.build_visgym --split val                  --images_root data --out data/visgym_eval.jsonl
```

VisGym uses the older ShareGPT schema (`conversations` with `{"from","value"}`).
Map: human turn = `"<image>" + step.prompt`; gpt turn = `step.vlm_output`
(`<think>…</think><answer>(action)</answer>`); one decoded image per step.

## 4. Longitudinal Radiology — Medical-Diff-VQA

Paired chest X-rays from the same patient with annotated changes (130k/12.5k/16k),
DICOM→PNG downsized to ~768². **Obtain from PhysioNet** (credentialed; derives from
MIMIC-CXR). Prepare records as `{"images": [ref, cur], "target": "<finding list>"}`
and run `build_generic --task diffvqa`. Targets are templated finding lists
(added / missing / no-change).

## 5. Fine-grained Image Comparison — ImgEdit

Source→edited pairs; the model predicts the edit instruction (301k/1.4k), downsized
to ~384². We use **7 of ImgEdit's categories** (add, adjust, background, content
memory, content understanding, remove, replace — style excluded as shortcut-solvable),
which on the public upstream [`sysuyy/ImgEdit`](https://huggingface.co/datasets/sysuyy/ImgEdit)
are exactly **8 archives**:

```
Singleturn/results_add_laion_part0          Singleturn/results_remove_part0
Singleturn/results_adjust_canny_laion_part0 Singleturn/results_remove_laion_part1
Singleturn/results_background_laion_part0    Singleturn/results_replace_laion_part4
Multiturn/results_content_memory_part2       Multiturn/results_content_understanding_part2
```

Captions are at `zwcolin/sve-data` (`imgedit_{train,test}.jsonl`, paths
`ImgEdit/extracted/results_*/...`). Download + extract exactly this image subset:

```bash
python -m data_prep.download_imgedit --out data   # -> data/ImgEdit/extracted/results_*/...
```

(Or build the captions from your own records: `build_generic --task imgedit` with
`{"images": [orig, edited], "target": "<instruction>"}`.) Because instructions don't
always match the actual visual edit, the paper scores ImgEdit reference-free with an
MLLM judge.

## 6. Remote Sensing — LEVIR-CC

Bitemporal before/after satellite images (fixed **256×256**) with change captions;
~half of training captions are "no change" (34k/6.7k/1.9k). 5 reference captions per
pair → evaluate **multi-reference**.

Captions are on `zwcolin/sve-data` (`levircc_{train,val,test}.jsonl`, paths
`LEVIR-CC/images/{train,val,test}/{A,B}/*.png`). Images are at
[`lcybuaa/LEVIR-CC`](https://huggingface.co/datasets/lcybuaa/LEVIR-CC). The
`Levir-CC-dataset.zip` extracts to `images/...` at its root, so **extract it into
`data/LEVIR-CC/`** for the JSONL paths to resolve:

```bash
python -c "from huggingface_hub import hf_hub_download; hf_hub_download('lcybuaa/LEVIR-CC','Levir-CC-dataset.zip',repo_type='dataset',local_dir='data/_levir')"
mkdir -p data/LEVIR-CC && unzip -q data/_levir/Levir-CC-dataset.zip -d data/LEVIR-CC
```

To build the captions yourself, prepare records as `{"images": [before, after],
"target": "<caption>"}` and run `build_generic --task levircc`.

---

## Hugging Face mirrors

**Self-contained datasets (images + captions), public** — generator-rendered, no
public upstream, so the images are hosted in full (zip-sharded; unzip after download):

- CLEVR-Multi-Change → [`zwcolin/clevr-multichange`](https://huggingface.co/datasets/zwcolin/clevr-multichange)
- Dot Distance/Area → [`zwcolin/dot-distance-area`](https://huggingface.co/datasets/zwcolin/dot-distance-area)

**Caption JSONLs only** (image paths remapped relative to `data/`; fetch images from
each dataset's upstream source) at
[`zwcolin/sve-data`](https://huggingface.co/datasets/zwcolin/sve-data) —
`imgedit_{train,test}`, `levircc_{train,val,test}` (`.jsonl`).
Download with `python -m data_prep.download`.

**VisGym** is reproduced directly from its public upstream with
`python -m data_prep.build_visgym` (no separate mirror; see §3).

Medical-Diff-VQA (MIMIC-CXR) is **not** mirrored — obtain it from PhysioNet.
