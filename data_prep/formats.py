"""ShareGPT conversation templates for the SVE training tasks.

Each task is formatted as a ShareGPT sample
``{"messages": [...], "images": [...]}`` where the k-th ``<image>`` tag binds to
``images[k]``. Intermediate "filler" assistant turns provide context but are
masked during training (`mask_history=True`); only the final assistant turn — the
answer — is supervised. The exact prompts below match the paper (Appendix B).

These builders take **already-prepared records** (image paths + the target
answer) and wrap them in the right conversation. Where the target itself must be
derived from raw annotations (e.g. CLEVR scene graphs), see the task-specific
builder (`build_clevr_multichange.py`).
"""

# --- System prompts -------------------------------------------------------

DIFFVQA_SYSTEM = (
    "You are a medical imaging expert. Given a reference chest X-ray and a "
    "current chest X-ray from the same patient, describe what findings have changed."
)
LEVIRCC_SYSTEM = (
    "You are an expert at detecting and describing changes between two images. "
    "Given a before and after image, describe what has changed."
)
IMGEDIT_SYSTEM = "You are given a pair of images. Describe the edit made between them."

# Spatial-aggregation system prompts, keyed by number of images (2..5).
DISTANCE_AREA_SYSTEM = {
    2: ("You are a visual distance estimator. You are shown two screenshots, each "
        "with a red dot. Your task is to estimate the normalized Euclidean distance "
        "between the red dots across the two images. The distance is normalized to "
        "[0, 1] where 0 means the dots are at the same position and 1 means they are "
        "at opposite corners. Output only the distance as a decimal number rounded to "
        "4 decimal places."),
    3: ("You are a visual area estimator. You are shown three screenshots, each with a "
        "red dot. Your task is to estimate the normalized area of the triangle formed "
        "by the red dots across the three images. The area is normalized by the full "
        "image area, so it ranges from 0 to 0.5. Output only the area as a decimal "
        "number rounded to 4 decimal places."),
    4: ("You are a visual area estimator. You are shown four screenshots, each with a "
        "red dot. Your task is to estimate the area of the convex hull formed by the "
        "red dots across the four images. The area is normalized by the full image "
        "area. Output only the area as a decimal number rounded to 4 decimal places."),
    5: ("You are a visual area estimator. You are shown five screenshots, each with a "
        "red dot. Your task is to estimate the area of the convex hull formed by the "
        "red dots across the five images. The area is normalized by the full image "
        "area. Output only the area as a decimal number rounded to 4 decimal places."),
}
_DISTANCE_AREA_QUANTITY = {2: "distance", 3: "area", 4: "area", 5: "area"}
_NUM_WORD = {2: "two", 3: "three", 4: "four", 5: "five"}


def _msg(role, content):
    return {"role": role, "content": content}


def build_sample(task: str, images, target: str, subtask_images: int = None) -> dict:
    """Generic ShareGPT builder for all single-shot tasks.

    Args:
        task: one of "clevr_multichange", "diffvqa", "imgedit", "levircc",
              "distance_area".
        images: list of image paths (2 for the comparison tasks; 2..5 for
                distance_area).
        target: the final answer string (caption / findings / instruction / number).
        subtask_images: for distance_area, the number of images (2..5); defaults to
                len(images).
    """
    if task == "clevr_multichange":
        a, b = images
        return {"messages": [
            _msg("user", "<image>\nHere is an image of a scene with objects."),
            _msg("assistant", "I see the scene. Please show me the next image."),
            _msg("user", "<image>\nWhat changed between the two images?"),
            _msg("assistant", target),
        ], "images": [a, b]}

    if task == "diffvqa":
        a, b = images
        return {"messages": [
            _msg("system", DIFFVQA_SYSTEM),
            _msg("user", "<image>\nThis is the reference (prior) chest X-ray."),
            _msg("assistant", "Understood. Please provide the current chest X-ray."),
            _msg("user", "<image>\nThis is the current chest X-ray. What has changed compared to the reference image?"),
            _msg("assistant", target),
        ], "images": [a, b]}

    if task == "imgedit":
        a, b = images
        return {"messages": [
            _msg("system", IMGEDIT_SYSTEM),
            _msg("user", "<image>\nHere is the original image."),
            _msg("assistant", "I see the image. Please show me the edited version."),
            _msg("user", "<image>\nWhat was edited between the two images?"),
            _msg("assistant", target),
        ], "images": [a, b]}

    if task == "levircc":
        a, b = images
        return {"messages": [
            _msg("system", LEVIRCC_SYSTEM),
            _msg("user", "<image>\nThis is the before satellite image."),
            _msg("assistant", "Understood. Please provide the after image."),
            _msg("user", "<image>\nThis is the after satellite image. Describe what has changed."),
            _msg("assistant", target),
        ], "images": list(images)}

    if task == "distance_area":
        n = subtask_images or len(images)
        if n not in DISTANCE_AREA_SYSTEM:
            raise ValueError(f"distance_area expects 2..5 images, got {n}")
        msgs = [_msg("system", DISTANCE_AREA_SYSTEM[n])]
        for i, img in enumerate(images):
            last = i == len(images) - 1
            if last:
                q = (f"<image>\nA red dot is placed on this screenshot.\n"
                     f"What is the {_DISTANCE_AREA_QUANTITY[n]} formed by the red dots "
                     f"across the {_NUM_WORD[n]} images?")
                msgs.append(_msg("user", q))
                msgs.append(_msg("assistant", target))
            else:
                msgs.append(_msg("user", "<image>\nA red dot is placed on this screenshot."))
                msgs.append(_msg("assistant", "I see the red dot on the screenshot."))
        return {"messages": msgs, "images": list(images)}

    raise ValueError(f"Unknown task {task!r}")
