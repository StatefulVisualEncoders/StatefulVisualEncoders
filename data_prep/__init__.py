"""Minimal dataset preparation for SVE training.

Ways to get data:
  * ``python -m data_prep.download`` — pull caption JSONLs from ``zwcolin/sve-data``
    (ImgEdit, LEVIR-CC). CLEVR-Multi-Change and Dot Distance/Area are hosted with
    images at ``zwcolin/clevr-multichange`` / ``zwcolin/dot-distance-area``.
  * ``data_prep.build_visgym`` — reproduce VisGym from its public upstream
    (downloads + decodes the embedded per-step images).
  * ``data_prep.download_imgedit`` — fetch + extract the exact ImgEdit image subset
    (8 archives) from upstream ``sysuyy/ImgEdit``.
  * ``data_prep.build_generic`` / ``data_prep.build_clevr_multichange`` — build the
    JSONLs from raw sources yourself.

The exact conversation templates (system prompts, filler turns, masking) live in
``data_prep.formats``. See ``docs/DATA.md`` for per-dataset sources and licenses.
"""

from .formats import build_sample

__all__ = ["build_sample"]
