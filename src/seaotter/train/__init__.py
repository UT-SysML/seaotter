"""Training package for the six paper-relevant SEA OTTER configurations.

Recipes live under `seaotter.train.recipes`:

  | codec  | task | entry point                          |
  |--------|------|--------------------------------------|
  | FRAPPE | cls  | seaotter.train.recipes.frappe_cls    |
  | FRAPPE | seg  | seaotter.train.recipes.frappe_seg    |
  | FRAPPE | clip | seaotter.train.recipes.frappe_clip   |
  | WaLLoC | cls  | seaotter.train.recipes.walloc_cls    |
  | WaLLoC | seg  | seaotter.train.recipes.walloc_seg    |
  | WaLLoC | clip | seaotter.train.recipes.walloc_clip   |

Each recipe locks its hyperparameters per
`prompts/organize_src_code.md` and calls into the shared `Trainer`.

Open recipe questions (intentionally not resolved here; flagged in the
recipe-file docstrings):

  1. Seg recipes use LSDIR + pseudolabels (iter-1 / iter-7 inheritance).
     An ADE20k + GT refresh has never been validated end-to-end.
  2. WaLLoC seg LR refresh (iter-10-style) has not been run.
  3. `walloc_clip` is forward-projected from iter-10 + iter-9 — its
     first end-to-end run is the validation.
"""

from seaotter.train.config import TrainerConfig
from seaotter.train.trainer import Trainer

__all__ = ["Trainer", "TrainerConfig"]
