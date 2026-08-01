"""
cli_paths.py — command-line path resolution shared by the generator scripts
===========================================================================
The generators read the dataset and three checkpoint directories and write
their output to a fourth. Those locations have defaults that match this
repository's layout, but none of them is fixed: every script accepts

  --data_dir        dataset root, holding test_synth/ test_real/ ...
  --ckpt_dir        pre-revision checkpoints (checkpoints/full/)
  --rev_ckpt_dir    +/-12 dB revision checkpoints
  --eval_ckpt_dir   evaluation staging directory
  --out_dir         where the script writes; created if missing

so the scripts can be pointed at copies unpacked anywhere.

On the staging directory: --eval_ckpt_dir holds the checkpoints each model is
evaluated from. Most of its entries are the pre-revision files; A0_Proposed.pt
and A2_withPrefLoss.pt are the seed-7 +/-12 checkpoints instead, which is how
the "representative seed 7" convention is applied. The release ships that as
an explicit mapping (evaluation_staging in checkpoints_manifest.json) rather
than as substituted files, because two different checkpoints under one name is
the same trap as a +/-6 model loaded through a +/-12 instance.
"""
import argparse
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

DEFAULTS = {
    "data_dir": ROOT / "data" / "dataset_v3",
    "ckpt_dir": ROOT / "checkpoints" / "full",
    "rev_ckpt_dir": HERE / "checkpoints",
    "eval_ckpt_dir": HERE / "ckpt_eval",
    "out_dir": HERE / "results",
}

_HELP = {
    "data_dir": "dataset root (test_synth/, test_real/, ...)",
    "ckpt_dir": "pre-revision checkpoints",
    "rev_ckpt_dir": "+/-12 dB revision checkpoints",
    "eval_ckpt_dir": "evaluation staging checkpoints",
    "out_dir": "output directory (created if missing)",
}


def add_path_args(ap):
    """Attach the five path options to an existing parser."""
    for k, v in DEFAULTS.items():
        ap.add_argument(f"--{k}", default=str(v), help=f"{_HELP[k]} (default: {v})")
    return ap


def parse(description=None, configure=None, require=("data_dir",)):
    """Parse the path options and return (paths, args).

    `paths` is a namespace of Path objects; `args` is the full parse result so
    a caller that added its own options can read them. Pass `configure` to
    register extra arguments before parsing.

    `require` names the inputs this script cannot run without. A missing one
    raises here rather than half-way through, because the failure modes it
    prevents are silent: a mistyped dataset path used to fall back to synthetic
    data, and a mistyped checkpoint path used to leave the model randomly
    initialised. Both produce output files that look entirely normal.
    """
    ap = argparse.ArgumentParser(description=description)
    add_path_args(ap)
    if configure is not None:
        configure(ap)
    args = ap.parse_args()
    paths = argparse.Namespace(**{k: Path(getattr(args, k)) for k in DEFAULTS})
    for k in require:
        if not getattr(paths, k).is_dir():
            raise FileNotFoundError(f"--{k} not found: {getattr(paths, k)}")
    paths.out_dir.mkdir(parents=True, exist_ok=True)
    return paths, args
