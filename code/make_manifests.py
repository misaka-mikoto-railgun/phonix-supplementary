"""
make_manifests.py — manifests for the checkpoints and the dataset
=========================================================================
Neither the trained checkpoints nor the audio is distributed. These two files
describe them instead, so that a copy obtained elsewhere can be identified as
the one the published results came from:

  checkpoints_manifest.json
      One entry per published checkpoint: which model it is, the seed it was
      trained with, the bounds it was trained under, its SHA-256 and size.

      The bounds matter because a checkpoint does not record them. Loading a
      +/-12 dB checkpoint into a model built with gain_max=6.0 succeeds and then
      silently clamps, so the manifest is the only place the pairing is stated.

      It also carries `evaluation_staging`: the mapping from a model name to
      the file it is actually evaluated from. In the working tree that mapping
      was applied by substituting files inside a staging directory, which left
      two different checkpoints sharing the name A0_Proposed.pt. The mapping is
      stated here instead; code/make_eval_staging.py rebuilds the directory
      from it.

  dataset_manifest.json
      The audio is third-party (FMA, BUT ReverbDB, OpenAIR) and is not
      redistributed. The generator sorts its file list and then splits it, so a
      different local corpus yields a different split even at the same seed;
      this file records the track and RIR lists per split, together with the
      full generation config, so an identical split can be reconstructed.

Usage:
  python make_manifests.py --data_dir ../data/dataset_v3 \\
      --ckpt_dir ../checkpoints/full --rev_ckpt_dir ./checkpoints \\
      --out_dir ..
"""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

SPLITS = ["train", "val", "test_synth", "test_real", "paired_mode_test"]

# Manifest group -> source directory. ckpt_eval/ is not described separately:
# 13 of its 15 entries are byte-identical to checkpoints_original/ and the other
# two are in checkpoints_revision/ under their own names.
STAGING = {
    "A0_Proposed": "checkpoints_revision/A0_g12_f16k_s7.pt",
    "A2_withPrefLoss": "checkpoints_revision/A2_g12_f16k_s7.pt",
}

def model_bounds(stem):
    """(gain_max, fc_max, model_name) for a checkpoint, or (None, None, stem).

    Revision checkpoints carry their configuration in the filename
    (A0_g12_f16k_s42 -> +/-12 dB, 16 kHz). Pre-revision ones do not, so the
    bound is the +/-6 dB the paper reports for that round — but only for the
    models that have a per-section gain at all. E3/E4/E5 and the dense AC
    variants emit a full response and have no such bound; they get null rather
    than a number that would look like a constraint they never had.
    """
    parts = stem.split("_")
    gain = fc = None
    name = stem
    for p in parts:
        if p == "g6":
            gain = 6.0
        elif p == "g12":
            gain = 12.0
        elif p == "f16k":
            fc = 16000.0
        elif p == "f20k":
            fc = 20000.0
    if gain is not None:                       # revision checkpoint
        head = parts[0]
        name = {"A0": "A0_Proposed", "A2": "A2_withPrefLoss"}.get(head, stem)
        if stem.endswith("_g12") or stem.endswith("_g6"):   # AC*_Biquad_g12
            name = stem.rsplit("_", 1)[0]
            fc = fc or 16000.0
        return gain, fc or 16000.0, name
    return (6.0, 16000.0, stem) if HAS_GAIN_BOUND.get(stem) else (None, None, stem)


def _discover_bounded_models():
    """Which pre-revision checkpoints belong to a model with a gain bound."""
    found = {}
    try:
        import train_full as TF
        from arch_biquad import BIQUAD_REGISTRY
        for k, v in TF.build_registry().items():
            found[k] = hasattr(v["model"], "gain_max")
        for k, build in BIQUAD_REGISTRY.items():
            found[k] = True
    except Exception as exc:      # manifest generation must not need a GPU
        print(f"  [warn] model registries unavailable ({exc}); bounds left null")
    return found


HAS_GAIN_BOUND = _discover_bounded_models()


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def git_sha():
    """Repository HEAD when the manifest is written.

    The commit that carries the manifest is necessarily this one's child, since
    writing the file changes the tree. It identifies the code the description
    was produced from, not the commit the file appears in — for that, use
    `git log` on the manifest itself.
    """
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return None


def describe(path, release_dir):
    stem = path.stem
    seed = None
    for part in stem.split("_"):
        if part.startswith("s") and part[1:].isdigit():
            seed = int(part[1:])
    gain, fc, model = model_bounds(stem)
    return {
        "file": f"{release_dir}/{path.name}",
        "model": model,
        "checkpoint_stem": stem,
        "train_seed": seed if seed is not None else 42,
        "gain_max": gain,
        "fc_max": fc,
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def build_checkpoints(ckpt_dir, rev_ckpt_dir, commit):
    entries = []
    for d, release_dir in ((rev_ckpt_dir, "checkpoints_revision"),
                           (ckpt_dir, "checkpoints_original")):
        for p in sorted(Path(d).glob("*.pt")):
            entries.append(describe(p, release_dir))
    return {
        "source_commit": commit,
        "note": ("These checkpoints are not redistributed. gain_max / fc_max are not "
                 "stored inside a .pt file, so this manifest is where the pairing is "
                 "recorded: instantiate the model with the values given here, because "
                 "a mismatched bound loads without error and then clamps the output. "
                 "The sha256 of each entry identifies exactly the file the published "
                 "results were produced from."),
        "evaluation_staging": STAGING,
        "evaluation_staging_note": ("Model names not listed above are evaluated from "
                                    "checkpoints_original/<name>.pt. "
                                    "code/make_eval_staging.py rebuilds the staging "
                                    "directory from this mapping and checks each file "
                                    "against its sha256."),
        "checkpoints": entries,
    }


def build_dataset(data_dir, commit):
    data_dir = Path(data_dir)
    out = {
        "source_commit": commit,
        "generator": "code/dataset_generator_v4_tracklevel.py",
        "note": ("The audio is third-party and is not redistributed. The generator "
                 "sorts the file list before splitting, so a different local corpus "
                 "produces a different split even at the same seed; the per-split "
                 "track and RIR lists below are what make the split reconstructible."),
        "splits": {},
    }
    for s in SPLITS:
        d = data_dir / s
        if not (d / "meta.json").is_file():
            continue
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        entry = {k: meta.get(k) for k in
                 ("n_samples", "n_pairs", "n_chunks", "is_real_rir",
                  "n_unique_tracks", "version")
                 if meta.get(k) is not None}
        tm = d / "track_map.json"
        if tm.is_file():
            entry["tracks"] = [t["value"] for t in
                               json.loads(tm.read_text(encoding="utf-8"))["tracks"]]
        rm = d / "rir_map.json"
        if rm.is_file():
            entry["rirs"] = [r["value"] for r in
                             json.loads(rm.read_text(encoding="utf-8"))["rirs"]]
        if "config" in meta and "config" not in out:
            out["config"] = meta["config"]
        out["splits"][s] = entry
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.strip().split("\n")[1])
    ap.add_argument("--data_dir", default=str(ROOT / "data" / "dataset_v3"))
    ap.add_argument("--ckpt_dir", default=str(ROOT / "checkpoints" / "full"))
    ap.add_argument("--rev_ckpt_dir", default=str(HERE / "checkpoints"))
    ap.add_argument("--out_dir", default=str(ROOT))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    commit = git_sha()

    for name, payload in (
            ("checkpoints_manifest.json",
             build_checkpoints(args.ckpt_dir, args.rev_ckpt_dir, commit)),
            ("dataset_manifest.json", build_dataset(args.data_dir, commit))):
        p = out_dir / name
        p.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                     encoding="utf-8")
        print(f"  {name}: {p}")

    n = len(build_checkpoints(args.ckpt_dir, args.rev_ckpt_dir, commit)["checkpoints"])
    print(f"  checkpoints described: {n}")


if __name__ == "__main__":
    main()
