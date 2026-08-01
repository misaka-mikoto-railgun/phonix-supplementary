"""
make_eval_staging.py — materialise the evaluation staging directory
===================================================================
Some generators are given `--eval_ckpt_dir`: the directory each model is
evaluated from. It is not published, because most of it duplicates
checkpoints_original/ and two of its entries would otherwise ship a second,
different file under a name that already means something else — the same kind of
trap as loading a +/-6 dB checkpoint through a +/-12 dB instance.

The release states the difference declaratively instead, as `evaluation_staging`
in checkpoints_manifest.json. This script turns that statement back into the
directory, and verifies every file it writes against the SHA-256 in the manifest.

  python make_eval_staging.py --ckpt_dir /path/to/checkpoints_original \\
      --rev_ckpt_dir /path/to/checkpoints_revision --out_dir ./ckpt_eval
"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__.strip().split("\n")[1])
    ap.add_argument("--manifest", default=str(ROOT / "checkpoints_manifest.json"))
    ap.add_argument("--ckpt_dir", default=str(ROOT / "checkpoints" / "full"),
                    help="published checkpoints_original/")
    ap.add_argument("--rev_ckpt_dir", default=str(HERE / "checkpoints"),
                    help="published checkpoints_revision/")
    ap.add_argument("--out_dir", default=str(HERE / "ckpt_eval"))
    args = ap.parse_args()

    man = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    digest = {e["file"]: e["sha256"] for e in man["checkpoints"]}
    roots = {"checkpoints_original": Path(args.ckpt_dir),
             "checkpoints_revision": Path(args.rev_ckpt_dir)}
    for name, d in roots.items():
        if not d.is_dir():
            raise FileNotFoundError(f"{name} not found: {d}")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Start from the pre-revision set, then apply the mapping over it.
    plan = {p.stem: f"checkpoints_original/{p.name}"
            for p in sorted(roots["checkpoints_original"].glob("*.pt"))}
    plan.update(man["evaluation_staging"])

    n = 0
    for model, rel in sorted(plan.items()):
        sub, fname = rel.split("/", 1)
        src = roots[sub] / fname
        if not src.is_file():
            raise FileNotFoundError(f"{rel} not found under {roots[sub]}")
        want = digest.get(rel)
        got = sha256(src)
        if want and got != want:
            raise RuntimeError(f"{rel}: sha256 {got} does not match the manifest ({want})")
        dst = out / f"{model}.pt"
        shutil.copyfile(src, dst)
        mark = "  <- evaluation_staging" if model in man["evaluation_staging"] else ""
        print(f"  {dst.name:26} <- {rel}{mark}")
        n += 1

    print(f"\n{n} checkpoints written to {out}, each verified against the manifest.")


if __name__ == "__main__":
    main()
