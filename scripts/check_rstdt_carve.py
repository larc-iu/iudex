"""Verify an RST-DT data root against the pinned carve manifest.

check_rstdt_carve.py [DATA_ROOT] [--build PINNED_ROOT]

DATA_ROOT defaults to data/rstdt. Exits nonzero (listing drift) if the doc
IDs in {train,dev,test} do not match configs/lib/rstdt_carve.json. With
--build, instead constructs PINNED_ROOT/{train,dev,test} as symlinks into
DATA_ROOT's files, rearranged to match the manifest (non-destructive, the
source dirs are untouched). Run before any cross-machine dev comparison.
"""

import argparse
import json
import os
import sys
from glob import glob
from pathlib import Path

MANIFEST = Path(__file__).resolve().parent.parent / "configs" / "lib" / "rstdt_carve.json"


def doc_ids(d):
    return {os.path.basename(p).split(".")[0] for p in glob(os.path.join(d, "*.rs3"))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_root", nargs="?", default="data/rstdt")
    ap.add_argument("--build", default=None, help="build a pinned symlink tree here instead of checking")
    args = ap.parse_args()

    with open(MANIFEST) as f:
        manifest = json.load(f)
    splits = {s: set(manifest[s]) for s in ("train", "dev", "test")}

    if args.build:
        by_id = {}
        for split in ("train", "dev", "test"):
            for p in glob(os.path.join(args.data_root, split, "*")):
                by_id.setdefault(os.path.basename(p).split(".")[0], []).append(os.path.abspath(p))
        missing = set().union(*splits.values()) - set(by_id)
        if missing:
            sys.exit(f"docs in manifest but not under {args.data_root}: {sorted(missing)}")
        for split, ids in splits.items():
            out = Path(args.build) / split
            out.mkdir(parents=True, exist_ok=True)
            for doc in ids:
                for src in by_id[doc]:
                    dst = out / os.path.basename(src)
                    if not dst.exists():
                        dst.symlink_to(src)
        print(f"built pinned tree at {os.path.abspath(args.build)}")
        return

    ok = True
    for split, want in splits.items():
        have = doc_ids(os.path.join(args.data_root, split))
        extra, gone = sorted(have - want), sorted(want - have)
        if extra or gone:
            ok = False
            print(f"{split}: NOT pinned ({len(have)} docs, manifest {len(want)})")
            if extra:
                print(f"  not in manifest {split}: {extra}")
            if gone:
                print(f"  missing from {split}: {gone}")
        else:
            print(f"{split}: ok ({len(have)} docs)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
