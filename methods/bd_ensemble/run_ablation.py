# -*- coding: utf-8 -*-
"""
Driver for the band-depth ablation studies (paper Section 5.4).

Sweeps the two band-depth-specific hyperparameters and writes AUROCs to
output/ablation/ (output/metric/ is never touched):

  1. Band subset size J  → output/ablation/subset_size.json   (vBD and gBD_v2)
  2. gBD random-walk len → output/ablation/walk_length.json   (gBD_v2 only)

Each run reuses the existing output/<model>/<dataset>/ensemble_v1.json and
output_v1_w_labels.json produced by the main pipeline. Confidence files for
ablation runs are written under confidences/ablation/ so they never collide
with the main results.

Usage (from repo root, with the cotuq env active):
    python methods/bd_ensemble/run_ablation.py
    python methods/bd_ensemble/run_ablation.py --dry-run
    python methods/bd_ensemble/run_ablation.py --model_engine llama3-1_8B \
        --datasets gsm8k svamp hotpotQA
"""

import argparse
import os
import subprocess
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
VBD = os.path.join("methods", "bd_ensemble", "vBD.py")
GBD = os.path.join("methods", "bd_ensemble", "gBD_v2.py")

J_VALUES = [2, 3, 4, 5]
L_VALUES = ["3", "5", "7", "10", "adaptive"]


def _run(script: str, extra: list[str], dry_run: bool) -> bool:
    """Invoke a method script as a subprocess; return True on success."""
    cmd = [sys.executable, script] + extra
    print("  $ " + " ".join(cmd))
    if dry_run:
        return True
    env = dict(os.environ, PYTHONPATH="./")
    completed = subprocess.run(cmd, cwd=REPO_ROOT, env=env)
    if completed.returncode != 0:
        print(f"  ! failed (exit {completed.returncode})")
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Band-depth ablation driver.")
    parser.add_argument("--model_engine", default="llama3-1_8B",
                        choices=["llama3-1_8B", "llama2-13b"])
    parser.add_argument("--datasets", nargs="+",
                        default=["gsm8k", "svamp", "hotpotQA"])
    parser.add_argument("--dry-run", action="store_true",
                        help="print the commands without running them")
    opts = parser.parse_args()

    base = ["--model_engine", opts.model_engine]
    failures = []

    # ── Ablation 1: band subset size J (vBD and gBD_v2) ──────────────────────
    print("=== Ablation 1: band subset size J ===")
    for dataset in opts.datasets:
        for j in J_VALUES:
            for script in (VBD, GBD):
                extra = base + ["--dataset", dataset, "--ablation", "subset_size",
                                "--subset_size", str(j)]
                if not _run(script, extra, opts.dry_run):
                    failures.append((script, dataset, f"J={j}"))

    # ── Ablation 2: gBD random-walk length L (gBD_v2 only) ───────────────────
    print("=== Ablation 2: gBD random-walk length L ===")
    for dataset in opts.datasets:
        for length in L_VALUES:
            extra = base + ["--dataset", dataset, "--ablation", "walk_length",
                            "--walk_length", length]
            if not _run(GBD, extra, opts.dry_run):
                failures.append((GBD, dataset, f"L={length}"))

    print("\n=== Done ===")
    if failures:
        print(f"{len(failures)} run(s) failed:")
        for script, dataset, param in failures:
            print(f"  {os.path.basename(script)}  {dataset}  {param}")
    else:
        print("All runs completed. Results in output/ablation/"
              "{subset_size,walk_length}.json")


if __name__ == "__main__":
    main()
