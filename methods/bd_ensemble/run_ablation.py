# -*- coding: utf-8 -*-
"""
Driver for the band-depth ablation studies (paper Section 5.4).

Sweeps the band-depth-specific hyperparameters and writes AUROCs to
output/ablation/ (output/metric/ is never touched). Ablations run on a single
small dataset (SVAMP) with one model (llama3-1_8B) to bound cost.

Sweeps (→ output file):
  1. Band subset size J        → subset_size.json     (vBD and gBD_v2)
  2. gBD random-walk length L   → walk_length.json     (gBD_v2 only)
  3. Clustering threshold theta_c → sim_threshold.json  (vBD and gBD_v2)
  4. Semantic-edge theta_e      → cross_threshold.json (gBD_v2 only)
  5. Monte-Carlo trials         → n_trials.json        (vBD and gBD_v2)

Each run reuses the existing output/<model>/<dataset>/ensemble_v1.json and
output_v1_w_labels.json produced by the main pipeline. Confidence files for
ablation runs are written under confidences/ablation/ so they never collide
with the main results.

Usage (from repo root, with the cotuq env active):
    python methods/bd_ensemble/run_ablation.py
    python methods/bd_ensemble/run_ablation.py --dry-run
    python methods/bd_ensemble/run_ablation.py --datasets svamp gsm8k
"""

import argparse
import os
import subprocess
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
PBD = os.path.join("methods", "bd_ensemble", "pBD.py")
VBD = os.path.join("methods", "bd_ensemble", "vBD.py")
GBD1 = os.path.join("methods", "bd_ensemble", "gBD_v1.py")
GBD2 = os.path.join("methods", "bd_ensemble", "gBD_v2.py")

# (ablation name, scripts, CLI flag, swept values)
# Each hyperparameter is run only for the methods it actually affects:
#   - subset_size J : vBD (vertex pairs), gBD-v2 (reference walks). gBD-v1 uses a
#                     hardcoded path pair; pBD has no graph.
#   - walk_length L : gBD-v2 only (gBD-v1 uses anchored simple paths, no length).
#   - theta_c       : every graph method (vBD, gBD-v1, gBD-v2).
#   - theta_e       : gBD-v1, gBD-v2 (vBD has no soft cross edges).
#   - n_trials T    : every Monte-Carlo method (vBD, gBD-v1, gBD-v2).
#   - alpha         : ALL four methods (the blend weight is shared by everyone).
SWEEPS = [
    ("subset_size",     [VBD, GBD2],            "--subset_size",     ["2", "3", "5", "8", "12", "16", "20"]),
    ("walk_length",     [GBD2],                 "--walk_length",     ["2", "3", "5", "8", "12", "16", "20", "adaptive"]),
    ("sim_threshold",   [VBD, GBD1, GBD2],      "--sim_threshold",   ["0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9"]),
    ("cross_threshold", [GBD1, GBD2],           "--cross_threshold", ["0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9"]),
    ("n_trials",        [VBD, GBD1, GBD2],      "--n_trials",        ["10", "50", "100", "200", "400", "800"]),
    ("alpha",           [PBD, VBD, GBD1, GBD2], "--alpha",           ["0.0", "0.1", "0.2", "0.3", "0.4", "0.5",
                                                                      "0.6", "0.7", "0.8", "0.9", "1.0"]),
]


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
    parser.add_argument("--datasets", nargs="+", default=["svamp"],
                        help="ablation dataset(s); default is the smallest (SVAMP)")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the commands without running them")
    opts = parser.parse_args()

    base = ["--model_engine", opts.model_engine]
    failures = []

    for name, scripts, flag, values in SWEEPS:
        print(f"=== Ablation: {name} ===")
        for dataset in opts.datasets:
            for value in values:
                for script in scripts:
                    extra = base + ["--dataset", dataset, "--ablation", name, flag, value]
                    if not _run(script, extra, opts.dry_run):
                        failures.append((script, dataset, f"{flag}={value}"))

    print("\n=== Done ===")
    if failures:
        print(f"{len(failures)} run(s) failed:")
        for script, dataset, param in failures:
            print(f"  {os.path.basename(script)}  {dataset}  {param}")
    else:
        print("All runs completed. Results in output/ablation/*.json")


if __name__ == "__main__":
    main()
