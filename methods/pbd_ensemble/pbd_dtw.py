# -*- coding: utf-8 -*-
"""
UQ via Path Band Depth with DTW alignment (Approach A — order-sensitive).

Reads:  output/<model_engine>/<dataset>/ensemble_v1.json
Writes: output/<model_engine>/<dataset>/confidences/ensemble_v1_pbd_dtw.json

Each output line:
    {"id": ..., "question": ..., "correct answer": ...,
     "llm answer": <majority-vote answer>, "confidence": <mean pBD>}

confidence = mean pBD across all chains in the ensemble.
High confidence → chains are central/consistent. Low → scattered reasoning.
"""

import json
import os
import random
import re
import sys
from collections import Counter
from typing import Optional

import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../"))
from config import args
from utils import parse_response_to_dict, setup_log, print_exp

# ── Hyperparameters ────────────────────────────────────────────────────────────
ENCODER_MODEL = "all-MiniLM-L6-v2"  # 384-d, fast, good for sentence similarity
J = 2           # band formed by j reference chains
N_TRIALS = 100  # random j-subsets to sample per chain
# ──────────────────────────────────────────────────────────────────────────────


# ── Step 1: Parse CoT response into ordered step texts ────────────────────────

_STEP_PREFIX = re.compile(r"^Step\s+\d+\s*:\s*", re.IGNORECASE)


def _parse_steps(llm_response: Optional[str]) -> list[str]:
    """
    Extract ordered step texts from a CoT response.

    Uses parse_response_to_dict, then strips 'Step N:' prefixes so the
    embedding captures the reasoning content only.
    Returns empty list if the response is None or malformed.
    """
    if not isinstance(llm_response, str) or not llm_response.strip():
        return []
    result = parse_response_to_dict(llm_response)
    if result is None or result[1] is None:
        return []
    _, steps_dict, _ = result
    if not steps_dict:
        return []
    # sort by step number; keys are "Step 1", "Step 2", ...
    def _step_num(key: str) -> int:
        parts = key.split()
        return int(parts[-1]) if parts[-1].isdigit() else 0

    ordered = sorted(steps_dict.items(), key=lambda kv: _step_num(kv[0]))
    texts = []
    for _, segment in ordered:
        # strip the "Step N: " prefix so the number doesn't influence embeddings
        clean = _STEP_PREFIX.sub("", segment.strip(), count=1).strip()
        if clean:
            texts.append(clean)
    return texts


# ── Step 2: Batch-embed all steps for a question ──────────────────────────────

def _embed_chains(chains: list[list[str]], encoder) -> list[np.ndarray]:
    """
    Encode all steps across all chains in a single batch call.

    Returns a list of (n_steps_i, d) float32 arrays, one per chain.
    L2-normalised so dot product = cosine similarity.
    Falls back to a zero vector for empty chains.
    """
    flat_texts = [step for chain in chains for step in chain]
    if not flat_texts:
        d = 384
        return [np.zeros((1, d), dtype=np.float32) for _ in chains]

    all_embs = encoder.encode(
        flat_texts,
        normalize_embeddings=True,
        show_progress_bar=False,
        batch_size=256,
    ).astype(np.float32)

    result, offset = [], 0
    for chain in chains:
        n = max(len(chain), 1)
        if not chain:
            result.append(np.zeros((1, all_embs.shape[1]), dtype=np.float32))
        else:
            result.append(all_embs[offset:offset + n])
        offset += len(chain)
    return result


# ── Step 3: DTW alignment ─────────────────────────────────────────────────────

def _dtw_path(emb1: np.ndarray, emb2: np.ndarray) -> list[tuple[int, int]]:
    """
    Classic DTW on L2-normalised embeddings.

    Cost at each cell = cosine distance = 1 - dot(emb1[i], emb2[j]).
    Returns the optimal alignment path as (i, j) pairs from (0,0) to (n-1, m-1).
    """
    n, m = len(emb1), len(emb2)
    # cosine distance matrix: since both are L2-normalised, dot = cosine sim
    cost = (1.0 - emb1 @ emb2.T).astype(np.float64)  # (n, m)

    # accumulated cost DP
    D = np.full((n, m), np.inf)
    D[0, 0] = cost[0, 0]
    for i in range(1, n):
        D[i, 0] = D[i - 1, 0] + cost[i, 0]
    for j in range(1, m):
        D[0, j] = D[0, j - 1] + cost[0, j]
    for i in range(1, n):
        for j in range(1, m):
            D[i, j] = cost[i, j] + min(D[i - 1, j - 1], D[i - 1, j], D[i, j - 1])

    # backtrack from (n-1, m-1) to (0, 0)
    path = []
    i, j = n - 1, m - 1
    while i > 0 or j > 0:
        path.append((i, j))
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            move = int(np.argmin([D[i - 1, j - 1], D[i - 1, j], D[i, j - 1]]))
            if move == 0:
                i -= 1
                j -= 1
            elif move == 1:
                i -= 1
            else:
                j -= 1
    path.append((0, 0))
    return list(reversed(path))


# ── Step 4: Soft band membership at one aligned index ─────────────────────────

def _soft_membership(p_step: np.ndarray, ref_steps: list[np.ndarray]) -> float:
    """
    Soft band membership score in [0, 1] for one aligned index.

    Geometric convex-hull containment is ill-conditioned in high-D embedding
    space (any point far from the segment's axis scores 0 by distance). Instead
    we use cosine similarity to the centroid of the reference steps — the
    "mean direction" in the unit hypersphere, which is the natural analogue
    of "between A and B" for normalised vectors.

    Score is 1.0 if p_step points exactly in the mean reference direction,
    and falls toward 0 as it becomes more orthogonal or opposed.
    All input vectors are assumed L2-normalised.
    """
    centroid = np.mean(np.stack(ref_steps), axis=0)
    norm = float(np.linalg.norm(centroid))
    if norm < 1e-9:
        return 0.0
    # Cosine similarity in [−1, 1]; clip to [0, 1] (negative = opposite direction)
    return float(np.clip(np.dot(p_step, centroid / norm), 0.0, 1.0))


# ── Step 5: pBD score for one chain ───────────────────────────────────────────

def _pbd_score(
    emb_p: np.ndarray,
    others: list[np.ndarray],
    j: int = J,
    n_trials: int = N_TRIALS,
) -> float:
    """
    Estimate the modified pBD score for chain emb_p against the ensemble.

    For each trial:
      1. Sample j reference chains from others.
      2. DTW-align emb_p against each reference.
      3. At each position k in emb_p, gather the DTW-matched steps from each ref.
         (DTW is many-to-one: average matched ref steps when multiple map to k.)
      4. Compute soft band membership at each k; average over k → trial score.
    Final pBD = mean over all trials.
    """
    if len(others) < j:
        return 1.0  # degenerate: treat as perfectly central
    # skip chains that are zero vectors (failed parse → no valid steps)
    if not np.any(emb_p):
        return 0.0

    trial_scores = []
    for _ in range(n_trials):
        refs = random.sample(others, j)

        # Build mapping: position k in p → list of matched ref step indices, per ref
        alignments = [_dtw_path(emb_p, ref) for ref in refs]
        n = len(emb_p)
        # p_ref_map[k][r] = list of ref indices matched to p[k] by reference r
        p_ref_map: list[list[list[int]]] = [[[] for _ in range(j)] for _ in range(n)]
        for r, path in enumerate(alignments):
            for i_p, i_ref in path:
                p_ref_map[i_p][r].append(i_ref)

        memberships = []
        for k in range(n):
            # For each reference, average the embeddings of all matched steps
            ref_embs_at_k = []
            for r in range(j):
                matched_indices = p_ref_map[k][r] if p_ref_map[k][r] else [0]
                ref_embs_at_k.append(
                    refs[r][matched_indices].mean(axis=0)
                )
            memberships.append(_soft_membership(emb_p[k], ref_embs_at_k))

        trial_scores.append(float(np.mean(memberships)))

    return float(np.mean(trial_scores))


# ── Step 6: Majority-vote answer ──────────────────────────────────────────────

def _majority_answer(samples: list[dict]) -> str:
    """Return the most common llm_answer in the ensemble."""
    answers = [s.get("llm answer", "") for s in samples]
    if not answers:
        return ""
    return Counter(answers).most_common(1)[0][0]


# ── Step 7: I/O helpers ───────────────────────────────────────────────────────

def _load_processed_ids(output_path: str) -> set:
    """Return IDs already written to the output file (for resume support)."""
    processed: set = set()
    if not os.path.exists(output_path):
        return processed
    with open(output_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                processed.add(json.loads(line)["id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return processed


# ── Main ──────────────────────────────────────────────────────────────────────

def pbd_dtw_uq() -> None:
    """Run DTW-pBD uncertainty quantification over the ensemble."""
    from sentence_transformers import SentenceTransformer

    log = setup_log(args)

    input_path = os.path.join(args.output_path, "ensemble_v1.json")
    confidences_dir = os.path.join(args.output_path, "confidences")
    os.makedirs(confidences_dir, exist_ok=True)
    output_path = os.path.join(confidences_dir, "ensemble_v1_pbd_dtw.json")

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Ensemble file not found: {input_path}")

    processed_ids = _load_processed_ids(output_path)
    if processed_ids:
        log.info(f"Resuming: skipping {len(processed_ids)} already-processed questions.")

    log.info(f"Loading encoder: {ENCODER_MODEL}")
    encoder = SentenceTransformer(ENCODER_MODEL)

    with open(input_path, encoding="utf-8") as f_in:
        lines = [line.strip() for line in f_in if line.strip()]

    log.info(f"Computing DTW-pBD UQ for {len(lines)} questions "
             f"(j={J}, trials={N_TRIALS})")

    with open(output_path, "a", encoding="utf-8") as f_out:
        for line in tqdm(lines, total=len(lines)):
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            qid = record["id"]
            if qid in processed_ids:
                continue

            samples = record.get("samples", [])

            if len(samples) < 2:
                confidence = 0.5
            else:
                # Parse steps and embed in one batch
                chains_text = [_parse_steps(s.get("llm response", "")) for s in samples]
                chains_emb = _embed_chains(chains_text, encoder)

                # Compute pBD for each chain; confidence = mean pBD
                depths = []
                for i, emb_p in enumerate(chains_emb):
                    others = [chains_emb[k] for k in range(len(chains_emb)) if k != i]
                    depths.append(_pbd_score(emb_p, others, j=J, n_trials=N_TRIALS))

                confidence = float(np.mean(depths))

            result = {
                "id": qid,
                "question": record.get("question", ""),
                "correct answer": record.get("correct answer", ""),
                "llm answer": _majority_answer(samples),
                "confidence": confidence,
            }
            f_out.write(json.dumps(result, ensure_ascii=False) + "\n")


# ── Step 8: AUROC against existing labels ────────────────────────────────────

def compute_auroc() -> None:
    """
    Compute AUROC for pBD-DTW and append the result to output/<dataset>/result.json.

    Mirrors the pattern in analyze_result.py: join labels and confidences by
    question text, then write results[model_engine]["pbd-dtw"] = auroc.
    Silently skips if the labels file does not yet exist.
    """
    import torch
    from torchmetrics import AUROC

    labels_path = os.path.join(args.output_path, "output_v1_w_labels.json")
    conf_path = os.path.join(args.output_path, "confidences", "ensemble_v1_pbd_dtw.json")

    if not os.path.exists(labels_path):
        print(f"Labels file not found, skipping AUROC: {labels_path}")
        return
    if not os.path.exists(conf_path):
        print(f"Confidence file not found, skipping AUROC: {conf_path}")
        return

    label_by_question: dict[str, int] = {}
    with open(labels_path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            label_by_question[d["question"]] = 1 if d["label"] else 0

    confidences, targets = [], []
    with open(conf_path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            q = d.get("question", "")
            if q in label_by_question:
                confidences.append(float(d["confidence"]))
                targets.append(label_by_question[q])

    if not confidences:
        print("No matching questions between labels and confidence file.")
        return

    auroc_fn = AUROC(task="binary")
    auroc_value = auroc_fn(torch.tensor(confidences), torch.tensor(targets))
    print(f"AUROC (pbd-dtw, {args.dataset}): {auroc_value.item():.6f}  (n={len(confidences)})")

    result_path = os.path.join("output", args.dataset, "result.json")
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    results = {}
    if os.path.exists(result_path):
        with open(result_path, encoding="utf-8") as f:
            results = json.load(f)

    results.setdefault(args.model_engine, {})["pbd-dtw"] = round(auroc_value.item(), 6)

    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    print_exp(args)

    if args.model_engine in ["llama3-1_8B", "llama2-13b"]:
        pbd_dtw_uq()
        compute_auroc()
    else:
        raise ValueError(f"Unsupported model engine: {args.model_engine}")
