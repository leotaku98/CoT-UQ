# -*- coding: utf-8 -*-
"""
UQ via Path Band Depth with DTW alignment (pBD — order-sensitive).

Reads:  output/<model_engine>/<dataset>/ensemble_v1.json
Writes: output/<model_engine>/<dataset>/confidences/ensemble_v1_pBD.json

Each output line:
    {"id": ..., "question": ..., "correct answer": ...,
     "llm answer": <majority-vote answer>, "confidence": <mean pBD>}

confidence = mean pBD across all chains in the ensemble.
High confidence → chains are central/consistent. Low → scattered reasoning.
"""

import json
import os
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
ALPHA = 0.5  # weight on process-level (pBD) vs outcome-level (majority vote)
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

def _pbd_score(emb_p: np.ndarray, others: list[np.ndarray]) -> float:
    """
    Compute the pBD centrality score for chain emb_p against the full ensemble.

    For a finite ensemble, the centroid of all remaining chains is a more stable
    reference than random pair sampling. For each step position k in emb_p:
      1. DTW-align emb_p against every other chain.
      2. Collect the DTW-matched step embeddings from all other chains at k.
         (Average when DTW maps multiple steps of a reference to position k.)
      3. Compute the centroid of those collected embeddings.
      4. Score = cosine similarity of emb_p[k] to that centroid.
    Final score = mean over all step positions k.
    """
    if not others:
        return 1.0
    if not np.any(emb_p):
        return 0.0

    n = len(emb_p)
    # p_ref_steps[k] = list of reference embeddings matched to p[k], across all refs
    p_ref_steps: list[list[np.ndarray]] = [[] for _ in range(n)]

    for ref in others:
        path = _dtw_path(emb_p, ref)
        ref_map: dict[int, list[int]] = {}
        for i_p, i_ref in path:
            ref_map.setdefault(i_p, []).append(i_ref)
        for i_p, i_refs in ref_map.items():
            p_ref_steps[i_p].append(ref[i_refs].mean(axis=0))

    memberships = []
    for k in range(n):
        if not p_ref_steps[k]:
            continue
        centroid = np.mean(np.stack(p_ref_steps[k]), axis=0)
        memberships.append(_soft_membership(emb_p[k], [centroid]))

    return float(np.mean(memberships)) if memberships else 0.0


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

def pBD_uq() -> None:
    """Run DTW-pBD uncertainty quantification over the ensemble."""
    from sentence_transformers import SentenceTransformer

    log = setup_log(args)

    input_path = os.path.join(args.output_path, "ensemble_v1.json")
    confidences_dir = os.path.join(args.output_path, "confidences")
    os.makedirs(confidences_dir, exist_ok=True)
    output_path = os.path.join(confidences_dir, "ensemble_v1_pBD.json")

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Ensemble file not found: {input_path}")

    processed_ids = _load_processed_ids(output_path)
    if processed_ids:
        log.info(f"Resuming: skipping {len(processed_ids)} already-processed questions.")

    log.info(f"Loading encoder: {ENCODER_MODEL}")
    encoder = SentenceTransformer(ENCODER_MODEL)

    with open(input_path, encoding="utf-8") as f_in:
        lines = [line.strip() for line in f_in if line.strip()]

    log.info(f"Computing pBD UQ for {len(lines)} questions "
             f"(centroid of all remaining chains)")

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

                # Process-level: mean pBD across all chains
                depths = []
                for i, emb_p in enumerate(chains_emb):
                    others = [chains_emb[k] for k in range(len(chains_emb)) if k != i]
                    depths.append(_pbd_score(emb_p, others))
                pbd_score = float(np.mean(depths))

                # Outcome-level: majority vote fraction
                answers = [s.get("llm answer", "") for s in samples]
                majority_count = Counter(answers).most_common(1)[0][1]
                majority_frac = majority_count / len(samples)

                # Combined confidence
                confidence = ALPHA * pbd_score + (1 - ALPHA) * majority_frac

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
    Compute AUROC for pBD and append the result to output/<dataset>/result.json.

    Mirrors the pattern in analyze_result.py: join labels and confidences by
    question text, then write results[model_engine]["pBD"] = auroc.
    Silently skips if the labels file does not yet exist.
    """
    import torch
    from torchmetrics import AUROC

    labels_path = os.path.join(args.output_path, "output_v1_w_labels.json")
    conf_path = os.path.join(args.output_path, "confidences", "ensemble_v1_pBD.json")

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
    print(f"AUROC (pBD, {args.dataset}): {auroc_value.item():.6f}  (n={len(confidences)})")

    result_path = os.path.join("output", args.dataset, "result.json")
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    results = {}
    if os.path.exists(result_path):
        with open(result_path, encoding="utf-8") as f:
            results = json.load(f)

    results.setdefault(args.model_engine, {})["pBD"] = round(auroc_value.item(), 6)

    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    print_exp(args)

    if args.model_engine in ["llama3-1_8B", "llama2-13b"]:
        pBD_uq()
        compute_auroc()
    else:
        raise ValueError(f"Unsupported model engine: {args.model_engine}")
