# -*- coding: utf-8 -*-
"""
UQ via Discrete Semantic Entropy (Kuhn et al., 2023).

Reads:  output/<model_engine>/<dataset>/ensemble_v1.json
Writes: output/<model_engine>/<dataset>/confidences/ensemble_v1_semantic_entropy.json
        output/<dataset>/result.json  (key: "semantic-entropy")

Algorithm (per question):
  1. Collect the N sampled answers.
  2. Cluster answers into semantic equivalence classes:
       answers i and j are in the same class iff
       NLI(i→j) = entailment  AND  NLI(j→i) = entailment  (bidirectional).
     Union-Find is used to propagate transitivity.
  3. Compute Shannon entropy over cluster sizes:
       H = -Σ p_k * log(p_k),  p_k = |cluster_k| / N
  4. confidence = exp(-H)
       → 1.0 when all answers are semantically identical (H=0)
       → near 0 when answers are maximally scattered (H=log N)
"""

import json
import os
import sys
from collections import Counter

import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../"))
from config import args
from utils import setup_log, print_exp
from methods.basic_method._nli import (
    NLI_MODEL,
    load_nli_model,
    get_label_indices,
    build_ordered_pairs,
    score_pairs,
)

# Minimum softmax probability to count a direction as "entailment"
ENTAILMENT_THRESHOLD = 0.5


def _find(parent: list[int], x: int) -> int:
    """Path-compressed find for union-find."""
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def _union(parent: list[int], x: int, y: int) -> None:
    """Union two sets in the union-find structure."""
    parent[_find(parent, x)] = _find(parent, y)


def _cluster_answers(
    answers: list[str],
    probs: np.ndarray,
    pair_idx: dict[tuple[int, int], int],
    entailment_idx: int,
) -> list[int]:
    """
    Cluster N answers by bidirectional entailment using union-find.

    Returns a list of cluster IDs (root indices) of length N.
    """
    n = len(answers)
    parent = list(range(n))
    for i in range(n):
        for j in range(n):
            if i >= j:
                continue
            p_ij = probs[pair_idx[(i, j)], entailment_idx]
            p_ji = probs[pair_idx[(j, i)], entailment_idx]
            if p_ij >= ENTAILMENT_THRESHOLD and p_ji >= ENTAILMENT_THRESHOLD:
                _union(parent, i, j)
    return [_find(parent, i) for i in range(n)]


def _semantic_entropy(cluster_ids: list[int]) -> float:
    """Compute Shannon entropy over cluster size distribution."""
    n = len(cluster_ids)
    if n == 0:
        return 0.0
    counts = Counter(cluster_ids)
    probs = np.array(list(counts.values()), dtype=np.float64) / n
    return float(-np.sum(probs * np.log(probs + 1e-12)))


def _majority_answer(samples: list[dict]) -> str:
    """Return the most common llm_answer in the ensemble."""
    answers = [s.get("llm answer", "") for s in samples]
    return Counter(answers).most_common(1)[0][0] if answers else ""


def _load_processed_ids(path: str) -> set:
    """Return IDs already written to an output file (for resume support)."""
    processed: set = set()
    if not os.path.exists(path):
        return processed
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                processed.add(json.loads(line)["id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return processed


def _compute_auroc() -> None:
    """Compute AUROC and write to output/<dataset>/result.json under 'semantic-entropy'."""
    import torch
    from torchmetrics import AUROC

    labels_path = os.path.join(args.output_path, "output_v1_w_labels.json")
    conf_path = os.path.join(args.output_path, "confidences", "ensemble_v1_semantic_entropy.json")

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
    print(f"AUROC (semantic-entropy, {args.dataset}): {auroc_value.item():.6f}  (n={len(confidences)})")

    result_path = os.path.join("output", args.dataset, "result.json")
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    results: dict = {}
    if os.path.exists(result_path):
        with open(result_path, encoding="utf-8") as f:
            results = json.load(f)

    results.setdefault(args.model_engine, {})["semantic-entropy"] = round(auroc_value.item(), 6)

    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def semantic_entropy_uq() -> None:
    """Run Discrete Semantic Entropy UQ over the ensemble."""
    log = setup_log(args)

    input_path = os.path.join(args.output_path, "ensemble_v1.json")
    confidences_dir = os.path.join(args.output_path, "confidences")
    os.makedirs(confidences_dir, exist_ok=True)
    output_path = os.path.join(confidences_dir, "ensemble_v1_semantic_entropy.json")

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Ensemble file not found: {input_path}")

    processed_ids = _load_processed_ids(output_path)
    if processed_ids:
        log.info(f"Resuming: skipping {len(processed_ids)} already-processed questions.")

    log.info(f"Loading NLI model: {NLI_MODEL}")
    model = load_nli_model()
    contradiction_idx, entailment_idx = get_label_indices(model)
    log.info(f"Label indices — contradiction: {contradiction_idx}, entailment: {entailment_idx}")
    log.info(f"Entailment threshold: {ENTAILMENT_THRESHOLD}")

    with open(input_path, encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    log.info(f"Computing Semantic Entropy UQ for {len(lines)} questions.")

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
            answers = [s.get("llm response", "") for s in samples]
            answers = [a if isinstance(a, str) else "" for a in answers]

            if len(answers) < 2:
                confidence = 0.5
            else:
                pairs, pair_idx = build_ordered_pairs(answers)
                probs = score_pairs(model, pairs)

                cluster_ids = _cluster_answers(answers, probs, pair_idx, entailment_idx)
                entropy = _semantic_entropy(cluster_ids)
                confidence = float(np.exp(-entropy))

            result = {
                "id": qid,
                "question": record.get("question", ""),
                "correct answer": record.get("correct answer", ""),
                "llm answer": _majority_answer(samples),
                "confidence": confidence,
            }
            f_out.write(json.dumps(result, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    print_exp(args)

    if args.model_engine in ["llama3-1_8B", "llama2-13b"]:
        semantic_entropy_uq()
        _compute_auroc()
    else:
        raise ValueError(f"Unsupported model engine: {args.model_engine}")
