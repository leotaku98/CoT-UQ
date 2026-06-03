# -*- coding: utf-8 -*-
"""
UQ via NLI-based consistency scoring (Entailment Probability + Non-Contradiction).

Reads:  output/<model_engine>/<dataset>/ensemble_v1.json
Writes:
  output/<model_engine>/<dataset>/confidences/ensemble_v1_entailment.json
  output/<model_engine>/<dataset>/confidences/ensemble_v1_non_contradiction.json
  output/<dataset>/result.json  (keys: "entailment-prob", "non-contradiction")

Both metrics are computed from the same NLI forward pass over all ordered answer pairs.

Entailment Probability:
    confidence = mean P(entailment) over all ordered pairs (i, j), i != j.
    High → answers mutually entail each other → model is consistent.

Non-Contradiction Probability:
    confidence = mean (1 - P(contradiction)) over all ordered pairs.
    High → answers do not contradict each other.
"""

import json
import os
import sys

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


def _majority_answer(samples: list[dict]) -> str:
    """Return the most common llm_answer in the ensemble."""
    from collections import Counter
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


def _compute_auroc(method_name: str, conf_filename: str) -> None:
    """Compute AUROC and write to output/<dataset>/result.json."""
    import torch
    from torchmetrics import AUROC

    labels_path = os.path.join(args.output_path, "output_v1_w_labels.json")
    conf_path = os.path.join(args.output_path, "confidences", conf_filename)

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
        print(f"No matching questions for {method_name}, skipping AUROC.")
        return

    auroc_fn = AUROC(task="binary")
    auroc_value = auroc_fn(torch.tensor(confidences), torch.tensor(targets))
    print(f"AUROC ({method_name}, {args.dataset}): {auroc_value.item():.6f}  (n={len(confidences)})")

    result_path = os.path.join("output", args.dataset, "result.json")
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    results: dict = {}
    if os.path.exists(result_path):
        with open(result_path, encoding="utf-8") as f:
            results = json.load(f)

    results.setdefault(args.model_engine, {})[method_name] = round(auroc_value.item(), 6)

    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def nli_uq() -> None:
    """Run NLI-based UQ, writing both entailment-prob and non-contradiction outputs."""
    log = setup_log(args)

    input_path = os.path.join(args.output_path, "ensemble_v1.json")
    confidences_dir = os.path.join(args.output_path, "confidences")
    os.makedirs(confidences_dir, exist_ok=True)

    entail_path = os.path.join(confidences_dir, "ensemble_v1_entailment.json")
    noncon_path = os.path.join(confidences_dir, "ensemble_v1_non_contradiction.json")

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Ensemble file not found: {input_path}")

    # Resume: skip questions already written to the entailment file.
    # Both files are always written together, so one file is sufficient to check.
    processed_ids = _load_processed_ids(entail_path)
    if processed_ids:
        log.info(f"Resuming: skipping {len(processed_ids)} already-processed questions.")

    log.info(f"Loading NLI model: {NLI_MODEL}")
    model = load_nli_model()
    contradiction_idx, entailment_idx = get_label_indices(model)
    log.info(f"Label indices — contradiction: {contradiction_idx}, entailment: {entailment_idx}")

    with open(input_path, encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    log.info(f"Computing NLI UQ for {len(lines)} questions.")

    with (
        open(entail_path, "a", encoding="utf-8") as f_entail,
        open(noncon_path, "a", encoding="utf-8") as f_noncon,
    ):
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
            # Treat missing/empty responses as distinct empty strings.
            answers = [a if isinstance(a, str) else "" for a in answers]

            majority = _majority_answer(samples)

            if len(answers) < 2:
                entail_conf = 0.5
                noncon_conf = 0.5
            else:
                pairs, pair_idx = build_ordered_pairs(answers)
                probs = score_pairs(model, pairs)  # (n_pairs, n_labels)

                entail_scores = probs[:, entailment_idx]
                contra_scores = probs[:, contradiction_idx]
                entail_conf = float(np.mean(entail_scores))
                noncon_conf = float(np.mean(1.0 - contra_scores))

            base = {
                "id": qid,
                "question": record.get("question", ""),
                "correct answer": record.get("correct answer", ""),
                "llm answer": majority,
            }
            f_entail.write(json.dumps({**base, "confidence": entail_conf}, ensure_ascii=False) + "\n")
            f_noncon.write(json.dumps({**base, "confidence": noncon_conf}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    print_exp(args)

    if args.model_engine in ["llama3-1_8B", "llama2-13b"]:
        nli_uq()
        _compute_auroc("entailment-prob", "ensemble_v1_entailment.json")
        _compute_auroc("non-contradiction", "ensemble_v1_non_contradiction.json")
    else:
        raise ValueError(f"Unsupported model engine: {args.model_engine}")
