# -*- coding: utf-8 -*-
"""
UQ via Cross-Model Disagreement (Hamidieh et al., 2026).

Reads:  output/<model_engine>/<dataset>/ensemble_v1.json          (reference model)
        output/<aux_engine>/<dataset>/ensemble_v1.json            (auxiliary models)
Writes: output/<model_engine>/<dataset>/confidences/ensemble_v1_cross_model.json
        output/metric/<dataset>.json  (key: "cross-model-disagreement")

Total predictive uncertainty is decomposed into two terms, mirroring the classical
deep-ensemble aleatoric/epistemic split but computed in semantic response space so
that it needs only black-box text output:

  1. Aleatoric (within-model): mean pairwise semantic dissimilarity among the
     reference model's own sampled responses -- the quantity self-consistency
     methods already estimate.
  2. Epistemic (cross-model): the gap between how much the reference model agrees
     with itself and how much it agrees with an auxiliary ensemble of independently
     trained models. A model that is internally consistent but diverges from other
     families is confidently wrong -- a failure self-consistency cannot see, since
     it never leaves the reference model's own distribution.

  confidence = 1 - (aleatoric + epistemic) / 2

Deviations from the published method, both forced by available compute:
  * The auxiliary ensemble is the project's other three backbones (leave-one-out)
    rather than five independently trained 7-9B families, so it spans two families
    (Llama, Qwen) at 3B-13B rather than five families at 7-9B.
  * Every model is truncated to the first N_SAMPLES responses so the sampling
    budget is matched; the stored ensembles are otherwise ragged (5 or 10).
"""

import json
import os
import sys

import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../"))
from config import args
from utils import setup_log, print_exp

ENCODER_MODEL = "all-MiniLM-L6-v2"

# All backbones with a stored ensemble; the reference model is excluded at runtime.
ALL_ENGINES = ["llama3-1_8B", "llama2-13b", "qwen2.5-3b", "qwen3-8B"]

# Matched sampling budget: responses per model per question.
N_SAMPLES = 5


def _majority_answer(samples: list[dict]) -> str:
    """Most common final answer across the reference model's samples."""
    from collections import Counter

    answers = [s.get("llm answer", "") for s in samples]
    answers = [a for a in answers if a]
    if not answers:
        return ""
    return Counter(answers).most_common(1)[0][0]


def _load_ensemble(engine: str) -> dict[str, list[str]]:
    """Map question id -> up to N_SAMPLES response strings for one engine."""
    path = os.path.join("output", engine, args.dataset, "ensemble_v1.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing auxiliary ensemble: {path}")

    by_id: dict[str, list[str]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            responses = [s.get("llm response", "") for s in record.get("samples", [])]
            responses = [r for r in responses if isinstance(r, str) and r.strip()]
            by_id[record["id"]] = responses[:N_SAMPLES]
    return by_id


def _mean_pairwise_similarity(embs: np.ndarray) -> float:
    """Mean cosine similarity over distinct unordered pairs of unit-norm rows."""
    if len(embs) < 2:
        return 1.0
    sims = embs @ embs.T
    iu = np.triu_indices(len(embs), k=1)
    return float(sims[iu].mean())


def _mean_cross_similarity(ref_embs: np.ndarray, aux_embs: np.ndarray) -> float:
    """Mean cosine similarity over all reference-auxiliary pairs."""
    if len(ref_embs) == 0 or len(aux_embs) == 0:
        return 1.0
    return float((ref_embs @ aux_embs.T).mean())


def cross_model_uq() -> None:
    """Score every question by aleatoric + epistemic semantic disagreement."""
    from sentence_transformers import SentenceTransformer

    log = setup_log(args)

    aux_engines = [e for e in ALL_ENGINES if e != args.model_engine]
    log.info(f"Reference: {args.model_engine} | auxiliary ensemble: {aux_engines}")

    aux_data = {engine: _load_ensemble(engine) for engine in aux_engines}

    input_path = os.path.join(args.output_path, "ensemble_v1.json")
    confidences_dir = os.path.join(args.output_path, "confidences")
    os.makedirs(confidences_dir, exist_ok=True)
    output_path = os.path.join(confidences_dir, "ensemble_v1_cross_model.json")

    processed_ids: set = set()
    if os.path.exists(output_path):
        with open(output_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    processed_ids.add(json.loads(line)["id"])

    with open(input_path, encoding="utf-8") as f:
        lines = [line for line in f if line.strip()]

    log.info(f"Loading encoder: {ENCODER_MODEL}")
    encoder = SentenceTransformer(ENCODER_MODEL)

    log.info(f"Computing Cross-Model Disagreement UQ for {len(lines)} questions.")
    missing_aux = 0

    with open(output_path, "a", encoding="utf-8") as f_out:
        for line in tqdm(lines, total=len(lines)):
            record = json.loads(line)
            qid = record["id"]
            if qid in processed_ids:
                continue

            samples = record.get("samples", [])[:N_SAMPLES]
            ref_responses = [s.get("llm response", "") for s in samples]
            ref_responses = [r for r in ref_responses if isinstance(r, str) and r.strip()]

            aux_responses: list[str] = []
            for engine in aux_engines:
                aux_responses.extend(aux_data[engine].get(qid, []))

            if len(ref_responses) < 2 or not aux_responses:
                missing_aux += 1
                confidence = 0.5
            else:
                embs = encoder.encode(
                    ref_responses + aux_responses,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
                ref_embs = np.asarray(embs[:len(ref_responses)])
                aux_embs = np.asarray(embs[len(ref_responses):])

                self_sim = _mean_pairwise_similarity(ref_embs)
                cross_sim = _mean_cross_similarity(ref_embs, aux_embs)

                # Both terms are mapped from cosine [-1, 1] onto [0, 1].
                aleatoric = (1.0 - self_sim) / 2.0
                epistemic = max(0.0, self_sim - cross_sim) / 2.0
                confidence = 1.0 - (aleatoric + epistemic) / 2.0

            result = {
                "id": qid,
                "question": record.get("question", ""),
                "correct answer": record.get("correct answer", ""),
                "llm answer": _majority_answer(samples),
                "confidence": float(confidence),
            }
            f_out.write(json.dumps(result, ensure_ascii=False) + "\n")

    if missing_aux:
        log.info(f"{missing_aux} questions fell back to confidence 0.5 (too few responses).")


def _compute_auroc() -> None:
    """Compute AUROC and write to output/metric/<dataset>.json under 'cross-model-disagreement'."""
    import torch
    from torchmetrics import AUROC

    labels_path = os.path.join(args.output_path, "output_v1_w_labels.json")
    conf_path = os.path.join(args.output_path, "confidences", "ensemble_v1_cross_model.json")

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
    print(f"AUROC (cross-model-disagreement, {args.dataset}): {auroc_value.item():.6f}  (n={len(confidences)})")

    result_path = os.path.join("output", "metric", args.dataset + ".json")
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    results: dict = {}
    if os.path.exists(result_path):
        with open(result_path, encoding="utf-8") as f:
            results = json.load(f)

    results.setdefault(args.model_engine, {})["cross-model-disagreement"] = round(auroc_value.item(), 6)

    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    print_exp(args)

    if args.model_engine in ALL_ENGINES:
        cross_model_uq()
        _compute_auroc()
    else:
        raise ValueError(f"Unsupported model engine: {args.model_engine}")
