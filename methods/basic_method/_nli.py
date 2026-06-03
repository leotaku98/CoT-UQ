# -*- coding: utf-8 -*-
"""Shared NLI model loader and pair-scoring utilities for basic_method UQ scripts."""

import numpy as np
from scipy.special import softmax

# cross-encoder/nli-deberta-v3-base: smaller (86M vs 400M params), cached on /data.
NLI_MODEL = "cross-encoder/nli-deberta-v3-base"


def load_nli_model(model_name: str = NLI_MODEL):
    """Load a CrossEncoder NLI model."""
    from sentence_transformers.cross_encoder import CrossEncoder
    return CrossEncoder(model_name)


def get_label_indices(model) -> tuple[int, int]:
    """
    Return (contradiction_idx, entailment_idx) by reading the model's id2label.
    Falls back to (0, 2) if the config is unavailable.
    """
    try:
        id2label = model.model.config.id2label
        label2id = {v.lower(): int(k) for k, v in id2label.items()}
        contradiction_idx = label2id.get("contradiction", 0)
        entailment_idx = label2id.get("entailment", 2)
    except AttributeError:
        contradiction_idx, entailment_idx = 0, 2
    return contradiction_idx, entailment_idx


def build_ordered_pairs(
    answers: list[str],
) -> tuple[list[tuple[str, str]], dict[tuple[int, int], int]]:
    """
    Build all ordered (i, j) pairs where i != j.

    Returns:
        pairs: flat list of (text_i, text_j) tuples for CrossEncoder.predict
        pair_idx: maps (i, j) → row index in the pairs list
    """
    pairs: list[tuple[str, str]] = []
    pair_idx: dict[tuple[int, int], int] = {}
    n = len(answers)
    for i in range(n):
        for j in range(n):
            if i != j:
                pair_idx[(i, j)] = len(pairs)
                pairs.append((answers[i], answers[j]))
    return pairs, pair_idx


def score_pairs(model, pairs: list[tuple[str, str]]) -> np.ndarray:
    """
    Run NLI forward pass on all pairs in one batch.

    Returns (n_pairs, n_labels) softmax probability array.
    """
    if not pairs:
        return np.zeros((0, 3), dtype=np.float32)
    logits = model.predict(pairs, show_progress_bar=False)
    if logits.ndim == 1:
        logits = logits.reshape(1, -1)
    return softmax(logits, axis=-1).astype(np.float32)
