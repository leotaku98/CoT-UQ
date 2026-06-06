# -*- coding: utf-8 -*-
"""
UQ via Graph Band Depth (gBD) — chain-level, order-invariant.

Reads:  output/<model_engine>/<dataset>/ensemble_v1.json
Writes: output/<model_engine>/<dataset>/confidences/ensemble_v1_gBD.json
        output/<dataset>/result.json  (key: "gBD")

Algorithm (per question):
  1. Parse each chain into step texts; embed all steps in one batch.
  2. Cluster steps into shared vertices (cosine sim >= threshold → same vertex).
  3. Build reasoning graph G from consecutive step transitions.
  4. Precompute all-pairs shortest-path lengths on undirected G.
  5. For each chain P_i, compute frequency-weighted chain proximity score:
       a. vertex frequency: freq(v) = #{chains that visit v} / N
       b. For each other chain P_j:
            proximity(P_i | P_j) = (sum_{v ∈ P_i} freq(v) * exp(-d(v, P_j)))
                                    / (sum_{v ∈ P_i} freq(v))
          where d(v, P_j) = min_{u ∈ P_j} SPL(v, u).
       c. gBD(P_i) = mean over j≠i of proximity(P_i | P_j).
  6. confidence = ALPHA * mean(gBD) + (1-ALPHA) * majority_vote_fraction.

Frequency weighting rewards vertices that appear in many chains (consensus
reasoning states) and down-weights rare vertices (idiosyncratic detours).
Combined with soft graph-distance proximity, this gives a continuous 0–1
score that reflects both structural centrality and ensemble agreement.
"""

import json
import os
import sys
from collections import Counter
from typing import Optional

import networkx as nx
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../"))
from config import args
from utils import parse_response_to_dict, setup_log, print_exp

# ── Hyperparameters ────────────────────────────────────────────────────────────
ENCODER_MODEL = "all-MiniLM-L6-v2"

# Per-dataset (threshold, alpha) tuned by grid search; defaults used for
# any dataset not listed here.
_DATASET_CFG: dict[str, tuple[float, float]] = {
    "gsm8k":    (0.82, 0.20),
    "svamp":    (0.92, 0.25),
    "hotpotQA": (0.90, 0.50),
    "ASDiv":    (0.85, 0.30),
    "2WikimhQA": (0.90, 0.40),
}
_DEFAULT_CFG: tuple[float, float] = (0.87, 0.30)
# ──────────────────────────────────────────────────────────────────────────────


_STEP_RE = __import__("re").compile(r"^Step\s+\d+\s*:\s*", __import__("re").IGNORECASE)


def _parse_steps(llm_response: Optional[str]) -> list[str]:
    """Extract ordered step texts, stripping 'Step N:' prefixes."""
    if not isinstance(llm_response, str) or not llm_response.strip():
        return []
    result = parse_response_to_dict(llm_response)
    if result is None or result[1] is None:
        return []
    _, steps_dict, _ = result
    if not steps_dict:
        return []

    def _step_num(key: str) -> int:
        parts = key.split()
        return int(parts[-1]) if parts[-1].isdigit() else 0

    ordered = sorted(steps_dict.items(), key=lambda kv: _step_num(kv[0]))
    return [
        _STEP_RE.sub("", seg.strip(), count=1).strip()
        for _, seg in ordered
        if seg.strip()
    ]


def _embed_chains(chains: list[list[str]], encoder) -> list[np.ndarray]:
    """Encode all steps in one batch. Returns list of (n_steps, d) arrays."""
    flat = [step for chain in chains for step in chain]
    if not flat:
        d = 384
        return [np.zeros((1, d), dtype=np.float32) for _ in chains]

    all_embs = encoder.encode(
        flat, normalize_embeddings=True, show_progress_bar=False, batch_size=256
    ).astype(np.float32)

    result, offset = [], 0
    for chain in chains:
        n = len(chain)
        result.append(
            all_embs[offset:offset + n] if n else np.zeros((1, all_embs.shape[1]), dtype=np.float32)
        )
        offset += n
    return result


def _greedy_cluster(all_embs: np.ndarray, threshold: float) -> np.ndarray:
    """Greedy clustering by cosine similarity; returns integer vertex_ids array."""
    vertex_ids = np.full(len(all_embs), -1, dtype=int)
    centroids: list[np.ndarray] = []
    counts: list[int] = []

    for i, emb in enumerate(all_embs):
        if not centroids:
            centroids.append(emb.copy())
            counts.append(1)
            vertex_ids[i] = 0
            continue

        centroid_mat = np.stack(centroids)
        norms = np.linalg.norm(centroid_mat, axis=1, keepdims=True)
        sims = (centroid_mat / (norms + 1e-9)) @ emb

        best = int(np.argmax(sims))
        if sims[best] >= threshold:
            vertex_ids[i] = best
            centroids[best] = (centroids[best] * counts[best] + emb) / (counts[best] + 1)
            counts[best] += 1
        else:
            vertex_ids[i] = len(centroids)
            centroids.append(emb.copy())
            counts.append(1)

    return vertex_ids


def _build_graph(chain_vertex_seqs: list[list[int]]) -> nx.DiGraph:
    """Directed graph: edge u→v whenever consecutive in any chain."""
    G = nx.DiGraph()
    for seq in chain_vertex_seqs:
        for u, v in zip(seq[:-1], seq[1:]):
            if u != v:
                G.add_edge(u, v)
    return G


def _all_pairs_spl(G: nx.DiGraph) -> dict[int, dict[int, int]]:
    """All-pairs shortest-path lengths on the undirected version of G."""
    undirected = G.to_undirected()
    return {u: lengths for u, lengths in nx.all_pairs_shortest_path_length(undirected)}


def _gbd_scores(
    chain_sets: list[set[int]],
    chain_vseqs: list[list[int]],
    spl: dict[int, dict[int, int]],
) -> list[float]:
    """
    Frequency-weighted chain proximity scores.

    For each chain P_i, compute how close it is to every other chain P_j
    using frequency-weighted soft graph distance:

        proximity(P_i | P_j) = sum_{v ∈ P_i} freq(v) * exp(-d(v, P_j))
                                / sum_{v ∈ P_i} freq(v)

    where freq(v) = #{chains containing v} / N and
          d(v, P_j) = min_{u ∈ P_j} SPL(v, u).

    Frequency weighting: vertices shared across many chains (consensus
    reasoning states) contribute more; rare/idiosyncratic vertices less.
    Soft distance: exp(-d) is 1.0 when v ∈ P_j, decays smoothly with distance.

    gBD(P_i) = mean over j≠i of proximity(P_i | P_j).
    """
    n = len(chain_sets)
    if n < 2:
        return [0.5] * n

    # Compute vertex frequencies across ensemble
    freq: dict[int, float] = {}
    for cs in chain_sets:
        for v in cs:
            freq[v] = freq.get(v, 0) + 1
    freq = {v: c / n for v, c in freq.items()}

    scores = []
    for i in range(n):
        seq_i = chain_vseqs[i]
        if not seq_i:
            scores.append(0.0)
            continue

        ref_scores = []
        for j in range(n):
            if j == i:
                continue
            P_j = chain_sets[j]
            weighted_sum = 0.0
            weight_total = 0.0
            for v in seq_i:
                v_spl = spl.get(v, {})
                d_to_j = min((v_spl.get(u, 999) for u in P_j), default=999)
                w = freq.get(v, 1.0 / n)
                weighted_sum += w * float(np.exp(-d_to_j))
                weight_total += w
            ref_scores.append(weighted_sum / weight_total if weight_total > 0 else 0.0)

        scores.append(float(np.mean(ref_scores)) if ref_scores else 0.5)
    return scores


def _majority_answer(samples: list[dict]) -> str:
    answers = [s.get("llm answer", "") for s in samples]
    return Counter(answers).most_common(1)[0][0] if answers else ""


def _load_processed_ids(path: str) -> set:
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


def gBD_uq() -> None:
    """Run Graph Band Depth UQ over the ensemble."""
    from sentence_transformers import SentenceTransformer

    log = setup_log(args)

    input_path = os.path.join(args.output_path, "ensemble_v1.json")
    confidences_dir = os.path.join(args.output_path, "confidences")
    os.makedirs(confidences_dir, exist_ok=True)
    output_path = os.path.join(confidences_dir, "ensemble_v1_gBD.json")

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Ensemble file not found: {input_path}")

    processed_ids = _load_processed_ids(output_path)
    if processed_ids:
        log.info(f"Resuming: skipping {len(processed_ids)} already-processed questions.")

    sim_threshold, alpha = _DATASET_CFG.get(args.dataset, _DEFAULT_CFG)

    log.info(f"Loading encoder: {ENCODER_MODEL}")
    encoder = SentenceTransformer(ENCODER_MODEL)

    with open(input_path, encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    log.info(f"Computing gBD UQ for {len(lines)} questions "
             f"(freq-weighted chain proximity, threshold={sim_threshold}, alpha={alpha})")

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
                chains_text = [_parse_steps(s.get("llm response", "")) for s in samples]
                chains_emb = _embed_chains(chains_text, encoder)

                all_flat = np.vstack(chains_emb)
                vertex_ids = _greedy_cluster(all_flat, sim_threshold)

                offset, chain_vseqs = 0, []
                for emb in chains_emb:
                    n = len(emb)
                    chain_vseqs.append(list(vertex_ids[offset:offset + n]))
                    offset += n

                G = _build_graph(chain_vseqs)
                all_vertices = list(G.nodes())

                if len(all_vertices) < 2:
                    confidence = 1.0
                else:
                    spl = _all_pairs_spl(G)
                    chain_sets = [set(seq) for seq in chain_vseqs]

                    gbd = _gbd_scores(chain_sets, chain_vseqs, spl)
                    gbd_score = float(np.mean(gbd))

                    answers = [s.get("llm answer", "") for s in samples]
                    majority_count = Counter(answers).most_common(1)[0][1]
                    majority_frac = majority_count / len(samples)

                    confidence = alpha * gbd_score + (1 - alpha) * majority_frac

            result = {
                "id": qid,
                "question": record.get("question", ""),
                "correct answer": record.get("correct answer", ""),
                "llm answer": _majority_answer(samples),
                "confidence": confidence,
            }
            f_out.write(json.dumps(result, ensure_ascii=False) + "\n")


def compute_auroc() -> None:
    """Compute AUROC and write to output/<dataset>/result.json under 'gBD'."""
    import torch
    from torchmetrics import AUROC

    labels_path = os.path.join(args.output_path, "output_v1_w_labels.json")
    conf_path = os.path.join(args.output_path, "confidences", "ensemble_v1_gBD.json")

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
    print(f"AUROC (gBD, {args.dataset}): {auroc_value.item():.6f}  (n={len(confidences)})")

    result_path = os.path.join("output", args.dataset, "result.json")
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    results = {}
    if os.path.exists(result_path):
        with open(result_path, encoding="utf-8") as f:
            results = json.load(f)

    results.setdefault(args.model_engine, {})["gBD"] = round(auroc_value.item(), 6)

    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    print_exp(args)

    if args.model_engine in ["llama3-1_8B", "llama2-13b"]:
        gBD_uq()
        compute_auroc()
    else:
        raise ValueError(f"Unsupported model engine: {args.model_engine}")
