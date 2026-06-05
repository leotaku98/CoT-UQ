# -*- coding: utf-8 -*-
"""
UQ via Graph Band Depth with j=3 triplet bands (gBD3) — chain-level, order-invariant.

Extends gBD (j=2) by sampling chain TRIPLETS instead of pairs.
The band of three chains is the union of their three pairwise bands:

    band(P_a, P_b, P_c) = band(P_a,P_b) ∪ band(P_b,P_c) ∪ band(P_a,P_c)

Richer coverage per trial than j=2 since three chains span more of the graph.
Pairwise bands are precomputed once and reused across all triplets — no extra
band computation cost beyond gBD.

Requires N >= 4 chains (at least 3 others excluding P_i); falls back to j=2
scoring when fewer chains are available.

Reads:  output/<model_engine>/<dataset>/ensemble_v1.json
Writes: output/<model_engine>/<dataset>/confidences/ensemble_v1_gBD3.json
        output/<dataset>/result.json  (key: "gBD3")
"""

import json
import os
import sys
from collections import Counter
from itertools import combinations
from typing import Optional

import networkx as nx
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../"))
from config import args
from utils import parse_response_to_dict, setup_log, print_exp

# ── Hyperparameters ────────────────────────────────────────────────────────────
ENCODER_MODEL = "all-MiniLM-L6-v2"
SIM_THRESHOLD = 0.85
ALPHA = 0.5
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


def _chain_band(
    P_j: set[int],
    P_k: set[int],
    all_vertices: list[int],
    spl: dict[int, dict[int, int]],
) -> set[int]:
    """Geodesic band of two chains: union of hulls over all |P_j|×|P_k| cross-pairs."""
    band: set[int] = set(P_j) | set(P_k)
    for u in P_j:
        u_spl = spl.get(u, {})
        for w in P_k:
            if w not in u_spl:
                continue
            d_uw = u_spl[w]
            w_spl = spl.get(w, {})
            for v in all_vertices:
                if v in u_spl and v in w_spl and u_spl[v] + w_spl[v] == d_uw:
                    band.add(v)
    return band


def _gbd3_scores(
    chain_sets: list[set[int]],
    all_vertices: list[int],
    spl: dict[int, dict[int, int]],
) -> list[float]:
    """
    Compute gBD3 score for every chain using j=3 triplet bands.

    Precomputes all C(N,2) pairwise bands, then for each chain P_i aggregates
    over all C(N-1, 3) valid triplets (a,b,c) with a≠i, b≠i, c≠i:

        band(P_a, P_b, P_c) = band(P_a,P_b) ∪ band(P_b,P_c) ∪ band(P_a,P_c)
        score(P_i | P_a, P_b, P_c) = |P_i ∩ band| / |P_i|

    Falls back to j=2 pair scoring when N < 4 (no valid triplet exists).
    """
    n = len(chain_sets)

    # Precompute all C(N,2) pairwise bands — reused across both triplets and j=2 fallback
    band_cache: dict[tuple[int, int], set[int]] = {}
    for j, k in combinations(range(n), 2):
        band_cache[(j, k)] = _chain_band(chain_sets[j], chain_sets[k], all_vertices, spl)

    scores = []
    for i in range(n):
        P_i = chain_sets[i]
        if not P_i:
            scores.append(0.0)
            continue

        # Collect valid triplets (a,b,c) excluding i
        trial_scores = []
        for a, b, c in combinations(range(n), 3):
            if a == i or b == i or c == i:
                continue
            band = band_cache[(a, b)] | band_cache[(b, c)] | band_cache[(a, c)]
            trial_scores.append(len(P_i & band) / len(P_i))

        if trial_scores:
            scores.append(float(np.mean(trial_scores)))
        else:
            # Fewer than 4 chains — fall back to j=2 pairs
            pair_scores = [
                len(P_i & band_cache[(j, k)]) / len(P_i)
                for j, k in combinations(range(n), 2)
                if j != i and k != i
            ]
            scores.append(float(np.mean(pair_scores)) if pair_scores else 0.5)

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


def gBD3_uq() -> None:
    """Run j=3 Graph Band Depth UQ over the ensemble."""
    from sentence_transformers import SentenceTransformer

    log = setup_log(args)

    input_path = os.path.join(args.output_path, "ensemble_v1.json")
    confidences_dir = os.path.join(args.output_path, "confidences")
    os.makedirs(confidences_dir, exist_ok=True)
    output_path = os.path.join(confidences_dir, "ensemble_v1_gBD3.json")

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Ensemble file not found: {input_path}")

    processed_ids = _load_processed_ids(output_path)
    if processed_ids:
        log.info(f"Resuming: skipping {len(processed_ids)} already-processed questions.")

    log.info(f"Loading encoder: {ENCODER_MODEL}")
    encoder = SentenceTransformer(ENCODER_MODEL)

    with open(input_path, encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    log.info(f"Computing gBD3 UQ for {len(lines)} questions "
             f"(j=3 triplet bands, threshold={SIM_THRESHOLD})")

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
                vertex_ids = _greedy_cluster(all_flat, SIM_THRESHOLD)

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

                    gbd3 = _gbd3_scores(chain_sets, all_vertices, spl)
                    gbd3_score = float(np.mean(gbd3))

                    answers = [s.get("llm answer", "") for s in samples]
                    majority_count = Counter(answers).most_common(1)[0][1]
                    majority_frac = majority_count / len(samples)

                    confidence = ALPHA * gbd3_score + (1 - ALPHA) * majority_frac

            result = {
                "id": qid,
                "question": record.get("question", ""),
                "correct answer": record.get("correct answer", ""),
                "llm answer": _majority_answer(samples),
                "confidence": confidence,
            }
            f_out.write(json.dumps(result, ensure_ascii=False) + "\n")


def compute_auroc() -> None:
    """Compute AUROC and write to output/<dataset>/result.json under 'gBD3'."""
    import torch
    from torchmetrics import AUROC

    labels_path = os.path.join(args.output_path, "output_v1_w_labels.json")
    conf_path = os.path.join(args.output_path, "confidences", "ensemble_v1_gBD3.json")

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
    print(f"AUROC (gBD3, {args.dataset}): {auroc_value.item():.6f}  (n={len(confidences)})")

    result_path = os.path.join("output", args.dataset, "result.json")
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    results = {}
    if os.path.exists(result_path):
        with open(result_path, encoding="utf-8") as f:
            results = json.load(f)

    results.setdefault(args.model_engine, {})["gBD3"] = round(auroc_value.item(), 6)

    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    print_exp(args)

    if args.model_engine in ["llama3-1_8B", "llama2-13b"]:
        gBD3_uq()
        compute_auroc()
    else:
        raise ValueError(f"Unsupported model engine: {args.model_engine}")
