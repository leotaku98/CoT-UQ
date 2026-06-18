# -*- coding: utf-8 -*-
"""
UQ via Graph Band Depth v2 (gBD_v2) — anchored random-walk path band depth.

Reads:  output/<model_engine>/<dataset>/ensemble_v1.json
Writes: output/<model_engine>/<dataset>/confidences/ensemble_v1_gBD_v2.json
        output/metric/<dataset>/result.json  (key: "gBD_v2")

Algorithm (per question):
  1. Parse each chain into step texts; embed all steps in one batch.
  2. Cluster steps into shared vertices (cosine sim >= threshold → same vertex).
  3. Build directed reasoning graph G from consecutive step transitions.
     Add two special anchor nodes:
       INPUT_VERTEX  (-1): connected → first step of every chain.
       OUTPUT_VERTEX (-2): connected ← last step of every chain.
     Add undirected similarity edges between distinct vertices whose centroid
     cosine similarity ∈ [CROSS_SIM_THRESHOLD, SIM_THRESHOLD): "similar but
     not identical" steps get a direct connection, reducing graph fragmentation.
  4. Precompute all-pairs shortest-path lengths on undirected G.
  5. For each chain P_i, estimate anchored path band depth (N_TRIALS):
       a. Sample two anchored random walks P_a, P_b:
            start = INPUT_VERTEX; follow random edges for walk_length steps;
            stop early if OUTPUT_VERTEX is reached; append OUTPUT_VERTEX if not.
       b. Compute geodesic convex hull of {P_a, P_b}:
            hull = { v : ∃ u ∈ P_a, w ∈ P_b s.t. d(u,v) + d(v,w) = d(u,w) }
       c. Score = fraction of P_i's vertices that lie inside the hull.
  6. BD(P_i) = mean score over N_TRIALS.
  7. confidence = ALPHA * mean(BD) + (1-ALPHA) * majority_vote_fraction.

Anchoring interpretation: each random walk is a plausible hypothetical reasoning
path for the same question (starts at INPUT) that produces some answer (ends at
OUTPUT). The band between two such paths captures the space of plausible
reasoning between them. A chain with high band depth lies "centrally" between
many pairs of hypothetical reasoning paths — i.e., its reasoning is consistent
with what the graph considers reachable from this specific question.
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
SIM_THRESHOLD: float = 0.92    # cosine sim >= this → merge into same vertex
CROSS_SIM_THRESHOLD: Optional[float] = 0.70  # cosine sim in [CROSS, SIM) → add soft edge
N_TRIALS: int = 200
ALPHA: float = 0.30
# ──────────────────────────────────────────────────────────────────────────────

INPUT_VERTEX: int = -1
OUTPUT_VERTEX: int = -2

# Ablation override: subset size J = number of reference walks forming each band.
N_WALKS: int = args.subset_size if args.subset_size else 2


def _conf_path(confidences_dir: str) -> str:
    """Confidence output path; suffixed and relocated under ablation mode."""
    if not args.ablation:
        return os.path.join(confidences_dir, "ensemble_v1_gBD_v2.json")
    abl_dir = os.path.join(confidences_dir, "ablation")
    os.makedirs(abl_dir, exist_ok=True)
    suffix = f"subset_size_{N_WALKS}" if args.ablation == "subset_size" else f"walk_length_{args.walk_length}"
    return os.path.join(abl_dir, f"ensemble_v1_gBD_v2_{suffix}.json")

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


def _greedy_cluster(
    all_embs: np.ndarray, threshold: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Greedy clustering by cosine similarity.

    Returns (vertex_ids, centroids) where centroids[k] is the mean embedding
    of all steps assigned to vertex k.
    """
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

    return vertex_ids, np.stack(centroids) if centroids else np.zeros((0, all_embs.shape[1]))


def _add_similarity_edges(
    G: nx.DiGraph,
    centroids: np.ndarray,
    cross_threshold: float,
    sim_threshold: float,
) -> None:
    """
    Add undirected edges between distinct vertices whose centroid cosine
    similarity falls in [cross_threshold, sim_threshold).

    These "soft" edges connect semantically related but non-identical reasoning
    steps, making the graph less fragmented and allowing random walks to
    traverse conceptually nearby steps even if they never co-occurred in the
    same chain transition.
    """
    n = len(centroids)
    if n < 2:
        return
    norms = np.linalg.norm(centroids, axis=1, keepdims=True)
    normed = centroids / (norms + 1e-9)
    sim_matrix = normed @ normed.T  # (n, n)

    for i in range(n):
        for j in range(i + 1, n):
            s = float(sim_matrix[i, j])
            if cross_threshold <= s < sim_threshold:
                G.add_edge(i, j)
                G.add_edge(j, i)


def _build_graph(chain_vertex_seqs: list[list[int]]) -> nx.DiGraph:
    """
    Directed graph with INPUT (-1) and OUTPUT (-2) anchor nodes.

    INPUT  → first step of every chain
    last step of every chain → OUTPUT
    consecutive step transitions within each chain
    """
    G = nx.DiGraph()
    for seq in chain_vertex_seqs:
        if not seq:
            continue
        G.add_edge(INPUT_VERTEX, seq[0])
        G.add_edge(seq[-1], OUTPUT_VERTEX)
        for u, v in zip(seq[:-1], seq[1:]):
            if u != v:
                G.add_edge(u, v)
    return G


def _all_pairs_spl(G: nx.DiGraph) -> dict[int, dict[int, int]]:
    """All-pairs shortest-path lengths on the undirected version of G."""
    undirected = G.to_undirected()
    return {u: lengths for u, lengths in nx.all_pairs_shortest_path_length(undirected)}


def _anchored_walk(
    G_undirected: nx.Graph,
    walk_length: int,
    rng: np.random.Generator,
) -> list[int]:
    """
    Random walk starting at INPUT_VERTEX, ending at OUTPUT_VERTEX.

    Follows random edges for up to walk_length steps from INPUT_VERTEX.
    Stops early if OUTPUT_VERTEX is reached naturally. If OUTPUT_VERTEX
    is not reached within walk_length steps, appends it explicitly —
    every reasoning path must eventually produce an answer.
    """
    path = [INPUT_VERTEX]
    current = INPUT_VERTEX
    for _ in range(walk_length):
        if current == OUTPUT_VERTEX:
            break
        neighbors = list(G_undirected.neighbors(current))
        if not neighbors:
            break
        current = int(rng.choice(neighbors))
        path.append(current)
    if path[-1] != OUTPUT_VERTEX:
        path.append(OUTPUT_VERTEX)
    return path


def _compute_hull(
    path_a: list[int],
    path_b: list[int],
    spl: dict[int, dict[int, int]],
    all_vertices: list[int],
) -> set[int]:
    """
    Geodesic convex hull of path_a ∪ path_b.

    v is in the hull iff ∃ u ∈ path_a, w ∈ path_b: d(u,v) + d(v,w) = d(u,w).
    """
    hull: set[int] = set()
    set_a = set(path_a)
    set_b = set(path_b)

    for u in set_a:
        u_spl = spl.get(u, {})
        for w in set_b:
            d_uw = u_spl.get(w)
            if d_uw is None:
                continue
            w_spl = spl.get(w, {})
            for v in all_vertices:
                if v in hull:
                    continue
                d_uv = u_spl.get(v)
                d_vw = w_spl.get(v)
                if d_uv is not None and d_vw is not None and d_uv + d_vw == d_uw:
                    hull.add(v)
    return hull


def _path_band_depths(
    chain_vseqs: list[list[int]],
    spl: dict[int, dict[int, int]],
    all_vertices: list[int],
    G_undirected: nx.Graph,
    n_trials: int,
    walk_length: int,
    n_walks: int = 2,
) -> list[float]:
    """
    For each chain P_i, estimate anchored band depth via random reference walks.

    Each trial:
      1. Generate n_walks anchored walks (INPUT → ... → OUTPUT).
      2. Compute the geodesic hull as the union over all walk pairs.
         (n_walks=2 reduces to a single pair, the default behaviour.)
      3. Score = fraction of P_i's step vertices inside the hull.
         (INPUT and OUTPUT anchors are excluded from P_i's vertex sequence.)
    BD(P_i) = mean score over n_trials.
    """
    rng = np.random.default_rng()
    scores = []

    for seq in chain_vseqs:
        if not seq:
            scores.append(0.0)
            continue

        trial_scores = []
        for _ in range(n_trials):
            walks = [_anchored_walk(G_undirected, walk_length, rng) for _ in range(n_walks)]
            hull: set[int] = set()
            for i in range(len(walks)):
                for k in range(i + 1, len(walks)):
                    hull |= _compute_hull(walks[i], walks[k], spl, all_vertices)
            # Score only on reasoning step vertices (not INPUT/OUTPUT anchors)
            step_verts = [v for v in seq if v != INPUT_VERTEX and v != OUTPUT_VERTEX]
            if step_verts:
                frac = sum(1 for v in step_verts if v in hull) / len(step_verts)
            else:
                frac = 0.0
            trial_scores.append(frac)

        scores.append(float(np.mean(trial_scores)))
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


def gBD_v2_uq() -> None:
    """Run Graph Band Depth v2 (anchored random walk) UQ over the ensemble."""
    from sentence_transformers import SentenceTransformer

    log = setup_log(args)

    input_path = os.path.join(args.output_path, "ensemble_v1.json")
    confidences_dir = os.path.join(args.output_path, "confidences")
    os.makedirs(confidences_dir, exist_ok=True)
    output_path = _conf_path(confidences_dir)

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Ensemble file not found: {input_path}")

    processed_ids = _load_processed_ids(output_path)
    if processed_ids:
        log.info(f"Resuming: skipping {len(processed_ids)} already-processed questions.")

    log.info(f"Loading encoder: {ENCODER_MODEL}")
    encoder = SentenceTransformer(ENCODER_MODEL)

    with open(input_path, encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    log.info(
        f"Computing gBD_v2 UQ for {len(lines)} questions "
        f"(anchored path BD: INPUT→walk→OUTPUT, trials={N_TRIALS}, "
        f"cluster_thr={SIM_THRESHOLD}, cross_thr={CROSS_SIM_THRESHOLD}, alpha={ALPHA})"
    )

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
                vertex_ids, centroids = _greedy_cluster(all_flat, SIM_THRESHOLD)

                offset, chain_vseqs = 0, []
                for emb in chains_emb:
                    n = len(emb)
                    chain_vseqs.append(list(vertex_ids[offset:offset + n]))
                    offset += n

                G = _build_graph(chain_vseqs)
                _add_similarity_edges(G, centroids, CROSS_SIM_THRESHOLD, SIM_THRESHOLD)
                # all_vertices excludes INPUT/OUTPUT anchors (scored separately)
                all_vertices = [v for v in G.nodes() if v not in (INPUT_VERTEX, OUTPUT_VERTEX)]

                if len(all_vertices) < 2:
                    confidence = 1.0
                else:
                    spl = _all_pairs_spl(G)
                    G_undirected = G.to_undirected()
                    if args.walk_length != "adaptive":
                        walk_length = int(args.walk_length)
                    else:
                        walk_length = max(2, int(round(
                            np.mean([len(seq) for seq in chain_vseqs if seq])
                        )))

                    bd = _path_band_depths(
                        chain_vseqs, spl, all_vertices,
                        G_undirected, N_TRIALS, walk_length, N_WALKS,
                    )
                    bd_score = float(np.mean(bd))

                    answers = [s.get("llm answer", "") for s in samples]
                    majority_count = Counter(answers).most_common(1)[0][1]
                    majority_frac = majority_count / len(samples)

                    confidence = ALPHA * bd_score + (1 - ALPHA) * majority_frac

            result = {
                "id": qid,
                "question": record.get("question", ""),
                "correct answer": record.get("correct answer", ""),
                "llm answer": _majority_answer(samples),
                "confidence": confidence,
            }
            f_out.write(json.dumps(result, ensure_ascii=False) + "\n")


def compute_auroc() -> None:
    """Compute AUROC and write to output/metric/<dataset>.json under 'gBD_v2'."""
    import torch
    from torchmetrics import AUROC

    labels_path = os.path.join(args.output_path, "output_v1_w_labels.json")
    conf_path = _conf_path(os.path.join(args.output_path, "confidences"))

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
    print(f"AUROC (gBD_v2, {args.dataset}): {auroc_value.item():.6f}  (n={len(confidences)})")

    if args.ablation:
        # Ablation: write to output/ablation/<ablation>.json, leaving output/metric untouched.
        result_path = os.path.join("output", "ablation", args.ablation + ".json")
        param_key = str(N_WALKS) if args.ablation == "subset_size" else str(args.walk_length)
        os.makedirs(os.path.dirname(result_path), exist_ok=True)
        results = {}
        if os.path.exists(result_path):
            with open(result_path, encoding="utf-8") as f:
                results = json.load(f)
        (results.setdefault(args.model_engine, {})
                .setdefault("gBD_v2", {})
                .setdefault(param_key, {}))[args.dataset] = round(auroc_value.item(), 6)
    else:
        result_path = os.path.join("output", "metric", args.dataset + ".json")
        os.makedirs(os.path.dirname(result_path), exist_ok=True)
        results = {}
        if os.path.exists(result_path):
            with open(result_path, encoding="utf-8") as f:
                results = json.load(f)
        results.setdefault(args.model_engine, {})["gBD_v2"] = round(auroc_value.item(), 6)

    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    print_exp(args)

    if args.model_engine in ["llama3-1_8B", "llama2-13b"]:
        gBD_v2_uq()
        compute_auroc()
    else:
        raise ValueError(f"Unsupported model engine: {args.model_engine}")
