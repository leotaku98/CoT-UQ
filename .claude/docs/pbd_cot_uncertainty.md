# Path Band Depth (pBD) — Summary and CoT Extension

## 1. pBD on Graphs — Method Summary

Band depth orders a set of paths from "most central" to "most outlying" — the graph analogue of a median and boxplot for curves.

### 1.1 Band for vertices

Given a graph $G=(V,E)$ with geodesic distance $d_g$, the **band** formed by $j$ vertices is their **geodesic-convex hull** $H[\mathcal{V}_j]$: the smallest set of vertices closed under all shortest paths among them. For $j=2$ this is simply every vertex on the shortest path between the two endpoints.

**Vertex band depth** of $v$:
$$v\text{BD}(v) = \mathbb{E}_{\mathcal{V}_j \sim S_j}\bigl[\chi(v \in H[\mathcal{V}_j])\bigr]$$

where $\mathcal{V}_j$ is a set of $j$ vertices drawn independently from a distribution over $V$. At $j=2$ with uniform distribution this reduces to classical betweenness centrality.

### 1.2 Band for paths — alignment

Paths may have different lengths, so a **common index set** $\mathcal{I}$ must be established before comparing them. The alignment is computed via dynamic programming (Needleman-Wunsch), minimising the total geodesic distance between corresponding vertices:

$$\min_{\text{monotonic mappings}} \sum_{k \in \mathcal{I}} d_g\!\left(p_l(k),\, p_m(k)\right)$$

For $>2$ paths, the same DP generalises to a tensor of all pairwise distances (NP-hard in general; tractable for small $j$, e.g. $j \le 3$).

### 1.3 Path band depth

A path $p$ lies inside the band of $\mathcal{P}_j = \{p_1,\ldots,p_j\}$ iff **at every index** $l \in \mathcal{I}$:
$$p(l) \in H[\{p_1(l), \ldots, p_j(l)\}]$$

**Path band depth:**
$$p\text{BD}(p) = \mathbb{E}\bigl[\chi(p \in B[\mathcal{P}_j])\bigr]$$

Estimated as a sample mean over many random $j$-subsets drawn from the ensemble. The **median path** achieves the highest pBD; **outliers** are paths whose pBD falls below:
$$p\text{BD}(p_{\text{median}}) - \alpha \cdot (p\text{BD}(p_{\text{median}}) - p\text{BD}(p_{50\%}))$$
with $\alpha \approx 1.5$–$3.7$.

---

## 2. Extending pBD to CoT Ensembles

### 2.1 Mapping

| Graph pBD concept | CoT ensemble analogue |
|---|---|
| Path $p$ on graph | One CoT reasoning chain (sequence of steps) |
| Vertex on path | One reasoning step (text) |
| Geodesic distance $d_g$ | Semantic distance between step embeddings |
| Geodesic-convex hull | Convex hull in embedding space |
| Ensemble of paths | 10 sampled CoT chains for one question |
| pBD score | How "central/representative" a chain is |

### 2.2 Step representation

Parse each `llm response` into a step sequence using `parse_response_to_dict`:

```
CoT chain i  →  [step_1^i, step_2^i, ..., step_{n_i}^i]
```

Embed each step into $\mathbb{R}^d$ using a sentence encoder (e.g. `all-MiniLM-L6-v2` — small, fast). This gives:

```
e_k^i = embed(step_k^i)  ∈ R^d
```

### 2.3 Step alignment (DTW)

Since chains have different lengths, use **Dynamic Time Warping** (a monotonic DP) to align chain $p$ against each reference chain. DTW minimises:

$$\sum_{k \in \mathcal{I}} \| e^p(k) - e^{q}(\sigma(k)) \|_2$$

over all monotonic mappings $\sigma$, producing a correspondence between every step in $p$ and one or more steps in $q$.

### 2.4 Centrality score via ensemble centroid

The original pBD formulation samples random $j$-subsets of reference chains and checks band membership. For a **finite ensemble** of $M$ chains, the expectation over random pairs is better replaced by a direct computation: use the centroid of **all remaining $M-1$ chains** as the single reference.

For chain $p_i$ and all other chains $\{p_r\}_{r \neq i}$:

1. DTW-align $p_i$ against each $p_r$ → correspondence $\sigma_r$
2. At each step position $k$ in $p_i$, collect the matched embeddings from every reference:

$$\mathcal{R}_k = \left\{ \overline{e^{p_r}(\sigma_r(k))} \;\middle|\; r \neq i \right\}$$

where $\overline{e^{p_r}(\sigma_r(k))}$ is the mean of all embeddings in $p_r$ that DTW mapped to position $k$.

3. Compute the centroid and score via cosine similarity:

$$s_k = \cos\!\left(e^{p_i}(k),\; \frac{1}{M-1}\sum_{r \neq i} \overline{e^{p_r}(\sigma_r(k))}\right)$$

4. Chain score:

$$s_i = \frac{1}{n_i} \sum_{k=1}^{n_i} s_k$$

This is deterministic (no sampling), uses all available ensemble information, and is equivalent to a leave-one-out cosine similarity to the ensemble mean trajectory.

### 2.5 Uncertainty score

Given pBD scores $\{s_1, \ldots, s_{10}\}$ for the 10 chains, several uncertainty formulations are possible:

| Name | Formula | Interpretation |
|---|---|---|
| **Outlier fraction** | $\frac{1}{10}\sum_i \mathbf{1}[s_i < \text{threshold}]$ | Fraction of chains that are outliers |
| **Mean depth** | $1 - \bar{s}$ | Low average centrality → uncertain |
| **Depth spread** | $\text{Var}(s_1,\ldots,s_{10})$ | Fragmented ensemble → uncertain |
| **Median depth** | $1 - s_{\text{median}}$ | Even the best representative is not deep → uncertain |

**Recommended starting point:** mean depth $= 1 - \bar{s}$. This is differentiable, easy to compute, and directly reflects how scattered the reasoning chains are.

### 2.6 Combining with answer agreement

pBD captures *reasoning* diversity, independent of whether chains reach the same answer. A natural two-factor score:

$$\text{uncertainty} = \alpha \cdot (1 - \bar{s}) + (1-\alpha) \cdot \left(1 - \frac{\text{majority vote count}}{10}\right)$$

where $\alpha$ balances process-level (reasoning) and outcome-level (answer) uncertainty.

---

## 3. Implementation Sketch

See `methods/bd_ensemble/pBD.py` for the full implementation.

```python
def _pbd_score(emb_p: np.ndarray, others: list[np.ndarray]) -> float:
    """Centrality of chain emb_p against all other chains in the ensemble."""
    n = len(emb_p)
    # p_ref_steps[k] = reference embeddings matched to p[k] across all other chains
    p_ref_steps: list[list[np.ndarray]] = [[] for _ in range(n)]

    for ref in others:
        path = _dtw_path(emb_p, ref)          # DTW alignment
        ref_map: dict[int, list[int]] = {}
        for i_p, i_ref in path:
            ref_map.setdefault(i_p, []).append(i_ref)
        for i_p, i_refs in ref_map.items():
            p_ref_steps[i_p].append(ref[i_refs].mean(axis=0))   # avg if many-to-one

    memberships = []
    for k in range(n):
        centroid = np.mean(np.stack(p_ref_steps[k]), axis=0)    # ensemble centroid at k
        memberships.append(cos_sim(emb_p[k], centroid))          # score at position k

    return float(np.mean(memberships))   # chain score s_i


def pbd_confidence(samples: list[dict], encoder) -> float:
    chains_text = [parse_steps(s["llm response"]) for s in samples]
    chains_emb  = embed_chains(chains_text, encoder)             # L2-normalised

    depths = []
    for i, emb_p in enumerate(chains_emb):
        others = [chains_emb[k] for k in range(len(chains_emb)) if k != i]
        depths.append(_pbd_score(emb_p, others))

    return float(np.mean(depths))   # confidence = s̄;  uncertainty = 1 - s̄
```

---

## 4. Three Approaches

The pBD formulation in §2 uses DTW for step alignment, which is a **sequential** alignment: it treats the CoT chain as an ordered path and penalises any step that appears out of position relative to reference chains. An order-invariant extension is proposed in §4.2, which preserves the band depth framework while removing the ordering constraint. Which approach performs best on which dataset is an empirical question.

### 4.1 pBD — Path Band Depth (DTW alignment)

**Core idea:** treat each CoT chain as an *ordered path*. Two chains are similar only if they visit the same vertices *in the same order*. This is the full pBD formulation from §2–3.

#### Permutation sensitivity

At each aligned index $k$, the band membership check is:

$$p(k) \in H[\{p_1(k), \ldots, p_j(k)\}]$$

Chains that swap steps (e.g. `B↔C`) will land outside the hull of positionally-matching vertices from other chains.

| Scenario | DTW+pBD |
|---|---|
| `A→B→C→D` vs `A→C→B→D`, same answer | may flag as different |
| `A→B→C→D` vs `A→B→E→D`, same answer | flags divergence at step 3 |
| `A→B→C→D` vs `A→B→C→F`, diff answer | flags divergence at step 4 |

#### Implementation

See §3 and `methods/bd_ensemble/pBD.py`. For a finite ensemble the centroid of all remaining chains replaces random pair sampling — see §2.4.

---

### 4.2 vBD — Vertex Band Depth on Reasoning Graphs (Order-Invariant)

**Core idea:** construct a *reasoning graph* from all sampled chains, compute a band depth score for each vertex (reasoning state) in the graph, then define a chain's depth as the mean depth of the vertices it visits. No alignment is needed — the representation depends only on *which* states are visited, not their order.

#### Reasoning graph construction

Given $M$ sampled CoT chains for one question, embed each step with a sentence encoder. Cluster semantically equivalent steps into shared vertices (cosine similarity > threshold → same vertex). Let $G=(V,E)$ be the resulting reasoning graph. Two vertices are connected by a directed edge if they appear consecutively in at least one chain.

Each chain is then represented as a subset of vertices:

$$P_i \subseteq V$$

Two chains that visit the same steps in different orders produce nearly identical subsets, so permutations are absorbed by the set representation.

#### Vertex band depth

For a random $j$-subset of vertices $\mathcal{V}_j = \{v_1,\ldots,v_j\}$ drawn from $V$, the geodesic-convex hull $H[\mathcal{V}_j]$ is the smallest vertex set closed under all shortest paths in $G$. The band depth of vertex $v$ is:

$$v\text{BD}(v) = \mathbb{E}_{\mathcal{V}_j}\bigl[\chi(v \in H[\mathcal{V}_j])\bigr]$$

Vertices that frequently lie inside geodesic-convex hulls are central reasoning states shared across many chains.

#### Chain depth

The depth of chain $P_i$ is the mean vertex depth over all reasoning states it visits:

$$c\text{BD}(P_i) = \frac{1}{|P_i|} \sum_{v \in P_i} v\text{BD}(v)$$

Chains traversing central regions of the reasoning graph receive high depth scores; chains containing rare or unusual steps receive lower scores.

#### Uncertainty score

$$U_\text{vBD} = 1 - \frac{1}{M} \sum_{i=1}^{M} c\text{BD}(P_i)$$

#### Order invariance

Chains $A\to B\to C\to D$ and $A\to C\to B\to D$ both yield vertex set $\{A,B,C,D\}$ and therefore the same chain depth. The method measures *which* reasoning states are visited, not *the order* in which they are visited.

#### Implementation sketch

```python
import networkx as nx

def vbd_uncertainty(samples: list[dict], j: int = 2, n_trials: int = 100,
                    sim_threshold: float = 0.85) -> float:
    encoder = SentenceTransformer("all-MiniLM-L6-v2")
    chains_text = [parse_steps(s["llm response"]) for s in samples]
    all_embs = [encoder.encode(steps, normalize_embeddings=True) for steps in chains_text]

    # Cluster all steps into shared vertices
    all_flat = np.vstack(all_embs)
    vertex_ids = _greedy_cluster(cosine_similarity(all_flat), sim_threshold)
    offset, chain_verts = 0, []
    for emb in all_embs:
        n = len(emb)
        chain_verts.append(list(vertex_ids[offset:offset + n]))
        offset += n

    # Build reasoning graph
    G = nx.DiGraph()
    for chain in chain_verts:
        for u, v in zip(chain[:-1], chain[1:]):
            G.add_edge(u, v)

    all_vertices = list(G.nodes())
    chain_sets = [set(c) for c in chain_verts]

    # Compute vertex band depth for each vertex
    vbd = {}
    for v in all_vertices:
        hits = []
        for _ in range(n_trials):
            sample_verts = random.sample(all_vertices, j)
            hull = _geodesic_hull(G, set(sample_verts))
            hits.append(float(v in hull))
        vbd[v] = float(np.mean(hits))

    # Chain depth = mean vertex depth over visited vertices
    chain_depths = [
        float(np.mean([vbd[v] for v in vset]) if vset else 0.0)
        for vset in chain_sets
    ]
    return float(1.0 - np.mean(chain_depths))
```

---

### 4.3 Summary

| | pBD (`methods/bd_ensemble/pBD.py`) | vBD (`methods/bd_ensemble/vBD.py`) |
|---|---|---|
| Alignment | Sequential DTW (monotonic) | None |
| Depth computed at | Chain level (band of chains) | Vertex level, averaged over chain |
| Order-sensitive | Yes | No |
| Graph construction required | No | Yes (per question) |
| Similarity threshold required | No | Yes (for clustering) |
| Sensitive to step repetition | Yes | No (set, not multiset) |
| gsm8k AUROC (llama3-1_8B) | 0.6285 | 0.6160 |

Which approach yields the best AUROC is an empirical question — no assumptions are made here.

---

## 5. Relation to Existing UQ Methods in This Repo

The five `self-probing-*` variants in `stepuq.py` all ask the model to *self-assess* its confidence. pBD-based uncertainty is **model-free** — it derives uncertainty purely from the geometric structure of the ensemble and requires no additional LLM calls.

| Property | self-probing | pBD ensemble |
|---|---|---|
| Extra LLM calls | Yes (1 per question) | No |
| Captures reasoning diversity | No (single chain) | Yes |
| Sensitive to answer diversity | No | Optionally (§2.6) |
| Requires embeddings | No | Yes (small encoder) |
