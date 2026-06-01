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

Since chains have different lengths, use **Dynamic Time Warping** (a monotonic DP) to produce a common index set $\mathcal{I}$ across all 10 chains. DTW minimises:

$$\sum_{k \in \mathcal{I}} \| e^{p_l}(k) - e^{p_m}(k) \|_2$$

For the multi-path case ($j > 2$), align all chains jointly or use a star-alignment against the chain with median length.

### 2.4 Band membership in embedding space

At each aligned index $k$, the "band" of $j$ chains is the **convex hull** of their embeddings:

$$B_k = \text{conv}\!\left(\{e^{p_1}(k), \ldots, e^{p_j}(k)\}\right) \subset \mathbb{R}^d$$

A chain $p$ is inside the band at index $k$ iff $e^p(k) \in B_k$.

For $j=2$: this reduces to checking whether $e^p(k)$ lies on the line segment between the two reference embeddings, i.e. $e^p(k) = \lambda e^{p_1}(k) + (1-\lambda) e^{p_2}(k)$ for some $\lambda \in [0,1]$. In high-dimensional space this almost never holds exactly, so use a **softened** version:

$$\chi_\epsilon\!\left(p \in B_k\right) = \mathbf{1}\!\left[d\!\left(e^p(k),\, B_k\right) \le \epsilon\right]$$

where $d(\cdot, B_k)$ is the distance from the point to the convex hull.

Alternatively, use the **proportion of the band** that $p$ falls inside averaged over indices — the modified band depth of López-Pintado & Romo (2009):

$$p\text{BD}_{\text{mod}}(p) = \mathbb{E}_{\mathcal{P}_j}\!\left[\frac{1}{|\mathcal{I}|}\sum_{k \in \mathcal{I}} \chi\!\left(e^p(k) \in B_k\right)\right]$$

This resolves ties and is more robust to the "strict for-all" condition.

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

```python
# For one question with 10 CoT samples
from sentence_transformers import SentenceTransformer
from dtaidistance import dtw_ndim
from scipy.spatial import ConvexHull
import numpy as np

def parse_steps(llm_response: str) -> list[str]:
    """Reuse parse_response_to_dict, return list of step texts."""
    ...

def pbd_uncertainty(samples: list[dict], j: int = 2) -> float:
    encoder = SentenceTransformer("all-MiniLM-L6-v2")
    chains = [parse_steps(s["llm response"]) for s in samples]   # list of step lists
    embedded = [encoder.encode(steps) for steps in chains]        # list of (n_i, d) arrays

    # Align all chains to common index set via DTW against reference
    ref = embedded[len(embedded) // 2]
    aligned = [dtw_ndim.warping_path(ref, e) for e in embedded]

    # Compute pBD for each chain via sample mean over random j-subsets
    depths = []
    for i, chain in enumerate(aligned):
        others = [embedded[k] for k in range(len(embedded)) if k != i]
        scores = []
        for _ in range(100):           # 100 random j-subsets
            subset = random.sample(others, j)
            # check membership at each aligned index ...
            scores.append(_band_membership(chain, subset))
        depths.append(np.mean(scores))

    return float(1.0 - np.mean(depths))   # uncertainty: lower depth → higher uncertainty
```

---

## 4. Approaches by Task Type

The pBD formulation in §2 uses DTW for step alignment, which is a **sequential** alignment: it treats the CoT chain as an ordered path and penalises any step that appears out of position relative to reference chains. This works well when reasoning order is meaningful (multi-hop QA), but breaks for math datasets where two valid chains can visit the same steps in different orders and should be considered identical reasoning.

For math datasets, two order-invariant extensions of pBD are proposed in §4.2 and §4.3, which preserve the band depth framework while removing the ordering assumption.

### 4.1 Approach A — DTW + pBD (order-sensitive, for QA)

**Core idea:** treat each CoT chain as an *ordered path* on the per-question graph. Two chains are similar only if they visit the same vertices *in the same order*. This is the full pBD formulation from §2–3.

Use this when step order encodes causal structure — e.g. in hotpotQA you must retrieve supporting fact A before you can use it to look up fact B. Reversing the order represents genuinely different (and possibly wrong) reasoning.

#### Permutation sensitivity

At each aligned index $k$, the band membership check is:

$$p(k) \in H[\{p_1(k), \ldots, p_j(k)\}]$$

Chains that swap independent steps (e.g. `B↔C`) will land outside the hull of positionally-matching vertices from other chains — correctly flagged as structurally different for QA tasks where that swap is meaningful, but incorrectly penalised for math tasks where order does not matter.

| Scenario | DTW+pBD |
|---|---|
| `A→B→C→D` vs `A→C→B→D`, same answer | may flag as different ✗ |
| `A→B→C→D` vs `A→B→E→D`, same answer | flags divergence at step 3 ✓ |
| `A→B→C→D` vs `A→B→C→F`, diff answer | flags divergence at step 4 ✓ |
| Multi-hop: wrong retrieval order | correctly flags ✓ |

#### Implementation sketch

```python
def pbd_uncertainty(samples: list[dict], j: int = 2, n_trials: int = 100) -> float:
    encoder = SentenceTransformer("all-MiniLM-L6-v2")
    chains = [parse_steps(s["llm response"]) for s in samples]
    embedded = [encoder.encode(steps) for steps in chains]  # list of (n_i, d)

    depths = []
    for i in range(len(embedded)):
        others = [embedded[k] for k in range(len(embedded)) if k != i]
        trial_scores = []
        for _ in range(n_trials):
            subset = random.sample(others, j)
            # DTW-align chain i against subset reference; check convex hull membership
            trial_scores.append(_pbd_trial(embedded[i], subset))
        depths.append(float(np.mean(trial_scores)))

    return float(1.0 - np.mean(depths))
```

---

### 4.2 Approach B — Optimal-Transport Alignment + pBD (order-invariant pBD)

**Core idea:** keep the full pBD machinery — convex hull membership check at each matched index — but replace the sequential DTW alignment with an **optimal bipartite matching** (Hungarian algorithm). The matching is free to pair any step of $p$ with any step of the reference chain, finding the globally minimum-cost correspondence without the monotonicity constraint.

#### Why this fixes permutations

For equal-length chains the Hungarian algorithm solves:

$$\sigma^* = \arg\min_{\sigma \in S_n} \sum_{k=1}^{n} d\!\left(e^p(k),\, e^q(\sigma(k))\right)$$

For $p = [A,B,C,D]$ and $q = [A,C,B,D]$:
- DTW is forced to match $B \leftrightarrow C$ and $C \leftrightarrow B$ (positional, high cost)
- Hungarian freely matches $A\leftrightarrow A$, $B\leftrightarrow B$, $C\leftrightarrow C$, $D\leftrightarrow D$ (cost 0)

After the optimal matching, the pBD hull-membership check at each aligned index proceeds exactly as in §2, so the full pBD formula is unchanged.

#### Handling unequal chain lengths

Use **Wasserstein optimal transport** (unbalanced or with dummy vertices) instead of the Hungarian algorithm. Treat each chain as a discrete uniform distribution over its step embeddings; the OT plan gives fractional correspondences that minimise total transport cost across all steps.

$$\pi^* = \arg\min_{\pi \in \Pi(\mu_p, \mu_q)} \int d(x, y)\, d\pi(x, y)$$

where $\mu_p, \mu_q$ are the empirical step distributions of chains $p$ and $q$. The support of $\pi^*$ defines the alignment.

#### pBD formula (unchanged, new alignment only)

$$p\text{BD}(p) = \mathbb{E}_{\mathcal{P}_j}\!\left[\frac{1}{n}\sum_{k=1}^{n} \chi\!\left(e^p(k) \in \text{conv}\!\left(\{e^{p_1}(\sigma_1^*(k)),\ldots,e^{p_j}(\sigma_j^*(k))\}\right)\right)\right]$$

where $\sigma_i^*$ is the optimal assignment matching $p$ to reference chain $p_i$.

#### Implementation sketch

```python
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

def _ot_align(emb_p: np.ndarray, emb_q: np.ndarray) -> np.ndarray:
    """Return index array sigma such that emb_q[sigma[k]] is matched to emb_p[k]."""
    n, m = len(emb_p), len(emb_q)
    cost = cdist(emb_p, emb_q, metric="cosine")       # (n, m) cost matrix
    if n == m:
        row_ind, col_ind = linear_sum_assignment(cost)
        sigma = col_ind                                 # sigma[k] = matched index in q
    else:
        # pad shorter chain with dummy vertices (cost = max_penalty)
        pad = abs(n - m)
        penalty = cost.max() * 2
        if n < m:
            cost = np.vstack([cost, np.full((pad, m), penalty)])
        else:
            cost = np.hstack([cost, np.full((n, pad), penalty)])
        _, col_ind = linear_sum_assignment(cost)
        sigma = col_ind[:n]
    return sigma

def _in_convex_hull(point: np.ndarray, vertices: np.ndarray) -> bool:
    """Check if point is inside the convex hull of vertices (j <= 3 for efficiency)."""
    if len(vertices) == 1:
        return np.allclose(point, vertices[0], atol=1e-6)
    if len(vertices) == 2:
        # point on segment: point = a*v0 + (1-a)*v1 for a in [0,1]
        v = vertices[1] - vertices[0]
        t = np.dot(point - vertices[0], v) / (np.dot(v, v) + 1e-12)
        proj = vertices[0] + np.clip(t, 0, 1) * v
        return np.linalg.norm(point - proj) < 1e-4
    from scipy.spatial import ConvexHull, Delaunay
    try:
        return Delaunay(vertices).find_simplex(point) >= 0
    except Exception:
        return False

def pbd_ot_uncertainty(samples: list[dict], j: int = 2, n_trials: int = 100) -> float:
    encoder = SentenceTransformer("all-MiniLM-L6-v2")
    chains = [parse_steps(s["llm response"]) for s in samples]
    embedded = [encoder.encode(steps) for steps in chains]

    depths = []
    for i, emb_p in enumerate(embedded):
        others = [embedded[k] for k in range(len(embedded)) if k != i]
        trial_scores = []
        for _ in range(n_trials):
            refs = random.sample(others, j)
            # Align p to each reference via OT
            sigmas = [_ot_align(emb_p, ref) for ref in refs]
            # At each step k of p, check hull of matched reference steps
            memberships = []
            for k in range(len(emb_p)):
                ref_steps = np.stack([refs[r][sigmas[r][k]] for r in range(j)])
                memberships.append(float(_in_convex_hull(emb_p[k], ref_steps)))
            trial_scores.append(float(np.mean(memberships)))
        depths.append(float(np.mean(trial_scores)))

    return float(1.0 - np.mean(depths))
```

#### Complexity vs DTW

| | DTW | Hungarian | Wasserstein OT |
|---|---|---|---|
| Equal lengths | $O(n^2)$ | $O(n^3)$ | $O(n^3)$ |
| Unequal lengths | $O(n \cdot m)$ | $O(\max(n,m)^3)$ | $O(\max(n,m)^3)$ |
| Permutation-invariant | No | Yes | Yes |

For CoT chains of length 5–10, the $O(n^3)$ cost is negligible (~microseconds per pair).

---

### 4.3 Approach C — Set-Level Geodesic Hull Band Depth (order-invariant pBD on graph)

**Core idea:** redefine band membership at the *set* level using the graph's geodesic hull, eliminating the alignment step entirely. This stays fully within the graph-based pBD framework from the original paper.

#### Redefined band membership

Chain $p$ is inside the band of $\mathcal{P}_j = \{p_1, \ldots, p_j\}$ iff its entire vertex set is contained in the geodesic-convex hull of the union of reference vertex sets:

$$p \in B[\mathcal{P}_j] \quad \text{iff} \quad V_p \subseteq H\!\left[\bigcup_{i=1}^{j} V_{p_i}\right]$$

where $H[\cdot]$ is the geodesic-convex hull on the per-question graph (smallest geodesic-closed set containing its argument).

#### Set band depth formula

$$p\text{sBD}(p) = \mathbb{E}_{\mathcal{P}_j \sim S_j}\!\left[\chi\!\left(V_p \subseteq H\!\left[\bigcup_{i=1}^{j} V_{p_i}\right]\right)\right]$$

#### Why permutations are handled

$V_{A\to B\to C\to D} = \{A,B,C,D\} = V_{A\to C\to B\to D}$. Both chains have identical vertex sets, so they always appear inside each other's band — identical psBD score regardless of step order.

#### What makes a chain an outlier

Chain $p$ is an outlier if it visits vertices that lie *outside the geodesic closure* of any $j$-subset of other chains. Concretely: it uses a reasoning step that is not reachable via shortest paths between any pair of steps that appear in the reference chains. This means the reasoning is genuinely structurally different, not just reordered.

#### Implementation sketch

```python
import networkx as nx

def _build_graph(vertex_ids: list[list[int]]) -> nx.DiGraph:
    """Build step-transition graph from per-chain vertex id sequences."""
    G = nx.DiGraph()
    for chain in vertex_ids:
        for u, v in zip(chain[:-1], chain[1:]):
            G.add_edge(u, v)
    return G

def _geodesic_hull(G: nx.Graph, vertex_set: set) -> set:
    """Geodesic-convex hull: add all vertices on shortest paths between any pair."""
    hull = set(vertex_set)
    pairs = [(u, v) for u in vertex_set for v in vertex_set if u != v]
    for u, v in pairs:
        try:
            for path in nx.all_shortest_paths(G.to_undirected(), u, v):
                hull.update(path)
        except nx.NetworkXNoPath:
            continue
    return hull

def psbd_uncertainty(samples: list[dict], j: int = 2, n_trials: int = 100,
                     sim_threshold: float = 0.85) -> float:
    encoder = SentenceTransformer("all-MiniLM-L6-v2")
    chains = [parse_steps(s["llm response"]) for s in samples]
    all_embs = [encoder.encode(steps) for steps in chains]

    # Cluster steps into vertices
    all_flat = np.vstack(all_embs)
    vertex_ids = _greedy_cluster(cosine_similarity(all_flat), sim_threshold)
    offset, chain_verts = 0, []
    for emb in all_embs:
        n = len(emb)
        chain_verts.append(list(vertex_ids[offset:offset + n]))
        offset += n

    G = _build_graph(chain_verts)
    chain_sets = [set(v) for v in chain_verts]

    depths = []
    for i, V_p in enumerate(chain_sets):
        others = [chain_sets[k] for k in range(len(chain_sets)) if k != i]
        trial_scores = []
        for _ in range(n_trials):
            subset = random.sample(others, j)
            union = set().union(*subset)
            hull = _geodesic_hull(G, union)
            trial_scores.append(float(V_p.issubset(hull)))
        depths.append(float(np.mean(trial_scores)))

    return float(1.0 - np.mean(depths))
```

#### Trade-off vs Approach B

| | Approach B (OT alignment) | Approach C (set hull) |
|---|---|---|
| Alignment needed | Yes (Hungarian/OT) | No |
| Checks step-level structure | Yes (per matched index) | No (set containment only) |
| Sensitive to which steps are visited | Yes | Yes |
| Sensitive to how many times a step is visited | Yes | No (set, not multiset) |
| Graph construction required | No (works in embedding space) | Yes |
| Hull computation | Convex hull in $\mathbb{R}^d$ | Geodesic closure on graph |
| Order-invariant | Yes | Yes |

Approach B is finer-grained: it still checks whether each step is "geometrically between" the matched reference steps, which can catch subtle divergences even among chains with the same vertex set. Approach C is coarser but simpler and more directly motivated by the original paper's graph formalism.

---

### 4.4 Summary: which to use

| Dataset | Recommended | Reason |
|---|---|---|
| `gsm8k`, `svamp`, `ASDiv` | **B** (OT+pBD) or **C** (set hull) | Order-invariant pBD; math steps are often commutative |
| `hotpotQA`, `2WikimhQA` | **A** (DTW+pBD) | Causal step order encodes retrieval dependencies |

For an initial experiment, run Approach B on math datasets and compare AUROC against the existing `self-probing-*` variants.

---

## 5. Relation to Existing UQ Methods in This Repo

The five `self-probing-*` variants in `stepuq.py` all ask the model to *self-assess* its confidence. pBD-based uncertainty is **model-free** — it derives uncertainty purely from the geometric structure of the ensemble and requires no additional LLM calls.

| Property | self-probing | pBD ensemble |
|---|---|---|
| Extra LLM calls | Yes (1 per question) | No |
| Captures reasoning diversity | No (single chain) | Yes |
| Sensitive to answer diversity | No | Optionally (§2.6) |
| Requires embeddings | No | Yes (small encoder) |
