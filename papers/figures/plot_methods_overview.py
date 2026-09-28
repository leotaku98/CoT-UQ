# -*- coding: utf-8 -*-
"""Band-of-curves visualization of the three CoT band-depth estimators.

The figure adapts the classical "band depth" picture from functional data
analysis to Chain-of-Thought reasoning. Each curve is one CoT chain drawn in
embedding space; each circle on a curve is one reasoning step (node). A *band*
is the region swept between two randomly chosen references, and a chain is
*deep* when it lies inside many such bands.

Three side-by-side panels show the same idea at increasing structure:

    pBD : separate curves over a reasoning-step axis; the band is the region
          between two whole curves (order-sensitive).
    vBD : curves share nodes (semantically merged steps); each vertex is sized
          by how many chains visit it (more visits = deeper).
    gBD : curves share nodes; the band is the region between two random walks,
          and a chain threading the band is deep.

Output: papers/resources/methods_overview.{pdf,png}
"""

import os

import matplotlib
import numpy as np

matplotlib.use("Agg")

from matplotlib import pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

try:
    from scipy.interpolate import PchipInterpolator, splev, splprep
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False

# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------
# Five categorical colors (M = 5 chains, matching the paper's ensemble size).
CHAIN_COLORS = ["#2f6fb3", "#e08a1e", "#2a9d8f", "#8e6bbf", "#c05780"]
BAND_COLOR = "#7f9bb3"
# Reference random walks are light (de-emphasized); the scored chain is dark.
WALK_A_COLOR = "#8fc9d9"
WALK_B_COLOR = "#a9d7b6"
DEEP_COLOR = "#d1495b"
# Plain node fill shared by vBD and gBD (depth shown by size, not colour).
NODE_FILL = "#f2f2f2"

# Shared visual weights so all three panels look consistent.
NODE_SIZE = 110
EDGE_LW = 2.6
CHAIN_ALPHA = 0.6

# Shared-node embedding layout for the vBD / gBD panels. Node y-values are
# deliberately varied so the chains trace diverse shapes (arc, dip, flat,
# rising) while still merging at shared vertices (notably the hub ``h``).
NODES = {
    "Q": (0.10, 0.50),
    "v1a": (1.00, 0.82),
    "v1b": (1.00, 0.54),
    "v1c": (1.00, 0.24),
    "v2t": (2.00, 0.74),
    "h": (2.00, 0.48),
    "v2b": (2.00, 0.28),
    "v3a": (3.00, 0.68),
    "v3b": (3.00, 0.44),
    "v3c": (3.00, 0.20),
    "O": (3.95, 0.50),
}
CHAINS = {
    "chain_1": ["Q", "v1a", "v2t", "v3a", "O"],   # high arc
    "chain_2": ["Q", "v1c", "h", "v3c", "O"],     # low, up to hub, down
    "chain_3": ["Q", "v1b", "h", "v3b", "O"],     # near-flat mid
    "chain_4": ["Q", "v1a", "h", "v3a", "O"],     # high then dives to hub
    "chain_5": ["Q", "v1c", "v2b", "v3b", "O"],   # low then rising
}


# ---------------------------------------------------------------------------
# Smoothing helpers
# ---------------------------------------------------------------------------
def smooth_func(xs, ys, n=220):
    """Smooth an overshoot-free curve y(x) through the control points."""
    xf = np.linspace(min(xs), max(xs), n)
    if HAVE_SCIPY:
        yf = PchipInterpolator(np.asarray(xs), np.asarray(ys))(xf)
    else:
        yf = np.interp(xf, xs, ys)
    return xf, yf


def smooth_path(names, n=220):
    """Smooth a 2D curve through a sequence of shared-node positions."""
    pts = np.array([NODES[k] for k in names])
    if HAVE_SCIPY and len(pts) >= 3:
        tck, _ = splprep([pts[:, 0], pts[:, 1]], s=0, k=min(3, len(pts) - 1))
        uu = np.linspace(0, 1, n)
        xf, yf = splev(uu, tck)
        return np.asarray(xf), np.asarray(yf)
    # Linear densification fallback.
    dist = np.r_[0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
    uu = np.linspace(0, dist[-1], n)
    return np.interp(uu, dist, pts[:, 0]), np.interp(uu, dist, pts[:, 1])


# Per-chain splines and an edge->chain-segment map, so a random walk can be
# drawn along the *existing* chain edges rather than as new node-to-node splines.
CHAIN_SPLINES = {}
EDGE_TO_CHAIN = {}


def build_graph_curves():
    """Precompute each chain's spline and map every edge to a chain segment."""
    CHAIN_SPLINES.clear()
    EDGE_TO_CHAIN.clear()
    for name, seq in CHAINS.items():
        for i, (a, b) in enumerate(zip(seq[:-1], seq[1:])):
            EDGE_TO_CHAIN.setdefault((a, b), (name, i))
        if HAVE_SCIPY:
            pts = np.array([NODES[k] for k in seq])
            tck, u = splprep([pts[:, 0], pts[:, 1]], s=0, k=min(3, len(pts) - 1))
            CHAIN_SPLINES[name] = (tck, u)


def walk_curve(walk, per_seg=60):
    """Trace a walk by following the existing chain edge for each of its steps."""
    xs, ys = [], []
    for a, b in zip(walk[:-1], walk[1:]):
        name, i = EDGE_TO_CHAIN[(a, b)]
        if HAVE_SCIPY:
            tck, u = CHAIN_SPLINES[name]
            xf, yf = splev(np.linspace(u[i], u[i + 1], per_seg), tck)
        else:
            (x0, y0), (x1, y1) = NODES[a], NODES[b]
            xf, yf = np.linspace(x0, x1, per_seg), np.linspace(y0, y1, per_seg)
        xs.append(np.asarray(xf))
        ys.append(np.asarray(yf))
    return np.concatenate(xs), np.concatenate(ys)


# ---------------------------------------------------------------------------
# Shared 5-curve ensemble used by the pBD and vBD panels.
# ---------------------------------------------------------------------------
PBD_XS = [0, 1, 2, 3, 4]
PBD_CURVES = {
    "chain_1": [0.68, 0.80, 0.82, 0.80, 0.78],   # upper ensemble member
    "chain_2": [0.32, 0.22, 0.18, 0.24, 0.26],   # lower ensemble member
    "chain_3": [0.50, 0.51, 0.52, 0.51, 0.50],   # inside the band -> deep
    "chain_4": [0.56, 0.60, 0.54, 0.44, 0.36],   # band boundary
    "chain_5": [0.44, 0.42, 0.50, 0.58, 0.66],   # band boundary (single crossing)
}
BAND_PAIR = ("chain_4", "chain_5")


def draw_curve_ensemble(ax):
    """Draw the shared 5-curve ensemble with the band between BAND_PAIR."""
    _, y_a = smooth_func(PBD_XS, PBD_CURVES[BAND_PAIR[0]])
    xf, y_b = smooth_func(PBD_XS, PBD_CURVES[BAND_PAIR[1]])
    ax.fill_between(xf, np.minimum(y_a, y_b), np.maximum(y_a, y_b),
                    color=BAND_COLOR, alpha=0.30, zorder=1)
    for name, color in zip(PBD_CURVES, CHAIN_COLORS):
        xf, yf = smooth_func(PBD_XS, PBD_CURVES[name])
        ax.plot(xf, yf, color=color, lw=EDGE_LW, alpha=CHAIN_ALPHA, zorder=3)
        ax.scatter(PBD_XS, PBD_CURVES[name], s=NODE_SIZE, color=color,
                   edgecolors="white", linewidths=1.2, zorder=4)
        ax.text(4.12, PBD_CURVES[name][-1], name.replace("_", r"$_") + "$",
                color=color, fontsize=13.5, va="center", fontweight="bold")


def _curve_frame(ax, title, xmax=4.9):
    ax.set_title(title, fontsize=17.25, fontweight="bold")
    ax.set_xlim(-0.3, xmax)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("reasoning step (position)", fontsize=13.5)
    ax.set_ylabel("embedding coordinate", fontsize=13.5)
    ax.set_xticks(PBD_XS)
    ax.set_xticklabels(["Q", "$s_1$", "$s_2$", "$s_3$", "ans"], fontsize=13.5)
    ax.set_yticks([])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


# ---------------------------------------------------------------------------
# Panel 1: pBD -- band between two separate curves
# ---------------------------------------------------------------------------
def panel_pbd(ax):
    """pBD: separate curves over a step axis; band between two of them."""
    draw_curve_ensemble(ax)
    ax.annotate("band(chain$_4$,chain$_5$)", xy=(0.76, 0.53), xytext=(0.72, 0.95),
                fontsize=13.5, color="#3a4a58",
                arrowprops=dict(arrowstyle="-|>", color="#3a4a58", lw=1.4,
                                shrinkA=4, shrinkB=3))
    ax.annotate("chain$_3\\subset$ band\n$\\Rightarrow$ deep",
                xy=(3.4, 0.505), xytext=(2.3, 0.01), ha="center", va="bottom",
                fontsize=13.5, color=DEEP_COLOR, fontweight="bold",
                arrowprops=dict(arrowstyle="-|>", color=DEEP_COLOR, lw=1.6,
                                shrinkA=7, shrinkB=3))
    _curve_frame(ax, "pBD: band between two curves\n(separate nodes, order-sensitive)")


# ---------------------------------------------------------------------------
# Shared-node drawing for vBD / gBD panels
# ---------------------------------------------------------------------------
def visit_counts():
    """How many chains pass through each shared node (visit frequency)."""
    counts = {v: 0 for v in NODES}
    for seq in CHAINS.values():
        for v in seq:
            counts[v] += 1
    return counts


def draw_chain_edges(ax, alpha=CHAIN_ALPHA):
    """Draw each chain as a curve through the shared nodes at pBD line width."""
    for (_, seq), color in zip(CHAINS.items(), CHAIN_COLORS):
        xf, yf = smooth_path(seq)
        ax.plot(xf, yf, color=color, lw=EDGE_LW, alpha=alpha, zorder=2)


def draw_shared_nodes(ax, colors, size=NODE_SIZE):
    """Draw the shared node circles (pBD size) with per-node face colors."""
    for v, (x, y) in NODES.items():
        ax.scatter([x], [y], s=size, c=[colors[v]], edgecolors="#222222",
                   linewidths=1.2, zorder=5)


def panel_vbd(ax):
    """vBD: shared-node graph; vertex SIZE encodes how often it is visited
    (more chains through a node -> bigger -> deeper). Plain nodes, no colour.
    The trivially-shared input Q and answer O are not drawn."""
    counts = visit_counts()
    draw_chain_edges(ax)
    for v, (x, y) in NODES.items():
        if v in ("Q", "O"):
            continue
        ax.scatter([x], [y], s=NODE_SIZE * counts[v], c=NODE_FILL,
                   edgecolors="#222222", linewidths=1.3, zorder=5)
    ax.annotate("visited by more chains\n$\\Rightarrow$ bigger $\\Rightarrow$ deeper",
                xy=(2.0, 0.40), xytext=(2.0, 0.01), ha="center", va="bottom",
                fontsize=13.5, color=DEEP_COLOR, fontweight="bold",
                arrowprops=dict(arrowstyle="-|>", color=DEEP_COLOR, lw=1.6,
                                shrinkA=7, shrinkB=3,
                                connectionstyle="arc3,rad=-0.45"))
    _curve_frame(ax, "vBD: vertices sized by visit count\n(shared nodes, order-invariant)",
                 xmax=4.25)
    n_max = max(v for k, v in counts.items() if k not in ("Q", "O"))
    handles = [Line2D([0], [0], marker="o", linestyle="none",
                      markerfacecolor=NODE_FILL, markeredgecolor="#222222",
                      markeredgewidth=1.1, markersize=(NODE_SIZE * n) ** 0.5 * 0.8,
                      label=f"{n}")
               for n in range(1, n_max + 1)]
    leg = ax.legend(handles=handles, loc="upper right", fontsize=11,
                    frameon=True, ncol=n_max, columnspacing=0.6,
                    handletextpad=0.3, borderpad=0.5, framealpha=1.0,
                    handleheight=1.8, title="chains through node",
                    title_fontsize=10)
    leg.get_frame().set_edgecolor("#cccccc")


def panel_gbd(ax):
    """gBD: shared-node curves; band between two random walks; deep chain inside."""
    draw_chain_edges(ax)
    # Reference walks follow real graph edges but switch chains at the hub h,
    # so each combines segments from two different chains.
    walk_a = ["Q", "v1a", "h", "v3c", "O"]   # chain_4 then chain_2
    walk_b = ["Q", "v1c", "h", "v3a", "O"]   # chain_2 then chain_4
    xa, ya = walk_curve(walk_a)
    xb, yb = walk_curve(walk_b)
    # Band = region enclosed between the two walks (both run Q -> O).
    poly_x = np.r_[xa, xb[::-1]]
    poly_y = np.r_[ya, yb[::-1]]
    ax.fill(poly_x, poly_y, color=BAND_COLOR, alpha=0.26, zorder=1)
    ax.plot(xa, ya, color=WALK_A_COLOR, lw=EDGE_LW, zorder=3)
    ax.plot(xb, yb, color=WALK_B_COLOR, lw=EDGE_LW, zorder=3)
    # A real chain threading the band's centre -> deep.
    xt, yt = smooth_path(["Q", "v1b", "h", "v3b", "O"])
    ax.plot(xt, yt, color=DEEP_COLOR, lw=EDGE_LW + 0.4, zorder=4)

    plain = {v: NODE_FILL for v in NODES}
    draw_shared_nodes(ax, plain)
    ax.annotate("walk-band", xy=(1.3, 0.62), xytext=(0.55, 0.95),
                fontsize=13.5, color="#3a4a58",
                arrowprops=dict(arrowstyle="-|>", color="#3a4a58", lw=1.4,
                                shrinkA=4, shrinkB=3))
    ax.annotate("chain inside band\n$\\Rightarrow$ deep",
                xy=(2.45, 0.46), xytext=(2.0, 0.01), ha="center", va="bottom",
                fontsize=13.5, color=DEEP_COLOR, fontweight="bold",
                arrowprops=dict(arrowstyle="-|>", color=DEEP_COLOR, lw=1.6,
                                shrinkA=7, shrinkB=3))
    _curve_frame(ax, "gBD: band between two random walks\n(shared nodes, order-invariant)",
                 xmax=4.25)
    legend = [
        Line2D([0], [0], color=WALK_A_COLOR, lw=EDGE_LW, label="walk $W_a$"),
        Line2D([0], [0], color=WALK_B_COLOR, lw=EDGE_LW, label="walk $W_b$"),
        Line2D([0], [0], color=DEEP_COLOR, lw=EDGE_LW + 0.4, label="chain (deep)"),
    ]
    ax.legend(handles=legend, loc="upper right", fontsize=12, frameon=True)


def main():
    build_graph_curves()
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.8))
    panel_pbd(axes[0])
    panel_vbd(axes[1])
    panel_gbd(axes[2])
    fig.tight_layout(w_pad=2.5)

    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..",
                                           "resources"))
    pdf_path = os.path.join(out_dir, "methods_overview.pdf")
    png_path = os.path.join(out_dir, "methods_overview.png")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=170, bbox_inches="tight")
    print(f"scipy smoothing: {HAVE_SCIPY}")
    print(f"wrote {pdf_path}")
    print(f"wrote {png_path}")


if __name__ == "__main__":
    main()
