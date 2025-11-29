import math
import random
from pathlib import Path
from typing import Dict, Set

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_CONNECTOME = REPO_ROOT / "patches" / "c302_data" / "herm_full_edgelist_MODIFIED.csv"
DEFAULT_METRICS    = REPO_ROOT / "metrics.csv"
DEFAULT_OUTDIR     = REPO_ROOT / "output" / "figures"

YAN_NEURONS_DEFAULT = {
    "AVAL", "AVAR", "AS08", "AS09", "AS10", "AS11",
    "DA07", "DA08", "DA09",
    "DB05", "DB06", "DB07",
    "DD04", "DD05", "DD06",
    "VA12", "VB11", "VD12", "VD13", "PDB",
}

# Static simplified sets
yan_set: Set[str]       = set(YAN_NEURONS_DEFAULT)
classes: Dict[str, str] = {n: "yan" for n in YAN_NEURONS_DEFAULT}

def tidy_axes(ax: plt.Axes) -> None:
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.yaxis.set_ticks_position("left")
    ax.xaxis.set_ticks_position("bottom")


random.seed(42)

connectome_path = DEFAULT_CONNECTOME
metrics_path    = DEFAULT_METRICS

# output directory
outdir = DEFAULT_OUTDIR
outdir.mkdir(parents=True, exist_ok=True)

# load connectome
g = nx.Graph()
with connectome_path.open() as handle:
    next(handle)  # skip header
    for line in handle:
        parts = [p.strip() for p in line.strip().split(",")]
        if len(parts) < 4:
            continue
        src, dst, weight, etype = parts
        try:
            w = float(weight)
        except ValueError:
            w = 1.0
        g.add_edge(src, dst, weight=w, type=etype)

# load metrics
metrics: Dict[str, Dict[str, float]] = {}
with metrics_path.open() as handle:
    header = handle.readline().strip().split(",")
    for line in handle:
        parts = line.strip().split(",")
        if len(parts) != len(header):
            continue
        row = dict(zip(header, parts))
        name = row.get("run_name", "")
        if name == "baseline":
            continue
        try:
            metrics[name] = {
                "curv_pct": abs(float(row.get("curvature_percent_change") or 0.0)),
                "vel_pct": abs(float(row.get("forward_velocity_percent_change") or 0.0)),
                "curv_delta": abs(float(row.get("curvature_delta") or 0.0)),
            }
        except ValueError:
            continue

# restrict graph to neurons present in metrics (ablated set)
metric_nodes = set(metrics.keys())
g = g.subgraph(metric_nodes).copy()

# degree, weight, and strength distributions
degs = [d for _, d in g.degree()]
if degs:
    # Degree distribution (from Connectomics-F25-W4L6.ipynb)
    kmin, kmax = min(degs), max(degs)
    bin_edges = np.logspace(math.log10(max(kmin, 1)), math.log10(max(kmax, 1)), num=12)
    density, _ = np.histogram(degs, bins=bin_edges, density=True)
    mid = 10 ** ((np.log10(bin_edges[1:]) + np.log10(bin_edges[:-1])) / 2)

    fig = plt.figure(figsize=(6, 4))
    ax = plt.gca()
    ax.loglog(mid, density, marker="o", linestyle="none", color="steelblue")
    ax.set_xlabel("degree k")
    ax.set_ylabel("P(k)")
    tidy_axes(ax)
    fig.tight_layout()
    fig.savefig(outdir / "figure1_degree_distribution.png", dpi=200)
    plt.close(fig)

    # Weight distribution (from Rich Club notebook)
    weights = [d.get("weight", 1.0) for _, _, d in g.edges(data=True)]
    if weights:
        wmin, wmax = min(weights), max(weights)
        bin_edges = np.logspace(
            math.log10(max(wmin, 1e-6)),
            math.log10(max(wmax, 1e-6)),
            num=20,
        )
        density, _ = np.histogram(weights, bins=bin_edges, density=True)
        mid = 10 ** ((np.log10(bin_edges[1:]) + np.log10(bin_edges[:-1])) / 2)

        fig = plt.figure(figsize=(6, 4))
        ax = plt.gca()
        ax.loglog(mid, density, marker="o", linestyle="none", color="darkorange")
        ax.set_xlabel("edge weight w")
        ax.set_ylabel("P(w)")
        tidy_axes(ax)
        fig.tight_layout()
        fig.savefig(outdir / "figure1b_weight_distribution.png", dpi=200)
        plt.close(fig)

    # Strength distribution
    strengths = [s for _, s in g.degree(weight="weight")]
    if strengths:
        smin, smax = min(strengths), max(strengths)
        bin_edges = np.logspace(
            math.log10(max(smin, 1e-6)),
            math.log10(max(smax, 1e-6)),
            num=20,
        )
        density, _ = np.histogram(strengths, bins=bin_edges, density=True)
        mid = 10 ** ((np.log10(bin_edges[1:]) + np.log10(bin_edges[:-1])) / 2)

        fig = plt.figure(figsize=(6, 4))
        ax = plt.gca()
        ax.loglog(mid, density, marker="o", linestyle="none", color="seagreen")
        ax.set_xlabel("node strength s")
        ax.set_ylabel("P(s)")
        tidy_axes(ax)
        fig.tight_layout()
        fig.savefig(outdir / "figure1c_strength_distribution.png", dpi=200)
        plt.close(fig)

# clustering and paths
samples = 20
seed = 42

rng = random.Random(seed)
# Use largest connected component for path length comparisons if the graph is disconnected
if nx.is_connected(g):
    g_for_paths = g
else:
    largest_cc_nodes = max(nx.connected_components(g), key=len)
    g_for_paths = g.subgraph(largest_cc_nodes).copy()

base_clust = nx.average_clustering(g_for_paths)
base_path  = nx.average_shortest_path_length(g_for_paths)

n = g_for_paths.number_of_nodes()
m = g_for_paths.number_of_edges()
p = m / (n * (n - 1) / 2)

er_clust = []
er_path = []
dp_clust = []
dp_path = []

for _ in range(samples):
    # ER null
    er = nx.erdos_renyi_graph(n, p, seed=rng.randint(0, 1_000_000))
    if nx.is_connected(er):
        er_clust.append(nx.average_clustering(er))
        er_path.append(nx.average_shortest_path_length(er))

    # Degree-preserving null (double-edge swap)
    dp = g_for_paths.copy()
    try:
        nx.double_edge_swap(
            dp,
            nswap=max(1, m),
            max_tries=m * 10,
            seed=rng.randint(0, 1_000_000),
        )
    except Exception:
        pass

    if nx.is_connected(dp):
        dp_clust.append(nx.average_clustering(dp))
        dp_path.append(nx.average_shortest_path_length(dp))

#box plots for clustering and path length ---

#clustering coefficient boxplot
fig = plt.figure(figsize=(6, 4))
ax = plt.gca()
clust_data = [er_clust, dp_clust, [base_clust]]
ax.boxplot(clust_data)

ax.scatter(
    [1, 2, 3],
    [np.mean(er_clust) if er_clust else np.nan,
     np.mean(dp_clust) if dp_clust else np.nan,
     base_clust],
    s=30,
    color="black",
    zorder=3,
)
ax.set_xticks([1, 2, 3])
ax.set_xticklabels(["ER", "DP", "Connectome"])
ax.set_ylabel("Average clustering coefficient")
tidy_axes(ax)
fig.tight_layout()
fig.savefig(outdir / "figure2_clustering_vs_nulls.png", dpi=200)
plt.close(fig)

#path length boxplot stuff also from the excersise 1 notebook
fig = plt.figure(figsize=(6, 4))
ax = plt.gca()
path_data = [er_path, dp_path, [base_path]]
ax.boxplot(path_data)
ax.scatter(
    [1, 2, 3],
    [np.mean(er_path) if er_path else np.nan,
     np.mean(dp_path) if dp_path else np.nan,
     base_path],
    s=30,
    color="black",
    zorder=3,
)
ax.set_xticks([1, 2, 3])
ax.set_xticklabels(["ER", "DP", "Connectome"])
ax.set_ylabel("Avg path length")
tidy_axes(ax)
fig.tight_layout()
fig.savefig(outdir / "figure3_pathlength_vs_nulls.png", dpi=200)
plt.close(fig)

# impact vs centrality
deg  = dict(g.degree())
betw = nx.betweenness_centrality(g, normalized=True)

xs = []
ys = []
for n_name, m_vals in metrics.items():
    vel = m_vals.get("vel_pct", 0.0)
    if vel > 0:
        xs.append(deg.get(n_name, 0))
        ys.append(vel)

fig = plt.figure()
ax = plt.gca()
ax.scatter(xs, ys, alpha=0.7, color="teal")
for n_name, m_vals in metrics.items():
    vel = m_vals.get("vel_pct", 0.0)
    if vel > 0:
        ax.text(deg.get(n_name, 0), vel, n_name, fontsize=6, color="#444444")
ax.set_xlabel("Degree")
ax.set_ylabel("|%Δforward velocity|")
tidy_axes(ax)
fig.tight_layout()
fig.savefig(outdir / "figure5_impact_vs_degree.png", dpi=200)
plt.close(fig)

xs = []
ys = []
for n_name, m_vals in metrics.items():
    vel = m_vals.get("vel_pct", 0.0)
    if vel > 0:
        xs.append(betw.get(n_name, 0.0))
        ys.append(vel)

fig = plt.figure()
ax = plt.gca()
ax.scatter(xs, ys, alpha=0.7, color="purple")
for n_name, m_vals in metrics.items():
    vel = m_vals.get("vel_pct", 0.0)
    if vel > 0:
        ax.text(betw.get(n_name, 0.0), vel, n_name, fontsize=6, color="#444444")
ax.set_xlabel("Betweenness centrality")
ax.set_ylabel("|%Δforward velocity|")
tidy_axes(ax)
fig.tight_layout()
fig.savefig(outdir / "figure6_impact_vs_betweenness.png", dpi=200)
plt.close(fig)

# ranked impacts
items = []
for name, m_vals in metrics.items():
    vel = m_vals.get("vel_pct", 0.0)
    if vel > 0:
        items.append((name, vel, classes.get(name, "unknown"), name in yan_set))

if items:
    items.sort(key=lambda x: x[1], reverse=True)

    palette = {
        "sensory": "tab:blue",
        "interneuron": "tab:orange",
        "motor": "tab:green",
        "unknown": "gray",
        "yan": "gray",
    }

    xs = list(range(len(items)))
    ys = [v[1] for v in items]
    colors = [palette.get(v[2], "gray") for v in items]

    fig = plt.figure(figsize=(10, 5))
    ax = plt.gca()
    ax.scatter(xs, ys, c=colors, alpha=0.8)

    yan_x = [i for i, v in enumerate(items) if v[3]]
    yan_y = [ys[i] for i in yan_x]
    ax.scatter(yan_x, yan_y, s=40, facecolors="none", edgecolors="red", linewidths=1.2)

    for i, (name, val, _, _) in enumerate(items):
        ax.text(xs[i], ys[i], name, fontsize=6, color="#444444", ha="center", va="bottom")

    ax.set_xlabel("Neurons (sorted)")
    ax.set_ylabel("|%Δforward velocity|")
    tidy_axes(ax)
    fig.tight_layout()
    fig.savefig(outdir / "figure7_rank_velocity.png", dpi=200)
    plt.close(fig)
