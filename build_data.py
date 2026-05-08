"""Preprocess the SNAP Facebook ego network into a single data.json the
front-end can render directly.

Outputs for the page:
  - whole-network stats (people, friendships, avg/max friends)
  - top-10 most-connected people
  - top-5 bridges (highest approximate betweenness centrality)
  - communities found by Louvain on the full network
  - a 2-D layout position for every node, pre-computed so the page doesn't
    have to run a force simulation on cold load
"""
import gzip
import json
import time
from collections import Counter

import networkx as nx
import numpy as np


def t(label, start):
    print(f"  {label}: {time.time() - start:.1f}s")


print("Loading edges …")
t0 = time.time()
G = nx.Graph()
with gzip.open("facebook_combined.txt.gz", "rt") as f:
    for line in f:
        a, b = line.split()
        G.add_edge(int(a), int(b))
t("loaded", t0)
n_nodes, n_edges = G.number_of_nodes(), G.number_of_edges()
print(f"  {n_nodes} people, {n_edges} friendships")

degree = dict(G.degree())
avg_deg = (2 * n_edges) / n_nodes

print("Detecting friend groups (Louvain) …")
t0 = time.time()
comms = nx.community.louvain_communities(G, seed=42, resolution=1.0)
# Sort by size descending; cap at top 8 + 'other' for legend readability.
comms.sort(key=len, reverse=True)
TOP_C = 8
node_community = {}
for i, c in enumerate(comms):
    label = i if i < TOP_C else TOP_C
    for n in c:
        node_community[n] = label
community_sizes = Counter(node_community.values())
n_groups_shown = min(TOP_C, len(comms))
n_groups_other = max(0, len(comms) - TOP_C)
t("louvain", t0)
print(f"  {len(comms)} groups; showing {n_groups_shown}, rolling up {n_groups_other} into 'other'")

print("Computing bridges (approximate betweenness, k=500) …")
t0 = time.time()
bc = nx.betweenness_centrality(G, k=500, seed=42, normalized=True)
t("betweenness", t0)

print("Computing layout (spring, 50 iterations) …")
t0 = time.time()
# Seed positions from a 2-D spectral layout to speed convergence and give
# communities a natural separation; then refine with spring layout.
init_pos = nx.spring_layout(G, iterations=50, seed=42, k=None, threshold=1e-3)
t("spring", t0)

# Normalise positions into a comfortable canvas range.
xy = np.array([init_pos[n] for n in G.nodes()])
xy -= xy.mean(axis=0)
scale = np.percentile(np.linalg.norm(xy, axis=1), 99) or 1.0
xy /= scale  # ~99% of nodes within unit radius
positions = {n: (float(x), float(y)) for n, (x, y) in zip(G.nodes(), xy)}

top10 = sorted(degree.items(), key=lambda kv: -kv[1])[:10]
top_bridges = sorted(bc.items(), key=lambda kv: -kv[1])[:5]

nodes = []
for n in G.nodes():
    x, y = positions[n]
    nodes.append({
        "id": n,
        "x": round(x, 4),
        "y": round(y, 4),
        "deg": degree[n],
        "c": node_community[n],
    })

# Edges as compact [a, b] pairs with stable node ordering.
node_index = {n: i for i, n in enumerate(G.nodes())}
edges = [[node_index[a], node_index[b]] for a, b in G.edges()]

data = {
    "stats": {
        "people": n_nodes,
        "friendships": n_edges,
        "avgFriends": round(avg_deg, 1),
        "maxFriends": max(degree.values()),
        "groupsFound": len(comms),
        "groupsShown": n_groups_shown,
    },
    "communities": [
        {"id": i, "size": community_sizes.get(i, 0)}
        for i in range(n_groups_shown)
    ] + ([{"id": TOP_C, "size": community_sizes.get(TOP_C, 0)}] if n_groups_other else []),
    "topFriends": [{"id": n, "deg": d} for n, d in top10],
    "topBridges": [{"id": n, "score": round(s, 4), "deg": degree[n]} for n, s in top_bridges],
    "nodes": nodes,
    "edges": edges,
}

with open("data.json", "w") as f:
    json.dump(data, f, separators=(",", ":"))

import os
size_kb = os.path.getsize("data.json") / 1024
print(f"\ndata.json written: {size_kb:.0f} KB")
print(f"  nodes={len(nodes)} edges={len(edges)}")
print(f"  top friends: {top10[0]} … {top10[-1]}")
print(f"  top bridges: {top_bridges[0]} … {top_bridges[-1]}")
