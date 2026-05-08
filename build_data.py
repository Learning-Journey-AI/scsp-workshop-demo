"""Preprocess the SNAP Facebook ego network into a single data.json the
front-end can render directly.

Outputs for the page:
  - whole-network stats (people, friendships, avg/max friends)
  - top-10 most-connected people, with the 10 ego users marked
  - top-5 bridges (highest approximate betweenness centrality)
  - communities found by Louvain on the full network, plus a comparison
    against the 193 hand-labeled ground-truth circles from the paper
  - feature-category browser for ego 107's network (most-connected ego):
    which alters share which kind of profile feature
  - a 2-D layout position for every node, pre-computed so the page doesn't
    have to run a force simulation on cold load
"""
import glob
import gzip
import json
import os
import re
import time
from collections import Counter, defaultdict

import networkx as nx
import numpy as np

# The 10 ego users from the McAuley & Leskovec paper. Each one installed a
# Facebook research app and shared the friendship structure of their friends.
EGO_IDS = [0, 107, 348, 414, 686, 698, 1684, 1912, 3437, 3980]
EGO_LABELS = {eid: f"Ego {chr(65 + i)}" for i, eid in enumerate(EGO_IDS)}

EGONETS_DIR = "facebook"
EGONETS_TAR = "facebook.tar.gz"
FEATURE_BROWSER_EGO = 107  # most-connected ego, used for the feature-browser section


def ensure_egonets():
    """Make sure the per-ego files are extracted. The tarball is small enough
    to keep in the repo; the extracted tree is too noisy, so we extract on
    demand.
    """
    if os.path.isdir(EGONETS_DIR) and os.path.exists(os.path.join(EGONETS_DIR, "0.circles")):
        return
    if not os.path.exists(EGONETS_TAR):
        print(f"  warning: {EGONETS_TAR} missing — sections B/C will be empty")
        return
    import tarfile
    with tarfile.open(EGONETS_TAR, "r:gz") as tf:
        tf.extractall(".")


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


def connected_groups(node, max_groups=3, min_share=0.05):
    """Which friend groups does this person have meaningful ties to?

    Counts a group only if at least `min_share` of the person's friendships
    go to it (filters out a single stray contact). Returns up to
    `max_groups` group ids ordered by share, descending.
    """
    group_count = Counter(node_community[nb] for nb in G.neighbors(node))
    total = sum(group_count.values()) or 1
    ranked = [
        (g, c / total) for g, c in group_count.most_common()
        if c / total >= min_share
    ]
    return [g for g, _ in ranked[:max_groups]]


# Pick the top-5 bridges by betweenness that actually span 2+ friend groups —
# otherwise the list would include high-betweenness hubs whose friendships
# stay inside a single group, which doesn't match the "bridge" label.
top_bridges = []
for n, s in sorted(bc.items(), key=lambda kv: -kv[1]):
    cg = connected_groups(n)
    if len(cg) >= 2:
        top_bridges.append((n, s, cg))
        if len(top_bridges) == 5:
            break


ego_set = set(EGO_IDS)

nodes = []
for n in G.nodes():
    x, y = positions[n]
    node = {
        "id": n,
        "x": round(x, 4),
        "y": round(y, 4),
        "deg": degree[n],
        "c": node_community[n],
    }
    if n in ego_set:
        node["ego"] = True
        node["egoLabel"] = EGO_LABELS[n]
    nodes.append(node)

# Edges as compact [a, b] pairs with stable node ordering.
node_index = {n: i for i, n in enumerate(G.nodes())}
edges = [[node_index[a], node_index[b]] for a, b in G.edges()]


# ── Section B: Algorithm vs. humans ──────────────────────────────────────
# Read the hand-labeled circles from the per-ego files in facebook/, then
# score each circle against the best-matching detected community using F1.

print("Ensuring per-ego bundle is extracted …")
ensure_egonets()

print("Loading hand-labeled circles …")
t0 = time.time()
all_circles = []  # [{egoId, name, members, ...}]
for ego_id in EGO_IDS:
    path = os.path.join(EGONETS_DIR, f"{ego_id}.circles")
    if not os.path.exists(path):
        continue
    with open(path) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            cname, members = parts[0], [int(x) for x in parts[1:]]
            # Filter to members that exist in the combined graph (a few may not).
            members = [m for m in members if m in node_community]
            if not members:
                continue
            all_circles.append({
                "egoId": ego_id,
                "egoLabel": EGO_LABELS[ego_id],
                "name": cname,
                "members": members,
            })


def score_circle(circle):
    """Match a hand-labeled circle to the best-fitting detected community."""
    member_set = set(circle["members"])
    if not member_set:
        return None
    # Tally which detected community holds each member.
    tally = Counter(node_community[m] for m in member_set)
    best_c, hits = tally.most_common(1)[0]
    community_size = community_sizes[best_c]
    precision = hits / community_size if community_size else 0.0
    recall = hits / len(member_set)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "bestCommunity": int(best_c),
        "hits": hits,
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "communitySize": community_size,
    }


for c in all_circles:
    c["match"] = score_circle(c)

f1_values = [c["match"]["f1"] for c in all_circles if c["match"]]
match_summary = {
    "totalCircles": len(all_circles),
    "matchedHigh": sum(1 for v in f1_values if v >= 0.5),
    "matchedAny": sum(1 for v in f1_values if v > 0.0),
    "avgF1": round(sum(f1_values) / len(f1_values), 3) if f1_values else 0.0,
    "medianF1": round(sorted(f1_values)[len(f1_values) // 2], 3) if f1_values else 0.0,
    "perfectMatches": sum(1 for v in f1_values if v >= 0.95),
}
t("circles", t0)
print(f"  loaded {len(all_circles)} circles, "
      f"{match_summary['matchedHigh']}/{len(all_circles)} matched at F1≥0.5, "
      f"avg F1 {match_summary['avgF1']}")


# ── Section C: feature browser for ego 107 ───────────────────────────────
# Each ego has its own feature space. We pick ego 107 (most friends + many
# features). Feature names look like:
#   "education;classes;id;anonymized feature 8"
# We bucket features by their *category* (the first token before ";"), then
# inside each category bucket users by which feature value they have.

print(f"Loading features for ego {FEATURE_BROWSER_EGO} …")
t0 = time.time()

featnames_path = os.path.join(EGONETS_DIR, f"{FEATURE_BROWSER_EGO}.featnames")
feat_path = os.path.join(EGONETS_DIR, f"{FEATURE_BROWSER_EGO}.feat")

# featnames: "<index> <category;...;anonymized feature N>"
feat_categories = {}  # idx -> {"category": "education", "label": "education;..."}
if os.path.exists(featnames_path):
    with open(featnames_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = re.match(r"^(\d+)\s+(.+)$", line)
            if not m:
                continue
            idx = int(m.group(1))
            label = m.group(2)
            category = label.split(";", 1)[0]
            feat_categories[idx] = {"category": category, "label": label}

# feat: "<userId> <bit_0> <bit_1> ... <bit_N>"
user_features = {}  # user_id -> set(idx where bit==1)
if os.path.exists(feat_path):
    with open(feat_path) as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            uid = int(parts[0])
            bits = [i - 1 for i, v in enumerate(parts[1:], start=1) if v == "1"]
            if bits:
                user_features[uid] = set(bits)

# Group features by category, then by feature index. For each feature, list
# the global node IDs of users in this ego's network who have that feature.
feature_categories_out = defaultdict(list)
for idx, info in feat_categories.items():
    members = [u for u, bits in user_features.items() if idx in bits]
    members = [m for m in members if m in node_community]
    if not members:
        continue
    # Strip the trailing "anonymized feature N" suffix for cleaner display.
    short_label = re.sub(r";anonymized feature \d+$", "", info["label"])
    feature_categories_out[info["category"]].append({
        "idx": idx,
        "label": short_label,
        "members": members,
    })

# Sort feature groups within each category by descending size.
for cat, groups in feature_categories_out.items():
    groups.sort(key=lambda g: -len(g["members"]))

# Top-line summary per category
feature_category_summary = []
for cat in sorted(feature_categories_out.keys()):
    groups = feature_categories_out[cat]
    feature_category_summary.append({
        "category": cat,
        "groups": len(groups),
        "topGroupSize": len(groups[0]["members"]) if groups else 0,
    })
t("features", t0)
print(f"  feature categories: {[c['category'] for c in feature_category_summary]}")

egos_summary = []
ego_circle_counts = Counter(c["egoId"] for c in all_circles)
for eid in EGO_IDS:
    egos_summary.append({
        "id": eid,
        "label": EGO_LABELS[eid],
        "friends": degree.get(eid, 0),
        "circles": ego_circle_counts.get(eid, 0),
    })

# Trim each circle's member list to keep data.json reasonable; keep up to 60.
def trim_circle(c):
    members = c["members"]
    return {
        "egoId": c["egoId"],
        "egoLabel": c["egoLabel"],
        "name": c["name"],
        "size": len(members),
        "members": members[:60],
        "match": c["match"],
    }

# Highlight a curated set of circles for the picker: top 12 by F1 (best matches),
# plus the 6 worst non-trivial matches, all sized ≥ 5 to keep the comparison
# meaningful.
big_enough = [c for c in all_circles if len(c["members"]) >= 5 and c["match"]]
big_enough.sort(key=lambda c: -c["match"]["f1"])
showcase_circles = big_enough[:12] + big_enough[-6:]

data = {
    "stats": {
        "people": n_nodes,
        "friendships": n_edges,
        "avgFriends": round(avg_deg, 1),
        "maxFriends": max(degree.values()),
        "groupsFound": len(comms),
        "groupsShown": n_groups_shown,
        "egoCount": len(EGO_IDS),
        "circleCount": len(all_circles),
    },
    "communities": [
        {"id": i, "size": community_sizes.get(i, 0)}
        for i in range(n_groups_shown)
    ] + ([{"id": TOP_C, "size": community_sizes.get(TOP_C, 0)}] if n_groups_other else []),
    "topFriends": [{"id": n, "deg": d, "ego": n in ego_set} for n, d in top10],
    "topBridges": [
        {"id": n, "score": round(s, 4), "deg": degree[n], "connects": cg}
        for n, s, cg in top_bridges
    ],
    "egos": egos_summary,
    "circles": {
        "summary": match_summary,
        "showcase": [trim_circle(c) for c in showcase_circles],
    },
    "featureBrowser": {
        "egoId": FEATURE_BROWSER_EGO,
        "egoLabel": EGO_LABELS.get(FEATURE_BROWSER_EGO, f"Ego {FEATURE_BROWSER_EGO}"),
        "summary": feature_category_summary,
        "byCategory": {
            cat: [{"label": g["label"], "members": g["members"]}
                  for g in groups[:20]]  # cap at 20 groups per category
            for cat, groups in feature_categories_out.items()
        },
    },
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
print(f"  top bridges (≥2 groups):")
for n, s, cg in top_bridges:
    print(f"    person {n}  score={s:.3f}  connects={cg}")
