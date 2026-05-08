import gzip, json
from collections import Counter, defaultdict

edges_full = []
with gzip.open("facebook_combined.txt.gz", "rt") as f:
    for line in f:
        a, b = line.split()
        edges_full.append((int(a), int(b)))

degree = Counter()
for a, b in edges_full:
    degree[a] += 1
    degree[b] += 1

n_nodes = len(degree)
n_edges = len(edges_full)
avg_deg = (2 * n_edges) / n_nodes

top10 = degree.most_common(10)

# Sample: top-K highest-degree nodes + the induced subgraph among them.
K = 200
top_nodes = set(n for n, _ in degree.most_common(K))

sub_edges = [(a, b) for a, b in edges_full if a in top_nodes and b in top_nodes]
sub_deg = Counter()
for a, b in sub_edges:
    sub_deg[a] += 1
    sub_deg[b] += 1

# Drop isolates from the sample.
visible = {n for n in top_nodes if sub_deg[n] > 0}
sub_edges = [(a, b) for a, b in sub_edges if a in visible and b in visible]

nodes = [
    {"id": str(n), "degree": degree[n], "subDegree": sub_deg[n]}
    for n in visible
]
links = [{"source": str(a), "target": str(b)} for a, b in sub_edges]

data = {
    "stats": {
        "nodes": n_nodes,
        "edges": n_edges,
        "avgDegree": round(avg_deg, 2),
        "maxDegree": max(degree.values()),
        "top10": [{"id": str(n), "degree": d} for n, d in top10],
    },
    "sample": {
        "k": K,
        "nodes": nodes,
        "links": links,
    },
}

with open("data.json", "w") as f:
    json.dump(data, f)

print(f"nodes={n_nodes} edges={n_edges} sampled_nodes={len(nodes)} sampled_links={len(links)}")
