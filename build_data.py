import gzip, json, random
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

adj = defaultdict(set)
for a, b in sub_edges:
    adj[a].add(b)
    adj[b].add(a)

# Label propagation on the sampled subgraph — vanilla Python, deterministic seed.
random.seed(42)
labels = {n: i for i, n in enumerate(visible)}
order = list(visible)
for _ in range(30):
    random.shuffle(order)
    changed = 0
    for n in order:
        if not adj[n]:
            continue
        counts = Counter(labels[m] for m in adj[n])
        top = max(counts.values())
        winners = [lbl for lbl, c in counts.items() if c == top]
        new_label = min(winners)
        if new_label != labels[n]:
            labels[n] = new_label
            changed += 1
    if changed == 0:
        break

# Re-index communities so the largest is 0, next is 1, etc. — for stable color mapping.
size_by_label = Counter(labels.values())
ordered_labels = [lbl for lbl, _ in size_by_label.most_common()]
remap = {old: new for new, old in enumerate(ordered_labels)}
community = {n: remap[labels[n]] for n in visible}

# Group anything past the top 6 communities into "other" so the legend stays readable.
TOP_C = 6
def cap(c): return c if c < TOP_C else TOP_C
community_capped = {n: cap(community[n]) for n in visible}
community_sizes = Counter(community_capped.values())

nodes = [
    {"id": str(n), "degree": degree[n], "subDegree": sub_deg[n], "community": community_capped[n]}
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
        "communitySizes": [community_sizes[i] for i in range(min(TOP_C + 1, len(community_sizes)))],
    },
}

with open("data.json", "w") as f:
    json.dump(data, f)

print(f"nodes={n_nodes} edges={n_edges} sampled_nodes={len(nodes)} sampled_links={len(links)}")
