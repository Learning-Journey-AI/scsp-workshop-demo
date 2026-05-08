# SCSP Workshop demo

An interactive visualization of the Stanford SNAP Facebook ego networks, built live with Claude Code at the SCSP AI+ Expo (May 9, 2026).

The page has three sections:

1. **The whole network** — all 4,039 people and 88,234 friendships, color-coded by friend group (Louvain community detection). The 10 survey participants ("egos") are marked with a gold ring. Hover, click, search, and filter by friend group.
2. **Algorithm vs. humans** — compares the algorithm's friend groups against the 193 social circles the participants labeled by hand ("high-school friends," "co-workers," "family"). Most don't match — and that's the interesting story.
3. **What ties friend groups together** — for the most-connected ego (person 107), browses anonymized profile fields (education, work, hometown, etc.) to see which feature values their friends share.

## Files

- `index.html` — the page (HTML canvas + d3-zoom, no build step)
- `data.json` — preprocessed network (positions, communities, bridges, circles, features)
- `build_data.py` — Python preprocessor (NetworkX for Louvain + betweenness + spring layout)
- `facebook_combined.txt.gz` — the combined edge list from SNAP (4,039 users, 88,234 friendships)
- `facebook.tar.gz` — the per-ego bundle from SNAP (10 ego networks with `.edges`, `.feat`, `.featnames`, `.circles`, `.egofeat`); auto-extracted by `build_data.py` on first run

## Source

Stanford SNAP, ego-Facebook: <https://snap.stanford.edu/data/ego-Facebook.html>

From McAuley & Leskovec, *"Learning to Discover Social Circles in Ego Networks,"* NIPS 2012. Collected via a custom Facebook research app from 10 survey participants who shared the friendship structure of their friend lists. All identities anonymized; the central user is excluded from each ego network.
