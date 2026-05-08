# SCSP Workshop demo

An interactive visualization of the Stanford SNAP `facebook_combined` social network, built live with Claude Code at the SCSP AI+ Expo (May 9, 2026).

- `index.html` — the page (D3 force-directed graph + stat tiles)
- `data.json` — preprocessed graph (full stats + induced subgraph among top-200 hubs)
- `build_data.py` — preprocessor that reads `facebook_combined.txt.gz` and emits `data.json`

Source data: <https://snap.stanford.edu/data/ego-Facebook.html>
