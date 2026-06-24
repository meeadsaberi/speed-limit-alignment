#!/usr/bin/env bash
# Speed-Limit Alignment — one-command reproduction (no API key needed for the core).
# The VLM imagery characterisation is optional and run separately (needs keys, has a cost).
set -euo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python
if [ ! -x "$PY" ]; then
  echo "[setup] creating venv + installing requirements ..."
  python3 -m venv .venv && .venv/bin/pip install -q --upgrade pip && .venv/bin/pip install -q -r requirements.txt
fi

echo "[1/4] harmonise networks      -> data/enriched.gpkg"
$PY model/load.py
echo "[2/4] enrich (land use + OSM POIs)"
$PY model/enrich.py
echo "[3/4] score: speed-limit alignment  (uses data/vlm_context.parquet if present)"
$PY score/limit_alignment.py
echo "[4/4] per-city map layers     -> map/data/align_{region}.geojson"
$PY map/build_alignment_map.py

echo
echo "Done."
echo "  Score (per link): score/limit_alignment.csv / .geojson"
echo "  Interactive map : node map/server.js  ->  http://localhost:8731"
