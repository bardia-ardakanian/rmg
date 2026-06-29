#!/bin/bash
# Fetch the rigged robot mannequin used by the visuals (Mixamo Xbot; ships with the three.js examples).
# The web viewer loads it straight from the CDN; render_robot.py needs the local file.
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
curl -sL https://cdn.jsdelivr.net/gh/mrdoob/three.js@r160/examples/models/gltf/Xbot.glb -o "$ROOT/assets_xbot.glb"
echo "saved $ROOT/assets_xbot.glb ($(wc -c < "$ROOT/assets_xbot.glb") bytes)"
