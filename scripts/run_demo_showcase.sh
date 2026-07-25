#!/usr/bin/env bash
# Start showcase Web UIs (offside review :6006, multiview dashboard :6008).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export DEMO_DASHBOARD_DIR="${DEMO_DASHBOARD_DIR:-$ROOT/demo_outputs/dashboard}"
export DEMO_OFFSIDE_REVIEW_DIR="${DEMO_OFFSIDE_REVIEW_DIR:-$ROOT/demo_outputs/offside/review}"
PY="${PYTHON:-python}"

mkdir -p "$DEMO_DASHBOARD_DIR" "$DEMO_OFFSIDE_REVIEW_DIR"

if [[ ! -f configs/demo_dashboard.yaml ]]; then
  cp configs/demo_dashboard.example.yaml configs/demo_dashboard.yaml
  echo "Created configs/demo_dashboard.yaml from example — edit paths before use."
fi
if [[ ! -f configs/offside_demo.yaml ]]; then
  cp configs/offside_demo.example.yaml configs/offside_demo.yaml
  echo "Created configs/offside_demo.yaml from example — edit paths before use."
fi

echo "Starting multiview dashboard on :6008 ..."
$PY main.py demo-dashboard --config configs/demo_dashboard.yaml --host 0.0.0.0 --port 6008 &
DASH_PID=$!

if [[ -f "${OFFSIDE_VIDEO:-demo_data/offside/offside.mp4}" ]]; then
  echo "Starting offside review on :6006 ..."
  $PY main.py demo-offside review \
    --input "${OFFSIDE_VIDEO:-demo_data/offside/offside.mp4}" \
    --discovery-dir "${OFFSIDE_DISCOVERY:-demo_outputs/offside/discovery}" \
    --config configs/offside_demo.yaml \
    --host 0.0.0.0 --port 6006 --device "${DEVICE:-cuda}" &
  OFF_PID=$!
else
  echo "Skip offside review (missing offside video). Set OFFSIDE_VIDEO=..."
  OFF_PID=""
fi

echo "Dashboard PID=$DASH_PID  Offside PID=${OFF_PID:-n/a}"
echo "Open http://127.0.0.1:6008/  and  http://127.0.0.1:6006/"
wait
