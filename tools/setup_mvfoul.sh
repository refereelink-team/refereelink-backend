#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
target="$repo_root/third_party/sn-mvfoul"

if [ -f "$target/VARS model/model.py" ]; then
  echo "SoccerNet VARS model source is already available."
  exit 0
fi

mkdir -p "$repo_root/third_party"
if [ ! -d "$target/.git" ]; then
  git clone --filter=blob:none --no-checkout \
    https://github.com/SoccerNet/sn-mvfoul.git "$target"
fi

git -C "$target" sparse-checkout init --cone
git -C "$target" sparse-checkout set "VARS model"
git -C "$target" checkout

echo "SoccerNet VARS model source is ready at: $target/VARS model"
echo "Place 14_model.pth.tar in: $repo_root/assets/weights"
