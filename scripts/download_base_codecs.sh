#!/usr/bin/env bash
# Download pretrained TIC base codecs into checkpoints/base_codec/
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${ROOT}/checkpoints/base_codec"
mkdir -p "${OUT}"

BASE_URL="https://github.com/NYCU-MAPL/TransTIC/releases/download/v1.0"
for q in 1 2 3 4; do
  f="base_codec_${q}.pth.tar"
  # remove text placeholder if present
  rm -f "${OUT}/PLACEHOLDER_${f}.txt"
  if [[ -f "${OUT}/${f}" ]]; then
    echo "exists ${f}"
  else
    echo "downloading ${f}"
    curl -L "${BASE_URL}/${f}" -o "${OUT}/${f}"
  fi
  # also keep a convenience copy at checkpoints/ for older configs
  ln -sfn "base_codec/${f}" "${ROOT}/checkpoints/${f}"
done
echo "Done. Files in ${OUT}"
