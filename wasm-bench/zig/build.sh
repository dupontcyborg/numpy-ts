#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DIST_DIR="$SCRIPT_DIR/../dist"

mkdir -p "$DIST_DIR"

KERNELS=(matmul reduction unary binary sort convolve)

for kern in "${KERNELS[@]}"; do
  echo "Building Zig WASM kernel: $kern..."
  zig build-exe \
    "$SCRIPT_DIR/$kern.zig" \
    -target wasm32-freestanding \
    -O ReleaseFast \
    -mcpu=generic+simd128 \
    -fno-entry \
    -rdynamic \
    -fstrip \
    -femit-bin="$DIST_DIR/${kern}_zig.wasm"
done

echo ""
ls -lh "$DIST_DIR"/*_zig.wasm
