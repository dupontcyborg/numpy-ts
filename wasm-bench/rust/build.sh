#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DIST_DIR="$SCRIPT_DIR/../dist"

mkdir -p "$DIST_DIR"

KERNELS=(matmul reduction unary binary sort convolve)

cd "$SCRIPT_DIR"

for kern in "${KERNELS[@]}"; do
  echo "Building Rust WASM kernel: $kern..."
  cargo build --target wasm32-unknown-unknown --release --no-default-features --features "kern-$kern"
  cp "$SCRIPT_DIR/target/wasm32-unknown-unknown/release/matmul_wasm.wasm" "$DIST_DIR/${kern}_rust.wasm"
done

echo ""
ls -lh "$DIST_DIR"/*_rust.wasm
