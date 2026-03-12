# WASM Acceleration Plan

WASM backend as an optional accelerator. Zero API duplication — same code paths, faster inner loops. Non-WASM users pay nothing. WASM size never appears in the main bundle badge.

## Architecture

Four entrypoints, two with WASM:

| Entrypoint | Exports | WASM | Tree-shakeable |
|---|---|---|---|
| `numpy-ts` | NDArray (wraps `/core`) | No | No |
| `numpy-ts/core` | Standalone functions | No | Yes |
| `numpy-ts/wasm` | NDArray (wraps `/wasm/core`) | Yes | No |
| `numpy-ts/wasm/core` | Standalone WASM functions | Yes | Yes |

```typescript
// JS-only (existing)
import { array, add } from 'numpy-ts/core';       // tree-shakeable
import { array } from 'numpy-ts';                  // full NDArray

// WASM-accelerated (new)
import { array, add } from 'numpy-ts/wasm/core';   // tree-shakeable, WASM
import { array } from 'numpy-ts/wasm';              // full NDArray, WASM
```

Users opt into WASM by changing their import path. Same API, same types, faster inner loops. Falls back to JS automatically for small arrays or unsupported ops/dtypes.

### How it works

`numpy-ts/wasm/core` re-exports everything from `/core`, overriding specific functions with WASM-accelerated wrappers:

```typescript
// src/wasm/core.ts
export * from '../core';
export { add } from './kernels/add';       // overrides core's add
export { matmul } from './kernels/matmul'; // overrides core's matmul
// un-overridden functions pass through from core as-is
```

Each WASM kernel wrapper imports its WASM binary module + the JS fallback, dispatches based on dtype, array size, and contiguity:

```typescript
// src/wasm/kernels/add.ts
import { add as jsAdd } from '../../core/arithmetic';
import { add_f64, add_f32, add_i32 } from '../bins/binary.wasm';
import { withWasm } from '../runtime';

const THRESHOLD = 256;

const wasmDispatch: Partial<Record<DType, (a, b, out, n) => void>> = {
  float64: add_f64,
  float32: add_f32,
  int32: add_i32,
  int16: add_i16,
  int8: add_i8,
  complex128: add_c128,
  complex64: add_c64,
  // missing dtypes (uint8, bigint64, etc.) fall through to JS
};

export function add(x1: NDArrayCore, x2: NDArrayCore | number): NDArrayCore {
  const fn = wasmDispatch[x1.dtype];
  if (fn && typeof x2 !== 'number' && isContiguous(x1) && isContiguous(x2) && x1.size > THRESHOLD) {
    return withWasm(fn, x1, x2);
  }
  return jsAdd(x1, x2); // JS handles small arrays, broadcasting, non-contiguous, unsupported dtypes
}
```

`numpy-ts/wasm` exports a generated NDArray class that wraps `/wasm/core` instead of `/core`. The existing NDArray generator runs twice — same template, different import path:

```typescript
// Generated: src/full/ndarray.ts (existing)
import * as core from '../core';
class NDArray {
  add(other) { return up(core.add(this, other)); }
}

// Generated: src/wasm/ndarray.ts (new — same template, different import)
import * as core from './core';   // -> numpy-ts/wasm/core (WASM-accelerated)
class NDArray {
  add(other) { return up(core.add(this, other)); }
}
```

One source of truth in the generator. Two outputs. No runtime indirection, no prototype patching, no backend registry.

### Tree-shaking behavior

**`numpy-ts/wasm/core`:** If a user imports `{ add, reshape }`, the bundler pulls in:
- `add` -> WASM kernel (.wasm base64) + JS fallback
- `reshape` -> just the JS version from core (no WASM overhead)
- All other WASM kernels -> tree-shaken away

**`numpy-ts/wasm`:** All WASM kernels included (same as how `/` includes all JS ops). Users who chose the full NDArray API already accepted no tree-shaking. WASM compilation is still lazy — base64 strings are in the bundle but `.wasm` only compiles on first call to each kernel group.

## Memory Management

WASM can't touch JS TypedArrays directly. Data must live in WASM linear memory. All kernels share a single runtime that handles the copy-in/copy-out pattern.

### Shared bump allocator (`src/wasm/runtime.ts`)

```typescript
// src/wasm/runtime.ts
let memory: WebAssembly.Memory;
let offset = 0;

export function ensureMemory(bytes: number): void {
  if (!memory) memory = new WebAssembly.Memory({ initial: 1 });
  const needed = Math.ceil(bytes / 65536); // pages
  const current = memory.buffer.byteLength / 65536;
  if (needed > current) memory.grow(needed - current);
}

function alloc(bytes: number): number {
  const aligned = (offset + 7) & ~7; // 8-byte align
  offset = aligned + bytes;
  return aligned;
}

/**
 * Copy inputs into WASM memory, run kernel, copy output back.
 * Bump allocator resets on every call (no fragmentation, no GC).
 * Memory grows monotonically (high-water-mark pattern).
 */
export function withWasm(
  kernel: Function,
  x1: NDArrayCore,
  x2: NDArrayCore,
): NDArrayCore {
  const bytesPerElement = x1.data.BYTES_PER_ELEMENT;
  const totalBytes = (x1.byteLength + x2.byteLength + x1.size * bytesPerElement);
  ensureMemory(totalBytes);

  offset = 0;

  // Copy inputs into WASM memory
  const aPtr = alloc(x1.byteLength);
  new Uint8Array(memory.buffer, aPtr, x1.byteLength)
    .set(new Uint8Array(x1.data.buffer, x1.data.byteOffset, x1.byteLength));

  const bPtr = alloc(x2.byteLength);
  new Uint8Array(memory.buffer, bPtr, x2.byteLength)
    .set(new Uint8Array(x2.data.buffer, x2.data.byteOffset, x2.byteLength));

  // Allocate output in WASM memory
  const outBytes = x1.size * bytesPerElement;
  const outPtr = alloc(outBytes);

  // Run kernel
  kernel(aPtr, bPtr, outPtr, x1.size);

  // Copy output back to JS
  const OutCtor = x1.data.constructor as TypedArrayConstructor;
  const result = new OutCtor(x1.size);
  result.set(new OutCtor(memory.buffer, outPtr, x1.size));

  return fromStorage(ArrayStorage.fromData(result, [...x1.shape], x1.dtype));
}
```

**Why copy-in/copy-out?** The alternative (allocating NDArrays directly in WASM memory) would require a completely different storage model for the entire library. Copy overhead is bounded: for `add` on 250K f64 elements (2MB), we copy ~6MB total. On modern hardware that's ~1-2ms, while SIMD saves more than that at scale. The per-kernel threshold ensures we only take the WASM path when it's a net win.

### Memory shared across all kernels

All `.wasm.ts` modules receive the same `WebAssembly.Memory` instance. The WASM instances are compiled with imported memory rather than their own:

```typescript
// In each .wasm.ts module
inst = new WebAssembly.Instance(new WebAssembly.Module(bytes), {
  env: { memory: getSharedMemory() }
});
```

This avoids each kernel having its own memory allocation and enables the single bump allocator in `runtime.ts` to manage all WASM memory.

## Language: Zig

Zig over Rust for this project because:
- **Native SIMD operators** (`+`, `*` on `@Vector` types) — cleaner than Rust intrinsics for a codebase that's 80% SIMD loops
- **No `unsafe`/inner function ceremony** — pointer-to-slice conversion at the FFI boundary gives bounds checking in debug builds, zero overhead in release, without the two-function dance
- **Built-in math** (`@sqrt`, `@exp`, `@log`) — no `libm` crate or `no_std` boilerplate
- **Less build friction** — no Cargo.toml, feature flags, or crate structure; single `zig build-lib` command
- **Explicit is appropriate** — for stateless numerical kernels, Zig's "C but better" philosophy fits better than Rust's ownership model (borrow checker is a no-op here anyway)

Rust is better suited for projects with complex allocations, shared state, and ownership graphs. These kernels are stateless functions that take pointers and do math.

## Kernel Granularity & Distribution

Each WASM kernel is compiled as a **separate, small .wasm binary** per function group (all dtypes for that group). Kernels are distributed as **base64-encoded strings inside generated .wasm.ts files** in `src/wasm/bins/`.

### Why base64-in-TS

- **Universal compatibility** — works in Node, Deno, Bun, browsers, and every bundler without special WASM loader plugins or config
- **Native tree-shaking** — bundlers drop unused modules automatically; unused kernels never enter the bundle
- **Simple npm distribution** — just .js files, no special `files` config for .wasm assets
- **~33% size overhead is negligible** — individual kernels are a few KB; base64 adds ~1KB per kernel

### Generated source files (same pattern as ndarray.ts)

The `.wasm.ts` files in `src/wasm/bins/` are **auto-generated TypeScript source files checked into git** — the same pattern as `src/full/ndarray.ts`. This means:

- **TypeScript/IDE always works** — real `.ts` files with correct types, autocomplete, go-to-definition
- **No stubs needed** — the real files are always there from git
- **No gitignore** — they're source files in `src/`
- **Only Zig kernel authors need Zig** — everyone else uses the committed `.wasm.ts` files
- **`npm run build:wasm`** regenerates them when Zig source changes (like `npm run generate` regenerates `ndarray.ts`)

### Kernel file structure

```
src/wasm/
  core.ts              # Re-exports /core, overrides WASM-accelerated functions
  ndarray.ts           # Generated NDArray wrapping /wasm/core
  index.ts             # Entrypoint for numpy-ts/wasm
  runtime.ts           # Shared memory management (bump allocator, copy-in/copy-out)
  kernels/
    add.ts             # WASM wrapper: dtype dispatch + threshold + JS fallback
    sub.ts             # ...
    matmul.ts          # ...
    ...
  bins/                # Generated .wasm.ts files (checked into git)
    binary.wasm.ts     # base64-encoded .wasm + lazy sync instantiation
    unary.wasm.ts
    matmul.wasm.ts
    reduction.wasm.ts
    ...
  zig/                 # Zig source (one file per kernel group)
    simd.zig           # Shared SIMD types and load/store helpers
    binary.zig
    unary.zig
    matmul.zig
    reduction.zig
    ...
```

Each `.wasm.ts` file follows this pattern:

```typescript
// src/wasm/bins/binary.wasm.ts
// AUTO-GENERATED by build:wasm — do not edit. Regenerate with: npm run build:wasm

import { getSharedMemory } from '../runtime';

const B64 = "AGFzbQEAAAA...";
let inst: WebAssembly.Instance;

function init() {
  if (inst) return;
  const bytes = Uint8Array.from(atob(B64), c => c.charCodeAt(0));
  inst = new WebAssembly.Instance(
    new WebAssembly.Module(bytes),
    { env: { memory: getSharedMemory() } }
  );
}

export function add_f64(aPtr: number, bPtr: number, outPtr: number, n: number): void {
  init();
  (inst.exports['add_f64'] as Function)(aPtr, bPtr, outPtr, n);
}
export function add_f32(aPtr: number, bPtr: number, outPtr: number, n: number): void {
  init();
  (inst.exports['add_f32'] as Function)(aPtr, bPtr, outPtr, n);
}
// ... all exports for this kernel group
```

**Fully synchronous — no top-level await.** Lazy init on first call. Sync `WebAssembly.Module()` works fine for small per-kernel binaries (a few KB each). This avoids Safari's top-level await bugs and works in all environments.

### Zig compilation: one .wasm per kernel group

```bash
# Each kernel group compiles independently
zig build-lib -target wasm32-freestanding -OReleaseFast src/wasm/zig/binary.zig  -o .wasm-tmp/binary.wasm
zig build-lib -target wasm32-freestanding -OReleaseFast src/wasm/zig/unary.zig   -o .wasm-tmp/unary.wasm
zig build-lib -target wasm32-freestanding -OReleaseFast src/wasm/zig/matmul.zig  -o .wasm-tmp/matmul.wasm
# ... etc
```

The build step then base64-encodes each .wasm and generates the `.wasm.ts` wrapper module.

### Zig kernel conventions

FFI exports convert pointers to slices at the boundary for debug-time bounds checking:

```zig
export fn add_f64(a_ptr: [*]const f64, b_ptr: [*]const f64, out_ptr: [*]f64, n: u32) void {
    const len = @as(usize, n);
    const a = a_ptr[0..len];
    const b = b_ptr[0..len];
    const out = out_ptr[0..len];
    // SIMD loop with bounds-checked slices in debug, zero-cost in release
}
```

No inner functions needed — Zig has no `unsafe` distinction. Explicit per-dtype implementations (no comptime generics for kernels) — explicitness is preferred for numerical code where SIMD lane widths and unroll factors differ by type.

WASM instances use **imported memory** (not their own) so the shared runtime can manage all allocations:

```zig
// Each kernel declares imported memory
extern var memory: [*]u8;
// Or equivalently, the Zig build uses --import-memory
```

### Dtype coverage

Each kernel wrapper has a per-kernel dispatch table mapping dtypes to WASM functions:

```typescript
// src/wasm/kernels/add.ts
const wasmDispatch: Partial<Record<DType, (aPtr, bPtr, outPtr, n) => void>> = {
  float64:    add_f64,
  float32:    add_f32,
  int32:      add_i32,
  int16:      add_i16,
  int8:       add_i8,
  complex128: add_c128,
  complex64:  add_c64,
  // uint8, bigint64, etc. — not in table, fall through to JS
};
```

Missing dtypes naturally fall through to the JS implementation. Some kernel/dtype combinations are intentionally excluded where JS is already fast enough (e.g., integer sort).

## Build Pipeline

Three phases, integrated into the existing `npm run build`:

```
npm run build
  |
  |-- 1. build:wasm  (Zig compile -> base64 -> generate .wasm.ts)
  |     SKIPPED with warning if Zig not installed
  |     Checked-in .wasm.ts files used as-is
  |
  |-- 2. generate    (existing + now also generates src/wasm/ndarray.ts)
  |
  |-- 3. esbuild     (existing + now also walks src/wasm/ for ESM output)
```

### Step 1: build:wasm (`scripts/build-wasm.ts`)

```typescript
// scripts/build-wasm.ts
import { execSync } from 'child_process';
import { readdirSync, readFileSync, writeFileSync, mkdirSync } from 'fs';

const ZIG_DIR = 'src/wasm/zig';
const BINS_DIR = 'src/wasm/bins';
const TMP_DIR = '.wasm-tmp';

// Check for Zig
try {
  execSync('zig version', { stdio: 'ignore' });
} catch {
  console.warn('WARNING: Zig not found — skipping WASM build.');
  console.warn('Checked-in .wasm.ts files will be used. Install Zig to rebuild.');
  process.exit(0);
}

mkdirSync(TMP_DIR, { recursive: true });

const zigFiles = readdirSync(ZIG_DIR)
  .filter(f => f.endsWith('.zig') && f !== 'simd.zig');

for (const file of zigFiles) {
  const name = file.replace('.zig', '');

  // Compile Zig -> .wasm
  execSync(
    `zig build-lib -target wasm32-freestanding -OReleaseFast ` +
    `--import-memory ` +
    `${ZIG_DIR}/${file} -femit-bin=${TMP_DIR}/${name}.wasm`,
    { stdio: 'inherit' }
  );

  // Read .wasm, base64 encode, parse exports
  const wasmBytes = readFileSync(`${TMP_DIR}/${name}.wasm`);
  const base64 = wasmBytes.toString('base64');
  const exports = parseWasmExports(wasmBytes);

  // Generate .wasm.ts
  writeFileSync(`${BINS_DIR}/${name}.wasm.ts`, generateWasmModule(name, base64, exports));
  console.log(`  ${name}.wasm.ts (${wasmBytes.length} bytes, ${exports.length} exports)`);
}

// Clean up
execSync(`rm -rf ${TMP_DIR}`);
```

### Step 2: Zig-optional build integration

In `build.ts`, the WASM step is conditional:

```typescript
function hasZig(): boolean {
  try { execSync('zig version', { stdio: 'ignore' }); return true; }
  catch { return false; }
}

if (hasZig()) {
  console.log('Building WASM kernels...');
  execSync('tsx scripts/build-wasm.ts', { stdio: 'inherit' });
} else {
  console.warn('Zig not found — using checked-in WASM binaries');
}
```

No stubs, no conditional compilation. The `.wasm.ts` files always exist in git. Zig is only needed to regenerate them after modifying `.zig` source.

### Step 3: Package config

**package.json exports:**
```jsonc
{
  ".": { /* existing — no WASM */ },
  "./core": { /* existing — no WASM */ },
  "./wasm": {
    "types": "./dist/types/wasm/index.d.ts",
    "import": "./dist/esm/wasm/index.js",
    "default": "./dist/esm/wasm/index.js"
  },
  "./wasm/core": {
    "types": "./dist/types/wasm/core.d.ts",
    "import": "./dist/esm/wasm/core.js",
    "default": "./dist/esm/wasm/core.js"
  }
}
```

**sideEffects** stays `false` — no side-effect imports needed. Users explicitly import from `/wasm` or `/wasm/core`. WASM kernels are lazy-init, not side-effectful.

## Implementation Steps

### 1. Shared runtime (`src/wasm/runtime.ts`)

Bump allocator + copy-in/copy-out logic. Shared `WebAssembly.Memory` instance used by all kernel modules. This is the foundation — build it first, test it independently.

### 2. First end-to-end kernel: `add`

Prove the full pipeline works with one kernel before scaling:
- Zig source (`src/wasm/zig/binary.zig` — already exists in `wasm-bench/`)
- Build script generates `src/wasm/bins/binary.wasm.ts`
- Kernel wrapper `src/wasm/kernels/add.ts` with dtype dispatch + threshold + fallback
- `src/wasm/core.ts` re-exports core + overrides add
- Unit test: WASM add matches JS add for all supported dtypes
- Benchmark: verify WASM is faster above threshold, JS is faster below

### 3. Build pipeline (`scripts/build-wasm.ts`)

Automate Zig compile -> base64 -> `.wasm.ts` generation. After this, adding a new kernel is just: write the `.zig` file, add the kernel wrapper, add to `wasm/core.ts`.

### 4. Remaining kernels

Port from existing `wasm-bench/zig/` implementations:
- Binary element-wise: sub, mul, div, power, maximum, minimum, copysign, mod, floor_divide, hypot, fmax, fmin, logaddexp
- Unary element-wise: sqrt, exp, log, sin, cos, tan, abs, negative, ceil, floor, round
- Matmul
- Reductions: sum, min, max, prod

### 5. Generated WASM NDArray (`src/wasm/ndarray.ts`)

Update `scripts/generate-full.ts` to produce two NDArray classes:
- `src/full/ndarray.ts` (existing) — imports from `../core`
- `src/wasm/ndarray.ts` (new) — imports from `./core` (WASM-accelerated)

Same template, same methods, different import path. One source of truth.

### 6. WASM entrypoints

- `src/wasm/core.ts` — re-export + override pattern
- `src/wasm/index.ts` — full NDArray + types

### 7. Package config

Add `/wasm` and `/wasm/core` exports to package.json.

### 8. Tests

- Unit tests: WASM kernel wrappers (correct results, fallback behavior, dtype dispatch)
- Integration tests: same test suite runs against `/core` and `/wasm/core`, results must match
- Bundle tests: verify `numpy-ts` bundle size unchanged (no WASM leakage)
- Tree-shaking tests: verify importing single function from `/wasm/core` only includes that kernel's WASM binary

### 9. Benchmarks

Extend existing benchmark suite to compare JS vs WASM backends across array sizes, dtypes, and operations. Use results to tune per-kernel thresholds.

## TODO

- [ ] Step 1: Shared runtime (bump allocator, copy-in/copy-out, shared memory)
- [ ] Step 2: First end-to-end kernel (add) — proves the full pipeline
- [ ] Step 3: Build pipeline (Zig compile + base64 + .wasm.ts generation)
- [ ] Step 4: Remaining kernels (binary, unary, matmul, reductions)
- [ ] Step 5: Update generator to produce WASM NDArray variant
- [ ] Step 6: WASM entrypoints (wasm/core.ts, wasm/index.ts)
- [ ] Step 7: Package config (exports for `/wasm` and `/wasm/core`)
- [ ] Step 8: Tests (unit, integration, bundle, tree-shaking)
- [ ] Step 9: Benchmarks (JS vs WASM comparison, threshold tuning)
- [ ] Update `docs/ARCHITECTURE.md` with WASM backend docs
- [ ] Update `docs/API-REFERENCE.md` with `/wasm` and `/wasm/core` entrypoint usage
- [ ] Update `README.md` with WASM installation/usage instructions

## Design Decisions

**Why Zig over Rust?** Stateless WASM kernels don't benefit from Rust's borrow checker. Zig's native SIMD operators, built-in math, and zero ceremony make it more ergonomic for this domain. Rust is better suited for projects with complex ownership.

**Why separate entrypoints, not a side-effect plugin?** A global backend registry (`import 'numpy-ts/wasm'`) would pull in ALL WASM kernels regardless of which ops are used. Separate entrypoints with re-export + override give bundlers the information they need to tree-shake unused WASM kernels.

**Why two generated NDArray classes?** NDArray methods are bound to core function imports at generation time. You can't tree-shake methods off a class, and you can't swap imports at runtime. Generating two classes (one wrapping `/core`, one wrapping `/wasm/core`) from the same template avoids runtime indirection, prototype patching, or backend registries. One source of truth in the generator — zero hand-maintained duplication.

**Why per-group .wasm, not one monolith?** Maximum tree-shaking granularity. If a user imports `add` from `/wasm/core`, they get the binary kernel group but not unary or matmul. Each group's WASM binary is small (a few KB) so the overhead of separate modules is negligible.

**Why base64-in-TS, not separate .wasm files?** Universal bundler/runtime compatibility without plugins. Tree-shakes naturally. Simple npm distribution. Size overhead is negligible for small kernels.

**Why synchronous, not top-level await?** Safari has a [WebKit bug](https://github.com/mdn/browser-compat-data/issues/20426) where simultaneous imports of modules with top-level await can deadlock. Sync `WebAssembly.Module()` works fine for small per-kernel binaries. Lazy init means WASM only compiles on first use — no upfront cost at import time.

**Why explicit per-dtype Zig functions, not comptime generics?** SIMD lane widths and unroll factors differ by type. Explicit code is easier to debug and profile for numerical kernels. The duplication is stable — once an op works, it rarely changes.

**Why `sideEffects: false` still works?** No side-effect imports needed. Users explicitly import from `/wasm` or `/wasm/core`. WASM kernel instantiation is lazy (inside function calls), not at module evaluation time.

**Why copy-in/copy-out, not shared memory?** Making NDArrays allocate directly in WASM memory would require a completely different storage model for the entire library. Copy overhead is bounded and predictable. The per-kernel threshold ensures WASM is only used when SIMD savings exceed copy cost.

**Why a shared bump allocator?** All WASM kernel calls are synchronous and non-reentrant. A bump allocator (reset offset to 0 on each call) has zero fragmentation and zero overhead. Memory grows monotonically (high-water-mark) which is fine for numerical workloads that stabilize quickly.

**Why per-kernel thresholds?** WASM call + memory copy overhead varies by operation complexity. A simple `add` might break even at 512 elements, while `matmul` (O(n^3)) breaks even at 32. Starting at 256, tuned per-kernel with benchmark data.

**Why generated .wasm.ts files are checked into git?** Same pattern as `src/full/ndarray.ts`. TypeScript/IDE always works. No stubs needed. Devs without Zig use the committed files. Only Zig kernel authors need Zig installed to regenerate.

**Why per-kernel dtype dispatch tables?** Each operation supports a different set of dtypes in WASM. `add` covers f64/f32/i32/i16/i8/c128/c64. `sort` might only cover f64/f32 (JS is fast enough for integers). A `Partial<Record<DType, fn>>` table makes this explicit — missing dtypes fall through to JS automatically.
