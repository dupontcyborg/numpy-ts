# numpy-ts Async Execution Design

## Motivation

numpy-ts operations are synchronous and single-threaded. For compute-intensive
operations on large arrays (matmul, FFT, convolution, linear algebra), this
blocks the JS thread for meaningful wall-clock time. On a 1024×1024 f64 matrix,
`np.matmul` takes ~90ms — long enough to drop frames if called on the main
thread.

The goal is an opt-in async API that offloads heavy compute to a worker pool,
with sub-worker parallelism for large ops, while keeping the existing sync API
completely unchanged.

Benchmarks motivating this work (wasm-bench results on Apple M-series):

| Op | Single-thread | 8-worker WASM | WebGPU (f32 only) |
|-|-|-|-|
| matmul 1024³ f64 | ~90ms / 24 GFLOP/s | ~14ms / 156 GFLOP/s | N/A |
| matmul 1024³ f32 | ~17ms / 16 GFLOP/s | ~2ms / 110 GFLOP/s | ~21ms / 104 GFLOP/s |

---

## Design Principles

1. **No breaking changes.** The sync API is untouched.
2. **Opt-in per call.** Async is a namespace, not a mode.
3. **No required server headers.** Workers work anywhere. SAB is a progressive
   enhancement, never required.
4. **Single implementation.** Async ops delegate to the same WASM kernels —
   no duplicate math code.

---

## API Surface

Async ops live under `np.async.*`. The function signatures mirror their sync
counterparts, returning `Promise<NdArray>` instead of `NdArray`.

```typescript
import { np } from 'numpy-ts';

// Sync (unchanged)
const c = np.matmul(a, b);

// Async
const c = await np.async.matmul(a, b);
const d = await np.async.fft(signal);
const e = await np.async.solve(A, b);
```

Operations covered in the first iteration (the ones where async pays off):
- `np.async.matmul`
- `np.async.fft` / `np.async.ifft`
- `np.async.convolve`
- `np.async.linalg.solve`
- `np.async.linalg.svd`
- `np.async.linalg.eig`

The namespace is extensible — any op can be added later. Small ops (element-wise
unary/binary) are not worth the round-trip cost and are intentionally excluded.

---

## Worker Pool

### Lifecycle

The worker pool is a **lazily-initialized singleton**. It is created on the
first `np.async.*` call and persists for the lifetime of the page/process. There
is no explicit teardown API in v1 (workers are cleaned up when the page unloads).

```typescript
// First call — pool initializes transparently, then executes
const c = await np.async.matmul(a, b);

// Subsequent calls — pool already warm, dispatches immediately
const d = await np.async.fft(x);
```

### Configuration

```typescript
// Optional — call before first async op to control pool size
// If not called, defaults to navigator.hardwareConcurrency (capped at 8)
np.async.init({ workers: 4 });
```

### Pool Workers vs Sub-Workers

There are two levels of workers:

**Pool workers** (1–N, default: `hardwareConcurrency`): receive dispatched ops
from the main thread. Each pool worker runs the full numpy-ts WASM module.

**Sub-workers** (spawned by pool workers): used for parallelizing a single large
op across CPU cores. A pool worker handling a 1024×1024 matmul will internally
divide the row space across sub-workers, wait via `Atomics.wait` (allowed in
worker context), and return when all sub-workers complete.

Sub-worker pools are owned by pool workers, not shared globally. A pool worker
initializes its sub-workers lazily on first large-op dispatch.

```
Main thread
  └── Pool worker 0  ──── Sub-worker 0-0
  │     (owns WASM        Sub-worker 0-1
  │      instance)        Sub-worker 0-2
  │                       Sub-worker 0-3
  └── Pool worker 1  ──── Sub-worker 1-0
        ...                ...
```

For most workloads, a pool of 1–2 pool workers with 4–8 sub-workers each is
optimal. The pool worker handles dispatch and coordination; sub-workers do pure
compute.

### Work Queue

Each pool worker has an internal FIFO queue. Concurrent `np.async.*` calls from
the main thread are enqueued and processed in order. There is no cross-worker
work-stealing in v1.

---

## Data Transport: Two Paths

The transport path is detected once at pool initialization and fixed for the
pool's lifetime.

### Detection

```typescript
function sabAvailable(): boolean {
  try {
    new SharedArrayBuffer(4);
    return true;
  } catch {
    return false;
  }
}
```

`SharedArrayBuffer` availability requires `Cross-Origin-Opener-Policy: same-origin`
and `Cross-Origin-Embedder-Policy: require-corp` response headers. These are not
required by Workers themselves and may not be present in all deployment
environments.

---

### Path A: SharedArrayBuffer (when available)

Arrays are backed by a shared WASM linear memory (a `SharedArrayBuffer`-backed
`WebAssembly.Memory` with `{ shared: true }`). The pool worker and all
sub-workers instantiate their WASM modules against this same memory object.

**Dispatch flow:**

```
Main thread                    Pool worker              Sub-workers
    │                              │                        │
    │  np.async.matmul(a, b)       │                        │
    │  write op descriptor         │                        │
    │  into SAB control region     │                        │
    │  Atomics.notify ──────────>  │                        │
    │                              │  divide rows           │
    │                              │  write sub-op descs    │
    │                              │  Atomics.notify ─────> │ compute
    │                              │  Atomics.wait          │
    │                              │ <── Atomics.notify ─── │ done
    │  Atomics.waitAsync           │                        │
    │ <── Atomics.notify ───────── │                        │
    │  return handle to c          │                        │
```

No array data is copied at any point. The result `c` is a view into the shared
WASM memory at the allocated offset.

**Signaling:** a small fixed-size SAB control region (separate from array data)
holds op descriptors and status words. Main thread uses `Atomics.waitAsync`
(non-blocking, allowed on main thread). Pool workers use `Atomics.wait`
(blocking, allowed in worker context).

---

### Path B: postMessage (universal fallback)

Arrays are backed by regular `ArrayBuffer`s (current default). The pool worker
has its own WASM module with its own linear memory.

**Dispatch flow:**

```
Main thread                    Pool worker
    │                              │
    │  np.async.matmul(a, b)       │
    │  structured clone a, b ────> │  (copy, ~0.2ms for 1024² f64)
    │                              │  copy into WASM memory
    │                              │  compute
    │                              │  transfer result buffer ──> │
    │ <── postMessage(result) ──── │
    │  wrap as NdArray             │
```

- **Inputs** are structured-cloned (copied). This preserves `a` and `b` on the
  main thread. For large matrices the copy is <0.5% of compute time.
- **Output** is transferred (zero-copy ownership move from worker to main
  thread). The pool worker allocates a fresh ArrayBuffer for the result.

Sub-worker parallelism in this path: the pool worker transfers row-band slices
to sub-workers, collects results, assembles the output buffer. There are no
`Atomics` in this path — coordination is pure `postMessage`/`onmessage`.

---

## Array Handles

From the main thread's perspective, the result of an async op is a standard
`NdArray`. Under the hood:

- **SAB path:** the `NdArray` wraps a typed array view into the shared WASM
  memory. `.data` is readable synchronously from the main thread without any
  round-trip.
- **postMessage path:** the `NdArray` wraps the transferred ArrayBuffer.
  `.data` is a standard typed array view.

In both cases the main-thread API is identical:

```typescript
const c = await np.async.matmul(a, b);
c.shape;    // [1024, 1024]
c.dtype;    // 'float64'
c.data[0];  // direct read, sync, no await needed
```

---

## Worker Script Loading

The pool worker script is bundled as a module worker using the
`new URL(..., import.meta.url)` pattern, which is understood by Vite, webpack 5,
and Rollup for automatic worker bundling:

```typescript
const worker = new Worker(
  new URL('../workers/async-pool-worker.js', import.meta.url),
  { type: 'module' }
);
```

No user configuration required. The worker bundle is tree-shaken to only include
the ops registered in `np.async.*`.

---

## Memory Management

### SAB path

Array data lives in a pooled allocator inside the shared WASM memory. In v1,
the allocator is a simple slab allocator with explicit free. Arrays returned
from async ops must be freed when no longer needed:

```typescript
const c = await np.async.matmul(a, b);
// ... use c ...
c.dispose(); // returns slab to allocator
```

A `FinalizationRegistry` is registered as a best-effort safety net, but explicit
`dispose()` is the contract.

### postMessage path

Array data is a transferred ArrayBuffer owned by the main thread. Standard JS
GC applies — no explicit lifecycle management needed.

---

## What Does Not Change

- The entire sync numpy-ts API (`np.matmul`, `np.fft`, etc.)
- The existing WASM kernels (Zig + Rust) — async ops run the same kernels
- The existing WASM runtime and memory management for sync ops
- Any existing user code

---

## Implementation Phases

### Phase 1 — postMessage path, single pool worker, no sub-workers
- Worker pool singleton (lazy init, configurable size)
- `np.async` namespace with `matmul`, `fft`, `ifft`
- Structured clone inputs, transfer output
- Basic op queue per worker

### Phase 2 — Sub-worker parallelism inside pool workers
- Pool workers spawn sub-worker pools
- Row-band dispatch for matmul
- `Atomics.wait`-based barrier in pool worker
- Extend to `convolve`, `linalg.*`

### Phase 3 — SAB path
- SAB detection at pool init
- Shared WASM memory compilation and instantiation
- Slab allocator for shared array storage
- `Atomics.waitAsync` signaling on main thread
- `dispose()` API

---

## Open Questions

- **Op fusion / batching:** multiple chained `await` calls each incur a
  round-trip. A future `np.async.exec(graph)` API could batch a sequence of ops
  into a single Worker round-trip. Out of scope for v1.
- **Transferable streams:** for ops that produce large outputs incrementally
  (e.g., iterative solvers), a streaming result via `ReadableStream` could avoid
  buffering the full output. Out of scope for v1.
- **WebGPU integration:** the pool worker context is also the right place to
  own a `GPUDevice` for f32 ops. WebGPU dispatch could be a fourth transport
  path sitting alongside SAB and postMessage. Out of scope for v1 but
  architecturally compatible.
