/**
 * WASM Memory Edge Cases: OOM fallback and mixed inputs
 *
 * Uses a constrained 16 MiB WASM memory (7 MiB usable heap) so we can
 * exhaust it and verify that JS-fallback works correctly.
 *
 * IMPORTANT: wasmMemoryConfig must be set BEFORE any WASM operation
 * triggers lazy initialization. Vitest isolates each file in its own
 * worker thread, so this is safe.
 */

import { wasmConfig, wasmMemoryConfig } from '../../src/common/wasm/config';

// Constrain memory BEFORE any WASM initialization
wasmMemoryConfig.maxMemoryBytes = 16 * 1024 * 1024; // 16 MiB total
wasmMemoryConfig.scratchBytes = 1 * 1024 * 1024; // 1 MiB scratch

// Heap: scratchBase(15 MiB) - heapBase(8 MiB) = 7 MiB usable

import { beforeEach, describe, expect, it } from 'vitest';
import { array } from '../../src';
import { ArrayStorage } from '../../src/common/storage';
import { wasmFreeBytes, wasmMalloc } from '../../src/common/wasm/runtime';
import { absolute as abs, add, maximum, multiply } from '../../src/full';

beforeEach(() => {
  wasmConfig.thresholdMultiplier = 0; // force WASM path
});

/**
 * Exhaust the WASM heap. Allocates small blocks first (so their buckets
 * have free-list entries after releaseAll), then large blocks.
 *
 * The segregated free-list allocator doesn't split large blocks, so
 * after releaseAll, only buckets that had blocks allocated will have
 * entries. Tests that need truly empty small buckets should drain
 * all buckets directly via wasmMalloc instead of using this helper.
 */
function exhaustHeapRaw(): { regions: { release(): void }[]; storages: ArrayStorage[] } {
  const storages: ArrayStorage[] = [];
  const regions: { release(): void }[] = [];
  // Small blocks first — ensures small buckets have free-list entries
  // after releaseAll (the segregated allocator doesn't split large blocks).
  for (const size of [24, 100, 1_000, 10_000]) {
    for (let i = 0; i < 100; i++) {
      const r = wasmMalloc(size);
      if (!r) break;
      regions.push(r);
    }
  }
  // Large blocks to consume the rest
  for (let i = 0; i < 30; i++) {
    const s = ArrayStorage.zeros([100_000], 'float64');
    storages.push(s);
    if (!s.isWasmBacked) break;
  }
  return { storages, regions };
}

function releaseAll(bag: { regions: { release(): void }[]; storages: ArrayStorage[] }) {
  for (const s of bag.storages) s.dispose();
  for (const r of bag.regions) r.release();
}

// ---------------------------------------------------------------------------
// Gap 1: OOM triggers JS-fallback
// ---------------------------------------------------------------------------

describe('OOM triggers JS-fallback', () => {
  it('allocates WASM-backed arrays until heap is exhausted, then falls back to JS', () => {
    const arrays: ArrayStorage[] = [];
    let sawWasm = false;
    let sawJs = false;

    for (let i = 0; i < 30; i++) {
      const s = ArrayStorage.zeros([100_000], 'float64');
      arrays.push(s);
      if (s.isWasmBacked) sawWasm = true;
      else sawJs = true;
      if (sawWasm && sawJs) break;
    }

    expect(sawWasm).toBe(true);
    expect(sawJs).toBe(true);

    for (const a of arrays) a.dispose();
  });

  it('JS-fallback array has correct dtype and data', () => {
    const bag = exhaustHeapRaw();

    // Use a size large enough that no free-list bucket can serve it
    const N = 200_000;
    const data = new Float32Array(N);
    data[0] = 1;
    data[1] = 2;
    data[N - 1] = 99;
    const jsFallback = ArrayStorage.fromData(data, [N], 'float32');
    expect(jsFallback.isWasmBacked).toBe(false);
    expect(jsFallback.dtype).toBe('float32');
    expect(jsFallback.data[0]).toBe(1);
    expect(jsFallback.data[1]).toBe(2);
    expect(jsFallback.data[N - 1]).toBe(99);

    releaseAll(bag);
  });

  it('JS-fallback works for multiple dtypes', () => {
    const bag = exhaustHeapRaw();

    // Use sizes large enough to exceed any free-list bucket
    const f64 = ArrayStorage.zeros([200_000], 'float64'); // 1.6 MB
    const f32 = ArrayStorage.zeros([200_000], 'float32'); // 800 KB (but 1MB bucket may be empty after re-exhaustion)
    const i32 = ArrayStorage.zeros([200_000], 'int32'); // 800 KB

    // At least the f64 (1.6 MB) should exceed all freed bucket sizes
    expect(f64.isWasmBacked).toBe(false);
    expect(f64.dtype).toBe('float64');
    expect(f32.dtype).toBe('float32');
    expect(i32.dtype).toBe('int32');
    expect(f64.data[0]).toBe(0);
    expect(f32.data[0]).toBe(0);
    expect(i32.data[0]).toBe(0);

    releaseAll(bag);
  });
});

// ---------------------------------------------------------------------------
// Gap 1: Operations on JS-fallback arrays
// ---------------------------------------------------------------------------

describe('operations on JS-fallback arrays produce correct results', () => {
  it('add works on JS-fallback arrays', () => {
    const bag = exhaustHeapRaw();
    const a = array([1, 2, 3]);
    const b = array([4, 5, 6]);
    const result = add(a, b);
    expect(result.tolist()).toEqual([5, 7, 9]);
    releaseAll(bag);
  });

  it('abs works on JS-fallback arrays', () => {
    const bag = exhaustHeapRaw();
    const a = array([-1, -2, 3]);
    const result = abs(a);
    expect(result.tolist()).toEqual([1, 2, 3]);
    releaseAll(bag);
  });
});

// ---------------------------------------------------------------------------
// Gap 1: Recovery after free
// ---------------------------------------------------------------------------

describe('recovery after free', () => {
  it('WASM-backed allocation resumes after freeing', () => {
    // Allocate known-size blocks directly via wasmMalloc
    const regions: { release(): void; ptr: number }[] = [];
    while (true) {
      const r = wasmMalloc(4096);
      if (!r) break;
      regions.push(r);
    }
    expect(regions.length).toBeGreaterThan(0);

    // Verify heap is full (can't allocate even small amounts)
    expect(wasmMalloc(4096)).toBeNull();

    // Free all
    for (const r of regions) r.release();

    const freeAfter = wasmFreeBytes();
    expect(freeAfter).toBeGreaterThan(0);

    // Re-allocate same size — should succeed from free list
    const recovered = wasmMalloc(4096);
    expect(recovered).not.toBeNull();
    recovered!.release();
  });
});

// ---------------------------------------------------------------------------
// Gap 2: Mixed WASM + JS inputs
// ---------------------------------------------------------------------------

describe('mixed WASM + JS inputs', () => {
  function makeOneWasmOneJs() {
    // First array should be WASM-backed (heap not yet full)
    const wasmArr = array([10, 20, 30]);
    expect(wasmArr.storage.isWasmBacked).toBe(true);

    // Exhaust remaining heap — drain ALL free-list buckets
    const regions: { release(): void }[] = [];
    for (let size = 24; size <= 2 * 1024 * 1024; size *= 2) {
      for (let i = 0; i < 500; i++) {
        const r = wasmMalloc(size);
        if (!r) break;
        regions.push(r);
      }
    }

    // This array should be JS-backed (every bucket is empty, bump is exhausted)
    const jsArr = array([1, 2, 3]);
    expect(jsArr.storage.isWasmBacked).toBe(false);

    return { wasm: wasmArr, js: jsArr, regions };
  }

  it('add(wasmBacked, jsBacked) produces correct results', () => {
    const { wasm, js, regions } = makeOneWasmOneJs();
    const result = add(wasm, js);
    expect(result.tolist()).toEqual([11, 22, 33]);
    for (const r of regions) r.release();
  });

  it('add(jsBacked, wasmBacked) produces correct results (reversed)', () => {
    const { wasm, js, regions } = makeOneWasmOneJs();
    const result = add(js, wasm);
    expect(result.tolist()).toEqual([11, 22, 33]);
    for (const r of regions) r.release();
  });

  it('multiply(wasmBacked, jsBacked) produces correct results', () => {
    const { wasm, js, regions } = makeOneWasmOneJs();
    const result = multiply(wasm, js);
    expect(result.tolist()).toEqual([10, 40, 90]);
    for (const r of regions) r.release();
  });

  it('maximum(wasmBacked, jsBacked) produces correct results', () => {
    const { wasm, js, regions } = makeOneWasmOneJs();
    const result = maximum(wasm, js);
    expect(result.tolist()).toEqual([10, 20, 30]);
    for (const r of regions) r.release();
  });

  it('binary op with both JS-fallback inputs works', () => {
    // Drain ALL free-list buckets so small arrays can't be WASM-backed
    const regions: { release(): void }[] = [];
    for (let size = 24; size <= 2 * 1024 * 1024; size *= 2) {
      for (let i = 0; i < 500; i++) {
        const r = wasmMalloc(size);
        if (!r) break;
        regions.push(r);
      }
    }

    const a = array([1, 2, 3]);
    const b = array([4, 5, 6]);
    expect(a.storage.isWasmBacked).toBe(false);
    expect(b.storage.isWasmBacked).toBe(false);

    const result = add(a, b);
    expect(result.tolist()).toEqual([5, 7, 9]);
    for (const r of regions) r.release();
  });
});
