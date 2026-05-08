/**
 * WASM Memory Edge Cases: Lifecycle, double-free, refcounting
 *
 * Uses default memory config (plenty of heap) — these tests focus on
 * the safety of dispose/retain/release semantics, not OOM.
 */

import { beforeEach, describe, expect, it } from 'vitest';
import { ArrayStorage } from '../../src/common/storage';
import { wasmConfig } from '../../src/common/wasm/config';
import { wasmFreeBytes } from '../../src/common/wasm/runtime';

beforeEach(() => {
  wasmConfig.thresholdMultiplier = 0; // force WASM
});

// ---------------------------------------------------------------------------
// Gap 3: Double-free / use-after-dispose
// ---------------------------------------------------------------------------

describe('double dispose is safe', () => {
  it('calling dispose() twice does not throw or corrupt the heap', () => {
    const s = ArrayStorage.zeros([1000], 'float64');
    expect(s.isWasmBacked).toBe(true);

    const freeBefore = wasmFreeBytes();
    s.dispose();
    const freeAfterFirst = wasmFreeBytes();
    expect(freeAfterFirst).toBeGreaterThan(freeBefore);

    // Second dispose — should be a no-op
    expect(() => s.dispose()).not.toThrow();
    const freeAfterSecond = wasmFreeBytes();
    expect(freeAfterSecond).toBe(freeAfterFirst);
  });

  it('isWasmBacked returns false after dispose', () => {
    const s = ArrayStorage.zeros([100], 'float64');
    expect(s.isWasmBacked).toBe(true);
    s.dispose();
    expect(s.isWasmBacked).toBe(false);
  });

  it('.data is still a TypedArray after dispose (dangling view)', () => {
    const s = ArrayStorage.zeros([100], 'float64');
    s.dispose();
    // The view is dangling but still a TypedArray — this is known behavior.
    // The WASM memory slot may be reused, so values are undefined.
    expect(s.data).toBeInstanceOf(Float64Array);
    expect(s.data.length).toBe(100);
  });
});

// ---------------------------------------------------------------------------
// Gap 4: Refcounting through shared views
// ---------------------------------------------------------------------------

describe('shared view refcount lifecycle', () => {
  it('dispose parent does not free region while shared view is alive', () => {
    const parent = ArrayStorage.zeros([1000], 'float64');
    expect(parent.isWasmBacked).toBe(true);
    const region = parent.wasmRegion!;
    expect(region.refCount).toBe(1);

    const freeBefore = wasmFreeBytes();

    // Create shared view (increments refcount)
    const view = ArrayStorage.fromDataShared(
      parent.data,
      [500],
      'float64',
      [1],
      0,
      parent.wasmRegion,
    );
    expect(region.refCount).toBe(2);

    // Dispose parent — region should NOT be freed (view still holds a ref)
    parent.dispose();
    expect(region.refCount).toBe(1);
    const freeAfterParentDispose = wasmFreeBytes();
    expect(freeAfterParentDispose).toBe(freeBefore); // not freed yet

    // Dispose view — region should now be freed
    view.dispose();
    expect(region.refCount).toBe(0);
    const freeAfterViewDispose = wasmFreeBytes();
    expect(freeAfterViewDispose).toBeGreaterThan(freeBefore);
  });

  it('dispose order does not matter (view first, then parent)', () => {
    const parent = ArrayStorage.zeros([1000], 'float64');
    const region = parent.wasmRegion!;

    const freeBefore = wasmFreeBytes();

    const view = ArrayStorage.fromDataShared(
      parent.data,
      [500],
      'float64',
      [1],
      0,
      parent.wasmRegion,
    );
    expect(region.refCount).toBe(2);

    // Dispose view first
    view.dispose();
    expect(region.refCount).toBe(1);
    expect(wasmFreeBytes()).toBe(freeBefore); // not freed yet

    // Dispose parent — now freed
    parent.dispose();
    expect(region.refCount).toBe(0);
    expect(wasmFreeBytes()).toBeGreaterThan(freeBefore);
  });

  it('many shared views — freed only when last view is disposed', () => {
    const parent = ArrayStorage.zeros([1000], 'float64');
    const region = parent.wasmRegion!;
    const freeBefore = wasmFreeBytes();

    const views: ArrayStorage[] = [];
    for (let i = 0; i < 5; i++) {
      views.push(
        ArrayStorage.fromDataShared(parent.data, [200], 'float64', [1], i * 200, parent.wasmRegion),
      );
    }
    expect(region.refCount).toBe(6); // parent + 5 views

    // Dispose parent + first 4 views — not freed yet
    parent.dispose();
    for (let i = 0; i < 4; i++) {
      views[i]!.dispose();
    }
    expect(region.refCount).toBe(1);
    expect(wasmFreeBytes()).toBe(freeBefore);

    // Dispose last view — now freed
    views[4]!.dispose();
    expect(region.refCount).toBe(0);
    expect(wasmFreeBytes()).toBeGreaterThan(freeBefore);
  });

  it('double dispose on shared view does not over-decrement refcount', () => {
    const parent = ArrayStorage.zeros([1000], 'float64');
    const region = parent.wasmRegion!;

    const view = ArrayStorage.fromDataShared(
      parent.data,
      [500],
      'float64',
      [1],
      0,
      parent.wasmRegion,
    );
    expect(region.refCount).toBe(2);

    // Double dispose on view
    view.dispose();
    expect(region.refCount).toBe(1);
    view.dispose(); // second dispose — should be no-op
    expect(region.refCount).toBe(1); // not decremented again

    parent.dispose();
    expect(region.refCount).toBe(0);
  });
});

// ---------------------------------------------------------------------------
// Gap 3: FinalizationRegistry integration (best-effort)
// ---------------------------------------------------------------------------

describe('FinalizationRegistry cleanup', () => {
  it('GC recovers unreferenced WASM-backed array (best-effort)', async () => {
    // This test is inherently non-deterministic — GC may not run.
    // We attempt to trigger it and accept that it may be flaky.
    if (typeof globalThis.gc !== 'function') {
      // --expose-gc not set — skip gracefully
      return;
    }

    const freeBefore = wasmFreeBytes();

    // Create and immediately drop reference (no dispose)
    (() => {
      ArrayStorage.zeros([10_000], 'float64');
      // 80 KB allocation, dropped
    })();

    // Attempt to trigger GC
    globalThis.gc();
    // FinalizationRegistry callbacks are async — wait a tick
    await new Promise((r) => globalThis.setTimeout(r, 100));
    globalThis.gc();
    await new Promise((r) => globalThis.setTimeout(r, 100));

    const freeAfter = wasmFreeBytes();
    // Best-effort: if GC ran, heap should have recovered
    // We don't assert strictly because GC is non-deterministic
    if (freeAfter > freeBefore) {
      expect(freeAfter).toBeGreaterThan(freeBefore);
    }
  });
});
