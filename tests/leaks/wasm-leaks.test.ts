/**
 * WASM Memory Leak Tests
 *
 * Validates that WASM operations properly free intermediate memory.
 * Each spec runs N iterations with dispose() on results; heap usage
 * should remain stable (no net leak beyond allocator overhead).
 *
 * Run standalone: npm run test:leaks
 * Run with suite:  npm test (included via 'leaks' project)
 */

import { describe, it, expect } from 'vitest';
import { wasmConfig } from '../../src/common/wasm/config';
import { wasmFreeBytes } from '../../src/common/wasm/runtime';
import { getBenchmarkSpecs } from '../../benchmarks/src/specs';
import { setupArrays, executeOperation } from '../../benchmarks/src/bench-utils';

// --- Config ---
// Use LEAK_SIZE_SCALE env var to override size scale (e.g. "large" for stress testing).
// Default is 'default' (100×100) for CI speed.
const SIZE_SCALE = (process.env.LEAK_SIZE_SCALE || 'default') as 'small' | 'default' | 'large';
const ITERS = 10; // enough to detect consistent leaks, fast enough for CI
const MAX_LEAK_PER_ITER = 0; // zero tolerance — any leak is a bug

// Force WASM for all sizes
wasmConfig.thresholdMultiplier = 0;

function releaseResult(result: unknown): void {
  if (result == null) return;
  if (typeof (result as any).dispose === 'function') {
    (result as any).dispose();
  } else if (result instanceof Map) {
    for (const val of result.values()) releaseResult(val);
  } else if (Array.isArray(result)) {
    for (const item of result) (item as any)?.dispose?.();
  } else if (
    typeof result === 'object' &&
    !(ArrayBuffer.isView(result) || result instanceof ArrayBuffer)
  ) {
    for (const val of Object.values(result as Record<string, unknown>)) {
      releaseResult(val);
    }
  }
}

// --- Test generation ---
const allSpecs = getBenchmarkSpecs('full', SIZE_SCALE);
const categories = [...new Set(allSpecs.map((s) => s.category))].sort();

for (const category of categories) {
  const categorySpecs = allSpecs.filter((s) => s.category === category);

  describe(`WASM memory leaks: ${category}`, () => {
    for (const spec of categorySpecs) {
      it(`${spec.name} (${ITERS} iters)`, () => {
        let arrays: Record<string, any>;
        try {
          arrays = setupArrays(spec.setup, spec.operation);
        } catch {
          return; // skip specs with unsupported setup
        }

        // Warmup — ensure WASM modules loaded, skip if operation errors
        try {
          for (let i = 0; i < 3; i++) {
            releaseResult(executeOperation(spec.operation, arrays));
          }
        } catch {
          for (const val of Object.values(arrays)) (val as any)?.dispose?.();
          return;
        }

        const freeBefore = wasmFreeBytes();

        for (let i = 0; i < ITERS; i++) {
          try {
            releaseResult(executeOperation(spec.operation, arrays));
          } catch {
            break;
          }
        }

        const freeAfter = wasmFreeBytes();
        const totalLeak = freeBefore - freeAfter;
        const leakPerIter = totalLeak / ITERS;

        // Release setup arrays
        for (const val of Object.values(arrays)) {
          (val as any)?.dispose?.();
        }

        expect(
          leakPerIter,
          `Leaks ${leakPerIter.toFixed(0)} bytes/iter (${(totalLeak / 1024).toFixed(1)} KB total over ${ITERS} iters)`
        ).toBeLessThanOrEqual(MAX_LEAK_PER_ITER);
      });
    }
  });
}
