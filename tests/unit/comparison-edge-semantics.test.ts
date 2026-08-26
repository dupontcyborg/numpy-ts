/**
 * IEEE-754 edge semantics for element-wise comparisons.
 *
 * These are the cases a fast path can silently get wrong:
 *   - +0 and -0 must compare equal despite differing bit patterns
 *   - every ordered comparison against NaN must be false, and NaN != NaN
 *   - infinities must order correctly
 *
 * float16 matters most here: it has no SIMD compare in WASM, so it is widened
 * to f32 and run through that kernel. Widening is exact, but a bit-level
 * shortcut (reinterpreting as u16 and comparing sign-magnitude keys) would pass
 * ordinary values and fail precisely these. Sizes are above the 32-element
 * kernel threshold so the WASM path is what gets exercised.
 */

import { describe, expect, it } from 'vitest';
import * as np from '../../src';

const EDGE = [0, -0, 1, -1, Number.POSITIVE_INFINITY, Number.NEGATIVE_INFINITY, NaN, 0.5];
const OTHER = [-0, 0, 1, 1, Number.POSITIVE_INFINITY, Number.POSITIVE_INFINITY, NaN, 0.25];
const PAD = 56; // push past the kernel threshold

const OPS = {
  equal: (x: number, y: number) => x === y,
  not_equal: (x: number, y: number) => x !== y,
  less: (x: number, y: number) => x < y,
  less_equal: (x: number, y: number) => x <= y,
  greater: (x: number, y: number) => x > y,
  greater_equal: (x: number, y: number) => x >= y,
} as const;

describe('comparison edge semantics (±0, NaN, infinity)', () => {
  for (const dtype of ['float64', 'float32', 'float16'] as const) {
    for (const [name, ref] of Object.entries(OPS)) {
      it(`${name} (${dtype})`, () => {
        const av = [...EDGE, ...Array(PAD).fill(1)];
        const bv = [...OTHER, ...Array(PAD).fill(1)];
        const a = np.array(av, dtype);
        const b = np.array(bv, dtype);

        const got = (
          np as unknown as Record<string, (x: unknown, y: unknown) => { toArray(): number[] }>
        )[name]!(a, b).toArray();

        // Compare against the same operator applied in plain JS, which follows
        // IEEE-754 — the semantics NumPy also implements.
        const want = av.map((x, i) => (ref(x, bv[i]!) ? 1 : 0));
        expect(got).toEqual(want);
      });
    }
  }
});
