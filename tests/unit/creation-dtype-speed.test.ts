/**
 * Speed must not depend on which dtypes a process has already used.
 *
 * A fill loop that stores through the array's own typed array is one store site
 * shared by every dtype that reaches it. V8 falls back to its generic store
 * path once a site has seen more than four typed-array classes, so the fifth
 * and later dtypes ran many times slower than the first four. The loops now go
 * through a Float64Array or a native fill, which keeps them uniform. This test
 * walks enough dtypes to cross that limit and compares them against each other.
 */

import { describe, expect, it } from 'vitest';
import * as np from '../../src';

type Numeric = 'float32' | 'float16' | 'int32' | 'int16' | 'int8' | 'uint32' | 'uint16' | 'uint8';

// float64 runs first, untimed, only to occupy one of the four fast slots.
const WARMUP_DTYPE = 'float64';
const TIMED: Numeric[] = [
  'float32',
  'float16',
  'int32',
  'int16',
  'int8',
  'uint32',
  'uint16',
  'uint8',
];

function perCallMicros(make: () => { dispose(): void }): number {
  for (let i = 0; i < 1500; i++) make().dispose();
  // The minimum over many short batches: under load, a timing run only needs one
  // quiet window to show a dtype's true cost, while a real slowdown shows in all of them.
  let best = Infinity;
  for (let s = 0; s < 8; s++) {
    const t0 = performance.now();
    let calls = 0;
    while (performance.now() - t0 < 12) {
      for (let i = 0; i < 20; i++) make().dispose();
      calls += 20;
    }
    best = Math.min(best, ((performance.now() - t0) / calls) * 1000);
  }
  return best;
}

const FILLS: [string, (dtype: Numeric | typeof WARMUP_DTYPE) => { dispose(): void }][] = [
  ['arange', (d) => np.arange(0, 1000, 1, d)],
  ['tri', (d) => np.tri(100, 100, 0, d)],
  ['linspace', (d) => np.linspace(0, 100, 1000, d)],
];

describe('fill speed is independent of dtype order', () => {
  for (const [name, make] of FILLS) {
    it(`${name} takes about the same time for every dtype`, { timeout: 30000 }, () => {
      perCallMicros(() => make(WARMUP_DTYPE));
      const times = TIMED.map((d) => [d, perCallMicros(() => make(d))] as const);
      const fastest = Math.min(...times.map(([, t]) => t));
      const slow = times
        .filter(([, t]) => t > fastest * 4)
        .map(([d, t]) => `${d} ${t.toFixed(1)}us`);
      expect(
        slow,
        `${name} has dtypes more than 4x slower than the fastest (${fastest.toFixed(1)}us). ` +
          'A fill loop is probably storing through a typed array shared by every dtype.',
      ).toEqual([]);
    });
  }
});

// The same limit applied to the copy, gather and conversion loops of these ops,
// which used to index the array's own typed array. Their per-dtype cost is
// uniform, so the dtypes are compared against each other with a tighter bound.
const SIDE = 100;
const mk2d = (d: string) => np.reshape(np.arange(0, SIDE * SIDE, 1, d as Numeric), [SIDE, SIDE]);
const plainList = Array.from({ length: SIDE * SIDE }, (_, i) => i % SIDE);

const GATHERS: [string, (d: Numeric | typeof WARMUP_DTYPE) => () => { dispose(): void }][] = [
  [
    'reshape of a transposed array',
    (d) => {
      const a = np.transpose(mk2d(d));
      return () => np.reshape(a, [SIDE * SIDE]);
    },
  ],
  [
    'ravel of a transposed array',
    (d) => {
      const a = np.transpose(mk2d(d));
      return () => np.ravel(a);
    },
  ],
  [
    'repeat along an axis',
    (d) => {
      const a = mk2d(d);
      return () => np.repeat(a, 2, 0);
    },
  ],
  [
    'roll along an axis',
    (d) => {
      const a = mk2d(d);
      return () => np.roll(a, 3, 0);
    },
  ],
  [
    'take',
    (d) => {
      const a = mk2d(d);
      const i = np.arange(0, 5000, 1, 'int32');
      return () => np.take(a, i);
    },
  ],
  [
    'take along an axis',
    (d) => {
      const a = mk2d(d);
      const i = np.arange(0, 50, 1, 'int32');
      return () => np.take(a, i, 0);
    },
  ],
  [
    'take_along_axis',
    (d) => {
      const a = mk2d(d);
      const i = np.zeros([SIDE, SIDE], 'int32');
      return () => np.take_along_axis(a, i, 1);
    },
  ],
  [
    'add with broadcasting',
    (d) => {
      const a = mk2d(d);
      const b = np.reshape(np.arange(0, SIDE, 1, d), [1, SIDE]);
      return () => np.add(a, b);
    },
  ],
  [
    'astype to float64',
    (d) => {
      const a = mk2d(d);
      return () => a.astype('float64');
    },
  ],
  ['array from a list', (d) => () => np.array(plainList, d)],
];

describe('copy and conversion speed is independent of dtype order', () => {
  for (const [name, setup] of GATHERS) {
    it(`${name} takes about the same time for every dtype`, { timeout: 30000 }, () => {
      perCallMicros(setup(WARMUP_DTYPE));
      const times = TIMED.map((d) => [d, perCallMicros(setup(d))] as const);
      const fastest = Math.min(...times.map(([, t]) => t));
      const slow = times
        .filter(([, t]) => t > fastest * 1.8)
        .map(([d, t]) => `${d} ${t.toFixed(1)}us`);
      expect(
        slow,
        `${name} has dtypes more than 1.8x slower than the fastest (${fastest.toFixed(1)}us). ` +
          'A loop is probably indexing a typed array shared by every dtype.',
      ).toEqual([]);
    });
  }
});
