/**
 * NumPy validation: argmin/argmax strided axis-reduction WASM kernel
 * (any axis of a C-contiguous array). Exercises the int64 fast path with sizes
 * above threshold across axis=0, the last axis, and a middle axis of a 3-D
 * array, verifying tie-break (first occurrence) and NaN semantics (first NaN
 * wins). The generalized kernel takes JS-computed (before, axisSize, inner).
 */

import { beforeAll, describe, expect, it } from 'vitest';
import * as np from '../../../src/index';
import { wasmConfig } from '../../../src/common/wasm/config';
import { arraysClose, runNumPy } from './_helpers';

const DTYPES = [
  'float64',
  'float32',
  'int64',
  'uint64',
  'int32',
  'uint32',
  'int16',
  'uint16',
  'int8',
  'uint8',
] as const;

// Recursive Python-literal builder that preserves NaN (JSON turns NaN → null).
function pyLiteral(x: unknown): string {
  if (Array.isArray(x)) return '[' + x.map(pyLiteral).join(',') + ']';
  return Number.isNaN(x as number) ? 'np.nan' : String(x);
}

function compareAxis(op: 'argmin' | 'argmax', data: unknown[], axis: number, dtype: string) {
  const a = np.array(data as never, dtype as never);
  const js = op === 'argmin' ? np.argmin(a, axis) : np.argmax(a, axis);
  const py = runNumPy(`
a = np.array(${pyLiteral(data)}, dtype=np.${dtype})
result = np.${op}(a, axis=${axis})
`);
  const jsArr = (js as { toArray(): unknown }).toArray();
  expect(arraysClose(jsArr, py.value)).toBe(true);
}

// 8x5 = 40 elements (> threshold of 32) → forces the WASM path.
const BIG2D: number[][] = [];
for (let r = 0; r < 8; r++) {
  const row: number[] = [];
  for (let c = 0; c < 5; c++) row.push((r * 7 + c * 3) % 11);
  BIG2D.push(row);
}

// 4x3x4 = 48 elements (> threshold) → exercises a middle axis (axis=1).
const BIG3D: number[][][] = [];
for (let i = 0; i < 4; i++) {
  const plane: number[][] = [];
  for (let j = 0; j < 3; j++) {
    const row: number[] = [];
    for (let k = 0; k < 4; k++) row.push((i * 13 + j * 5 + k * 3) % 7);
    plane.push(row);
  }
  BIG3D.push(plane);
}

const TIES: number[][] = [
  [5, 1, 5, 1, 5],
  [5, 1, 1, 5, 5],
  [1, 5, 5, 1, 1],
  [5, 1, 5, 5, 1],
  [1, 5, 1, 1, 5],
  [5, 1, 5, 1, 5],
  [1, 5, 1, 5, 1],
  [5, 1, 5, 1, 5],
];

const NAN2D: number[][] = [
  [1, 5, 3, 2, 4],
  [NaN, 2, NaN, 6, 1],
  [3, NaN, 8, 1, NaN],
  [7, 4, 2, NaN, 3],
  [2, 6, 1, 5, 8],
  [9, 1, 4, 3, 2],
  [4, 8, 6, 7, 5],
  [6, 3, 5, 2, 9],
];

describe('argmin/argmax strided axis WASM kernel (int64 output)', () => {
  beforeAll(() => {
    wasmConfig.thresholdMultiplier = 1;
  });

  for (const op of ['argmin', 'argmax'] as const) {
    for (const dtype of DTYPES) {
      // 2-D axis=0 (before=1) and axis=1 / last axis (inner=1)
      it(`${op} ${dtype} 2-D axis=0`, () => compareAxis(op, BIG2D, 0, dtype));
      it(`${op} ${dtype} 2-D axis=1`, () => compareAxis(op, BIG2D, 1, dtype));
    }

    // 3-D: first, middle, and last axes (middle exercises before>1 && inner>1)
    for (const dtype of ['int32', 'float64', 'uint64'] as const) {
      it(`${op} ${dtype} 3-D axis=0`, () => compareAxis(op, BIG3D, 0, dtype));
      it(`${op} ${dtype} 3-D axis=1 (middle)`, () => compareAxis(op, BIG3D, 1, dtype));
      it(`${op} ${dtype} 3-D axis=2 (last)`, () => compareAxis(op, BIG3D, 2, dtype));
    }

    it(`${op} ties return first occurrence (axis=0 & axis=1)`, () => {
      compareAxis(op, TIES, 0, 'int32');
      compareAxis(op, TIES, 1, 'int32');
    });

    for (const dtype of ['float64', 'float32'] as const) {
      it(`${op} ${dtype} NaN → first NaN wins (axis=0 & axis=1)`, () => {
        compareAxis(op, NAN2D, 0, dtype);
        compareAxis(op, NAN2D, 1, dtype);
      });
    }
  }
});
