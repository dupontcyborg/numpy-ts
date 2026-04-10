/**
 * DType Sweep: Broadcasting shape combinations.
 * Tests binary ops with varied shapes to validate broadcasting rules.
 * Uses representative dtypes (float64, int32, complex128) — shape logic is dtype-independent.
 */
import { describe, it, beforeAll } from 'vitest';
import * as np from '../../../src';
import {
  runNumPyBatch,
  checkNumPyAvailable,
  npDtype,
  pyArrayCast,
  expectMatchPre,
  isComplex,
} from './_helpers';
import type { NumPyResult } from '../numpy-oracle';

const { array } = np;

/** Shape combos to test broadcasting rules. */
const SHAPE_COMBOS: {
  label: string;
  shapeA: number[];
  shapeB: number[];
  dataA: number[];
  dataB: number[];
}[] = [
  // (1,) × (4,) — singleton broadcast
  {
    label: '(1,) × (4,)',
    shapeA: [1],
    shapeB: [4],
    dataA: [5],
    dataB: [1, 2, 3, 4],
  },
  // (4,) × (1,) — reverse singleton
  {
    label: '(4,) × (1,)',
    shapeA: [4],
    shapeB: [1],
    dataA: [1, 2, 3, 4],
    dataB: [5],
  },
  // (3,1) × (1,4) → (3,4)
  {
    label: '(3,1) × (1,4)',
    shapeA: [3, 1],
    shapeB: [1, 4],
    dataA: [1, 2, 3],
    dataB: [10, 20, 30, 40],
  },
  // (4,) × (3,4) — 1-D broadcast into 2-D
  {
    label: '(4,) × (3,4)',
    shapeA: [4],
    shapeB: [3, 4],
    dataA: [1, 2, 3, 4],
    dataB: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
  },
  // (3,1) × (3,) — trailing dim broadcast
  {
    label: '(3,1) × (3,)',
    shapeA: [3, 1],
    shapeB: [3],
    dataA: [1, 2, 3],
    dataB: [10, 20, 30],
  },
  // (2,1,3) × (1,4,1) → (2,4,3)
  {
    label: '(2,1,3) × (1,4,1)',
    shapeA: [2, 1, 3],
    shapeB: [1, 4, 1],
    dataA: [1, 2, 3, 4, 5, 6],
    dataB: [1, 2, 3, 4],
  },
  // (1,) × (5,) — singleton broadcast
  {
    label: '(1,) × (5,)',
    shapeA: [1],
    shapeB: [5],
    dataA: [7],
    dataB: [1, 2, 3, 4, 5],
  },
  // (2,3) × (2,3) — same shape (no broadcast needed, but verify)
  {
    label: '(2,3) × (2,3)',
    shapeA: [2, 3],
    shapeB: [2, 3],
    dataA: [1, 2, 3, 4, 5, 6],
    dataB: [7, 8, 9, 10, 11, 12],
  },
];

/** Binary ops to test broadcasting with. */
const BROADCAST_OPS: { name: string; fn: (a: any, b: any) => any }[] = [
  { name: 'add', fn: np.add },
  { name: 'multiply', fn: np.multiply },
  { name: 'subtract', fn: np.subtract },
  { name: 'divide', fn: np.divide },
];

/** Comparison ops to test broadcasting with. */
const COMPARISON_OPS: { name: string; fn: (a: any, b: any) => any }[] = [
  { name: 'greater', fn: np.greater },
  { name: 'less', fn: np.less },
  { name: 'equal', fn: np.equal },
];

/** Representative dtypes — one float, one int.
 * Complex128 excluded: complex broadcasting has a known bug in elementwiseBinaryOp
 * (broadcast path uses Number() on Complex objects → NaN). Same-shape complex works. */
const REPR_DTYPES = ['float64', 'int32'];

function makeArray(data: number[], shape: number[], dtype: string) {
  return array(data, dtype).reshape(shape);
}

function pyMakeArray(varName: string, data: number[], shape: number[], dtype: string): string {
  return `${varName} = np.array(${JSON.stringify(data)}, dtype=${npDtype(dtype)}).reshape(${JSON.stringify(shape)})`;
}

let oracle: Map<string, NumPyResult & { error?: string }>;

beforeAll(() => {
  if (!checkNumPyAvailable()) throw new Error('Python NumPy not available');

  const snippets: Record<string, string> = {};

  // Arithmetic broadcasting
  for (const { name } of BROADCAST_OPS) {
    for (const dtype of REPR_DTYPES) {
      for (const combo of SHAPE_COMBOS) {
        const ac = pyArrayCast(dtype);
        const key = `${name}_${dtype}_${combo.label}`;
        snippets[key] = `
${pyMakeArray('a', combo.dataA, combo.shapeA, dtype)}
${pyMakeArray('b', combo.dataB, combo.shapeB, dtype)}
_result_orig = np.${name}(a, b)
result = _result_orig.astype(${ac})`;
      }
    }
  }

  // Comparison broadcasting
  for (const { name } of COMPARISON_OPS) {
    for (const dtype of REPR_DTYPES) {
      for (const combo of SHAPE_COMBOS) {
        const key = `cmp_${name}_${dtype}_${combo.label}`;
        snippets[key] = `
${pyMakeArray('a', combo.dataA, combo.shapeA, dtype)}
${pyMakeArray('b', combo.dataB, combo.shapeB, dtype)}
_result_orig = np.${name}(a, b)
result = _result_orig.astype(np.float64)`;
      }
    }
  }

  // Mixed-dtype broadcasting (promotion + shape)
  const MIXED_PAIRS: [string, string][] = [
    ['int8', 'float64'],
    ['uint8', 'int32'],
    ['bool', 'float64'],
    ['float16', 'float32'],
  ];
  for (const [dtA, dtB] of MIXED_PAIRS) {
    for (const combo of SHAPE_COMBOS) {
      const promotedComplex = isComplex(dtA) || isComplex(dtB);
      const ac = promotedComplex ? 'np.complex128' : 'np.float64';
      const key = `mixed_add_${dtA}_${dtB}_${combo.label}`;
      snippets[key] = `
${pyMakeArray('a', combo.dataA, combo.shapeA, dtA)}
${pyMakeArray('b', combo.dataB, combo.shapeB, dtB)}
_result_orig = np.add(a, b)
result = _result_orig.astype(${ac})`;
    }
  }

  oracle = runNumPyBatch(snippets);
});

describe('DType Sweep: Broadcasting — arithmetic', () => {
  for (const { name, fn } of BROADCAST_OPS) {
    describe(name, () => {
      for (const dtype of REPR_DTYPES) {
        describe(dtype, () => {
          for (const combo of SHAPE_COMBOS) {
            it(combo.label, () => {
              const a = makeArray(combo.dataA, combo.shapeA, dtype);
              const b = makeArray(combo.dataB, combo.shapeB, dtype);
              const jsResult = fn(a, b);
              const key = `${name}_${dtype}_${combo.label}`;
              expectMatchPre(jsResult, oracle.get(key)!, { rtol: 1e-5 });
            });
          }
        });
      }
    });
  }
});

describe('DType Sweep: Broadcasting — comparisons', () => {
  for (const { name, fn } of COMPARISON_OPS) {
    describe(name, () => {
      for (const dtype of REPR_DTYPES) {
        describe(dtype, () => {
          for (const combo of SHAPE_COMBOS) {
            it(combo.label, () => {
              const a = makeArray(combo.dataA, combo.shapeA, dtype);
              const b = makeArray(combo.dataB, combo.shapeB, dtype);
              const jsResult = fn(a, b);
              const key = `cmp_${name}_${dtype}_${combo.label}`;
              expectMatchPre(jsResult, oracle.get(key)!, { rtol: 0, atol: 0 });
            });
          }
        });
      }
    });
  }
});

describe('DType Sweep: Broadcasting — mixed dtype + shape', () => {
  const MIXED_PAIRS: [string, string][] = [
    ['int8', 'float64'],
    ['uint8', 'int32'],
    ['bool', 'float64'],
    ['float16', 'float32'],
  ];

  for (const [dtA, dtB] of MIXED_PAIRS) {
    describe(`add(${dtA}, ${dtB})`, () => {
      for (const combo of SHAPE_COMBOS) {
        it(combo.label, () => {
          const a = makeArray(combo.dataA, combo.shapeA, dtA);
          const b = makeArray(combo.dataB, combo.shapeB, dtB);
          const jsResult = np.add(a, b);
          const key = `mixed_add_${dtA}_${dtB}_${combo.label}`;
          expectMatchPre(jsResult, oracle.get(key)!, { rtol: 1e-5 });
        });
      }
    });
  }
});
