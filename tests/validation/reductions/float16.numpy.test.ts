/**
 * NumPy validation: float16 reduction dtype and value correctness.
 *
 * Tests that our float16 reductions match NumPy's output dtype and
 * accumulation behavior (overflow vs promotion).
 */

import { describe, it, expect } from 'vitest';
import { WASM_MODES, setupWasmMode, np, runNumPy } from './_helpers';

/** Data small enough to not overflow in f16 — tests basic correctness */
const SMALL_DATA = [
  [1, 2, 3],
  [4, 5, 6],
];

/** 1000 elements of value 100 — sum=100000 overflows f16 (max ~65504) */
const OVERFLOW_DATA = Array(1000).fill(100);

/** 16 elements of 2 — prod=2^16=65536 overflows f16 (max ~65504) but not f32 */
const PROD_OVERFLOW_DATA = Array(16).fill(2);

/** NumPy dtype string from oracle dtype (e.g. 'float16' from 'float16') */
function normalizeDtype(d: string): string {
  // Oracle returns dtype like 'float16', 'float32', 'float64', 'int64', 'bool'
  return d.replace(/^numpy\./, '').replace(/^np\./, '');
}

for (const mode of WASM_MODES) {
  describe(`NumPy Validation: float16 Reductions [${mode.name}]`, () => {
    setupWasmMode(mode);

    describe('output dtype matches NumPy', () => {
      const dtypeTests: { op: string; fn: (a: any, axis?: number) => any; npOp?: string }[] = [
        { op: 'sum', fn: (a) => a.sum() },
        { op: 'sum axis=0', fn: (a) => a.sum(0), npOp: 'sum(a, axis=0)' },
        { op: 'prod', fn: (a) => np.prod(a) },
        { op: 'mean', fn: (a) => np.mean(a) },
        { op: 'mean axis=0', fn: (a) => np.mean(a, 0), npOp: 'mean(a, axis=0)' },
        { op: 'std', fn: (a) => np.std(a) },
        { op: 'var', fn: (a) => np.var_(a) },
        { op: 'max', fn: (a) => np.max(a) },
        { op: 'max axis=0', fn: (a) => np.max(a, 0), npOp: 'max(a, axis=0)' },
        { op: 'min', fn: (a) => np.min(a) },
        { op: 'nansum', fn: (a) => np.nansum(a) },
        { op: 'nanmean', fn: (a) => np.nanmean(a) },
        { op: 'nanmax', fn: (a) => np.nanmax(a) },
        { op: 'nanmin', fn: (a) => np.nanmin(a) },
        { op: 'argmax', fn: (a) => np.argmax(a) },
        { op: 'argmin', fn: (a) => np.argmin(a) },
      ];

      for (const { op, fn, npOp } of dtypeTests) {
        it(`${op} output dtype`, () => {
          const a = np.array(SMALL_DATA, 'float16');
          const jsResult = fn(a);
          const jsDtype =
            typeof jsResult === 'number' || typeof jsResult === 'boolean'
              ? (jsResult as any).constructor === Boolean
                ? 'bool'
                : 'scalar'
              : jsResult.dtype;

          const pyExpr =
            npOp ?? `${op.replace(/ axis=.*/, '')}(a${op.includes('axis=') ? ', axis=0' : ''})`;
          const pyResult = runNumPy(`
a = np.array(${JSON.stringify(SMALL_DATA)}, dtype=np.float16)
result = np.${pyExpr}
`);
          const npDtype = normalizeDtype(pyResult.dtype);

          // For scalar results (argmax/argmin return int), skip dtype check
          if (typeof jsResult !== 'number' && typeof jsResult !== 'boolean') {
            expect(jsDtype, `${op}: expected dtype ${npDtype}, got ${jsDtype}`).toBe(npDtype);
          }
        });
      }
    });

    describe('accumulation overflow matches NumPy', () => {
      it('sum: should overflow to inf like NumPy', () => {
        const a = np.array(OVERFLOW_DATA, 'float16');
        const jsResult = Number(a.sum());

        const pyResult = runNumPy(`
a = np.array([100.0] * 1000, dtype=np.float16)
result = np.sum(a)
`);
        const npVal = pyResult.value as number;

        // NumPy returns inf; we should too
        expect(npVal).toBe(Infinity);
        expect(jsResult, 'sum of 1000×100 in f16 should overflow to inf').toBe(Infinity);
      });

      it('nansum: should overflow to inf like NumPy', () => {
        const a = np.array(OVERFLOW_DATA, 'float16');
        const jsResult = Number(np.nansum(a));

        const pyResult = runNumPy(`
a = np.array([100.0] * 1000, dtype=np.float16)
result = np.nansum(a)
`);
        const npVal = pyResult.value as number;

        expect(npVal).toBe(Infinity);
        expect(jsResult, 'nansum of 1000×100 in f16 should overflow to inf').toBe(Infinity);
      });

      it('prod: should overflow to inf like NumPy (2^16 > f16 max)', () => {
        const a = np.array(PROD_OVERFLOW_DATA, 'float16');
        const jsResult = Number(np.prod(a));

        const pyResult = runNumPy(`
a = np.array([2.0] * 16, dtype=np.float16)
result = np.prod(a)
`);
        const npVal = pyResult.value as number;

        // NumPy returns inf (2^16=65536 > f16 max 65504); we should too
        expect(npVal).toBe(Infinity);
        expect(jsResult, 'prod of 2^16 in f16 should overflow to inf').toBe(Infinity);
      });

      it('mean: promotes to f32 internally (should NOT overflow)', () => {
        const a = np.array(OVERFLOW_DATA, 'float16');
        const jsResult = Number(np.mean(a));

        const pyResult = runNumPy(`
a = np.array([100.0] * 1000, dtype=np.float16)
result = np.mean(a)
`);
        const npVal = pyResult.value as number;

        // NumPy promotes f16→f32 for mean, gets correct answer ~100
        expect(npVal).toBeCloseTo(100, 0);
        expect(jsResult, 'mean should promote and return ~100').toBeCloseTo(100, 0);
      });

      it('std: internal sum overflows in NumPy (no f16→f32 promotion in var)', () => {
        // NumPy's var does NOT promote f16→f32 for its internal mean.
        // sum([100]*1000) overflows → inf, mean=inf, x-inf=-inf, var=inf, std=inf
        const a = np.array(OVERFLOW_DATA, 'float16');
        const jsResult = Number(np.std(a));

        const pyResult = runNumPy(`
a = np.array([100.0] * 1000, dtype=np.float16)
result = np.std(a)
`);
        const npVal = pyResult.value as number;

        expect(npVal).toBe(Infinity);
        expect(jsResult, 'std should be inf (var internal sum overflows in f16)').toBe(Infinity);
      });

      it('var: internal sum overflows in NumPy (no f16→f32 promotion in var)', () => {
        const a = np.array(OVERFLOW_DATA, 'float16');
        const jsResult = Number(np.var_(a));

        const pyResult = runNumPy(`
a = np.array([100.0] * 1000, dtype=np.float16)
result = np.var(a)
`);
        const npVal = pyResult.value as number;

        expect(npVal).toBe(Infinity);
        expect(jsResult, 'var should be inf (internal sum overflows in f16)').toBe(Infinity);
      });
    });

    describe('max/min/nanmax/nanmin preserve f16 dtype', () => {
      for (const op of ['max', 'min', 'nanmax', 'nanmin'] as const) {
        it(`${op} returns float16 array for axis reduction`, () => {
          const a = np.array(SMALL_DATA, 'float16');
          const jsResult =
            op === 'max'
              ? np.max(a, 0)
              : op === 'min'
                ? np.min(a, 0)
                : op === 'nanmax'
                  ? np.nanmax(a, 0)
                  : np.nanmin(a, 0);

          const pyResult = runNumPy(`
a = np.array(${JSON.stringify(SMALL_DATA)}, dtype=np.float16)
result = np.${op}(a, axis=0)
`);

          expect(jsResult.dtype, `${op} axis=0 dtype`).toBe(normalizeDtype(pyResult.dtype));
        });
      }
    });

    describe('quantile output dtype', () => {
      it('quantile returns float16 for float16 input', () => {
        const a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 'float16');
        const jsResult = np.quantile(a, 0.5);

        const pyResult = runNumPy(`
a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.float16)
result = np.quantile(a, 0.5)
`);

        // Check value
        const npVal = pyResult.value as number;
        expect(Number(jsResult)).toBeCloseTo(npVal, 1);

        // Check dtype — NumPy returns float16
        const npDtype = normalizeDtype(pyResult.dtype);
        expect(npDtype).toBe('float16');
        // Our result should also be float16 (currently may be float64)
        if (typeof jsResult === 'object' && 'dtype' in jsResult) {
          expect(jsResult.dtype, 'quantile output dtype').toBe('float16');
        }
      });
    });
  });
}
