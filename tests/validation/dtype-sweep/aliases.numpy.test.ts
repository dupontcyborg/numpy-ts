/**
 * DType Sweep: Alias functions.
 * Tests alias functions across ALL dtypes, validated against NumPy.
 * Uses batched oracle — all Python computations run in a single subprocess.
 */
import { describe, it, beforeAll } from 'vitest';
import * as np from '../../../src';
import {
  ALL_DTYPES,
  runNumPyBatch,
  checkNumPyAvailable,
  npDtype,
  pyArrayCast,
  pyScalarCast,
  expectBothRejectPre,
  expectMatchPre,
  scalarClose,
} from './_helpers';
import type { NumPyResult } from '../numpy-oracle';

const { array } = np;

let oracle: Map<string, NumPyResult & { error?: string }>;

beforeAll(() => {
  if (!checkNumPyAvailable()) throw new Error('Python NumPy not available');

  const snippets: Record<string, string> = {};

  // --- Unary aliases ---
  for (const dtype of ALL_DTYPES) {
    const ac = pyArrayCast(dtype);
    const data = dtype === 'bool' ? [1, 0, 1, 1] : [0.5, 1.0, 1.5, 2.0];
    const trigData = dtype === 'bool' ? [1, 0, 1, 0] : [0.1, 0.2, 0.3, 0.4];
    const acoshData = dtype === 'bool' ? [1, 0, 1, 0] : [1.1, 1.5, 2.0, 3.0];

    snippets[`abs_${dtype}`] = `
a = np.array(${JSON.stringify(data)}, dtype=${npDtype(dtype)})
_result_orig = np.abs(a)
result = _result_orig.astype(${ac})`;

    snippets[`round_${dtype}`] = `
a = np.array(${JSON.stringify(data)}, dtype=${npDtype(dtype)})
_result_orig = np.round(a)
result = _result_orig.astype(${ac})`;

    snippets[`conjugate_${dtype}`] = `
a = np.array(${JSON.stringify(data)}, dtype=${npDtype(dtype)})
_result_orig = np.conjugate(a)
result = _result_orig.astype(${ac})`;

    // Trig aliases
    snippets[`asin_${dtype}`] = `
a = np.array(${JSON.stringify(trigData)}, dtype=${npDtype(dtype)})
_result_orig = np.arcsin(a)
result = _result_orig.astype(${ac})`;

    snippets[`acos_${dtype}`] = `
a = np.array(${JSON.stringify(trigData)}, dtype=${npDtype(dtype)})
_result_orig = np.arccos(a)
result = _result_orig.astype(${ac})`;

    snippets[`atan_${dtype}`] = `
a = np.array(${JSON.stringify(trigData)}, dtype=${npDtype(dtype)})
_result_orig = np.arctan(a)
result = _result_orig.astype(${ac})`;

    snippets[`asinh_${dtype}`] = `
a = np.array(${JSON.stringify(trigData)}, dtype=${npDtype(dtype)})
_result_orig = np.arcsinh(a)
result = _result_orig.astype(${ac})`;

    snippets[`acosh_${dtype}`] = `
a = np.array(${JSON.stringify(acoshData)}, dtype=${npDtype(dtype)})
_result_orig = np.arccosh(a)
result = _result_orig.astype(${ac})`;

    snippets[`atanh_${dtype}`] = `
a = np.array(${JSON.stringify(trigData)}, dtype=${npDtype(dtype)})
_result_orig = np.arctanh(a)
result = _result_orig.astype(${ac})`;
  }

  // --- Binary aliases ---
  for (const dtype of ALL_DTYPES) {
    const ac = pyArrayCast(dtype);
    const data1 = dtype === 'bool' ? [1, 1, 0, 1] : [6, 7, 8, 9];
    const data2 = dtype === 'bool' ? [1, 0, 1, 0] : [1, 2, 3, 4];

    snippets[`pow_${dtype}`] = `
a = np.array(${JSON.stringify(data1)}, dtype=${npDtype(dtype)})
b = np.array(${JSON.stringify(data2)}, dtype=${npDtype(dtype)})
_result_orig = np.power(a, b)
result = _result_orig.astype(${ac})`;

    snippets[`true_divide_${dtype}`] = `
a = np.array(${JSON.stringify(data1)}, dtype=${npDtype(dtype)})
b = np.array(${JSON.stringify(data2)}, dtype=${npDtype(dtype)})
_result_orig = np.true_divide(a, b)
result = _result_orig.astype(${ac})`;

    snippets[`atan2_${dtype}`] = `
a = np.array(${JSON.stringify(data1)}, dtype=${npDtype(dtype)})
b = np.array(${JSON.stringify(data2)}, dtype=${npDtype(dtype)})
_result_orig = np.arctan2(a, b)
result = _result_orig.astype(${ac})`;
  }

  // --- Bitwise aliases ---
  for (const dtype of ALL_DTYPES) {
    const ac = pyArrayCast(dtype);
    const data1 = dtype === 'bool' ? [1, 1, 0, 1] : [6, 7, 8, 9];
    const data2 = dtype === 'bool' ? [1, 0, 1, 0] : [1, 2, 3, 4];

    snippets[`bitwise_invert_${dtype}`] = `
a = np.array(${JSON.stringify(data1)}, dtype=${npDtype(dtype)})
_result_orig = np.invert(a)
result = _result_orig.astype(${ac})`;

    snippets[`bitwise_left_shift_${dtype}`] = `
a = np.array(${JSON.stringify(data1)}, dtype=${npDtype(dtype)})
b = np.array(${JSON.stringify(data2)}, dtype=${npDtype(dtype)})
_result_orig = np.left_shift(a, b)
result = _result_orig.astype(${ac})`;

    snippets[`bitwise_right_shift_${dtype}`] = `
a = np.array(${JSON.stringify(data1)}, dtype=${npDtype(dtype)})
b = np.array(${JSON.stringify(data2)}, dtype=${npDtype(dtype)})
_result_orig = np.right_shift(a, b)
result = _result_orig.astype(${ac})`;
  }

  // --- Reduction aliases (scalar output) ---
  for (const dtype of ALL_DTYPES) {
    const sc = pyScalarCast(dtype);
    const data = dtype === 'bool' ? [1, 0, 1, 1, 0] : [3, 1, 4, 1, 5];

    snippets[`amax_${dtype}`] = `
a = np.array(${JSON.stringify(data)}, dtype=${npDtype(dtype)})
result = ${sc}(np.amax(a))`;

    snippets[`amin_${dtype}`] = `
a = np.array(${JSON.stringify(data)}, dtype=${npDtype(dtype)})
result = ${sc}(np.amin(a))`;

    snippets[`var_${dtype}`] = `
a = np.array(${JSON.stringify(data)}, dtype=${npDtype(dtype)})
result = ${sc}(np.var(a))`;
  }

  // --- Cumulative aliases (array output) ---
  for (const dtype of ALL_DTYPES) {
    const ac = pyArrayCast(dtype);
    const data = dtype === 'bool' ? [1, 0, 1, 1] : [1, 2, 3, 4];

    snippets[`cumulative_sum_${dtype}`] = `
a = np.array(${JSON.stringify(data)}, dtype=${npDtype(dtype)})
_result_orig = np.cumsum(a)
result = _result_orig.astype(${ac})`;

    snippets[`cumulative_prod_${dtype}`] = `
a = np.array(${JSON.stringify(data)}, dtype=${npDtype(dtype)})
_result_orig = np.cumprod(a)
result = _result_orig.astype(${ac})`;

    snippets[`nancumsum_${dtype}`] = `
a = np.array(${JSON.stringify(data)}, dtype=${npDtype(dtype)})
_result_orig = np.nancumsum(a)
result = _result_orig.astype(${ac})`;

    snippets[`nancumprod_${dtype}`] = `
a = np.array(${JSON.stringify(data)}, dtype=${npDtype(dtype)})
_result_orig = np.nancumprod(a)
result = _result_orig.astype(${ac})`;
  }

  // --- Stacking/shape aliases (array output) ---
  for (const dtype of ALL_DTYPES) {
    const ac = pyArrayCast(dtype);
    const data = dtype === 'bool' ? [1, 0, 1, 1] : [1, 2, 3, 4];

    snippets[`row_stack_${dtype}`] = `
a = np.array(${JSON.stringify(data)}, dtype=${npDtype(dtype)})
_result_orig = np.row_stack([a, a])
result = _result_orig.astype(${ac})`;

    snippets[`concat_${dtype}`] = `
a = np.array(${JSON.stringify(data)}, dtype=${npDtype(dtype)})
_result_orig = np.concatenate([a, a])
result = _result_orig.astype(${ac})`;

    snippets[`delete_${dtype}`] = `
a = np.array(${JSON.stringify(data)}, dtype=${npDtype(dtype)})
_result_orig = np.delete(a, 1)
result = _result_orig.astype(${ac})`;
  }

  oracle = runNumPyBatch(snippets);
});

// Tolerances
function getTols(dtype: string) {
  const isVeryLow = dtype === 'float16';
  const isLow = dtype === 'float32' || dtype === 'complex64';
  const rtol = isVeryLow ? 5e-2 : isLow ? 1e-2 : 1e-3;
  const atol = isVeryLow ? 1e-2 : isLow ? 1e-5 : 1e-8;
  return { rtol, atol };
}

describe('DType Sweep: Unary aliases', () => {
  const unaryOps: {
    name: string;
    pyName: string;
    fn: (a: any) => any;
    getData: (dtype: string) => number[];
  }[] = [
    {
      name: 'abs',
      pyName: 'abs',
      fn: np.abs,
      getData: (dtype) => (dtype === 'bool' ? [1, 0, 1, 1] : [0.5, 1.0, 1.5, 2.0]),
    },
    {
      name: 'round',
      pyName: 'round',
      fn: (a: any) => np.round(a),
      getData: (dtype) => (dtype === 'bool' ? [1, 0, 1, 1] : [0.5, 1.0, 1.5, 2.0]),
    },
    {
      name: 'conjugate',
      pyName: 'conjugate',
      fn: np.conjugate,
      getData: (dtype) => (dtype === 'bool' ? [1, 0, 1, 1] : [0.5, 1.0, 1.5, 2.0]),
    },
    {
      name: 'asin',
      pyName: 'asin',
      fn: np.asin,
      getData: (dtype) => (dtype === 'bool' ? [1, 0, 1, 0] : [0.1, 0.2, 0.3, 0.4]),
    },
    {
      name: 'acos',
      pyName: 'acos',
      fn: np.acos,
      getData: (dtype) => (dtype === 'bool' ? [1, 0, 1, 0] : [0.1, 0.2, 0.3, 0.4]),
    },
    {
      name: 'atan',
      pyName: 'atan',
      fn: np.atan,
      getData: (dtype) => (dtype === 'bool' ? [1, 0, 1, 0] : [0.1, 0.2, 0.3, 0.4]),
    },
    {
      name: 'asinh',
      pyName: 'asinh',
      fn: np.asinh,
      getData: (dtype) => (dtype === 'bool' ? [1, 0, 1, 0] : [0.1, 0.2, 0.3, 0.4]),
    },
    {
      name: 'acosh',
      pyName: 'acosh',
      fn: np.acosh,
      getData: (dtype) => (dtype === 'bool' ? [1, 0, 1, 0] : [1.1, 1.5, 2.0, 3.0]),
    },
    {
      name: 'atanh',
      pyName: 'atanh',
      fn: np.atanh,
      getData: (dtype) => (dtype === 'bool' ? [1, 0, 1, 0] : [0.1, 0.2, 0.3, 0.4]),
    },
  ];

  for (const { name, fn, getData } of unaryOps) {
    describe(name, () => {
      for (const dtype of ALL_DTYPES) {
        it(`${dtype}`, () => {
          const data = getData(dtype);
          const a = array(data, dtype);
          const py = oracle.get(`${name}_${dtype}`)!;
          const r = expectBothRejectPre(
            `${name} may not be supported for ${dtype}`,
            () => fn(a),
            py
          );
          if (r === 'both-reject') return;
          const jsResult = fn(a);
          const { rtol, atol } = getTols(dtype);
          expectMatchPre(jsResult, py, { rtol, atol });
        });
      }
    });
  }
});

describe('DType Sweep: Binary aliases', () => {
  const binaryOps: { name: string; fn: (a: any, b: any) => any; rejectComplex?: boolean }[] = [
    { name: 'pow', fn: np.pow },
    { name: 'true_divide', fn: np.true_divide },
    { name: 'atan2', fn: np.atan2, rejectComplex: true },
  ];

  for (const { name, fn } of binaryOps) {
    describe(name, () => {
      for (const dtype of ALL_DTYPES) {
        it(`${dtype}`, () => {
          const data1 = dtype === 'bool' ? [1, 1, 0, 1] : [6, 7, 8, 9];
          const data2 = dtype === 'bool' ? [1, 0, 1, 0] : [1, 2, 3, 4];
          const py = oracle.get(`${name}_${dtype}`)!;
          const r = expectBothRejectPre(
            `${name} may not be supported for ${dtype}`,
            () => fn(array(data1, dtype), array(data2, dtype)),
            py
          );
          if (r === 'both-reject') return;
          const jsResult = fn(array(data1, dtype), array(data2, dtype));
          const { rtol, atol } = getTols(dtype);
          expectMatchPre(jsResult, py, { rtol, atol });
        });
      }
    });
  }
});

describe('DType Sweep: Bitwise aliases', () => {
  describe('bitwise_invert', () => {
    for (const dtype of ALL_DTYPES) {
      it(`${dtype}`, () => {
        const data = dtype === 'bool' ? [1, 1, 0, 1] : [6, 7, 8, 9];
        const py = oracle.get(`bitwise_invert_${dtype}`)!;
        const r = expectBothRejectPre(
          'bitwise_invert requires integer or bool dtype',
          () => np.bitwise_invert(array(data, dtype)),
          py
        );
        if (r === 'both-reject') return;
        const jsResult = np.bitwise_invert(array(data, dtype));
        const { rtol, atol } = getTols(dtype);
        expectMatchPre(jsResult, py, { rtol, atol });
      });
    }
  });

  const bitwiseShiftOps: { name: string; fn: (a: any, b: any) => any }[] = [
    { name: 'bitwise_left_shift', fn: np.bitwise_left_shift },
    { name: 'bitwise_right_shift', fn: np.bitwise_right_shift },
  ];

  for (const { name, fn } of bitwiseShiftOps) {
    describe(name, () => {
      for (const dtype of ALL_DTYPES) {
        it(`${dtype}`, () => {
          const data1 = dtype === 'bool' ? [1, 1, 0, 1] : [6, 7, 8, 9];
          const data2 = dtype === 'bool' ? [1, 0, 1, 0] : [1, 2, 3, 4];
          const py = oracle.get(`${name}_${dtype}`)!;
          const r = expectBothRejectPre(
            `${name} requires integer dtype (no bool, float, or complex)`,
            () => fn(array(data1, dtype), array(data2, dtype)),
            py
          );
          if (r === 'both-reject') return;
          const jsResult = fn(array(data1, dtype), array(data2, dtype));
          const { rtol, atol } = getTols(dtype);
          expectMatchPre(jsResult, py, { rtol, atol });
        });
      }
    });
  }
});

describe('DType Sweep: Reduction aliases', () => {
  const reductionOps: { name: string; fn: (a: any) => any }[] = [
    { name: 'amax', fn: np.amax },
    { name: 'amin', fn: np.amin },
    { name: 'var', fn: (np as any).var },
  ];

  for (const { name, fn } of reductionOps) {
    describe(name, () => {
      for (const dtype of ALL_DTYPES) {
        it(`${dtype}`, () => {
          const data = dtype === 'bool' ? [1, 0, 1, 1, 0] : [3, 1, 4, 1, 5];
          const py = oracle.get(`${name}_${dtype}`)!;
          const r = expectBothRejectPre(
            `${name} may not be supported for ${dtype}`,
            () => fn(array(data, dtype)),
            py
          );
          if (r === 'both-reject') return;
          const jsResult = fn(array(data, dtype));
          const digits = dtype === 'float16' ? 2 : 3;
          scalarClose(jsResult, py.value, digits);
        });
      }
    });
  }
});

describe('DType Sweep: Cumulative aliases', () => {
  const cumulativeOps: { name: string; fn: (a: any) => any }[] = [
    { name: 'cumulative_sum', fn: np.cumulative_sum },
    { name: 'cumulative_prod', fn: np.cumulative_prod },
    { name: 'nancumsum', fn: np.nancumsum },
    { name: 'nancumprod', fn: np.nancumprod },
  ];

  for (const { name, fn } of cumulativeOps) {
    describe(name, () => {
      for (const dtype of ALL_DTYPES) {
        it(`${dtype}`, () => {
          const data = dtype === 'bool' ? [1, 0, 1, 1] : [1, 2, 3, 4];
          const py = oracle.get(`${name}_${dtype}`)!;
          const r = expectBothRejectPre(
            `${name} may not be supported for ${dtype}`,
            () => fn(array(data, dtype)),
            py
          );
          if (r === 'both-reject') return;
          const jsResult = fn(array(data, dtype));
          const { rtol, atol } = getTols(dtype);
          expectMatchPre(jsResult, py, { rtol, atol });
        });
      }
    });
  }
});

describe('DType Sweep: Stacking/shape aliases', () => {
  describe('row_stack', () => {
    for (const dtype of ALL_DTYPES) {
      it(`${dtype}`, () => {
        const data = dtype === 'bool' ? [1, 0, 1, 1] : [1, 2, 3, 4];
        const a = array(data, dtype);
        const py = oracle.get(`row_stack_${dtype}`)!;
        const r = expectBothRejectPre(
          `row_stack may not be supported for ${dtype}`,
          () => np.row_stack([a, a]),
          py
        );
        if (r === 'both-reject') return;
        const jsResult = np.row_stack([a, a]);
        const { rtol, atol } = getTols(dtype);
        expectMatchPre(jsResult, py, { rtol, atol });
      });
    }
  });

  describe('concat', () => {
    for (const dtype of ALL_DTYPES) {
      it(`${dtype}`, () => {
        const data = dtype === 'bool' ? [1, 0, 1, 1] : [1, 2, 3, 4];
        const a = array(data, dtype);
        const py = oracle.get(`concat_${dtype}`)!;
        const r = expectBothRejectPre(
          `concat may not be supported for ${dtype}`,
          () => np.concat([a, a]),
          py
        );
        if (r === 'both-reject') return;
        const jsResult = np.concat([a, a]);
        const { rtol, atol } = getTols(dtype);
        expectMatchPre(jsResult, py, { rtol, atol });
      });
    }
  });

  describe('delete', () => {
    for (const dtype of ALL_DTYPES) {
      it(`${dtype}`, () => {
        const data = dtype === 'bool' ? [1, 0, 1, 1] : [1, 2, 3, 4];
        const a = array(data, dtype);
        const py = oracle.get(`delete_${dtype}`)!;
        const r = expectBothRejectPre(
          `delete may not be supported for ${dtype}`,
          () => np.delete_(a, 1),
          py
        );
        if (r === 'both-reject') return;
        const jsResult = np.delete_(a, 1);
        const { rtol, atol } = getTols(dtype);
        expectMatchPre(jsResult, py, { rtol, atol });
      });
    }
  });
});
