/**
 * Coverage tests for src/common/wasm/* files.
 *
 * Sets wasmConfig.thresholdMultiplier = 0 so small arrays exercise the WASM
 * execution paths that are normally gated behind a size threshold.
 */

import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { array, hasFloat16 } from '../../src';
import type { DType } from '../../src/common/dtype';
import type { ArrayStorage } from '../../src/common/storage';
import { wasmArccos } from '../../src/common/wasm/arccos';
import { wasmArccosh } from '../../src/common/wasm/arccosh';
import { wasmArcsin } from '../../src/common/wasm/arcsin';
import { wasmArcsinh } from '../../src/common/wasm/arcsinh';
import { wasmArctan } from '../../src/common/wasm/arctan';
import { wasmArctanh } from '../../src/common/wasm/arctanh';
import { wasmConfig } from '../../src/common/wasm/config';
import { wasmCos } from '../../src/common/wasm/cos';
import { wasmCosh } from '../../src/common/wasm/cosh';
import { supportsRelaxedSimd, useRelaxedKernels } from '../../src/common/wasm/detect';
import { wasmDiv, wasmDivScalar } from '../../src/common/wasm/divide';
import { wasmExp } from '../../src/common/wasm/exp';
import { wasmExp2 } from '../../src/common/wasm/exp2';
import { wasmExpm1 } from '../../src/common/wasm/expm1';
import { wasmExtract, wasmTakeAlongAxis2D, wasmWhere } from '../../src/common/wasm/gather';
import { wasmGcd, wasmGcdScalar } from '../../src/common/wasm/gcd';
import { wasmHeaviside, wasmHeavisideScalar } from '../../src/common/wasm/heaviside';
import { wasmIndices } from '../../src/common/wasm/indices';
import { wasmLeftShift, wasmLeftShiftScalar } from '../../src/common/wasm/left_shift';
import { wasmLog } from '../../src/common/wasm/log';
import { wasmLog1p } from '../../src/common/wasm/log1p';
import { wasmLogaddexp, wasmLogaddexpScalar } from '../../src/common/wasm/logaddexp';
import { wasmMax, wasmMaxScalar } from '../../src/common/wasm/max';
import { wasmMin, wasmMinScalar } from '../../src/common/wasm/min';
import { wasmPad2D } from '../../src/common/wasm/pad';
import { wasmReduceCountNz } from '../../src/common/wasm/reduce_count_nz';
import { wasmReduceQuantile } from '../../src/common/wasm/reduce_quantile';
import { wasmRepeat } from '../../src/common/wasm/repeat';
import { wasmRightShift, wasmRightShiftScalar } from '../../src/common/wasm/right_shift';
import { wasmRoll } from '../../src/common/wasm/roll';
import { wasmSin } from '../../src/common/wasm/sin';
import { wasmSinc } from '../../src/common/wasm/sinc';
import { wasmSinh } from '../../src/common/wasm/sinh';
import { wasmSquare } from '../../src/common/wasm/square';
import { wasmTan } from '../../src/common/wasm/tan';
import { wasmTanh } from '../../src/common/wasm/tanh';
import { wasmTile2D } from '../../src/common/wasm/tile';
import { wasmVdotComplex } from '../../src/common/wasm/vdot';

beforeEach(() => {
  wasmConfig.thresholdMultiplier = 0;
});

afterEach(() => {
  wasmConfig.thresholdMultiplier = 1;
});

// ============================================================
// max / min
// ============================================================
describe('wasmMax / wasmMin', () => {
  it('wasmMax binary float64', () => {
    const a = array([1, 5, 3]);
    const b = array([4, 2, 6]);
    const r = wasmMax(a.storage, b.storage);
    expect(r).not.toBeNull();
    expect(Array.from(r!.data as Float64Array)).toEqual([4, 5, 6]);
  });

  it('wasmMax scalar float64', () => {
    const a = array([1, 5, 3]);
    const r = wasmMaxScalar(a.storage, 4);
    expect(r).not.toBeNull();
    expect(Array.from(r!.data as Float64Array)).toEqual([4, 5, 4]);
  });

  it('wasmMin binary float64', () => {
    const a = array([1, 5, 3]);
    const b = array([4, 2, 6]);
    const r = wasmMin(a.storage, b.storage);
    expect(r).not.toBeNull();
    expect(Array.from(r!.data as Float64Array)).toEqual([1, 2, 3]);
  });

  it('wasmMin scalar float64', () => {
    const a = array([1, 5, 3]);
    const r = wasmMinScalar(a.storage, 3);
    expect(r).not.toBeNull();
    expect(Array.from(r!.data as Float64Array)).toEqual([1, 3, 3]);
  });

  it('wasmMax returns null below threshold (default multiplier)', () => {
    wasmConfig.thresholdMultiplier = 1;
    const a = array([1, 2]);
    const r = wasmMax(a.storage, a.storage);
    expect(r).toBeNull();
  });
});

// ============================================================
// square
// ============================================================
describe('wasmSquare', () => {
  it('float64 values', () => {
    const a = array([2, 3, 4]);
    const r = wasmSquare(a.storage);
    expect(r).not.toBeNull();
    expect(Array.from(r!.data as Float64Array)).toEqual([4, 9, 16]);
  });

  it('int32 values', () => {
    const a = array([2, 3, 4], 'int32');
    const r = wasmSquare(a.storage);
    expect(r).not.toBeNull();
    expect(Array.from(r!.data as Int32Array)).toEqual([4, 9, 16]);
  });

  it('complex128 values', () => {
    const a = array([1, 2, 3], 'complex128');
    const r = wasmSquare(a.storage);
    expect(r).not.toBeNull();
    expect(r!.size).toBe(3);
  });
});

// ============================================================
// roll
// ============================================================
describe('wasmRoll', () => {
  it('shifts elements right', () => {
    const a = array([1, 2, 3, 4, 5]);
    const r = wasmRoll(a.storage, 2);
    expect(r).not.toBeNull();
    expect(Array.from(r!.data as Float64Array)).toEqual([4, 5, 1, 2, 3]);
  });

  it('negative shift (left)', () => {
    const a = array([1, 2, 3, 4, 5]);
    const r = wasmRoll(a.storage, -1);
    expect(r).not.toBeNull();
    expect(Array.from(r!.data as Float64Array)).toEqual([2, 3, 4, 5, 1]);
  });
});

// ============================================================
// repeat
// ============================================================
describe('wasmRepeat', () => {
  it('repeats each element', () => {
    const a = array([1, 2, 3]);
    const r = wasmRepeat(a.storage, 2);
    expect(r).not.toBeNull();
    expect(Array.from(r!.data as Float64Array)).toEqual([1, 1, 2, 2, 3, 3]);
  });

  it('int32 repeat', () => {
    const a = array([10, 20], 'int32');
    const r = wasmRepeat(a.storage, 3);
    expect(r).not.toBeNull();
    expect(r!.size).toBe(6);
  });
});

// ============================================================
// rot90
// ============================================================
// tile
// ============================================================
describe('wasmTile2D', () => {
  it('tiles 2D array', () => {
    const a = array([
      [1, 2],
      [3, 4],
    ]);
    const r = wasmTile2D(a.storage, 2, 3);
    expect(r).not.toBeNull();
    expect(Array.from(r!.shape)).toEqual([4, 6]);
  });

  it('returns null for 1D input', () => {
    const a = array([1, 2, 3]);
    const r = wasmTile2D(a.storage, 2, 2);
    expect(r).toBeNull();
  });
});

// ============================================================
// pad
// ============================================================
describe('wasmPad2D', () => {
  it('pads 2D array with zeros', () => {
    const a = array([
      [1, 2],
      [3, 4],
    ]);
    const r = wasmPad2D(a.storage, 1);
    expect(r).not.toBeNull();
    expect(Array.from(r!.shape)).toEqual([4, 4]);
    // corners and borders should be 0
    const data = Array.from(r!.data as Float64Array);
    expect(data[0]).toBe(0);
    expect(data[5]).toBe(1); // [1][1]
  });

  it('int32 padding', () => {
    const a = array(
      [
        [1, 2],
        [3, 4],
      ],
      'int32',
    );
    const r = wasmPad2D(a.storage, 2);
    expect(r).not.toBeNull();
    expect(Array.from(r!.shape)).toEqual([6, 6]);
  });

  it('returns null for 1D input', () => {
    const a = array([1, 2, 3]);
    const r = wasmPad2D(a.storage, 1);
    expect(r).toBeNull();
  });
});

// ============================================================
// heaviside
// ============================================================
describe('wasmHeaviside', () => {
  it('scalar: heaviside step function', () => {
    const a = array([-1, 0, 1]);
    const r = wasmHeavisideScalar(a.storage, 0.5, 'float64');
    expect(r).not.toBeNull();
    const data = Array.from(r!.data as Float64Array);
    expect(data[0]).toBe(0); // negative → 0
    expect(data[1]).toBe(0.5); // zero → x2
    expect(data[2]).toBe(1); // positive → 1
  });

  it('binary: heaviside with x2 array', () => {
    const x1 = array([-1, 0, 1]);
    const x2 = array([0.5, 0.5, 0.5]);
    const r = wasmHeaviside(x1.storage, x2.storage, 'float64');
    expect(r).not.toBeNull();
    const data = Array.from(r!.data as Float64Array);
    expect(data[0]).toBe(0);
    expect(data[1]).toBe(0.5);
    expect(data[2]).toBe(1);
  });

  it('float32 scalar', () => {
    const a = array([-1, 0, 2], 'float32');
    const r = wasmHeavisideScalar(a.storage, 0, 'float32');
    expect(r).not.toBeNull();
    expect(r!.size).toBe(3);
  });
});

// ============================================================
// gcd
// ============================================================
describe('wasmGcd', () => {
  it('gcd scalar int32', () => {
    const a = array([12, 8, 15], 'int32');
    const r = wasmGcdScalar(a.storage, 4);
    expect(r).not.toBeNull();
    const data = Array.from(r!.data as Int32Array);
    expect(data[0]).toBe(4); // gcd(12, 4) = 4
    expect(data[1]).toBe(4); // gcd(8, 4) = 4
    expect(data[2]).toBe(1); // gcd(15, 4) = 1
  });

  it('gcd binary int32', () => {
    const a = array([12, 8, 15], 'int32');
    const b = array([4, 6, 5], 'int32');
    const r = wasmGcd(a.storage, b.storage);
    expect(r).not.toBeNull();
    const data = Array.from(r!.data as Int32Array);
    expect(data[0]).toBe(4); // gcd(12, 4) = 4
    expect(data[1]).toBe(2); // gcd(8, 6) = 2
    expect(data[2]).toBe(5); // gcd(15, 5) = 5
  });

  it('returns null for float64 (unsupported dtype)', () => {
    const a = array([12, 8]);
    const r = wasmGcdScalar(a.storage, 4);
    expect(r).toBeNull();
  });
});

// ============================================================
// indices
// ============================================================
describe('wasmIndices', () => {
  it('2D grid indices', () => {
    const r = wasmIndices([3, 4], 'int32');
    expect(r).not.toBeNull();
    expect(Array.from(r!.shape)).toEqual([2, 3, 4]);
  });

  it('3D grid indices', () => {
    const r = wasmIndices([2, 3, 4], 'int32');
    expect(r).not.toBeNull();
    expect(Array.from(r!.shape)).toEqual([3, 2, 3, 4]);
  });

  it('returns null for 1D (unsupported)', () => {
    const r = wasmIndices([5], 'int32');
    expect(r).toBeNull();
  });

  it('returns null for unsupported dtype', () => {
    const r = wasmIndices([3, 4], 'float32');
    expect(r).toBeNull();
  });

  it('handles float64 dtype', () => {
    const r = wasmIndices([3, 4], 'float64');
    expect(r).not.toBeNull();
    expect(r!.dtype).toBe('float64');
  });
});

// ============================================================
// reduce_count_nz
// ============================================================
describe('wasmReduceCountNz', () => {
  it('counts nonzero float64', () => {
    const a = array([0, 1, 2, 0, 3]);
    const r = wasmReduceCountNz(a.storage);
    expect(r).not.toBeNull();
    expect(r).toBe(3);
  });

  it('counts nonzero int32', () => {
    const a = array([0, 0, 5, 7], 'int32');
    const r = wasmReduceCountNz(a.storage);
    expect(r).not.toBeNull();
    expect(r).toBe(2);
  });

  it('all zeros', () => {
    const a = array([0, 0, 0]);
    const r = wasmReduceCountNz(a.storage);
    expect(r).toBe(0);
  });
});

// ============================================================
// reduce_quantile
// ============================================================
describe('wasmReduceQuantile', () => {
  it('median of sorted data', () => {
    const a = array([1, 2, 3, 4, 5]);
    const r = wasmReduceQuantile(a.storage, 0.5);
    expect(r).not.toBeNull();
    expect(r).toBe(3);
  });

  it('min quantile (q=0)', () => {
    const a = array([3, 1, 4, 1, 5]);
    const r = wasmReduceQuantile(a.storage, 0);
    expect(r).not.toBeNull();
    expect(r).toBe(1);
  });

  it('max quantile (q=1)', () => {
    const a = array([3, 1, 4, 1, 5]);
    const r = wasmReduceQuantile(a.storage, 1);
    expect(r).not.toBeNull();
    expect(r).toBe(5);
  });

  it('int32 data', () => {
    const a = array([1, 2, 3, 4, 5], 'int32');
    const r = wasmReduceQuantile(a.storage, 0.5);
    expect(r).not.toBeNull();
    expect(r).toBe(3);
  });

  it('returns null for complex dtype', () => {
    const a = array([1, 2, 3], 'complex128');
    const r = wasmReduceQuantile(a.storage);
    expect(r).toBeNull();
  });
});

// ============================================================
// exp2
// ============================================================
describe('wasmExp2', () => {
  it('float64 path', () => {
    const a = array([0, 1, 2, 3]);
    const r = wasmExp2(a.storage);
    expect(r).not.toBeNull();
    const data = Array.from(r!.data as Float64Array);
    expect(data[0]).toBeCloseTo(1);
    expect(data[1]).toBeCloseTo(2);
    expect(data[2]).toBeCloseTo(4);
    expect(data[3]).toBeCloseTo(8);
  });

  it('float32 path', () => {
    const a = array([0, 1, 2], 'float32');
    const r = wasmExp2(a.storage);
    expect(r).not.toBeNull();
    expect(r!.size).toBe(3);
  });

  it('int64 native path', () => {
    const a = array([0n, 1n, 2n, 3n], 'int64');
    const r = wasmExp2(a.storage);
    expect(r).not.toBeNull();
    const data = Array.from(r!.data as Float64Array);
    expect(data[0]).toBeCloseTo(1);
    expect(data[1]).toBeCloseTo(2);
    expect(data[2]).toBeCloseTo(4);
  });

  it('uint64 native path', () => {
    const a = array([0n, 1n, 2n], 'uint64');
    const r = wasmExp2(a.storage);
    expect(r).not.toBeNull();
    expect(r!.size).toBe(3);
  });

  it('int32 integer path (converted to float64)', () => {
    const a = array([0, 1, 2, 3], 'int32');
    const r = wasmExp2(a.storage);
    expect(r).not.toBeNull();
    const data = Array.from(r!.data as Float64Array);
    expect(data[0]).toBeCloseTo(1);
    expect(data[1]).toBeCloseTo(2);
    expect(data[2]).toBeCloseTo(4);
  });

  it('int16 integer path', () => {
    const a = array([1, 2, 3], 'int16');
    const r = wasmExp2(a.storage);
    expect(r).not.toBeNull();
    expect(r!.dtype).toBe('float32');
  });

  it('complex128 path: 2^(a+0i) = 2^a', () => {
    const a = array([1, 2, 3], 'complex128');
    const r = wasmExp2(a.storage);
    expect(r).not.toBeNull();
    expect(r!.dtype).toBe('complex128');
    const data = Array.from(r!.data as Float64Array);
    expect(data[0]).toBeCloseTo(2); // 2^1, real
    expect(data[1]).toBeCloseTo(0); // imag
    expect(data[2]).toBeCloseTo(4); // 2^2, real
    expect(data[4]).toBeCloseTo(8); // 2^3, real
  });
});

// ============================================================
// logaddexp
// ============================================================
describe('wasmLogaddexp', () => {
  it('float64 binary path', () => {
    const a = array([0, 1]);
    const b = array([0, 1]);
    const r = wasmLogaddexp(a.storage, b.storage);
    expect(r).not.toBeNull();
    const data = Array.from(r!.data as Float64Array);
    expect(data[0]).toBeCloseTo(Math.log(2));
    expect(data[1]).toBeCloseTo(Math.log(Math.exp(1) + Math.exp(1)));
  });

  it('float32 binary path', () => {
    const a = array([0, 1], 'float32');
    const b = array([0, 1], 'float32');
    const r = wasmLogaddexp(a.storage, b.storage);
    expect(r).not.toBeNull();
    expect(r!.size).toBe(2);
  });

  it('int32 binary path → float64 output', () => {
    const a = array([1, 2, 3], 'int32');
    const b = array([1, 2, 3], 'int32');
    const r = wasmLogaddexp(a.storage, b.storage);
    expect(r).not.toBeNull();
    expect(r!.dtype).toBe('float64');
    expect(r!.size).toBe(3);
  });

  it('int16 binary path', () => {
    const a = array([1, 2, 3], 'int16');
    const b = array([1, 2, 3], 'int16');
    const r = wasmLogaddexp(a.storage, b.storage);
    expect(r).not.toBeNull();
    expect(r!.dtype).toBe('float32'); // NumPy: int16 → float32
  });

  it('float64 scalar path', () => {
    const a = array([0, 1]);
    const r = wasmLogaddexpScalar(a.storage, 0);
    expect(r).not.toBeNull();
    expect(r!.size).toBe(2);
  });

  it('float32 scalar path', () => {
    const a = array([0, 1], 'float32');
    const r = wasmLogaddexpScalar(a.storage, 1);
    expect(r).not.toBeNull();
    expect(r!.size).toBe(2);
  });

  it('int32 scalar path → float64 output', () => {
    const a = array([1, 2, 3], 'int32');
    const r = wasmLogaddexpScalar(a.storage, 1);
    expect(r).not.toBeNull();
    expect(r!.dtype).toBe('float64');
  });

  it('int8 scalar path', () => {
    const a = array([1, 2, 3], 'int8');
    const r = wasmLogaddexpScalar(a.storage, 2);
    expect(r).not.toBeNull();
    expect(r!.dtype).toBe(hasFloat16 ? 'float16' : 'float32'); // NumPy: int8 → float16
  });
});

// ============================================================
// divide
// ============================================================
describe('wasmDiv / wasmDivScalar', () => {
  it('float64 binary', () => {
    const a = array([4, 9, 16]);
    const b = array([2, 3, 4]);
    const r = wasmDiv(a.storage, b.storage);
    expect(r).not.toBeNull();
    const data = Array.from(r!.data as Float64Array);
    expect(data[0]).toBe(2);
    expect(data[1]).toBe(3);
    expect(data[2]).toBe(4);
  });

  it('int32 binary → float64 output', () => {
    const a = array([4, 9, 16], 'int32');
    const b = array([2, 3, 4], 'int32');
    const r = wasmDiv(a.storage, b.storage);
    expect(r).not.toBeNull();
    expect(r!.dtype).toBe('float64');
    const data = Array.from(r!.data as Float64Array);
    expect(data[0]).toBe(2);
    expect(data[1]).toBe(3);
  });

  it('int16 binary → float32 output', () => {
    const a = array([4, 9], 'int16');
    const b = array([2, 3], 'int16');
    const r = wasmDiv(a.storage, b.storage);
    expect(r).not.toBeNull();
    expect(r!.dtype).toBe('float64'); // NumPy: all int divide → float64
  });

  it('float64 scalar', () => {
    const a = array([4, 9, 16]);
    const r = wasmDivScalar(a.storage, 2);
    expect(r).not.toBeNull();
    const data = Array.from(r!.data as Float64Array);
    expect(data[0]).toBe(2);
    expect(data[1]).toBe(4.5);
    expect(data[2]).toBe(8);
  });

  it('int32 scalar → float64 output', () => {
    const a = array([4, 9, 16], 'int32');
    const r = wasmDivScalar(a.storage, 2);
    expect(r).not.toBeNull();
    expect(r!.dtype).toBe('float64');
  });

  it('int8 scalar → float64 output', () => {
    const a = array([4, 8, 12], 'int8');
    const r = wasmDivScalar(a.storage, 4);
    expect(r).not.toBeNull();
    expect(r!.dtype).toBe('float64'); // NumPy: all int divide → float64
  });

  it('returns null for mixed dtypes', () => {
    const a = array([1, 2, 3]);
    const b = array([1, 2, 3], 'float32');
    const r = wasmDiv(a.storage, b.storage);
    expect(r).toBeNull();
  });
});

// ============================================================
// left_shift / right_shift
// ============================================================
describe('wasmLeftShift / wasmRightShift', () => {
  it('left shift binary int32', () => {
    const a = array([1, 2, 4], 'int32');
    const b = array([1, 1, 1], 'int32');
    const r = wasmLeftShift(a.storage, b.storage);
    expect(r).not.toBeNull();
    expect(Array.from(r!.data as Int32Array)).toEqual([2, 4, 8]);
  });

  it('left shift scalar int32', () => {
    const a = array([1, 2, 4], 'int32');
    const r = wasmLeftShiftScalar(a.storage, 2);
    expect(r).not.toBeNull();
    expect(Array.from(r!.data as Int32Array)).toEqual([4, 8, 16]);
  });

  it('right shift binary int32', () => {
    const a = array([8, 16, 32], 'int32');
    const b = array([1, 2, 3], 'int32');
    const r = wasmRightShift(a.storage, b.storage);
    expect(r).not.toBeNull();
    expect(Array.from(r!.data as Int32Array)).toEqual([4, 4, 4]);
  });

  it('right shift scalar int32', () => {
    const a = array([8, 16, 32], 'int32');
    const r = wasmRightShiftScalar(a.storage, 2);
    expect(r).not.toBeNull();
    expect(Array.from(r!.data as Int32Array)).toEqual([2, 4, 8]);
  });

  it('left shift scalar int16', () => {
    const a = array([1, 2, 4], 'int16');
    const r = wasmLeftShiftScalar(a.storage, 1);
    expect(r).not.toBeNull();
    expect(r!.size).toBe(3);
  });

  it('right shift scalar uint32', () => {
    const a = array([8, 16, 32], 'uint32');
    const r = wasmRightShiftScalar(a.storage, 1);
    expect(r).not.toBeNull();
    expect(r!.size).toBe(3);
  });

  it('returns null for float64 (unsupported)', () => {
    const a = array([1, 2, 3]);
    const r = wasmLeftShiftScalar(a.storage, 1);
    expect(r).toBeNull();
  });
});

// ============================================================
// gather (extract, take_along_axis, where)
// ============================================================
describe('wasmExtract / wasmTakeAlongAxis2D / wasmWhere', () => {
  it('wasmExtract selects elements where condition is nonzero', () => {
    const cond = array([1, 0, 1, 0, 1], 'int32');
    const data = array([10, 20, 30, 40, 50]);
    const r = wasmExtract(cond.storage, data.storage);
    expect(r).not.toBeNull();
    expect(r!.size).toBe(3);
    const vals = Array.from(r!.data as Float64Array).slice(0, 3);
    expect(vals).toEqual([10, 30, 50]);
  });

  it('wasmExtract int32 data', () => {
    const cond = array([1, 0, 1], 'int32');
    const data = array([10, 20, 30], 'int32');
    const r = wasmExtract(cond.storage, data.storage);
    expect(r).not.toBeNull();
    expect(r!.size).toBe(2);
  });

  it('wasmExtract returns null for unsupported cond dtype', () => {
    const cond = array([1n, 0n, 1n], 'int64');
    const data = array([10, 20, 30]);
    const r = wasmExtract(cond.storage, data.storage);
    expect(r).toBeNull();
  });

  it('wasmTakeAlongAxis2D axis=0', () => {
    const data = array([
      [1, 2, 3],
      [4, 5, 6],
      [7, 8, 9],
      [10, 11, 12],
    ]);
    const indices = array(
      [
        [0, 2, 1],
        [1, 0, 2],
        [2, 1, 0],
        [3, 3, 3],
      ],
      'int32',
    );
    const r = wasmTakeAlongAxis2D(data.storage, indices.storage, 0);
    expect(r).not.toBeNull();
    expect(Array.from(r!.shape)).toEqual([4, 3]);
  });

  it('wasmTakeAlongAxis2D returns null for axis != 0', () => {
    const data = array([
      [1, 2],
      [3, 4],
    ]);
    const indices = array(
      [
        [0, 1],
        [1, 0],
      ],
      'int32',
    );
    const r = wasmTakeAlongAxis2D(data.storage, indices.storage, 1);
    expect(r).toBeNull();
  });

  it('wasmTakeAlongAxis2D returns null for 1D input', () => {
    const data = array([1, 2, 3, 4]);
    const indices = array([0, 1, 2, 3], 'int32');
    const r = wasmTakeAlongAxis2D(data.storage, indices.storage, 0);
    expect(r).toBeNull();
  });

  it('wasmWhere picks x or y based on condition', () => {
    const cond = array([1, 0, 1, 0], 'int32');
    const x = array([10, 20, 30, 40]);
    const y = array([100, 200, 300, 400]);
    const r = wasmWhere(cond.storage, x.storage, y.storage);
    expect(r).not.toBeNull();
    const data = Array.from(r!.data as Float64Array);
    expect(data[0]).toBe(10);
    expect(data[1]).toBe(200);
    expect(data[2]).toBe(30);
    expect(data[3]).toBe(400);
  });

  it('wasmWhere returns null for dtype mismatch', () => {
    const cond = array([1, 0, 1], 'int32');
    const x = array([1, 2, 3]);
    const y = array([4, 5, 6], 'float32');
    const r = wasmWhere(cond.storage, x.storage, y.storage);
    expect(r).toBeNull();
  });
});

// ============================================================
// vdot (complex)
// ============================================================
describe('wasmVdotComplex', () => {
  it('complex128 conjugate dot product', () => {
    // real arrays are complex with zero imaginary parts
    const a = array([1, 2, 3], 'complex128');
    const b = array([4, 5, 6], 'complex128');
    const r = wasmVdotComplex(a.storage, b.storage);
    expect(r).not.toBeNull();
    // conj([1,2,3]) · [4,5,6] = 1*4 + 2*5 + 3*6 = 32 (all im=0)
    expect(r!.re).toBeCloseTo(32);
    expect(r!.im).toBeCloseTo(0);
  });

  it('returns null for real dtype', () => {
    const a = array([1, 2, 3]);
    const b = array([4, 5, 6]);
    const r = wasmVdotComplex(a.storage, b.storage);
    expect(r).toBeNull();
  });

  it('returns null for length mismatch', () => {
    const a = array([1, 2], 'complex128');
    const b = array([1, 2, 3], 'complex128');
    const r = wasmVdotComplex(a.storage, b.storage);
    expect(r).toBeNull();
  });
});

// ============================================================
// Unary transcendentals (PR #135 WASM SIMD expansion)
//
// These kernels (sin/cos/tan/sinh/cosh/tanh/exp/log/expm1/log1p/
// arc{sin,cos,tan,sinh,cosh,tanh}) all share one shape: native f64/f32
// kernels, a float16 branch (f32 compute → f16 downcast, gated on
// effectiveDType/hasFloat16), small ints (i8/u8/i16/u16 → f32, with
// i8/u8 downcast to f16 when available) and large ints (i32/u32/i64/u64
// → f64). Exercising every dtype here covers each per-dtype kernel
// arrow-wrapper plus the float16 path.
//
// Relaxed vs baseline binary selection is governed by
// wasmConfig.useRelaxedSimd, left at its 'auto' default — on runtimes
// that support relaxed SIMD (CI), the relaxed kernels execute; the
// per-kernel `_bins` cache means a single process loads only one
// variant, so baseline/relaxed selection itself is covered by the
// dedicated detect.ts block below rather than per kernel.
// ============================================================

type UnaryWasm = (s: ArrayStorage) => ArrayStorage | null;

// Output dtype for the small/large-int + float paths shared by all 16 kernels.
const SMALL_TO_F16 = hasFloat16 ? 'float16' : 'float32';
const UNARY_DTYPE_CASES: Array<{
  dtype: DType;
  expected: DType;
  input: () => ReturnType<typeof array>;
}> = [
  { dtype: 'float64', expected: 'float64', input: () => array([0.5, 0.5, 0.5]) },
  { dtype: 'float32', expected: 'float32', input: () => array([0.5, 0.5, 0.5], 'float32') },
  { dtype: 'int8', expected: SMALL_TO_F16, input: () => array([1, 2, 3], 'int8') },
  { dtype: 'uint8', expected: SMALL_TO_F16, input: () => array([1, 2, 3], 'uint8') },
  { dtype: 'int16', expected: 'float32', input: () => array([1, 2, 3], 'int16') },
  { dtype: 'uint16', expected: 'float32', input: () => array([1, 2, 3], 'uint16') },
  { dtype: 'int32', expected: 'float64', input: () => array([1, 2, 3], 'int32') },
  { dtype: 'uint32', expected: 'float64', input: () => array([1, 2, 3], 'uint32') },
  { dtype: 'int64', expected: 'float64', input: () => array([1n, 2n, 3n], 'int64') },
  { dtype: 'uint64', expected: 'float64', input: () => array([1n, 2n, 3n], 'uint64') },
];

// fInput is in-domain for every kernel below (acosh needs ≥1, asin/acos
// need |x|≤1, atanh needs |x|<1, log needs >0) — 0.5 fails acosh/log, so
// the float reference is checked per-kernel with its own in-domain value.
// noF64: wasmLog keeps float64 INPUT on the JS fallback (V8's Math.log beats a
// 2-wide WASM kernel), so its float64 path returns null by design.
const TRANSCENDENTALS: Array<{
  name: string;
  fn: UnaryWasm;
  f: number;
  ref: (x: number) => number;
  noF64?: boolean;
}> = [
  { name: 'wasmSin', fn: wasmSin, f: 0.5, ref: Math.sin },
  { name: 'wasmCos', fn: wasmCos, f: 0.5, ref: Math.cos },
  { name: 'wasmTan', fn: wasmTan, f: 0.5, ref: Math.tan },
  { name: 'wasmSinh', fn: wasmSinh, f: 0.5, ref: Math.sinh },
  { name: 'wasmCosh', fn: wasmCosh, f: 0.5, ref: Math.cosh },
  { name: 'wasmTanh', fn: wasmTanh, f: 0.5, ref: Math.tanh },
  { name: 'wasmExp', fn: wasmExp, f: 0.5, ref: Math.exp },
  { name: 'wasmExpm1', fn: wasmExpm1, f: 0.5, ref: Math.expm1 },
  { name: 'wasmLog', fn: wasmLog, f: 2, ref: Math.log, noF64: true },
  { name: 'wasmLog1p', fn: wasmLog1p, f: 0.5, ref: Math.log1p },
  { name: 'wasmArcsin', fn: wasmArcsin, f: 0.5, ref: Math.asin },
  { name: 'wasmArccos', fn: wasmArccos, f: 0.5, ref: Math.acos },
  { name: 'wasmArctan', fn: wasmArctan, f: 0.5, ref: Math.atan },
  { name: 'wasmArcsinh', fn: wasmArcsinh, f: 0.5, ref: Math.asinh },
  { name: 'wasmArccosh', fn: wasmArccosh, f: 1.5, ref: Math.acosh },
  { name: 'wasmArctanh', fn: wasmArctanh, f: 0.5, ref: Math.atanh },
];

for (const { name, fn, f, ref, noF64 } of TRANSCENDENTALS) {
  describe(name, () => {
    if (noF64) {
      it('float64 input stays on JS fallback (returns null)', () => {
        expect(fn(array([f, f, f]).storage)).toBeNull();
      });
    } else {
      it('float64 path (value check)', () => {
        const r = fn(array([f, f, f]).storage);
        expect(r).not.toBeNull();
        expect(r!.dtype).toBe('float64');
        expect((r!.data as Float64Array)[0]).toBeCloseTo(ref(f), 4);
      });
    }

    it('float32 path (value check)', () => {
      const r = fn(array([f, f, f], 'float32').storage);
      expect(r).not.toBeNull();
      expect(r!.dtype).toBe('float32');
      expect(Number((r!.data as Float32Array)[0])).toBeCloseTo(ref(f), 2);
    });

    if (hasFloat16) {
      it('float16 path', () => {
        const r = fn(array([f, f, f], 'float16').storage);
        expect(r).not.toBeNull();
        expect(r!.dtype).toBe('float16');
        expect(r!.size).toBe(3);
      });
    }

    for (const { dtype, expected, input } of UNARY_DTYPE_CASES) {
      if (dtype === 'float64' || dtype === 'float32') continue; // covered above
      it(`${dtype} path → ${expected}`, () => {
        const r = fn(input().storage);
        expect(r).not.toBeNull();
        expect(r!.dtype).toBe(expected);
        expect(r!.size).toBe(3);
      });
    }
  });
}

// ============================================================
// sinc — distinct shape: float64/float32/float16 preserved,
// every integer dtype → float64.
// ============================================================
describe('wasmSinc', () => {
  const sincRef = (x: number) => (x === 0 ? 1 : Math.sin(Math.PI * x) / (Math.PI * x));

  it('float64 path (value check)', () => {
    const r = wasmSinc(array([0, 0.5, 1]).storage);
    expect(r).not.toBeNull();
    expect(r!.dtype).toBe('float64');
    const data = r!.data as Float64Array;
    expect(data[0]).toBeCloseTo(sincRef(0), 6);
    expect(data[1]).toBeCloseTo(sincRef(0.5), 6);
  });

  it('float32 path', () => {
    const r = wasmSinc(array([0, 0.5, 1], 'float32').storage);
    expect(r).not.toBeNull();
    expect(r!.dtype).toBe('float32');
  });

  if (hasFloat16) {
    it('float16 path', () => {
      const r = wasmSinc(array([0, 0.5, 1], 'float16').storage);
      expect(r).not.toBeNull();
      expect(r!.dtype).toBe('float16');
      expect(r!.size).toBe(3);
    });
  }

  for (const dtype of ['int8', 'uint8', 'int16', 'uint16', 'int32', 'uint32'] as const) {
    it(`${dtype} path → float64`, () => {
      const r = wasmSinc(array([0, 1, 2], dtype).storage);
      expect(r).not.toBeNull();
      expect(r!.dtype).toBe('float64');
    });
  }

  for (const dtype of ['int64', 'uint64'] as const) {
    it(`${dtype} path → float64`, () => {
      const r = wasmSinc(array([0n, 1n, 2n], dtype).storage);
      expect(r).not.toBeNull();
      expect(r!.dtype).toBe('float64');
    });
  }
});

// ============================================================
// Relaxed SIMD kernel selection (detect.ts)
// ============================================================
describe('relaxed SIMD selection', () => {
  afterEach(() => {
    wasmConfig.useRelaxedSimd = 'auto';
  });

  it('useRelaxedKernels honours forced true/false and auto', () => {
    wasmConfig.useRelaxedSimd = true;
    expect(useRelaxedKernels()).toBe(true);

    wasmConfig.useRelaxedSimd = false;
    expect(useRelaxedKernels()).toBe(false);

    wasmConfig.useRelaxedSimd = 'auto';
    expect(useRelaxedKernels()).toBe(supportsRelaxedSimd());
  });

  it('supportsRelaxedSimd is a stable boolean', () => {
    const first = supportsRelaxedSimd();
    expect(typeof first).toBe('boolean');
    expect(supportsRelaxedSimd()).toBe(first); // cached
  });
});
