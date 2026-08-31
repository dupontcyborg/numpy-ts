/**
 * @stdlib adapter (multi-package). Element-wise via scalar kernel + ndarray
 * dispatcher; reductions via stats-strided; sum via blas-ext; matmul/dot via BLAS.
 * See ../COVERAGE.md §stdlib.
 */

import ndarrayPkg from '@stdlib/ndarray';
import specialPkg from '@stdlib/math-base-special';
import opsPkg from '@stdlib/math-base-ops';
import unaryPkg from '@stdlib/ndarray-base-unary';
import binaryPkg from '@stdlib/ndarray-base-binary';
import statsPkg from '@stdlib/stats-strided';
import blasPkg from '@stdlib/blas';
import dsumPkg from '@stdlib/blas-ext-base-dsum';
import { is2D, toNested, toTyped } from '../lib/shape';
import type { ArrayData, JsLibAdapter, OpDef } from '../lib/types';

const nd: any = (ndarrayPkg as any).default ?? ndarrayPkg;
const special: any = (specialPkg as any).default ?? specialPkg;
const ops: any = (opsPkg as any).default ?? opsPkg;
const unary: any = (unaryPkg as any).default ?? unaryPkg;
const binary: any = (binaryPkg as any).default ?? binaryPkg;
const stats: any = (statsPkg as any).default ?? statsPkg;
const blas: any = (blasPkg as any).default ?? blasPkg;
const dsum: any = (dsumPkg as any).default ?? dsumPkg;

const mkND = (d: ArrayData) => nd.array(toNested(d) as any, { dtype: d.dtype, casting: 'unsafe' });
// stats-strided: float64 -> d*, float32 -> s*, otherwise the generic kernel.
const statsPfx = (d: ArrayData) => (d.dtype === 'float64' ? 'd' : d.dtype === 'float32' ? 's' : '');
// BLAS: float64 -> d, float32 -> s, otherwise generic (g) gemm/dot.
const blasPfx = (d: ArrayData) => (d.dtype === 'float64' ? 'd' : d.dtype === 'float32' ? 's' : 'g');

function unaryOp(kernel: (x: number) => number): OpDef {
  return {
    prepare: (d) => {
      const x = mkND(d.arrays['a']!);
      const out = nd.zeros(d.arrays['a']!.shape, { dtype: d.arrays['a']!.dtype });
      return { x, out };
    },
    run: (p: any) => {
      unary([p.x, p.out], kernel);
      return p.out;
    },
  };
}

function binaryOp(kernel: (x: number, y: number) => number): OpDef {
  return {
    prepare: (d) => {
      const a = d.arrays['a']!;
      const b = d.arrays['b']!;
      const x = mkND(a);
      const out = nd.zeros(a.shape, { dtype: a.dtype });
      if (b.data.length === 1) return { x, out, s: b.data[0], scalar: true }; // scalar broadcast
      return { x, y: mkND(b), out, scalar: false };
    },
    run: (p: any) => {
      if (p.scalar) unary([p.x, p.out], (v: number) => kernel(v, p.s));
      else binary([p.x, p.y, p.out], kernel);
      return p.out;
    },
  };
}

function reduceOp(name: string): OpDef {
  return {
    prepare: (d) => {
      if (d.params['axis'] != null) throw new Error('axis reduction unsupported');
      const a = d.arrays['a']!;
      const fn = stats[`${statsPfx(a)}${name}`];
      if (!fn) throw new Error(`no stats kernel ${statsPfx(a)}${name}`);
      return { buf: toTyped(a), n: a.data.length, fn };
    },
    run: (p: any) => p.fn(p.n, p.buf, 1),
  };
}

export const stdlib: JsLibAdapter = {
  name: 'stdlib',
  pkg: '@stdlib/ndarray',
  // No int64/uint64. complex bridging deferred in specdata, so they never run.
  dtypes: ['float16', 'float32', 'float64', 'int8', 'int16', 'int32', 'uint8', 'uint16', 'uint32', 'bool'],
  ops: {
    add: binaryOp(ops.add),
    subtract: binaryOp(ops.sub),
    multiply: binaryOp(ops.mul),
    divide: binaryOp(ops.div),
    power: binaryOp(special.pow),
    absolute: unaryOp(special.abs),
    reciprocal: unaryOp(special.inv),
    cbrt: unaryOp(special.cbrt),
    sqrt: unaryOp(special.sqrt),
    exp: unaryOp(special.exp),
    exp2: unaryOp(special.exp2),
    expm1: unaryOp(special.expm1),
    log: unaryOp(special.ln),
    log2: unaryOp(special.log2),
    log10: unaryOp(special.log10),
    log1p: unaryOp(special.log1p),
    floor: unaryOp(special.floor),
    ceil: unaryOp(special.ceil),
    trunc: unaryOp(special.trunc),
    round: unaryOp(special.round),
    deg2rad: unaryOp(special.deg2rad),
    rad2deg: unaryOp(special.rad2deg),
    sin: unaryOp(special.sin),
    cos: unaryOp(special.cos),
    tan: unaryOp(special.tan),
    arcsin: unaryOp(special.asin),
    arccos: unaryOp(special.acos),
    arctan: unaryOp(special.atan),
    arcsinh: unaryOp(special.asinh),
    arccosh: unaryOp(special.acosh),
    arctanh: unaryOp(special.atanh),
    sinh: unaryOp(special.sinh),
    cosh: unaryOp(special.cosh),
    tanh: unaryOp(special.tanh),
    sum: {
      // Installed sum kernel is blas-ext dsum (float64). float32/int sum -> N/A.
      prepare: (d) => {
        if (d.params['axis'] != null) throw new Error('axis reduction unsupported');
        const a = d.arrays['a']!;
        if (a.dtype !== 'float64') throw new Error('dsum is float64-only');
        return { buf: toTyped(a), n: a.data.length };
      },
      run: (p: any) => dsum(p.n, p.buf, 1),
    },
    mean: reduceOp('mean'),
    max: reduceOp('max'),
    min: reduceOp('min'),
    std: reduceOp('stdev'),
    var: reduceOp('variance'),
    matmul: {
      prepare: (d) => {
        const A = d.arrays['a']!;
        const B = d.arrays['b']!;
        if (!is2D(A) || !is2D(B)) throw new Error('matmul needs 2D');
        const [M, K] = A.shape as [number, number];
        const N = B.shape[1]!;
        const Ctor = A.dtype === 'float32' ? Float32Array : Float64Array;
        return { M, N, K, A: toTyped(A), B: toTyped(B), C: new Ctor(M * N), gemm: blas.base[`${blasPfx(A)}gemm`] };
      },
      run: (p: any) => {
        p.gemm('row-major', 'no-transpose', 'no-transpose', p.M, p.N, p.K, 1.0, p.A, p.K, p.B, p.N, 0.0, p.C, p.N);
        return p.C;
      },
    },
    transpose: {
      prepare: (d) => mkND(d.arrays['a']!),
      run: (x: any) => nd.copy(nd.transpose(x)), // transpose is a view; copy to materialize
    },
    concatenate: { prepare: (d) => [mkND(d.arrays['a']!), mkND(d.arrays['b']!)], run: (p: any) => nd.concat([p[0], p[1]]) },
    concat: { prepare: (d) => [mkND(d.arrays['a']!), mkND(d.arrays['b']!)], run: (p: any) => nd.concat([p[0], p[1]]) },
    dot: {
      // Opt into stdlib's WASM backend for float dot (BLAS L1 wasm has [s|d]dot,
      // no gdot); integer dot falls back to the JS generic gdot. There is no WASM
      // gemm in this version, so matmul stays on JS [d|s|g]gemm above.
      prepare: (d) => {
        const a = d.arrays['a']!;
        const p = blasPfx(a);
        const ddot = p === 'g' ? blas.base.gdot : blas.base.wasm[`${p}dot`];
        return { x: toTyped(a), y: toTyped(d.arrays['b']!), n: a.data.length, ddot };
      },
      run: (p: any) => p.ddot(p.n, p.x, 1, p.y, 1),
    },
  },
};
