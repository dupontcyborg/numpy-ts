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
const pfx = (d: ArrayData) => (d.dtype === 'float32' ? 's' : 'd'); // stats/blas kernel prefix

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
      const x = mkND(d.arrays['a']!);
      const y = mkND(d.arrays['b']!);
      const out = nd.zeros(d.arrays['a']!.shape, { dtype: d.arrays['a']!.dtype });
      return { x, y, out };
    },
    run: (p: any) => {
      binary([p.x, p.y, p.out], kernel);
      return p.out;
    },
  };
}

function reduceOp(fnName: (d: ArrayData) => string): OpDef {
  return {
    prepare: (d) => {
      if (d.params['axis'] != null) throw new Error('axis reduction unsupported');
      const a = d.arrays['a']!;
      return { buf: toTyped(a), n: a.data.length, fn: stats[fnName(a)] };
    },
    run: (p: any) => p.fn(p.n, p.buf, 1),
  };
}

export const stdlib: JsLibAdapter = {
  name: 'stdlib',
  pkg: '@stdlib/ndarray',
  regimes: ['float64', 'float32'],
  dtypes: ['float16', 'float32', 'float64', 'int8', 'int16', 'int32', 'uint8', 'uint16', 'uint32', 'bool', 'complex64', 'complex128'],
  ops: {
    add: binaryOp(ops.add),
    subtract: binaryOp(ops.sub),
    multiply: binaryOp(ops.mul),
    divide: binaryOp(ops.div),
    power: binaryOp(special.pow),
    absolute: unaryOp(special.abs),
    sqrt: unaryOp(special.sqrt),
    exp: unaryOp(special.exp),
    log: unaryOp(special.ln),
    log2: unaryOp(special.log2),
    log10: unaryOp(special.log10),
    sin: unaryOp(special.sin),
    cos: unaryOp(special.cos),
    tan: unaryOp(special.tan),
    sum: {
      prepare: (d) => {
        if (d.params['axis'] != null) throw new Error('axis reduction unsupported');
        const a = d.arrays['a']!;
        return { buf: toTyped(a), n: a.data.length };
      },
      run: (p: any) => dsum(p.n, p.buf, 1),
    },
    mean: reduceOp((d) => `${pfx(d)}mean`),
    max: reduceOp((d) => `${pfx(d)}max`),
    min: reduceOp((d) => `${pfx(d)}min`),
    std: reduceOp((d) => `${pfx(d)}stdev`),
    var: reduceOp((d) => `${pfx(d)}variance`),
    matmul: {
      prepare: (d) => {
        const A = d.arrays['a']!;
        const B = d.arrays['b']!;
        if (!is2D(A) || !is2D(B)) throw new Error('matmul needs 2D');
        const [M, K] = A.shape as [number, number];
        const N = B.shape[1]!;
        return { M, N, K, A: toTyped(A), B: toTyped(B), C: new Float64Array(M * N), gemm: blas.base[`${pfx(A)}gemm`] };
      },
      run: (p: any) => {
        p.gemm('row-major', 'no-transpose', 'no-transpose', p.M, p.N, p.K, 1.0, p.A, p.K, p.B, p.N, 0.0, p.C, p.N);
        return p.C;
      },
    },
    transpose: {
      prepare: (d) => mkND(d.arrays['a']!),
      run: (x: any) => nd.transpose(x),
    },
    dot: {
      // Opt into stdlib's WASM backend. In this version stdlib's WASM surface is
      // BLAS level-1 (dot, nrm2) — there is no WASM gemm, so matmul stays on the
      // JS dgemm above, and element-wise/reductions are JS scalar kernels.
      prepare: (d) => {
        const a = d.arrays['a']!;
        return { x: toTyped(a), y: toTyped(d.arrays['b']!), n: a.data.length, ddot: blas.base.wasm[`${pfx(a)}dot`] };
      },
      run: (p: any) => p.ddot(p.n, p.x, 1, p.y, 1),
    },
  },
};
