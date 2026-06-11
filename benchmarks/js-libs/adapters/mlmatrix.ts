/**
 * ml-matrix adapter — pure-JS float64, 2-D only (1-D specs are skipped).
 * Instance math mutates in place, so element-wise/unary clone first (numpy-ts
 * also allocates a new array), keeping the comparison fair.
 */

import { Matrix, determinant, inverse, solve, SVD, QR, EigenvalueDecomposition } from 'ml-matrix';
import { is2D, toNested } from '../lib/shape';
import type { ArrayData, JsLibAdapter, OpDef } from '../lib/types';

const mk = (d: ArrayData) => {
  if (!is2D(d)) throw new Error('ml-matrix is 2D-only');
  return new Matrix(toNested(d) as number[][]);
};

function unary(fn: (a: Matrix) => any): OpDef {
  return { prepare: (d) => mk(d.arrays['a']!), run: (a: any) => fn((a as Matrix).clone()) };
}
function binary(fn: (a: Matrix, b: Matrix) => any): OpDef {
  return {
    prepare: (d) => [mk(d.arrays['a']!), mk(d.arrays['b']!)],
    run: (p: any) => fn((p[0] as Matrix).clone(), p[1]),
  };
}
function reduce(full: (a: Matrix) => any, by?: (a: Matrix, axis: number) => any): OpDef {
  return {
    prepare: (d) => ({ m: mk(d.arrays['a']!), axis: d.params['axis'] as number | undefined }),
    run: (p: any) => {
      if (p.axis == null) return full(p.m);
      if (!by) throw new Error('axis unsupported');
      return by(p.m, p.axis);
    },
  };
}
const dim = (axis: number) => (axis === 0 ? 'column' : 'row');

export const mlmatrix: JsLibAdapter = {
  name: 'ml-matrix',
  pkg: 'ml-matrix',
  dtypes: ['float64'],
  ops: {
    add: binary((a, b) => a.add(b)),
    subtract: binary((a, b) => a.sub(b)),
    multiply: binary((a, b) => a.mul(b)),
    divide: binary((a, b) => a.div(b)),
    power: binary((a, b) => a.pow(b)),
    negative: unary((a) => a.mul(-1)),
    absolute: unary((a) => a.abs()),
    sign: unary((a) => a.sign()),
    cbrt: unary((a) => a.cbrt()),
    sqrt: unary((a) => a.sqrt()),
    exp: unary((a) => a.exp()),
    expm1: unary((a) => a.expm1()),
    log: unary((a) => a.log()),
    log2: unary((a) => a.log2()),
    log1p: unary((a) => a.log1p()),
    floor: unary((a) => a.floor()),
    ceil: unary((a) => a.ceil()),
    round: unary((a) => a.round()),
    trunc: unary((a) => a.trunc()),
    sin: unary((a) => a.sin()),
    cos: unary((a) => a.cos()),
    tan: unary((a) => a.tan()),
    arcsin: unary((a) => a.asin()),
    arccos: unary((a) => a.acos()),
    arctan: unary((a) => a.atan()),
    arcsinh: unary((a) => a.asinh()),
    arccosh: unary((a) => a.acosh()),
    arctanh: unary((a) => a.atanh()),
    sinh: unary((a) => a.sinh()),
    cosh: unary((a) => a.cosh()),
    tanh: unary((a) => a.tanh()),
    sum: reduce((a) => a.sum(), (a, ax) => a.sum(dim(ax))),
    max: reduce((a) => a.max(), (a, ax) => a.max(dim(ax))),
    min: reduce((a) => a.min(), (a, ax) => a.min(dim(ax))),
    mean: reduce((a) => a.mean(), (a, ax) => a.mean(dim(ax))),
    prod: reduce((a) => a.product()),
    var: reduce((a) => a.variance()),
    std: reduce((a) => a.standardDeviation()),
    cumsum: unary((a) => a.cumulativeSum()),
    matmul: { prepare: (d) => [mk(d.arrays['a']!), mk(d.arrays['b']!)], run: (p: any) => (p[0] as Matrix).mmul(p[1]) },
    transpose: { prepare: (d) => mk(d.arrays['a']!), run: (a: any) => (a as Matrix).transpose() },
    trace: reduce((a) => a.trace()),
    // linalg decompositions (ml-matrix's strength)
    linalg_det: { prepare: (d) => mk(d.arrays['a']!), run: (a: any) => determinant(a) },
    linalg_inv: { prepare: (d) => mk(d.arrays['a']!), run: (a: any) => inverse(a) },
    linalg_svd: { prepare: (d) => mk(d.arrays['a']!), run: (a: any) => new SVD(a) },
    linalg_qr: { prepare: (d) => mk(d.arrays['a']!), run: (a: any) => new QR(a) },
    linalg_eig: { prepare: (d) => mk(d.arrays['a']!), run: (a: any) => new EigenvalueDecomposition(a) },
    linalg_norm: { prepare: (d) => mk(d.arrays['a']!), run: (a: any) => (a as Matrix).norm() },
    linalg_solve: {
      prepare: (d) => ({ a: mk(d.arrays['a']!), b: Matrix.columnVector(d.arrays['b']!.data) }),
      run: (p: any) => solve(p.a, p.b),
    },
  },
};
