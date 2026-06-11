/**
 * ml-matrix adapter — pure-JS float64, 2-D only. 1-D specs are skipped.
 * Instance math mutates in place, so element-wise/unary clone first (numpy-ts
 * also allocates a new array), keeping the comparison fair.
 */

import { Matrix } from 'ml-matrix';
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
function reduce(fn: (a: Matrix) => any): OpDef {
  return {
    prepare: (d) => {
      if (d.params['axis'] != null) throw new Error('axis reduction unsupported');
      return mk(d.arrays['a']!);
    },
    run: (a: any) => fn(a as Matrix),
  };
}

export const mlmatrix: JsLibAdapter = {
  name: 'ml-matrix',
  pkg: 'ml-matrix',
  regimes: ['float64'],
  dtypes: ['float64'],
  ops: {
    add: binary((a, b) => a.add(b)),
    subtract: binary((a, b) => a.sub(b)),
    multiply: binary((a, b) => a.mul(b)),
    divide: binary((a, b) => a.div(b)),
    power: binary((a, b) => a.pow(b)),
    negative: unary((a) => a.mul(-1)),
    absolute: unary((a) => a.abs()),
    sqrt: unary((a) => a.sqrt()),
    exp: unary((a) => a.exp()),
    log: unary((a) => a.log()),
    log2: unary((a) => a.log2()),
    sin: unary((a) => a.sin()),
    cos: unary((a) => a.cos()),
    tan: unary((a) => a.tan()),
    sum: reduce((a) => a.sum()),
    max: reduce((a) => a.max()),
    min: reduce((a) => a.min()),
    mean: reduce((a) => a.mean()),
    prod: reduce((a) => a.product()),
    var: reduce((a) => a.variance()),
    std: reduce((a) => a.standardDeviation()),
    matmul: {
      prepare: (d) => [mk(d.arrays['a']!), mk(d.arrays['b']!)],
      run: (p: any) => (p[0] as Matrix).mmul(p[1]),
    },
    transpose: { prepare: (d) => mk(d.arrays['a']!), run: (a: any) => (a as Matrix).transpose() },
  },
};
