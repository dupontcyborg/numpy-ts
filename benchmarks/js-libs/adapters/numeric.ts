/**
 * numeric.js adapter — pure-JS float64, classic linalg + element-wise.
 * Element-wise/matmul on nested arrays; reductions on the flat array.
 */

import numericPkg from 'numeric';
import { is2D, toNested } from '../lib/shape';
import type { JsLibAdapter, OpDef } from '../lib/types';

const numeric: any = (numericPkg as any).default ?? numericPkg;

function unary(fn: (a: any) => any): OpDef {
  return { prepare: (d) => toNested(d.arrays['a']!), run: (a) => fn(a) };
}
function binary(fn: (a: any, b: any) => any): OpDef {
  return { prepare: (d) => [toNested(d.arrays['a']!), toNested(d.arrays['b']!)], run: (p: any) => fn(p[0], p[1]) };
}
function reduceFlat(fn: (a: number[]) => any): OpDef {
  return {
    prepare: (d) => {
      if (d.params['axis'] != null) throw new Error('axis reduction unsupported');
      return d.arrays['a']!.data;
    },
    run: (a: any) => fn(a),
  };
}

export const numericAdapter: JsLibAdapter = {
  name: 'numeric',
  pkg: 'numeric',
  regimes: ['float64'],
  dtypes: ['float64'],
  ops: {
    add: binary((a, b) => numeric.add(a, b)),
    subtract: binary((a, b) => numeric.sub(a, b)),
    multiply: binary((a, b) => numeric.mul(a, b)),
    divide: binary((a, b) => numeric.div(a, b)),
    power: binary((a, b) => numeric.pow(a, b)),
    negative: unary((a) => numeric.neg(a)),
    absolute: unary((a) => numeric.abs(a)),
    sqrt: unary((a) => numeric.sqrt(a)),
    exp: unary((a) => numeric.exp(a)),
    log: unary((a) => numeric.log(a)),
    sin: unary((a) => numeric.sin(a)),
    cos: unary((a) => numeric.cos(a)),
    tan: unary((a) => numeric.tan(a)),
    sum: reduceFlat((a) => numeric.sum(a)),
    prod: reduceFlat((a) => numeric.prod(a)),
    max: reduceFlat((a) => numeric.sup(a)),
    min: reduceFlat((a) => numeric.inf(a)),
    mean: reduceFlat((a) => numeric.sum(a) / a.length),
    matmul: {
      prepare: (d) => {
        if (!is2D(d.arrays['a']!) || !is2D(d.arrays['b']!)) throw new Error('matmul needs 2D');
        return [toNested(d.arrays['a']!), toNested(d.arrays['b']!)];
      },
      run: (p: any) => numeric.dot(p[0], p[1]),
    },
    transpose: {
      prepare: (d) => toNested(d.arrays['a']!),
      run: (a) => numeric.transpose(a),
    },
    dot: {
      prepare: (d) => [d.arrays['a']!.data, d.arrays['b']!.data],
      run: (p: any) => numeric.dot(p[0], p[1]),
    },
  },
};
