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
  return {
    prepare: (d) => {
      const b = d.arrays['b']!;
      return [toNested(d.arrays['a']!), b.data.length === 1 ? b.data[0] : toNested(b)]; // scalar broadcast
    },
    run: (p: any) => fn(p[0], p[1]),
  };
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
function mat2D(d: any, k: string): number[][] {
  if (!is2D(d.arrays[k])) throw new Error('needs 2D');
  return toNested(d.arrays[k]) as number[][];
}

export const numericAdapter: JsLibAdapter = {
  name: 'numeric',
  pkg: 'numeric',
  dtypes: ['float64'],
  ops: {
    add: binary((a, b) => numeric.add(a, b)),
    subtract: binary((a, b) => numeric.sub(a, b)),
    multiply: binary((a, b) => numeric.mul(a, b)),
    divide: binary((a, b) => numeric.div(a, b)),
    power: binary((a, b) => numeric.pow(a, b)),
    mod: binary((a, b) => numeric.mod(a, b)),
    negative: unary((a) => numeric.neg(a)),
    absolute: unary((a) => numeric.abs(a)),
    sqrt: unary((a) => numeric.sqrt(a)),
    exp: unary((a) => numeric.exp(a)),
    log: unary((a) => numeric.log(a)),
    floor: unary((a) => numeric.floor(a)),
    ceil: unary((a) => numeric.ceil(a)),
    round: unary((a) => numeric.round(a)),
    sin: unary((a) => numeric.sin(a)),
    cos: unary((a) => numeric.cos(a)),
    tan: unary((a) => numeric.tan(a)),
    arcsin: unary((a) => numeric.asin(a)),
    arccos: unary((a) => numeric.acos(a)),
    arctan: unary((a) => numeric.atan(a)),
    arctan2: binary((a, b) => numeric.atan2(a, b)),
    sum: reduceFlat((a) => numeric.sum(a)),
    prod: reduceFlat((a) => numeric.prod(a)),
    max: reduceFlat((a) => numeric.sup(a)),
    min: reduceFlat((a) => numeric.inf(a)),
    mean: reduceFlat((a) => numeric.sum(a) / a.length),
    matmul: {
      prepare: (d) => [mat2D(d, 'a'), mat2D(d, 'b')],
      run: (p: any) => numeric.dot(p[0], p[1]),
    },
    transpose: { prepare: (d) => mat2D(d, 'a'), run: (a) => numeric.transpose(a) },
    dot: {
      prepare: (d) => [d.arrays['a']!.data, d.arrays['b']!.data],
      run: (p: any) => numeric.dot(p[0], p[1]),
    },
    // linalg decompositions (numeric's strength)
    linalg_det: { prepare: (d) => mat2D(d, 'a'), run: (a) => numeric.det(a) },
    linalg_inv: { prepare: (d) => mat2D(d, 'a'), run: (a) => numeric.inv(a) },
    linalg_eig: { prepare: (d) => mat2D(d, 'a'), run: (a) => numeric.eig(a) },
    linalg_svd: { prepare: (d) => mat2D(d, 'a'), run: (a) => numeric.svd(a) },
    linalg_norm: { prepare: (d) => mat2D(d, 'a'), run: (a) => numeric.norm2(a) },
    linalg_solve: {
      prepare: (d) => ({ a: mat2D(d, 'a'), b: d.arrays['b']!.data }),
      run: (p: any) => numeric.solve(p.a, p.b),
    },
  },
};
