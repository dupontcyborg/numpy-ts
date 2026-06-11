/**
 * @d4c/numjs adapter — NumPy-shaped pure-JS on scijs/ndarray. Eager.
 * Supports f32/f64 + integer storage; no var/prod/log2/log10.
 */

import njPkg from '@d4c/numjs';
import { is2D, toNested } from '../lib/shape';
import type { ArrayData, JsLibAdapter, OpDef } from '../lib/types';

const nj: any = (njPkg as any).default ?? njPkg;
const mk = (d: ArrayData) => nj.array(toNested(d) as any, d.dtype === 'float32' ? 'float32' : 'float64');

function unary(fn: (a: any) => any): OpDef {
  return { prepare: (d) => mk(d.arrays['a']!), run: (a) => fn(a) };
}
function binary(fn: (a: any, b: any) => any): OpDef {
  return { prepare: (d) => [mk(d.arrays['a']!), mk(d.arrays['b']!)], run: (p: any) => fn(p[0], p[1]) };
}
function reduce(fn: (a: any) => any): OpDef {
  return {
    prepare: (d) => {
      if (d.params['axis'] != null) throw new Error('axis reduction unsupported');
      return mk(d.arrays['a']!);
    },
    run: (a) => fn(a),
  };
}

export const numjsAdapter: JsLibAdapter = {
  name: 'numjs',
  pkg: '@d4c/numjs',
  regimes: ['float64', 'float32'],
  dtypes: ['float32', 'float64', 'int8', 'int16', 'int32', 'uint8', 'uint16', 'uint32'],
  ops: {
    add: binary((a, b) => nj.add(a, b)),
    subtract: binary((a, b) => nj.subtract(a, b)),
    multiply: binary((a, b) => nj.multiply(a, b)),
    divide: binary((a, b) => nj.divide(a, b)),
    power: binary((a, b) => nj.power(a, b)),
    negative: unary((a) => nj.negative(a)),
    absolute: unary((a) => nj.abs(a)),
    sqrt: unary((a) => nj.sqrt(a)),
    exp: unary((a) => nj.exp(a)),
    log: unary((a) => nj.log(a)),
    sin: unary((a) => nj.sin(a)),
    cos: unary((a) => nj.cos(a)),
    tan: unary((a) => nj.tan(a)),
    sum: reduce((a) => a.sum()),
    mean: reduce((a) => a.mean()),
    max: reduce((a) => a.max()),
    min: reduce((a) => a.min()),
    std: reduce((a) => a.std()),
    matmul: {
      prepare: (d) => {
        if (!is2D(d.arrays['a']!) || !is2D(d.arrays['b']!)) throw new Error('matmul needs 2D');
        return [mk(d.arrays['a']!), mk(d.arrays['b']!)];
      },
      run: (p: any) => nj.dot(p[0], p[1]),
    },
    transpose: unary((a) => a.transpose()),
    dot: {
      prepare: (d) => [mk(d.arrays['a']!), mk(d.arrays['b']!)],
      run: (p: any) => nj.dot(p[0], p[1]),
    },
  },
};
