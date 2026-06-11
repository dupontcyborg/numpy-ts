/**
 * jax-js adapter — WASM backend, float32-native, lazy/async.
 * Ownership: ops consume operands, so reused inputs are passed as `.ref` borrows
 * and outputs are materialized (`await out.data()`) then disposed inside timing.
 */

import * as jax from '@jax-js/jax';
import { is2D, toNested } from '../lib/shape';
import type { JsLibAdapter, OpDef } from '../lib/types';

const np: any = (jax as any).numpy;

const mk = (d: any) => np.array(toNested(d) as any);
// .data() forces compute AND consumes the result's refcount (frees it), so there
// is no separate disposeResult. Operands survive: `a.ref` bumps refcount +1 and
// the op consumes one, leaving the prepared inputs stable across iterations.
const materialize = async (r: any) => {
  await r.data();
};

function unary(fn: (a: any) => any): OpDef {
  return { prepare: (d) => mk(d.arrays['a']), run: (a: any) => fn(a.ref), materialize,
    disposePrepared: (a: any) => a?.dispose?.() };
}
function binary(fn: (a: any, b: any) => any): OpDef {
  return {
    prepare: (d) => [mk(d.arrays['a']), mk(d.arrays['b'])],
    run: (p: any) => fn(p[0].ref, p[1].ref),
    materialize,
    disposePrepared: (p: any) => { p[0]?.dispose?.(); p[1]?.dispose?.(); },
  };
}
function reduce(fn: (a: any) => any): OpDef {
  return {
    prepare: (d) => {
      if (d.params['axis'] != null) throw new Error('axis reduction unsupported');
      return mk(d.arrays['a']);
    },
    run: (a: any) => fn(a.ref), materialize,
    disposePrepared: (a: any) => a?.dispose?.(),
  };
}

export const jaxjs: JsLibAdapter = {
  name: 'jax-js',
  pkg: '@jax-js/jax',
  regimes: ['float32'],
  dtypes: ['float16', 'float32', 'float64', 'int32', 'uint32', 'bool'],
  async init() {
    // Initializes the WASM runtime + default device; required before .ref works.
    await (jax as any).init();
  },
  ops: {
    add: binary((a, b) => np.add(a, b)),
    subtract: binary((a, b) => np.subtract(a, b)),
    multiply: binary((a, b) => np.multiply(a, b)),
    divide: binary((a, b) => np.divide(a, b)),
    power: binary((a, b) => np.power(a, b)),
    negative: unary((a) => np.negative(a)),
    absolute: unary((a) => np.absolute(a)),
    square: unary((a) => np.square(a)),
    sqrt: unary((a) => np.sqrt(a)),
    exp: unary((a) => np.exp(a)),
    log: unary((a) => np.log(a)),
    log2: unary((a) => np.log2(a)),
    log10: unary((a) => np.log10(a)),
    sin: unary((a) => np.sin(a)),
    cos: unary((a) => np.cos(a)),
    tan: unary((a) => np.tan(a)),
    sum: reduce((a) => np.sum(a)),
    mean: reduce((a) => np.mean(a)),
    max: reduce((a) => np.max(a)),
    min: reduce((a) => np.min(a)),
    prod: reduce((a) => np.prod(a)),
    std: reduce((a) => np.std(a)),
    var: reduce((a) => np.var_(a)),
    matmul: {
      prepare: (d) => {
        if (!is2D(d.arrays['a']!) || !is2D(d.arrays['b']!)) throw new Error('matmul needs 2D');
        return [mk(d.arrays['a']), mk(d.arrays['b'])];
      },
      run: (p: any) => np.matmul(p[0].ref, p[1].ref),
      materialize,
      disposePrepared: (p: any) => { p[0]?.dispose?.(); p[1]?.dispose?.(); },
    },
    transpose: unary((a) => np.transpose(a)),
    dot: binary((a, b) => np.dot(a, b)),
  },
};
