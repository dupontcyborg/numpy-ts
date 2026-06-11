/**
 * mathjs adapter — broad pure-JS, sync, float64 (JS number) storage.
 * Element-wise products use dotMultiply/dotDivide/dotPow; `multiply` is matmul.
 */

import { create, all } from 'mathjs';
import { is2D, toNested } from '../lib/shape';
import type { JsLibAdapter, OpDef, SpecData } from '../lib/types';

const math = create(all, {});

/**
 * Element-wise unary via matrix.map. mathjs scalar fns aren't auto-mapped over
 * matrices, and passing them straight to .map() leaks the (value, index, matrix)
 * args into the fn (e.g. log reads index as a base) — so wrap to arity-1.
 */
function unary(fn: (x: any) => any): OpDef {
  return {
    prepare: (d) => math.matrix(toNested(d.arrays['a']!) as any),
    run: (m) => (m as any).map((x: any) => fn(x)),
  };
}
/** transpose / other whole-matrix unary (no element map). */
function matrixUnary(fn: (m: any) => any): OpDef {
  return {
    prepare: (d) => math.matrix(toNested(d.arrays['a']!) as any),
    run: (m) => fn(m),
  };
}
/** element-wise binary over two matrices. */
function binary(fn: (a: any, b: any) => any): OpDef {
  return {
    prepare: (d) => [math.matrix(toNested(d.arrays['a']!) as any), math.matrix(toNested(d.arrays['b']!) as any)],
    run: (p) => fn((p as any[])[0], (p as any[])[1]),
  };
}
/** full-array reduction (no axis). Throws on axis specs -> recorded N/A. */
function reduce(fn: (m: any) => any): OpDef {
  return {
    prepare: (d) => {
      if (d.params['axis'] != null) throw new Error('axis reduction unsupported');
      return math.matrix(toNested(d.arrays['a']!) as any);
    },
    run: (m) => fn(m),
  };
}

export const mathjsAdapter: JsLibAdapter = {
  name: 'mathjs',
  pkg: 'mathjs',
  regimes: ['float64'],
  dtypes: ['float64'],
  ops: {
    add: binary((a, b) => math.add(a, b)),
    subtract: binary((a, b) => math.subtract(a, b)),
    multiply: binary((a, b) => math.dotMultiply(a, b)),
    divide: binary((a, b) => math.dotDivide(a, b)),
    power: binary((a, b) => math.dotPow(a, b)),
    negative: unary((m) => math.unaryMinus(m)),
    absolute: unary((m) => math.abs(m)),
    square: unary((m) => math.square(m)),
    sqrt: unary((m) => math.sqrt(m)),
    exp: unary((m) => math.exp(m)),
    log: unary((m) => math.log(m)),
    log2: unary((m) => math.log2(m)),
    log10: unary((m) => math.log10(m)),
    sin: unary((m) => math.sin(m)),
    cos: unary((m) => math.cos(m)),
    tan: unary((m) => math.tan(m)),
    sum: reduce((m) => math.sum(m)),
    mean: reduce((m) => math.mean(m)),
    max: reduce((m) => math.max(m)),
    min: reduce((m) => math.min(m)),
    prod: reduce((m) => math.prod(m)),
    std: reduce((m) => math.std(m)),
    var: reduce((m) => math.variance(m)),
    matmul: {
      prepare: (d: SpecData) => {
        if (!is2D(d.arrays['a']!) || !is2D(d.arrays['b']!)) throw new Error('matmul needs 2D');
        return [math.matrix(toNested(d.arrays['a']!) as any), math.matrix(toNested(d.arrays['b']!) as any)];
      },
      run: (p) => math.multiply((p as any[])[0], (p as any[])[1]),
    },
    transpose: matrixUnary((m) => math.transpose(m)),
    dot: {
      prepare: (d: SpecData) => [toNested(d.arrays['a']!), toNested(d.arrays['b']!)],
      run: (p) => math.dot((p as any[])[0], (p as any[])[1]),
    },
  },
};
