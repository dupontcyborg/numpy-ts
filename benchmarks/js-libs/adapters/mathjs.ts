/**
 * mathjs adapter — broad pure-JS, sync, float64 (JS number) storage.
 * Element-wise products use dotMultiply/dotDivide/dotPow; `multiply` is matmul.
 * Scalar fns aren't auto-mapped over matrices, so element-wise unary uses
 * matrix.map with arity-1 wrappers (mathjs leaks (value,index,matrix) otherwise).
 */

import { create, all } from 'mathjs';
import { is2D, toNested } from '../lib/shape';
import type { JsLibAdapter, OpDef, SpecData } from '../lib/types';

const math = create(all, {});
const mat = (d: SpecData, k: string) => math.matrix(toNested(d.arrays[k]!) as any);

function unary(fn: (x: any) => any): OpDef {
  return { prepare: (d) => mat(d, 'a'), run: (m) => (m as any).map((x: any) => fn(x)) };
}
function matrixUnary(fn: (m: any) => any): OpDef {
  return { prepare: (d) => mat(d, 'a'), run: (m) => fn(m) };
}
function binary(fn: (a: any, b: any) => any): OpDef {
  return { prepare: (d) => [mat(d, 'a'), mat(d, 'b')], run: (p) => fn((p as any[])[0], (p as any[])[1]) };
}
/** reduction with optional axis (mathjs accepts a dim arg). */
function reduce(full: (m: any) => any, axisFn?: (m: any, dim: number) => any): OpDef {
  return {
    prepare: (d) => ({ m: mat(d, 'a'), axis: d.params['axis'] as number | undefined }),
    run: (p: any) => {
      if (p.axis == null) return full(p.m);
      if (!axisFn) throw new Error('axis unsupported');
      return axisFn(p.m, p.axis);
    },
  };
}

export const mathjsAdapter: JsLibAdapter = {
  name: 'mathjs',
  pkg: 'mathjs',
  dtypes: ['float64'],
  ops: {
    add: binary((a, b) => math.add(a, b)),
    subtract: binary((a, b) => math.subtract(a, b)),
    multiply: binary((a, b) => math.dotMultiply(a, b)),
    divide: binary((a, b) => math.dotDivide(a, b)),
    power: binary((a, b) => math.dotPow(a, b)),
    mod: binary((a, b) => math.mod(a, b)),
    negative: unary((x) => math.unaryMinus(x)),
    absolute: unary((x) => math.abs(x)),
    square: unary((x) => math.square(x)),
    reciprocal: unary((x) => math.divide(1, x)),
    sign: unary((x) => math.sign(x)),
    cbrt: unary((x) => math.cbrt(x)),
    sqrt: unary((x) => math.sqrt(x)),
    exp: unary((x) => math.exp(x)),
    expm1: unary((x) => math.expm1(x)),
    log: unary((x) => math.log(x)),
    log2: unary((x) => math.log2(x)),
    log10: unary((x) => math.log10(x)),
    log1p: unary((x) => math.log1p(x)),
    floor: unary((x) => math.floor(x)),
    ceil: unary((x) => math.ceil(x)),
    round: unary((x) => math.round(x)),
    trunc: unary((x) => math.fix(x)),
    sin: unary((x) => math.sin(x)),
    cos: unary((x) => math.cos(x)),
    tan: unary((x) => math.tan(x)),
    arcsin: unary((x) => math.asin(x)),
    arccos: unary((x) => math.acos(x)),
    arctan: unary((x) => math.atan(x)),
    arcsinh: unary((x) => math.asinh(x)),
    arccosh: unary((x) => math.acosh(x)),
    arctanh: unary((x) => math.atanh(x)),
    sinh: unary((x) => math.sinh(x)),
    cosh: unary((x) => math.cosh(x)),
    tanh: unary((x) => math.tanh(x)),
    sum: reduce((m) => math.sum(m), (m, d) => math.sum(m, d)),
    mean: reduce((m) => math.mean(m), (m, d) => math.mean(m, d)),
    max: reduce((m) => math.max(m), (m, d) => math.max(m, d)),
    min: reduce((m) => math.min(m), (m, d) => math.min(m, d)),
    prod: reduce((m) => math.prod(m)),
    std: reduce((m) => math.std(m), (m, d) => math.std(m, d)),
    var: reduce((m) => math.variance(m), (m, d) => math.variance(m, d)),
    cumsum: matrixUnary((m) => math.cumsum(m)),
    median: matrixUnary((m) => math.median(m)),
    matmul: {
      prepare: (d) => {
        if (!is2D(d.arrays['a']!) || !is2D(d.arrays['b']!)) throw new Error('matmul needs 2D');
        return [mat(d, 'a'), mat(d, 'b')];
      },
      run: (p: any) => math.multiply(p[0], p[1]),
    },
    transpose: matrixUnary((m) => math.transpose(m)),
    concatenate: binary((a, b) => math.concat(a, b, 0)),
    concat: binary((a, b) => math.concat(a, b, 0)),
    dot: { prepare: (d) => [toNested(d.arrays['a']!), toNested(d.arrays['b']!)], run: (p: any) => math.dot(p[0], p[1]) },
    trace: matrixUnary((m) => math.trace(m)),
    kron: binary((a, b) => math.kron(a, b)),
    // linalg decompositions
    linalg_det: matrixUnary((m) => math.det(m)),
    linalg_inv: matrixUnary((m) => math.inv(m)),
    linalg_norm: matrixUnary((m) => math.norm(m)),
    linalg_qr: matrixUnary((m) => math.qr(m)),
    linalg_eig: matrixUnary((m) => math.eigs(m)),
    linalg_solve: { prepare: (d) => [mat(d, 'a'), mat(d, 'b')], run: (p: any) => math.lusolve(p[0], p[1]) },
  },
};
