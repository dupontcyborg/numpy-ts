/**
 * jax-js adapter — WASM backend, float32-native, lazy/async.
 * Ownership: ops consume operands, so reused inputs are passed as `.ref` borrows
 * and outputs are materialized (`await out.data()`, which also frees the result).
 */

import * as jax from '@jax-js/jax';
import { is2D, toNested } from '../lib/shape';
import type { JsLibAdapter, OpDef } from '../lib/types';

const np: any = (jax as any).numpy;
const DType: any = (jax as any).DType;
const JAX_DT: Record<string, any> = {
  float64: DType.Float64, float32: DType.Float32, float16: DType.Float16,
  int32: DType.Int32, uint32: DType.Uint32, bool: DType.Bool,
};

const mk = (d: any) => {
  const base = np.array(toNested(d) as any);
  return d.dtype === 'float32' ? base : base.astype(JAX_DT[d.dtype]);
};
const materialize = async (r: any) => { await r.data(); };

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
/** reduction with optional axis (jax axis is positional: np.fn(a, axis)). */
function reduce(fn: (a: any, axis?: number) => any): OpDef {
  return {
    prepare: (d) => ({ a: mk(d.arrays['a']), axis: d.params['axis'] as number | undefined }),
    run: (p: any) => (p.axis == null ? fn(p.a.ref) : fn(p.a.ref, p.axis)),
    materialize,
    disposePrepared: (p: any) => p.a?.dispose?.(),
  };
}

export const jaxjs: JsLibAdapter = {
  name: 'jax-js',
  pkg: '@jax-js/jax',
  dtypes: ['float16', 'float32', 'float64', 'int32', 'uint32', 'bool'],
  async init() {
    await (jax as any).init();
  },
  ops: {
    // arithmetic
    add: binary((a, b) => np.add(a, b)),
    subtract: binary((a, b) => np.subtract(a, b)),
    multiply: binary((a, b) => np.multiply(a, b)),
    divide: binary((a, b) => np.divide(a, b)),
    power: binary((a, b) => np.power(a, b)),
    mod: binary((a, b) => np.remainder(a, b)),
    remainder: binary((a, b) => np.remainder(a, b)),
    floor_divide: binary((a, b) => np.floorDivide(a, b)),
    negative: unary((a) => np.negative(a)),
    absolute: unary((a) => np.absolute(a)),
    square: unary((a) => np.square(a)),
    reciprocal: unary((a) => np.reciprocal(a)),
    sign: unary((a) => np.sign(a)),
    cbrt: unary((a) => np.cbrt(a)),
    // math / exp / log
    sqrt: unary((a) => np.sqrt(a)),
    exp: unary((a) => np.exp(a)),
    exp2: unary((a) => np.exp2(a)),
    expm1: unary((a) => np.expm1(a)),
    log: unary((a) => np.log(a)),
    log2: unary((a) => np.log2(a)),
    log10: unary((a) => np.log10(a)),
    log1p: unary((a) => np.log1p(a)),
    floor: unary((a) => np.floor(a)),
    ceil: unary((a) => np.ceil(a)),
    round: unary((a) => np.round(a)),
    trunc: unary((a) => np.trunc(a)),
    // trig / hyperbolic
    sin: unary((a) => np.sin(a)),
    cos: unary((a) => np.cos(a)),
    tan: unary((a) => np.tan(a)),
    arcsin: unary((a) => np.arcsin(a)),
    arccos: unary((a) => np.arccos(a)),
    arctan: unary((a) => np.arctan(a)),
    arcsinh: unary((a) => np.arcsinh(a)),
    arccosh: unary((a) => np.arccosh(a)),
    arctanh: unary((a) => np.arctanh(a)),
    sinh: unary((a) => np.sinh(a)),
    cosh: unary((a) => np.cosh(a)),
    tanh: unary((a) => np.tanh(a)),
    arctan2: binary((a, b) => np.arctan2(a, b)),
    hypot: binary((a, b) => np.hypot(a, b)),
    deg2rad: unary((a) => np.deg2rad(a)),
    rad2deg: unary((a) => np.rad2deg(a)),
    maximum: binary((a, b) => np.maximum(a, b)),
    minimum: binary((a, b) => np.minimum(a, b)),
    clip: { prepare: (d) => mk(d.arrays['a']), run: (a: any) => np.clip(a.ref, 10, 100), materialize,
      disposePrepared: (a: any) => a?.dispose?.() },
    // reductions (+axis)
    sum: reduce((a, ax) => np.sum(a, ax)),
    mean: reduce((a, ax) => np.mean(a, ax)),
    max: reduce((a, ax) => np.max(a, ax)),
    min: reduce((a, ax) => np.min(a, ax)),
    prod: reduce((a, ax) => np.prod(a, ax)),
    std: reduce((a, ax) => np.std(a, ax)),
    var: reduce((a, ax) => np.var_(a, ax)),
    argmin: reduce((a, ax) => np.argmin(a, ax)),
    argmax: reduce((a, ax) => np.argmax(a, ax)),
    cumsum: unary((a) => np.cumsum(a)),
    ptp: unary((a) => np.ptp(a)),
    // logic / bitwise
    logical_and: binary((a, b) => np.logicalAnd(a, b)),
    logical_or: binary((a, b) => np.logicalOr(a, b)),
    logical_xor: binary((a, b) => np.logicalXor(a, b)),
    logical_not: unary((a) => np.logicalNot(a)),
    isfinite: unary((a) => np.isfinite(a)),
    isinf: unary((a) => np.isinf(a)),
    isnan: unary((a) => np.isnan(a)),
    bitwise_and: binary((a, b) => np.bitwiseAnd(a, b)),
    bitwise_or: binary((a, b) => np.bitwiseOr(a, b)),
    bitwise_xor: binary((a, b) => np.bitwiseXor(a, b)),
    bitwise_not: unary((a) => np.bitwiseInvert(a)),
    invert: unary((a) => np.bitwiseInvert(a)),
    left_shift: binary((a, b) => np.bitwiseLeftShift(a, b)),
    right_shift: binary((a, b) => np.bitwiseRightShift(a, b)),
    // linalg products
    matmul: {
      prepare: (d) => {
        if (!is2D(d.arrays['a']!) || !is2D(d.arrays['b']!)) throw new Error('matmul needs 2D');
        return [mk(d.arrays['a']), mk(d.arrays['b'])];
      },
      run: (p: any) => np.matmul(p[0].ref, p[1].ref),
      materialize,
      disposePrepared: (p: any) => { p[0]?.dispose?.(); p[1]?.dispose?.(); },
    },
    transpose: unary((a) => np.transpose(a)), // lazy; .data() materializes a copy
    dot: binary((a, b) => np.dot(a, b)),
    inner: binary((a, b) => np.inner(a, b)),
    outer: binary((a, b) => np.outer(a, b)),
    trace: unary((a) => np.trace(a)),
    tensordot: binary((a, b) => np.tensordot(a, b)),
    // manipulation (data-moving)
    concatenate: binary((a, b) => np.concatenate([a, b], 0)),
    concat: binary((a, b) => np.concatenate([a, b], 0)),
    stack: binary((a, b) => np.stack([a, b], 0)),
    vstack: binary((a, b) => np.vstack([a, b])),
    hstack: binary((a, b) => np.hstack([a, b])),
    tile: unary((a) => np.tile(a, [2, 2])),
    repeat: unary((a) => np.repeat(a, 2)),
    // linalg decompositions
    linalg_det: unary((a) => np.linalg.det(a)),
    linalg_inv: unary((a) => np.linalg.inv(a)),
    linalg_solve: binary((a, b) => np.linalg.solve(a, b)),
    linalg_norm: unary((a) => np.linalg.vectorNorm(a)),
  },
};
