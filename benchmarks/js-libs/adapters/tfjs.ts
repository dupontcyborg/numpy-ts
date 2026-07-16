/**
 * TensorFlow.js adapter — WASM backend, float32-only, functional API.
 * Tensors are immutable (inputs persist); each result is materialized
 * (`dataSync`) then disposed inside timing.
 */

import * as tf from '@tensorflow/tfjs-core';
import '@tensorflow/tfjs-backend-wasm';
import { is2D, tfjsDtype } from '../lib/shape';
import type { ArrayData, JsLibAdapter, OpDef } from '../lib/types';

const mk = (d: ArrayData) => tf.tensor(d.data, d.shape.length ? d.shape : [d.data.length], tfjsDtype(d.dtype));
const materialize = (r: any) => { (r as tf.Tensor).dataSync(); };
const disposeResult = (r: any) => (r as tf.Tensor)?.dispose?.();

function unary(fn: (a: any) => any): OpDef {
  return { prepare: (d) => mk(d.arrays['a']!), run: (a) => fn(a), materialize, disposeResult,
    disposePrepared: (a: any) => a?.dispose?.() };
}
function binary(fn: (a: any, b: any) => any): OpDef {
  return {
    prepare: (d) => [mk(d.arrays['a']!), mk(d.arrays['b']!)],
    run: (p: any) => fn(p[0], p[1]),
    materialize, disposeResult,
    disposePrepared: (p: any) => { p[0]?.dispose?.(); p[1]?.dispose?.(); },
  };
}
function reduce(fn: (a: any, axis?: number) => any): OpDef {
  return {
    prepare: (d) => ({ a: mk(d.arrays['a']!), axis: d.params['axis'] as number | undefined }),
    run: (p: any) => (p.axis == null ? fn(p.a) : fn(p.a, p.axis)),
    materialize, disposeResult,
    disposePrepared: (p: any) => p.a?.dispose?.(),
  };
}

export const tfjs: JsLibAdapter = {
  name: 'tfjs',
  pkg: '@tensorflow/tfjs-core',
  // complex64 exists in tfjs but needs tf.complex(re,im) construction — not
  // bridged here, so we don't claim it (avoids a coerced comparison).
  dtypes: ['float32', 'int32', 'bool'],
  async init() {
    await tf.setBackend('wasm');
    await tf.ready();
  },
  ops: {
    add: binary((a, b) => tf.add(a, b)),
    subtract: binary((a, b) => tf.sub(a, b)),
    multiply: binary((a, b) => tf.mul(a, b)),
    divide: binary((a, b) => tf.div(a, b)),
    power: binary((a, b) => tf.pow(a, b)),
    mod: binary((a, b) => tf.mod(a, b)),
    floor_divide: binary((a, b) => tf.floorDiv(a, b)),
    negative: unary((a) => tf.neg(a)),
    absolute: unary((a) => tf.abs(a)),
    square: unary((a) => tf.square(a)),
    reciprocal: unary((a) => tf.reciprocal(a)),
    sign: unary((a) => tf.sign(a)),
    sqrt: unary((a) => tf.sqrt(a)),
    exp: unary((a) => tf.exp(a)),
    expm1: unary((a) => tf.expm1(a)),
    log: unary((a) => tf.log(a)),
    log1p: unary((a) => tf.log1p(a)),
    floor: unary((a) => tf.floor(a)),
    ceil: unary((a) => tf.ceil(a)),
    round: unary((a) => tf.round(a)),
    sin: unary((a) => tf.sin(a)),
    cos: unary((a) => tf.cos(a)),
    tan: unary((a) => tf.tan(a)),
    arcsin: unary((a) => tf.asin(a)),
    arccos: unary((a) => tf.acos(a)),
    arctan: unary((a) => tf.atan(a)),
    arcsinh: unary((a) => tf.asinh(a)),
    arccosh: unary((a) => tf.acosh(a)),
    arctanh: unary((a) => tf.atanh(a)),
    sinh: unary((a) => tf.sinh(a)),
    cosh: unary((a) => tf.cosh(a)),
    tanh: unary((a) => tf.tanh(a)),
    arctan2: binary((a, b) => tf.atan2(a, b)),
    maximum: binary((a, b) => tf.maximum(a, b)),
    minimum: binary((a, b) => tf.minimum(a, b)),
    clip: unary((a) => tf.clipByValue(a, 10, 100)),
    sum: reduce((a, ax) => tf.sum(a, ax)),
    mean: reduce((a, ax) => tf.mean(a, ax)),
    max: reduce((a, ax) => tf.max(a, ax)),
    min: reduce((a, ax) => tf.min(a, ax)),
    prod: reduce((a, ax) => tf.prod(a, ax)),
    argmin: reduce((a, ax) => tf.argMin(a, ax ?? 0)),
    argmax: reduce((a, ax) => tf.argMax(a, ax ?? 0)),
    cumsum: unary((a) => tf.cumsum(a)),
    cumprod: unary((a) => tf.cumprod(a)),
    logical_and: binary((a, b) => tf.logicalAnd(a, b)),
    logical_or: binary((a, b) => tf.logicalOr(a, b)),
    logical_xor: binary((a, b) => tf.logicalXor(a, b)),
    logical_not: unary((a) => tf.logicalNot(a)),
    isnan: unary((a) => tf.isNaN(a)),
    isinf: unary((a) => tf.isInf(a)),
    isfinite: unary((a) => tf.isFinite(a)),
    bitwise_and: binary((a, b) => tf.bitwiseAnd(a, b)),
    matmul: {
      prepare: (d) => {
        if (!is2D(d.arrays['a']!) || !is2D(d.arrays['b']!)) throw new Error('matmul needs 2D');
        return [mk(d.arrays['a']!), mk(d.arrays['b']!)];
      },
      run: (p: any) => tf.matMul(p[0], p[1]),
      materialize, disposeResult,
      disposePrepared: (p: any) => { p[0]?.dispose?.(); p[1]?.dispose?.(); },
    },
    transpose: unary((a) => tf.transpose(a)), // real op; dataSync materializes
    dot: binary((a, b) => tf.dot(a, b)),
    // manipulation (data-moving)
    concatenate: binary((a, b) => tf.concat([a, b], 0)),
    concat: binary((a, b) => tf.concat([a, b], 0)),
    stack: binary((a, b) => tf.stack([a, b], 0)),
    tile: unary((a) => tf.tile(a, [2, 2])),
  },
};
