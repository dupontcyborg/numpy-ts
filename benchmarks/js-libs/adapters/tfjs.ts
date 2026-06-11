/**
 * TensorFlow.js adapter — WASM backend, float32-only, functional API.
 * Tensors are immutable (inputs persist); each result is materialized
 * (`dataSync`) then disposed inside timing. log2/log10/std/var absent in core.
 */

import * as tf from '@tensorflow/tfjs-core';
import '@tensorflow/tfjs-backend-wasm';
import { is2D, toNested } from '../lib/shape';
import type { ArrayData, JsLibAdapter, OpDef } from '../lib/types';

const mk = (d: ArrayData) => tf.tensor(d.data, d.shape.length ? d.shape : [d.data.length], 'float32');
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
function reduce(fn: (a: any) => any): OpDef {
  return {
    prepare: (d) => {
      if (d.params['axis'] != null) throw new Error('axis reduction unsupported');
      return mk(d.arrays['a']!);
    },
    run: (a) => fn(a), materialize, disposeResult,
    disposePrepared: (a: any) => a?.dispose?.(),
  };
}

export const tfjs: JsLibAdapter = {
  name: 'tfjs',
  pkg: '@tensorflow/tfjs-core',
  regimes: ['float32'],
  dtypes: ['float32', 'int32', 'bool', 'complex64'],
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
    negative: unary((a) => tf.neg(a)),
    absolute: unary((a) => tf.abs(a)),
    square: unary((a) => tf.square(a)),
    sqrt: unary((a) => tf.sqrt(a)),
    exp: unary((a) => tf.exp(a)),
    log: unary((a) => tf.log(a)),
    sin: unary((a) => tf.sin(a)),
    cos: unary((a) => tf.cos(a)),
    tan: unary((a) => tf.tan(a)),
    sum: reduce((a) => tf.sum(a)),
    mean: reduce((a) => tf.mean(a)),
    max: reduce((a) => tf.max(a)),
    min: reduce((a) => tf.min(a)),
    prod: reduce((a) => tf.prod(a)),
    matmul: {
      prepare: (d) => {
        if (!is2D(d.arrays['a']!) || !is2D(d.arrays['b']!)) throw new Error('matmul needs 2D');
        return [mk(d.arrays['a']!), mk(d.arrays['b']!)];
      },
      run: (p: any) => tf.matMul(p[0], p[1]),
      materialize, disposeResult,
      disposePrepared: (p: any) => { p[0]?.dispose?.(); p[1]?.dispose?.(); },
    },
    transpose: unary((a) => tf.transpose(a)),
    dot: binary((a, b) => tf.dot(a, b)),
  },
};
