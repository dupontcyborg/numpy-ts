/**
 * Pyodide-NumPy adapter — real CPython NumPy running in WASM, driven by the
 * SAME JS-side async timer as every other library. Inputs come from the shared
 * SpecData; per spec we build numpy arrays once and compile a Python op closure,
 * then the timer invokes it once per iteration over the JS<->WASM FFI boundary
 * (that boundary cost is the honest "NumPy-from-JS" measurement). NumPy is eager,
 * so invoking the closure fully computes the result (no separate materialize).
 *
 * Resolves `pyodide` from the repo-root node_modules (already a dev dependency
 * for the existing bench:pyodide path) — not duplicated into js-libs.
 */

import type { Dtype, JsLibAdapter, OpDef, SpecData } from '../lib/types';

let py: any;

const REDUCTIONS = new Set([
  'sum', 'mean', 'max', 'min', 'prod', 'std', 'var', 'argmin', 'argmax',
]);

/** Python expression for one operation (a, optional b, optional ax in scope). */
function pyExpr(op: string, hasB: boolean, hasAxis: boolean): string {
  if (op.startsWith('linalg_')) {
    const f = op.slice('linalg_'.length);
    return hasB ? `np.linalg.${f}(a, b)` : `np.linalg.${f}(a)`;
  }
  switch (op) {
    case 'clip': return 'np.clip(a, 10, 100)';
    case 'tile': return 'np.tile(a, (2, 2))';
    case 'repeat': return 'np.repeat(a, 2)';
    case 'concatenate': case 'concat': return 'np.concatenate([a, b], 0)';
    case 'stack': return 'np.stack([a, b], 0)';
    case 'vstack': return 'np.vstack([a, b])';
    case 'hstack': return 'np.hstack([a, b])';
    case 'transpose': return 'np.ascontiguousarray(np.transpose(a))'; // materialize like the others
    case 'bitwise_not': return 'np.invert(a)';
    case 'var': return hasAxis ? 'np.var(a, axis=ax)' : 'np.var(a)';
  }
  if (hasB) return `np.${op}(a, b)`;
  if (hasAxis && REDUCTIONS.has(op)) return `np.${op}(a, axis=ax)`;
  return `np.${op}(a)`;
}

function makeOpDef(op: string): OpDef {
  return {
    prepare(d: SpecData) {
      const a = d.arrays['a'];
      if (!a) throw new Error('no operand a');
      py.globals.set('_ad', py.toPy(a.data));
      let setup = `a = np.array(_ad, dtype='${a.dtype}')`;
      if (a.shape.length > 1) setup += `.reshape(${a.shape.join(', ')})`;
      const b = d.arrays['b'];
      if (b) {
        py.globals.set('_bd', py.toPy(b.data));
        setup += `\nb = np.array(_bd, dtype='${b.dtype}')`;
        if (b.shape.length > 1) setup += `.reshape(${b.shape.join(', ')})`;
      }
      const ax = d.params['axis'] as number | undefined;
      if (ax != null) setup += `\nax = ${ax}`;
      setup += `\ndef _op():\n    return ${pyExpr(op, !!b, ax != null)}`;
      py.runPython(setup);
      return py.globals.get('_op');
    },
    run: (opFn: any) => opFn(), // NumPy is eager — this fully computes the result
    disposeResult: (r: any) => r?.destroy?.(),
    disposePrepared: (opFn: any) => opFn?.destroy?.(),
  };
}

const ALL: Dtype[] = [
  'float64', 'float32', 'float16', 'int8', 'int16', 'int32', 'int64',
  'uint8', 'uint16', 'uint32', 'uint64', 'bool', 'complex64', 'complex128',
];

export const pyodideNumpy: JsLibAdapter = {
  name: 'numpy(pyodide)',
  pkg: 'pyodide',
  dtypes: ALL, // CPython NumPy supports every dtype
  async init() {
    const { loadPyodide } = await import('pyodide');
    py = await loadPyodide({ stderr: () => {} });
    await py.loadPackage('numpy');
    py.runPython('import numpy as np');
  },
  // NumPy implements every benchmarked op; resolve an OpDef on demand for any.
  ops: new Proxy({} as Record<string, OpDef>, {
    get: (_t, prop) => (typeof prop === 'string' ? makeOpDef(prop) : undefined),
    has: () => true,
  }),
};
