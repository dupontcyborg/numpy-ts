/**
 * Shared helpers for reduction validation tests.
 */

import { expect, beforeAll, afterEach } from 'vitest';
import * as np from '../../../src/full/index';
import { wasmConfig } from '../../../src';
import { runNumPy, arraysClose, checkNumPyAvailable, getPythonInfo } from '../numpy-oracle';

/** NumPy dtype string for a given numpy-ts dtype */
export const NP_DTYPE: Record<string, string> = {
  float64: 'np.float64',
  float32: 'np.float32',
  int64: 'np.int64',
  int32: 'np.int32',
  int16: 'np.int16',
  int8: 'np.int8',
  uint64: 'np.uint64',
  uint32: 'np.uint32',
  uint16: 'np.uint16',
  uint8: 'np.uint8',
};

export const ALL_DTYPES = Object.keys(NP_DTYPE);
export const FLOAT_DTYPES = ['float64', 'float32'];
export const INT_DTYPES = ['int64', 'int32', 'int16', 'int8', 'uint64', 'uint32', 'uint16', 'uint8'];

export const WASM_MODES = [
  { name: 'default thresholds', multiplier: 1 },
  { name: 'forced WASM (threshold=0)', multiplier: 0 },
] as const;

/** Small 2x3 test array with values that fit in all dtypes */
export const SMALL_DATA = [
  [1, 2, 3],
  [4, 5, 6],
];

/** Helper: create array and run numpy comparison */
export function compareReduction(
  op: string,
  data: number[][] | number[][][],
  dtype: string,
  axis: number | undefined,
  npOp: string = op,
  tolerance?: number
) {
  const a = np.array(data, dtype as any);
  const npDtype = NP_DTYPE[dtype]!;
  const npAxisArg = axis !== undefined ? `, axis=${axis}` : '';

  // Call JS function
  let jsResult: any;
  if (op === 'sum') jsResult = axis !== undefined ? a.sum(axis) : a.sum();
  else if (op === 'mean') jsResult = axis !== undefined ? np.mean(a, axis) : np.mean(a);
  else if (op === 'max') jsResult = axis !== undefined ? np.max(a, axis) : np.max(a);
  else if (op === 'min') jsResult = axis !== undefined ? np.min(a, axis) : np.min(a);
  else if (op === 'prod') jsResult = axis !== undefined ? np.prod(a, axis) : np.prod(a);
  else if (op === 'argmin') jsResult = axis !== undefined ? np.argmin(a, axis) : np.argmin(a);
  else if (op === 'argmax') jsResult = axis !== undefined ? np.argmax(a, axis) : np.argmax(a);
  else if (op === 'var') jsResult = axis !== undefined ? np.var_(a, axis) : np.var_(a);
  else if (op === 'std') jsResult = axis !== undefined ? np.std(a, axis) : np.std(a);
  else if (op === 'all') jsResult = axis !== undefined ? np.all(a, axis) : np.all(a);
  else if (op === 'any') jsResult = axis !== undefined ? np.any(a, axis) : np.any(a);
  else if (op === 'nansum') jsResult = axis !== undefined ? np.nansum(a, axis) : np.nansum(a);
  else if (op === 'nanmean') jsResult = axis !== undefined ? np.nanmean(a, axis) : np.nanmean(a);
  else if (op === 'nanmin') jsResult = axis !== undefined ? np.nanmin(a, axis) : np.nanmin(a);
  else if (op === 'nanmax') jsResult = axis !== undefined ? np.nanmax(a, axis) : np.nanmax(a);
  else throw new Error(`Unknown op: ${op}`);

  // Call NumPy
  const pyResult = runNumPy(`
a = np.array(${JSON.stringify(data)}, dtype=${npDtype})
result = np.${npOp}(a${npAxisArg})
`);

  // Compare
  if (
    typeof jsResult === 'number' ||
    typeof jsResult === 'boolean' ||
    typeof jsResult === 'bigint'
  ) {
    const jsNum = Number(jsResult);
    const npNum = typeof pyResult.value === 'number' ? pyResult.value : Number(pyResult.value);
    const tol = tolerance ?? 1e-6;
    expect(Math.abs(jsNum - npNum)).toBeLessThanOrEqual(tol * Math.max(1, Math.abs(npNum)));
  } else {
    // Array result
    const jsArr = jsResult.toArray();
    expect(arraysClose(jsArr, pyResult.value, tolerance)).toBe(true);
  }
}

/** Setup hooks for a WASM mode describe block */
export function setupWasmMode(mode: typeof WASM_MODES[number]) {
  beforeAll(() => {
    wasmConfig.thresholdMultiplier = mode.multiplier;
    if (!checkNumPyAvailable()) {
      throw new Error('Python NumPy not available');
    }
    if (mode.multiplier === 1) {
      const info = getPythonInfo();
      console.log(`\n  Using Python ${info.python} with NumPy ${info.numpy} (${info.command})\n`);
    }
  });

  afterEach(() => {
    wasmConfig.thresholdMultiplier = mode.multiplier;
  });
}

export { np, runNumPy, arraysClose };
