/**
 * Shared helpers for dtype-sweep tests.
 */
import { expect } from 'vitest';
import {
  runNumPy as _runNumPy,
  runNumPyBatch as _runNumPyBatch,
  arraysClose as _arraysClose,
  checkNumPyAvailable as _checkNumPyAvailable,
  type NumPyResult,
} from '../numpy-oracle';
import { wasmConfig } from '../../../src/common/wasm/config';

// Force JS or WASM backend via FORCE_BACKEND env var.
// 'js' → thresholdMultiplier=Infinity (all JS), 'wasm' → 0 (all WASM), unset → auto.
const forcedBackend = process.env.FORCE_BACKEND as 'js' | 'wasm' | undefined;
if (forcedBackend === 'js') {
  wasmConfig.thresholdMultiplier = Infinity;
} else if (forcedBackend === 'wasm') {
  wasmConfig.thresholdMultiplier = 0;
}

/** Label for the active backend — use in describe() blocks for clear failure output. */
export const BACKEND: string = forcedBackend ?? 'auto';

export { ALL_DTYPES } from './_dtype-matrix';
export const runNumPy = _runNumPy;
export const runNumPyBatch = _runNumPyBatch;
export const arraysClose = _arraysClose;
export const checkNumPyAvailable = _checkNumPyAvailable;

export function npDtype(d: string) {
  return d === 'int64' ? 'np.int64' : d === 'uint64' ? 'np.uint64' : `np.${d}`;
}

/** Python cast expression for scalar results: complex() for complex dtypes, float() for real */
export function pyScalarCast(dtype: string): string {
  return isComplex(dtype) ? 'complex' : 'float';
}

/** Python astype target for array results: np.complex128 for complex, np.float64 for real */
export function pyArrayCast(dtype: string): string {
  return isComplex(dtype) ? 'np.complex128' : 'np.float64';
}

/** Convert NDArray to comparable value array — handles BigInt→Number, preserves Complex */
export function toComparable(arr: any): any {
  function walk(v: any): any {
    if (typeof v === 'bigint') return Number(v);
    if (Array.isArray(v)) return v.map(walk);
    return v; // Complex objects pass through for arraysClose
  }
  return walk(arr.toArray());
}

/** Compare scalar results — handles both real (Number) and Complex */
export function scalarClose(js: any, py: any, digits = 4): void {
  if (js && typeof js === 'object' && 're' in js) {
    const pyRe = py?.re ?? Number(py);
    const pyIm = py?.im ?? 0;
    expect(js.re).toBeCloseTo(pyRe, digits);
    expect(js.im).toBeCloseTo(pyIm, digits);
  } else if (py && typeof py === 'object' && 're' in py) {
    // JS returned a real scalar but oracle returned complex (e.g. any/all/argmax on complex input).
    // Compare JS result against the real part; imaginary should be zero.
    expect(py.im).toBeCloseTo(0, digits);
    expect(Number(js)).toBeCloseTo(py.re, digits);
  } else {
    expect(Number(js)).toBeCloseTo(Number(py), digits);
  }
}

export const isInt = (d: string) => d.startsWith('int') || d.startsWith('uint');
export const isFloat = (d: string) => d === 'float64' || d === 'float32';
export const isComplex = (d: string) => d.startsWith('complex');
export const isBool = (d: string) => d === 'bool';

/**
 * Handle operations where certain dtypes are expected to be rejected by both
 * JS and NumPy. Call this at the start of a test for known-unsupported dtype combos.
 *
 * Returns:
 * - `'both-reject'`  — both JS and NumPy error → caller should `return` (test passes)
 * - `'both-succeed'` — both succeed → caller should continue with value comparison
 * - fails the test   — if only one side errors (genuine bug in either direction)
 *
 * @param reason  Human-readable explanation of WHY this dtype is rejected
 * @param jsFn    Thunk that runs the JS operation
 * @param pyCode  Python code for the NumPy oracle
 */
export function expectBothReject(
  reason: string,
  jsFn: () => any,
  pyCode: string
): 'both-reject' | 'both-succeed' {
  let jsErr: Error | null = null;
  let pyErr: Error | null = null;

  try {
    jsFn();
  } catch (e: any) {
    jsErr = e;
  }

  try {
    _runNumPy(pyCode);
  } catch (e: any) {
    pyErr = e;
  }

  if (jsErr && pyErr) {
    return 'both-reject';
  }

  if (!jsErr && !pyErr) {
    return 'both-succeed';
  }

  if (jsErr && !pyErr) {
    expect.unreachable(
      `JS throws but NumPy succeeds (bug: we reject valid input).\n` +
        `Reason: ${reason}\n` +
        `JS error: ${jsErr.message?.slice(0, 150)}`
    );
  }

  // pyErr && !jsErr — we're too permissive (bug: we should reject like NumPy)
  expect.unreachable(
    `NumPy rejects but JS succeeds (bug: we should also reject).\n` +
      `Reason: ${reason}\n` +
      `NumPy error: ${pyErr!.message?.slice(0, 150)}`
  );

  return 'both-reject'; // unreachable, satisfies TS
}

/**
 * Like expectBothReject but uses a pre-computed batch result instead of calling runNumPy.
 * The batch result has an `error` field if Python threw.
 */
export function expectBothRejectPre(
  reason: string,
  jsFn: () => any,
  pyResult: NumPyResult & { error?: string }
): 'both-reject' | 'both-succeed' {
  let jsErr: Error | null = null;

  try {
    jsFn();
  } catch (e: any) {
    jsErr = e;
  }

  const pyErr = pyResult.error ? new Error(pyResult.error) : null;

  if (jsErr && pyErr) {
    return 'both-reject';
  }

  if (!jsErr && !pyErr) {
    return 'both-succeed';
  }

  if (jsErr && !pyErr) {
    expect.unreachable(
      `JS throws but NumPy succeeds (bug: we reject valid input).\n` +
        `Reason: ${reason}\n` +
        `JS error: ${jsErr.message?.slice(0, 150)}`
    );
  }

  expect.unreachable(
    `NumPy rejects but JS succeeds (bug: we should also reject).\n` +
      `Reason: ${reason}\n` +
      `NumPy error: ${pyErr!.message?.slice(0, 150)}`
  );

  return 'both-reject'; // unreachable, satisfies TS
}

/**
 * Map NumPy dtype string to our DType string.
 * NumPy reports e.g. 'float64', 'int32', 'bool', 'complex128'.
 */
const NP_DTYPE_MAP: Record<string, string> = {
  float64: 'float64',
  float32: 'float32',
  float16: 'float16',
  int64: 'int64',
  int32: 'int32',
  int16: 'int16',
  int8: 'int8',
  uint64: 'uint64',
  uint32: 'uint32',
  uint16: 'uint16',
  uint8: 'uint8',
  bool: 'bool',
  bool_: 'bool',
  complex128: 'complex128',
  complex64: 'complex64',
};

/**
 * Full oracle match: check value, dtype, and shape against NumPy.
 *
 * Python code MUST set `_result_orig` to the raw result (before any .astype() cast),
 * then set `result` to the cast version for value comparison. Example:
 *
 *   _result_orig = np.add(a, b)
 *   result = _result_orig.astype(np.float64)
 *
 * If there's no cast (scalar, already float64), just set `result` normally —
 * dtype/shape will be read from `result` itself.
 *
 * @param jsResult  The JS NDArray (must have .toArray(), .dtype, .shape)
 * @param pyCode    Python code for the oracle
 * @param opts      Tolerance options
 */
export function expectMatch(
  jsResult: any,
  pyCode: string,
  opts: { rtol?: number; atol?: number; indexResult?: boolean } = {}
): void {
  const { rtol = 1e-5, atol = 1e-8, indexResult = false } = opts;
  const pyResult = _runNumPy(pyCode);

  // --- Shape check ---
  const jsShape: number[] = Array.isArray(jsResult.shape) ? Array.from(jsResult.shape) : [];
  const pyShape: number[] = pyResult.orig_shape ?? pyResult.shape;
  expect(jsShape, 'shape mismatch').toEqual(pyShape);

  // --- Dtype check ---
  const jsDtype: string = jsResult.dtype ?? typeof jsResult;
  const pyDtypeRaw: string = pyResult.orig_dtype ?? pyResult.dtype;
  const pyDtype = NP_DTYPE_MAP[pyDtypeRaw] ?? pyDtypeRaw;
  // Index-returning functions: we return float64 (for JS ergonomics), NumPy returns int64
  if (indexResult && jsDtype === 'float64' && (pyDtype === 'int64' || pyDtype === 'uint64')) {
    // Accepted intentional difference
  } else {
    expect(jsDtype, `dtype mismatch: JS '${jsDtype}' vs NumPy '${pyDtype}'`).toBe(pyDtype);
  }

  // --- Value check ---
  if (jsShape.length === 0 && typeof jsResult !== 'object') {
    // Scalar result
    expect(Number(jsResult)).toBeCloseTo(Number(pyResult.value), 4);
  } else {
    expect(_arraysClose(jsResult.toArray(), pyResult.value, rtol, atol), 'value mismatch').toBe(
      true
    );
  }
}

/**
 * Like expectMatch but uses a pre-computed NumPyResult from a batch run.
 * If the result has an error field, the test fails with that error.
 */
export function expectMatchPre(
  jsResult: any,
  pyResult: NumPyResult & { error?: string },
  opts: { rtol?: number; atol?: number; indexResult?: boolean } = {}
): void {
  const { rtol = 1e-5, atol = 1e-8, indexResult = false } = opts;

  if (pyResult.error) {
    throw new Error(`NumPy error: ${pyResult.error}`);
  }

  // --- Shape check ---
  const jsShape: number[] = Array.isArray(jsResult.shape) ? Array.from(jsResult.shape) : [];
  const pyShape: number[] = pyResult.orig_shape ?? pyResult.shape;
  expect(jsShape, 'shape mismatch').toEqual(pyShape);

  // --- Dtype check ---
  const jsDtype: string = jsResult.dtype ?? typeof jsResult;
  const pyDtypeRaw: string = pyResult.orig_dtype ?? pyResult.dtype;
  const pyDtype = NP_DTYPE_MAP[pyDtypeRaw] ?? pyDtypeRaw;
  // Index-returning functions: we return float64 (for JS ergonomics), NumPy returns int64
  if (indexResult && jsDtype === 'float64' && (pyDtype === 'int64' || pyDtype === 'uint64')) {
    // Accepted intentional difference
  } else {
    expect(jsDtype, `dtype mismatch: JS '${jsDtype}' vs NumPy '${pyDtype}'`).toBe(pyDtype);
  }

  // --- Value check ---
  if (jsShape.length === 0 && typeof jsResult !== 'object') {
    // Scalar real result
    expect(Number(jsResult)).toBeCloseTo(Number(pyResult.value), 4);
  } else if (jsShape.length === 0 && jsResult?.re !== undefined) {
    // Scalar complex result
    const pyVal = pyResult.value;
    const pyRe = pyVal?.re ?? Number(pyVal);
    const pyIm = pyVal?.im ?? 0;
    expect(jsResult.re).toBeCloseTo(pyRe, 4);
    expect(jsResult.im).toBeCloseTo(pyIm, 4);
  } else {
    expect(_arraysClose(jsResult.toArray(), pyResult.value, rtol, atol), 'value mismatch').toBe(
      true
    );
  }
}
