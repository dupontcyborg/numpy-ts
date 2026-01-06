/**
 * Python NumPy Test Oracle
 *
 * Executes Python code with NumPy and returns results for comparison
 *
 * Environment Variables:
 * - NUMPY_PYTHON: Python command to use (default: 'python3')
 *   Examples:
 *     NUMPY_PYTHON='python3' npm test
 *     NUMPY_PYTHON='conda run -n myenv python' npm test
 *     NUMPY_PYTHON='python' npm test
 */

import { execSync } from 'child_process';
import { writeFileSync, unlinkSync } from 'fs';
import { tmpdir } from 'os';
import { join } from 'path';

// Get Python command from environment or use default
const PYTHON_CMD = process.env.NUMPY_PYTHON || 'python3';

export interface NumPyResult {
  value: any;
  dtype: string;
  shape: number[];
}

/**
 * Deserialize values from Python, converting special markers back to Infinity/NaN
 */

import { Complex } from '../../src/core/complex';

function deserializeValue(val: any): any {
  if (Array.isArray(val)) {
    return val.map((v) => deserializeValue(v));
  } else if (val === '__Infinity__') {
    return Infinity;
  } else if (val === '__-Infinity__') {
    return -Infinity;
  } else if (val === '__NaN__') {
    return NaN;
  } else if (typeof val === 'object' && val !== null && '__complex__' in val) {
    // Deserialize complex numbers from Python
    return new Complex(val.re, val.im);
  } else {
    return val;
  }
}

/**
 * Execute Python NumPy code and return the result
 */
export function runNumPy(code: string): NumPyResult {
  // Indent user code properly for try block
  const indentedCode = code
    .trim()
    .split('\n')
    .map((line) => `    ${line}`)
    .join('\n');

  const pythonCode = `import numpy as np
import json
import sys
import math

# Require NumPy 2.0+
NUMPY_VERSION = tuple(map(int, np.__version__.split('.')[:2]))
if NUMPY_VERSION < (2, 0):
    print(f"Error: NumPy 2.0+ is required for validation tests. Found NumPy {np.__version__}", file=sys.stderr)
    print("Please upgrade: pip install --upgrade 'numpy>=2.0'", file=sys.stderr)
    sys.exit(1)

def serialize_value(val):
    """Convert NumPy arrays/values to JSON-serializable format, handling inf/nan/complex"""
    if isinstance(val, np.ndarray):
        # Convert array to nested lists, then recursively serialize
        # tolist() handles multi-dimensional arrays correctly
        return serialize_value(val.tolist())
    elif isinstance(val, list):
        return [serialize_value(v) for v in val]
    # Check bool BEFORE int (Python bool is subclass of int)
    elif isinstance(val, (bool, np.bool_)):
        return bool(val)
    elif isinstance(val, (complex, np.complexfloating)):
        # Serialize complex numbers as object with re/im
        return {"__complex__": True, "re": val.real, "im": val.imag}
    elif isinstance(val, (float, np.floating)):
        if math.isnan(val):
            return "__NaN__"
        elif math.isinf(val):
            return "__Infinity__" if val > 0 else "__-Infinity__"
        else:
            return float(val)
    elif isinstance(val, (int, np.integer)):
        return int(val)
    else:
        return val

try:
${indentedCode}
    if isinstance(result, np.ndarray):
        output = {'value': serialize_value(result), 'dtype': str(result.dtype), 'shape': list(result.shape)}
    elif isinstance(result, (np.integer, np.floating, np.complexfloating)):
        output = {'value': serialize_value(result), 'dtype': str(type(result).__name__), 'shape': []}
    else:
        output = {'value': serialize_value(result), 'dtype': str(type(result).__name__), 'shape': []}
    print(json.dumps(output))
except Exception as e:
    print(json.dumps({'error': str(e)}), file=sys.stderr)
    sys.exit(1)`;

  // Write to temp file to avoid shell escaping issues
  const tmpFile = join(
    tmpdir(),
    `numpy-test-${Date.now()}-${Math.random().toString(36).slice(2)}.py`
  );

  try {
    writeFileSync(tmpFile, pythonCode, 'utf-8');
    const result = execSync(`${PYTHON_CMD} ${tmpFile}`, {
      encoding: 'utf-8',
      stdio: ['pipe', 'pipe', 'pipe'],
    });

    const parsed = JSON.parse(result);
    if ('error' in parsed) {
      throw new Error(`NumPy error: ${parsed.error}`);
    }
    // Deserialize special float values
    parsed.value = deserializeValue(parsed.value);
    return parsed as NumPyResult;
  } catch (error: unknown) {
    const err = error as { stderr?: Buffer; message?: string };
    if (err.stderr) {
      const stderrStr = err.stderr.toString();
      try {
        const parsed = JSON.parse(stderrStr);
        if ('error' in parsed) {
          throw new Error(`NumPy error: ${parsed.error}`);
        }
      } catch {
        // Not JSON, throw original error
      }
    }
    throw new Error(`Failed to run Python: ${err.message || 'Unknown error'}`);
  } finally {
    try {
      unlinkSync(tmpFile);
    } catch {
      // Ignore cleanup errors
    }
  }
}

/**
 * Compare two values with tolerance for floating point
 */
export function closeEnough(
  a: number,
  b: number,
  rtol: number = 1e-5,
  atol: number = 1e-8
): boolean {
  // Handle special values
  if (Number.isNaN(a) && Number.isNaN(b)) return true;
  if (a === Infinity && b === Infinity) return true;
  if (a === -Infinity && b === -Infinity) return true;
  if (!Number.isFinite(a) || !Number.isFinite(b)) return false;

  return Math.abs(a - b) <= atol + rtol * Math.abs(b);
}

/**
 * Recursively compare arrays/nested arrays with tolerance
 */

export function arraysClose(a: any, b: any, rtol: number = 1e-5, atol: number = 1e-8): boolean {
  if (Array.isArray(a) && Array.isArray(b)) {
    if (a.length !== b.length) return false;
    return a.every((val, i) => arraysClose(val, b[i], rtol, atol));
  } else if (a instanceof Complex && b instanceof Complex) {
    // Compare complex numbers by comparing real and imaginary parts
    return closeEnough(a.re, b.re, rtol, atol) && closeEnough(a.im, b.im, rtol, atol);
  } else if (a instanceof Complex || b instanceof Complex) {
    // One is complex, one is not - they should both be complex for comparison
    // Convert real number to complex for comparison
    const aComplex = a instanceof Complex ? a : new Complex(Number(a), 0);
    const bComplex = b instanceof Complex ? b : new Complex(Number(b), 0);
    return (
      closeEnough(aComplex.re, bComplex.re, rtol, atol) &&
      closeEnough(aComplex.im, bComplex.im, rtol, atol)
    );
  } else if (typeof a === 'number' && typeof b === 'number') {
    return closeEnough(a, b, rtol, atol);
  } else if (typeof a === 'bigint' || typeof b === 'bigint') {
    // Convert both to Number for comparison
    // This is necessary because NumPy returns regular numbers via JSON
    const aNum = typeof a === 'bigint' ? Number(a) : a;
    const bNum = typeof b === 'bigint' ? Number(b) : b;
    // For exact integer comparisons, use strict equality
    // (rtol/atol don't make sense for exact integer values)
    if (Number.isInteger(aNum) && Number.isInteger(bNum)) {
      return aNum === bNum;
    }
    return closeEnough(aNum, bNum, rtol, atol);
  } else if (typeof a === 'boolean' || typeof b === 'boolean') {
    // Handle boolean comparisons
    // Our JS implementation uses 0/1 for boolean arrays, NumPy uses false/true
    const aNum = typeof a === 'boolean' ? (a ? 1 : 0) : a;
    const bNum = typeof b === 'boolean' ? (b ? 1 : 0) : b;
    return aNum === bNum;
  } else {
    return a === b;
  }
}

/**
 * Check if Python NumPy is available
 */
export function checkNumPyAvailable(): boolean {
  try {
    execSync(`${PYTHON_CMD} -c "import numpy"`, { stdio: 'ignore' });
    return true;
  } catch {
    return false;
  }
}

/**
 * Get information about the Python/NumPy environment being used
 */
export function getPythonInfo(): { python: string; numpy: string; command: string } {
  try {
    const info = execSync(
      `${PYTHON_CMD} -c "import sys; import numpy; print(f'{sys.version.split()[0]}|{numpy.__version__}')"`,
      { encoding: 'utf-8' }
    ).trim();
    const [python, numpy] = info.split('|');
    return { python, numpy, command: PYTHON_CMD };
  } catch {
    return { python: 'unknown', numpy: 'unknown', command: PYTHON_CMD };
  }
}
