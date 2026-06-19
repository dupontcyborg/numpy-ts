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

import { type ChildProcess, execSync, spawn } from 'node:child_process';
import { readSync, writeFileSync, writeSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

// Get Python command from environment or use default
const PYTHON_CMD = process.env.NUMPY_PYTHON || 'python3';

// ---------------------------------------------------------------------------
// Persistent NumPy worker
//
// Spawning `python3` + `import numpy` per assertion costs ~80 ms; with thousands
// of validation assertions that dominates the suite. Instead we keep ONE long-
// lived Python process per Node (vitest) worker that imports NumPy once, then
// services requests over stdin/stdout. Each `runNumPy` call is a synchronous
// round-trip (~0.05 ms): write a JSON request line, block-read the JSON reply.
//
// Sync round-trip: tests call runNumPy synchronously, so we cannot use async
// streams. We write to the child's stdin fd and busy-read its stdout fd with
// fs.readSync (retrying on EAGAIN) until the newline-terminated reply arrives.
// ---------------------------------------------------------------------------

// Python server: imports numpy once, then loops over JSON request lines
// {"code": "..."} where the code sets `result` (and optionally `_result_orig`).
// Replies with one JSON line per request. The serialize_value logic mirrors the
// previous per-call script exactly so output is byte-for-byte unchanged.
const SERVER_PY = `import sys, json, math
import numpy as np

if tuple(map(int, np.__version__.split('.')[:2])) < (2, 0):
    sys.stderr.write("Error: NumPy 2.0+ is required. Found %s\\n" % np.__version__)
    sys.exit(3)

def serialize_value(val):
    if isinstance(val, np.ndarray):
        return serialize_value(val.tolist())
    elif isinstance(val, list):
        return [serialize_value(v) for v in val]
    elif isinstance(val, (bool, np.bool_)):
        return bool(val)
    elif isinstance(val, (complex, np.complexfloating)):
        return {"__complex__": True, "re": serialize_value(float(val.real)), "im": serialize_value(float(val.imag))}
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

def handle(code):
    # Expose the same names the previous inline script made available to
    # injected snippets (some snippets call serialize_value / use json, sys).
    ns = {"np": np, "math": math, "json": json, "sys": sys, "serialize_value": serialize_value}
    exec(code, ns)
    result = ns["result"]
    if isinstance(result, np.ndarray):
        out = {"value": serialize_value(result), "dtype": str(result.dtype), "shape": list(result.shape)}
    elif isinstance(result, (np.integer, np.floating, np.complexfloating)):
        out = {"value": serialize_value(result), "dtype": str(type(result).__name__), "shape": []}
    else:
        out = {"value": serialize_value(result), "dtype": str(type(result).__name__), "shape": []}
    if "_result_orig" in ns:
        ro = ns["_result_orig"]
        out["orig_dtype"] = str(ro.dtype) if hasattr(ro, "dtype") else out["dtype"]
        out["orig_shape"] = list(ro.shape) if hasattr(ro, "shape") else out["shape"]
    return out

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    try:
        out = handle(json.loads(line)["code"])
    except Exception as e:
        out = {"error": str(e)}
    sys.stdout.write(json.dumps(out) + "\\n")
    sys.stdout.flush()
`;

interface Worker {
  proc: ChildProcess;
  inFd: number;
  outFd: number;
}

let _worker: Worker | null = null;
let _serverPath: string | null = null;

function getWorker(): Worker {
  if (_worker && _worker.proc.exitCode === null && !_worker.proc.killed) return _worker;

  if (!_serverPath) {
    _serverPath = join(tmpdir(), `numpy-oracle-server-${process.pid}.py`);
    writeFileSync(_serverPath, SERVER_PY, 'utf-8');
  }

  // Support multi-word commands like "conda run -n env python".
  const parts = PYTHON_CMD.split(' ').filter(Boolean);
  const cmd = parts[0]!;
  const args = [...parts.slice(1), '-u', _serverPath];
  const proc = spawn(cmd, args, { stdio: ['pipe', 'pipe', 'inherit'] });
  proc.stdout!.pause();
  // The worker blocks on stdin, so its live pipes would keep the vitest worker's
  // event loop alive and hang teardown. unref the child and both pipes — we read
  // via fs.readSync on the raw fds, which doesn't need the handles ref'd.
  proc.unref();
  (proc.stdin as unknown as { unref?: () => void }).unref?.();
  (proc.stdout as unknown as { unref?: () => void }).unref?.();
  proc.on('exit', () => {
    if (_worker?.proc === proc) _worker = null;
  });
  process.once('exit', () => {
    try {
      proc.kill();
    } catch {
      // ignore
    }
  });

  // Public `.fd` is set under plain Node but undefined inside vitest's pool;
  // the libuv handle's fd is present in both. Fall back to it.
  const fdOf = (s: unknown): number | undefined => {
    const o = s as { fd?: number; _handle?: { fd?: number } };
    return typeof o.fd === 'number' ? o.fd : o._handle?.fd;
  };
  const inFd = fdOf(proc.stdin);
  const outFd = fdOf(proc.stdout);
  if (typeof inFd !== 'number' || inFd < 0 || typeof outFd !== 'number' || outFd < 0) {
    throw new Error('numpy-oracle: could not obtain worker pipe file descriptors');
  }
  _worker = { proc, inFd, outFd };
  return _worker;
}

/**
 * Terminate the persistent worker. Called from an `afterAll` teardown (see
 * tests/validation/_oracle-teardown.ts) so no live child process keeps the
 * vitest worker's event loop alive at pool shutdown. The worker respawns lazily
 * on the next `runNumPy` call, so persistence is retained within each test file.
 */
export function killNumpyWorker(): void {
  if (_worker) {
    try {
      _worker.proc.kill();
    } catch {
      // ignore
    }
    _worker = null;
  }
}

// Pipe I/O on non-blocking fds can do partial writes / EAGAIN; the response can
// arrive in several chunks. Both are bounded by a wall-clock deadline so a stuck
// worker surfaces as a thrown error (a named test failure) instead of an
// uninterruptible synchronous busy-spin that hangs the whole run.
const WORKER_TIMEOUT_MS = 30_000;

/** Write the full request, looping over partial writes / EAGAIN. */
function writeAll(fd: number, data: Buffer, deadline: number): void {
  let off = 0;
  while (off < data.length) {
    let n: number;
    try {
      n = writeSync(fd, data, off, data.length - off, null);
    } catch (e: unknown) {
      if ((e as { code?: string }).code === 'EAGAIN') {
        if (Date.now() > deadline) throw new Error('numpy-oracle: worker write timed out');
        continue;
      }
      throw e;
    }
    off += n;
  }
}

/** Synchronous request/response with the persistent worker. */
function evalOnWorker(code: string): Record<string, unknown> {
  const w = getWorker();
  const deadline = Date.now() + WORKER_TIMEOUT_MS;
  writeAll(w.inFd, Buffer.from(`${JSON.stringify({ code })}\n`, 'utf8'), deadline);

  const buf = Buffer.alloc(65536);
  let s = '';
  while (!s.includes('\n')) {
    let n: number;
    try {
      n = readSync(w.outFd, buf, 0, buf.length, null);
    } catch (e: unknown) {
      const ec = (e as { code?: string }).code;
      if (ec === 'EAGAIN') {
        if (Date.now() > deadline) {
          _worker = null;
          throw new Error(
            `numpy-oracle: worker timed out after ${WORKER_TIMEOUT_MS}ms on: ${code}`,
          );
        }
        continue; // no data yet — spin until the reply arrives
      }
      if (ec === 'EOF') break;
      throw e;
    }
    if (n > 0) s += buf.toString('utf8', 0, n);
    else break;
  }
  if (!s.includes('\n')) {
    _worker = null; // worker died (e.g. NumPy missing / <2.0) — force respawn next time
    throw new Error(
      'numpy-oracle: worker produced no response (NumPy 2.0+ required and importable). ' +
        `Python command: ${PYTHON_CMD}`,
    );
  }
  return JSON.parse(s.slice(0, s.indexOf('\n'))) as Record<string, unknown>;
}

export interface NumPyResult {
  value: any;
  dtype: string;
  shape: number[];
  /** Original dtype before any .astype() cast (set via _result_orig in Python code) */
  orig_dtype?: string;
  /** Original shape before any .astype() cast */
  orig_shape?: number[];
}

/**
 * Deserialize values from Python, converting special markers back to Infinity/NaN
 */

import { Complex } from '../../src/common/complex';

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
    // Deserialize complex numbers from Python (re/im may be special markers)
    return new Complex(deserializeValue(val.re) as number, deserializeValue(val.im) as number);
  } else {
    return val;
  }
}

/**
 * Execute Python NumPy code and return the result
 */
export function runNumPy(code: string): NumPyResult {
  const parsed = evalOnWorker(code);
  if ('error' in parsed) {
    throw new Error(`NumPy error: ${parsed.error as string}`);
  }
  parsed.value = deserializeValue(parsed.value);
  return parsed as unknown as NumPyResult;
}

/**
 * Execute multiple Python NumPy computations in a single subprocess.
 * Each entry maps a key to a Python code snippet that sets `result`
 * (and optionally `_result_orig` for dtype/shape capture).
 *
 * Returns a Map<string, NumPyResult> keyed by the same keys.
 * Entries that error are stored with a special `error` field.
 */
export function runNumPyBatch(
  snippets: Record<string, string>,
): Map<string, NumPyResult & { error?: string }> {
  const results = new Map<string, NumPyResult & { error?: string }>();
  // Per-call round-trips to the persistent worker are ~0.05 ms, so the former
  // single-script batching is no longer needed — just evaluate each snippet.
  for (const key of Object.keys(snippets)) {
    const parsed = evalOnWorker(snippets[key]!);
    if ('error' in parsed) {
      results.set(key, {
        error: parsed.error as string,
        value: undefined,
        dtype: '',
        shape: [],
      });
    } else {
      parsed.value = deserializeValue(parsed.value);
      results.set(key, parsed as unknown as NumPyResult);
    }
  }
  return results;
}

/**
 * Compare two values with tolerance for floating point
 */
export function closeEnough(
  a: number,
  b: number,
  rtol: number = 1e-5,
  atol: number = 1e-8,
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
      { encoding: 'utf-8' },
    ).trim();
    const [python, numpy] = info.split('|');
    return { python, numpy, command: PYTHON_CMD };
  } catch {
    return { python: 'unknown', numpy: 'unknown', command: PYTHON_CMD };
  }
}
