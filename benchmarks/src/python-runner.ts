/**
 * Python/NumPy benchmark runner
 * Spawns Python script and collects results
 */

import { spawn, spawnSync } from 'node:child_process';

/**
 * Python interpreter to run the NumPy side with.
 *
 * Mirrors `NUMPY_PYTHON` from the test oracle. Without this the harness spawned
 * a bare `python3`, so a machine whose default python lacks NumPy saw the child
 * die and the parent crash with an unhandled EPIPE while writing its stdin —
 * which says nothing about the actual cause.
 *
 * Accepts a command with arguments, e.g. `NUMPY_PYTHON='conda run -n env python'`.
 */
export function resolvePython(): { cmd: string; prefixArgs: string[] } {
  const raw = (process.env.NUMPY_PYTHON || 'python3').trim();
  const parts = raw.split(/\s+/);
  return { cmd: parts[0] as string, prefixArgs: parts.slice(1) };
}

/**
 * Is there a usable NumPy on the resolved interpreter?
 *
 * Checked up front so a missing NumPy degrades to a JS-only run instead of
 * aborting. Benchmarking numpy-ts against itself is still useful; only the
 * comparison and the correctness validation need Python.
 */
export function checkNumpy(): { ok: true } | { ok: false; cmd: string; detail: string } {
  const { cmd, prefixArgs } = resolvePython();
  try {
    const probe = spawnSync(cmd, [...prefixArgs, '-c', 'import numpy'], {
      encoding: 'utf-8',
      timeout: 30_000,
    });
    if (probe.error) return { ok: false, cmd, detail: probe.error.message };
    if (probe.status !== 0) {
      return {
        ok: false,
        cmd,
        detail: (probe.stderr || '').trim().split('\n').pop() || 'import numpy failed',
      };
    }
    return { ok: true };
  } catch (err) {
    return { ok: false, cmd, detail: String(err) };
  }
}

/** Turn a dead-child EPIPE into a message that names the likely cause. */
export function describeSpawnFailure(cmd: string, stderr: string): string {
  const hint = /ModuleNotFoundError|No module named/i.test(stderr)
    ? `'${cmd}' has no numpy installed`
    : `'${cmd}' exited before reading its input`;
  return (
    `NumPy benchmark helper failed: ${hint}.\n` +
    `Set NUMPY_PYTHON to an interpreter with NumPy, e.g.\n` +
    `  NUMPY_PYTHON=/path/to/env/bin/python3 pnpm run bench:node\n` +
    (stderr.trim() ? `\nPython stderr:\n${stderr.trim()}\n` : '')
  );
}

import { resolve } from 'node:path';
import type { BenchmarkCase, BenchmarkTiming } from './types';

export async function runPythonBenchmarks(
  specs: BenchmarkCase[],
  minSampleTimeMs: number = 100,
  targetSamples: number = 5,
  singleThread: boolean = false,
): Promise<{ results: BenchmarkTiming[]; pythonVersion?: string; numpyVersion?: string }> {
  const scriptPath = resolve(__dirname, '../scripts/numpy_benchmark.py');

  return new Promise((resolve, reject) => {
    const env = { ...process.env };
    if (singleThread) {
      env.OMP_NUM_THREADS = '1';
      env.MKL_NUM_THREADS = '1';
      env.OPENBLAS_NUM_THREADS = '1';
      env.NUMEXPR_NUM_THREADS = '1';
      env.VECLIB_MAXIMUM_THREADS = '1'; // Apple Accelerate
    }

    const { cmd, prefixArgs } = resolvePython();
    const python = spawn(cmd, [...prefixArgs, scriptPath], { env });
    // A child that exits before reading stdin makes the write below raise EPIPE
    // on the socket, which is an unhandled 'error' event and kills the process.
    python.stdin.on('error', () => {});

    let stdout = '';
    let stderr = '';
    let pythonVersion: string | undefined;
    let numpyVersion: string | undefined;

    python.stdout.on('data', (data) => {
      stdout += data.toString();
    });

    python.stderr.on('data', (data) => {
      const text = data.toString();
      stderr += text;

      // Extract version info
      const pythonMatch = text.match(/Python ([\d.]+)/);
      if (pythonMatch) {
        pythonVersion = pythonMatch[1];
      }

      const numpyMatch = text.match(/NumPy ([\d.]+)/);
      if (numpyMatch) {
        numpyVersion = numpyMatch[1];
      }

      // Print progress
      process.stderr.write(text);
    });

    python.on('close', (code) => {
      if (code !== 0) {
        reject(new Error(describeSpawnFailure(cmd, stderr)));
        return;
      }

      try {
        const results = JSON.parse(stdout) as BenchmarkTiming[];
        resolve({ results, pythonVersion, numpyVersion });
      } catch (err) {
        reject(new Error(`Failed to parse Python output: ${err}\n${stdout}`));
      }
    });

    python.on('error', (err) => {
      reject(new Error(`Failed to spawn Python: ${err.message}`));
    });

    // Send specs and config to Python via stdin
    python.stdin.write(
      JSON.stringify({
        specs,
        config: {
          minSampleTimeMs,
          targetSamples,
        },
      }),
    );
    python.stdin.end();
  });
}
