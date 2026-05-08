/**
 * Python/NumPy benchmark runner
 * Spawns Python script and collects results
 */

import { spawn } from 'child_process';
import { resolve } from 'path';
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

    const python = spawn('python3', [scriptPath], { env });

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
        reject(new Error(`Python script exited with code ${code}\n${stderr}`));
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
