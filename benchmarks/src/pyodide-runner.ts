/**
 * Pyodide (WASM NumPy) benchmark runner
 * Runs NumPy benchmarks inside Pyodide for a WASM-vs-WASM comparison
 */

import * as fs from 'fs';
import * as path from 'path';
import type { BenchmarkCase, BenchmarkTiming } from './types';

export async function runPyodideBenchmarks(
  specs: BenchmarkCase[],
  minSampleTimeMs: number = 100,
  targetSamples: number = 5
): Promise<{ results: BenchmarkTiming[]; pythonVersion: string; numpyVersion: string }> {
  // Dynamic import — pyodide is optional
  let loadPyodide: (typeof import('pyodide'))['loadPyodide'];
  try {
    const mod = await import('pyodide');
    loadPyodide = mod.loadPyodide;
  } catch {
    throw new Error(
      'pyodide package not found. Install it with:\n  npm install --save-dev pyodide'
    );
  }

  console.log('Loading Pyodide runtime...');
  const pyodide = await loadPyodide({
    stderr: (text: string) => {
      process.stderr.write(text + '\n');
    },
  });

  console.log('Loading NumPy package in Pyodide...');
  await pyodide.loadPackage('numpy');

  // Read the existing numpy_benchmark.py script
  const scriptPath = path.resolve(__dirname, '../scripts/numpy_benchmark.py');
  let script = fs.readFileSync(scriptPath, 'utf-8');

  // Strip the __main__ guard so we can call main() directly
  script = script.replace(/if __name__ == "__main__":\s*\n\s*main\(\)\s*$/, '');

  // Run the script to define all functions
  await pyodide.runPythonAsync(script);

  // Inject specs and config via globals, then run benchmarks
  const inputData = JSON.stringify({
    specs,
    config: { minSampleTimeMs, targetSamples },
  });

  const pythonRunner = `
import json, sys

# Override stdin reading — inject specs directly
_input_data = json.loads(${JSON.stringify(inputData)})

# Apply config
specs = _input_data["specs"]
config = _input_data.get("config", {})
MIN_SAMPLE_TIME_MS = config.get("minSampleTimeMs", MIN_SAMPLE_TIME_MS)
TARGET_SAMPLES = config.get("targetSamples", TARGET_SAMPLES)

_results = []

print(f"Python {sys.version.split()[0]}", file=sys.stderr)
import numpy as np
print(f"NumPy {np.__version__} (Pyodide/WASM)", file=sys.stderr)
print(f"Running {len(specs)} benchmarks with auto-calibration...", file=sys.stderr)
print(f"Target: {MIN_SAMPLE_TIME_MS}ms per sample, {TARGET_SAMPLES} samples per benchmark\\n", file=sys.stderr)

for i, spec in enumerate(specs, 1):
    result = run_benchmark(spec)
    _results.append(result)

    name_padded = spec["name"].ljust(40)
    mean_padded = f"{result['mean_ms']:.3f}".rjust(8)
    ops_formatted = f"{int(result['ops_per_sec']):,}".rjust(12)
    print(f"  [{i}/{len(specs)}] {name_padded} {mean_padded}ms  {ops_formatted} ops/sec", file=sys.stderr)

_python_version = sys.version.split()[0]
_numpy_version = np.__version__
_results_json = json.dumps(_results)
`;

  await pyodide.runPythonAsync(pythonRunner);

  // Extract results from Python globals
  const resultsJson = pyodide.globals.get('_results_json') as string;
  const pythonVersion = pyodide.globals.get('_python_version') as string;
  const numpyVersion = pyodide.globals.get('_numpy_version') as string;

  const results = JSON.parse(resultsJson) as BenchmarkTiming[];

  return { results, pythonVersion, numpyVersion };
}
