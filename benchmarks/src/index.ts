/**
 * Main benchmark orchestrator
 * Runs benchmarks for both Python NumPy and numpy-ts across multiple JS runtimes
 */

import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';
import { getBenchmarkSpecs, filterByCategory } from './specs';
import { setBenchmarkConfig } from './runner';
import { runPythonBenchmarks } from './python-runner';
import { detectRuntimes, spawnRuntimeBenchmark } from './runtime-spawner';
import {
  compareResults,
  calculateSummary,
  printResults,
  compareMultiRuntime,
  calculateMultiRuntimeSummaries,
  printMultiRuntimeResults,
} from './analysis';
import { generateHTMLReport, generateMultiRuntimeHTMLReport } from './visualization';
import { generatePNGChart, generateMultiRuntimePNGChart } from './chart-generator';
import { validateBenchmarks } from './validation';
import type {
  BenchmarkOptions,
  BenchmarkReport,
  BenchmarkTiming,
  RuntimeName,
  MultiRuntimeReport,
} from './types';

// Read version from root package.json
const packageJson = JSON.parse(
  fs.readFileSync(path.resolve(__dirname, '../../package.json'), 'utf-8')
);

function getMachineInfo(): string {
  const cpu = os.cpus()[0]?.model ?? 'Unknown CPU';
  const cores = os.cpus().length;
  const ramGb = Math.round(os.totalmem() / (1024 ** 3));
  const arch = os.arch();
  return `${cpu} (${cores} cores, ${ramGb} GB, ${arch})`;
}

const CACHE_MAX_AGE_MS = 24 * 60 * 60 * 1000; // 24 hours

interface CachedPython {
  results: BenchmarkTiming[];
  pythonVersion: string;
  numpyVersion: string;
}

/**
 * Try to load cached Python benchmark results from a previous run.
 * Returns null if cache is invalid or --fresh was passed.
 */
function tryLoadCachedPython(
  specs: { name: string }[],
  modeSuffix: string,
  resultsDir: string
): CachedPython | null {
  const jsonPath = path.join(resultsDir, `latest${modeSuffix}.json`);
  if (!fs.existsSync(jsonPath)) return null;

  const stat = fs.statSync(jsonPath);
  const ageMs = Date.now() - stat.mtimeMs;
  if (ageMs > CACHE_MAX_AGE_MS) return null;

  try {
    const raw = JSON.parse(fs.readFileSync(jsonPath, 'utf-8'));
    const env = raw.environment;
    if (!env?.numpy_version || !env?.machine) return null;

    // Check machine matches
    if (env.machine !== getMachineInfo()) return null;

    // Extract Python timings — handle both single-runtime and multi-runtime formats
    const results: BenchmarkTiming[] = [];
    const cachedResults: { name: string; numpy?: BenchmarkTiming }[] = raw.results ?? [];

    if (cachedResults.length !== specs.length) return null;

    for (let i = 0; i < specs.length; i++) {
      const cached = cachedResults[i]!;
      if (cached.name !== specs[i]!.name) return null; // name mismatch
      if (!cached.numpy) return null;
      results.push(cached.numpy);
    }

    const ageHours = Math.round(ageMs / (60 * 60 * 1000) * 10) / 10;
    console.log(
      `Using cached NumPy results (${ageHours}h old). Pass --fresh to re-run Python benchmarks.`
    );

    return {
      results,
      pythonVersion: env.python_version ?? 'unknown',
      numpyVersion: env.numpy_version,
    };
  } catch {
    return null;
  }
}

async function main() {
  // Parse command line arguments
  const args = process.argv.slice(2);
  const options: BenchmarkOptions = {
    mode: 'standard',
  };

  for (let i = 0; i < args.length; i++) {
    const arg = args[i];

    if (arg === '--quick') {
      options.mode = 'quick';
    } else if (arg === '--standard') {
      options.mode = 'standard';
    } else if (arg === '--full') {
      options.mode = 'full';
    } else if (arg === '--large') {
      options.mode = 'full'; // deprecated alias for --full
    } else if (arg === '--category' && i + 1 < args.length) {
      options.category = args[++i];
    } else if (arg === '--output' && i + 1 < args.length) {
      options.output = args[++i];
    } else if (arg === '--single-thread') {
      options.singleThread = true;
    } else if (arg === '--runtime' && i + 1 < args.length) {
      options.runtimes = args[++i]!.split(',').map((s) => s.trim()) as RuntimeName[];
    } else if (arg === '--fresh') {
      options.fresh = true;
    } else if (arg === '--help' || arg === '-h') {
      printHelp();
      process.exit(0);
    }
  }

  // Configure benchmark settings based on mode
  let minSampleTimeMs: number;
  let targetSamples: number;

  if (options.mode === 'quick') {
    minSampleTimeMs = 50;
    targetSamples = 1;
    setBenchmarkConfig(minSampleTimeMs, targetSamples);
  } else if (options.mode === 'full') {
    minSampleTimeMs = 100;
    targetSamples = 5;
    setBenchmarkConfig(minSampleTimeMs, targetSamples);
  } else {
    minSampleTimeMs = 100;
    targetSamples = 5;
    setBenchmarkConfig(minSampleTimeMs, targetSamples);
  }

  console.log('NumPy vs numpy-ts Benchmark Suite\n');
  console.log(`Mode: ${options.mode}`);
  if (options.singleThread) {
    console.log('NumPy threading: single-threaded (OMP/MKL/OpenBLAS threads = 1)');
  }

  // Detect available runtimes
  const allRuntimes = await detectRuntimes();
  let selectedRuntimes = allRuntimes;

  if (options.runtimes) {
    // Filter to requested runtimes
    selectedRuntimes = [];
    for (const requested of options.runtimes) {
      const found = allRuntimes.find((r) => r.name === requested);
      if (found) {
        selectedRuntimes.push(found);
      } else {
        console.log(`Warning: ${requested} not found on PATH, skipping`);
      }
    }
    if (selectedRuntimes.length === 0) {
      console.error('No requested runtimes are available');
      process.exit(1);
    }
  }

  const runtimeStr = selectedRuntimes
    .map((r) => `${r.name} v${r.version}`)
    .join(', ');
  console.log(`Runtimes: ${runtimeStr}`);

  // Get benchmark specifications
  let specs = getBenchmarkSpecs(options.mode || 'standard');

  // In quick mode, run only the representative tagged specs
  if (options.mode === 'quick') {
    specs = specs.filter((s) => s.includeInQuick);
  }

  if (options.category) {
    console.log(`Category filter: ${options.category}`);
    specs = filterByCategory(specs, options.category);

    if (specs.length === 0) {
      console.error('No benchmarks found for category: ' + options.category);
      process.exit(1);
    }
  }

  console.log(`Total benchmarks: ${specs.length}\n`);

  try {
    // Validate correctness before benchmarking
    const nonValidatableOperations = new Set([
      'linalg_det', 'linalg_qr', 'linalg_cholesky', 'linalg_svd', 'linalg_eig', 'linalg_eigh',
      'linalg_eigvals', 'linalg_eigvalsh', 'linalg_matrix_rank', 'linalg_pinv', 'linalg_cond',
      'linalg_lstsq', 'linalg_matrix_power'
    ]);
    const validatableSpecs = specs.filter(
      (spec) =>
        !Object.values(spec.setup).some((s) => s.dtype === 'int64' || s.dtype === 'uint64') &&
        spec.category !== 'io' &&
        !nonValidatableOperations.has(spec.operation)
    );
    if (validatableSpecs.length > 0) {
      console.log('Validating correctness against NumPy...');
      await validateBenchmarks(validatableSpecs);
      console.log('');
    }
    const skippedCount = specs.length - validatableSpecs.length;
    if (skippedCount > 0) {
      console.log(
        `Skipping validation for ${skippedCount} benchmarks (int64/uint64/IO/Complex linalg)\n`
      );
    }

    // Determine file suffix based on mode and threading (needed for cache lookup)
    const modeSuffix =
      (options.mode === 'full' ? '-full' : '') + (options.singleThread ? '_single' : '');
    const resultsDir = path.resolve(__dirname, '../results');

    // Run Python NumPy benchmarks (or use cache)
    let numpyResults: BenchmarkTiming[];
    let pythonVersion: string;
    let numpyVersion: string;

    const cached = options.fresh ? null : tryLoadCachedPython(specs, modeSuffix, resultsDir);
    if (cached) {
      numpyResults = cached.results;
      pythonVersion = cached.pythonVersion;
      numpyVersion = cached.numpyVersion;
    } else {
      console.log('Running Python NumPy benchmarks...');
      const pyResult = await runPythonBenchmarks(
        specs,
        minSampleTimeMs,
        targetSamples,
        options.singleThread ?? false
      );
      numpyResults = pyResult.results;
      pythonVersion = pyResult.pythonVersion;
      numpyVersion = pyResult.numpyVersion;
    }

    // Run each JS runtime via subprocess
    const runtimeResultsMap = new Map<string, BenchmarkTiming[]>();
    const runtimeVersions: Record<string, string> = {};

    for (const runtime of selectedRuntimes) {
      console.log(`\nRunning numpy-ts benchmarks under ${runtime.name}...`);
      try {
        const { results, version } = await spawnRuntimeBenchmark(
          runtime.name,
          specs,
          minSampleTimeMs,
          targetSamples,
        );
        runtimeResultsMap.set(runtime.name, results);
        runtimeVersions[runtime.name] = version;
      } catch (err) {
        console.error(`Warning: ${runtime.name} benchmarks failed: ${err}`);
        console.error(`Skipping ${runtime.name}\n`);
      }
    }

    if (runtimeResultsMap.size === 0) {
      console.error('All runtime benchmarks failed');
      process.exit(1);
    }

    // Save results
    const plotsDir = path.resolve(resultsDir, 'plots');
    if (!fs.existsSync(resultsDir)) fs.mkdirSync(resultsDir, { recursive: true });
    if (!fs.existsSync(plotsDir)) fs.mkdirSync(plotsDir, { recursive: true });

    if (runtimeResultsMap.size === 1) {
      // Single runtime: use legacy BenchmarkReport format for backward compatibility
      const [runtimeName, jsResults] = [...runtimeResultsMap.entries()][0]!;
      const comparisons = compareResults(specs, numpyResults, jsResults);
      const summary = calculateSummary(comparisons);

      printResults(comparisons, summary);

      const report: BenchmarkReport = {
        timestamp: new Date().toISOString(),
        environment: {
          node_version: runtimeVersions[runtimeName] || process.version,
          python_version: pythonVersion,
          numpy_version: numpyVersion,
          numpyjs_version: packageJson.version,
          machine: getMachineInfo(),
        },
        results: comparisons,
        summary,
      };

      const jsonPath = options.output
        ? path.resolve(options.output)
        : path.join(resultsDir, `latest${modeSuffix}.json`);
      fs.writeFileSync(jsonPath, JSON.stringify(report, null, 2));
      console.log(`Results saved to: ${jsonPath}`);

      const historyDir = path.join(resultsDir, 'history');
      if (!fs.existsSync(historyDir)) fs.mkdirSync(historyDir, { recursive: true });
      const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
      fs.writeFileSync(
        path.join(historyDir, `benchmark${modeSuffix}-${timestamp}.json`),
        JSON.stringify(report, null, 2)
      );

      const htmlPath = path.join(plotsDir, `latest${modeSuffix}.html`);
      generateHTMLReport(report, htmlPath);
      console.log(`HTML report saved to: ${htmlPath}`);

      const pngPath = path.join(plotsDir, `latest${modeSuffix}.png`);
      await generatePNGChart(report, pngPath);
      console.log(`PNG chart saved to: ${pngPath}`);

      console.log(`\nView report: open ${htmlPath}`);
    } else {
      // Multiple runtimes: use MultiRuntimeReport format
      const comparisons = compareMultiRuntime(specs, numpyResults, runtimeResultsMap);
      const summaries = calculateMultiRuntimeSummaries(comparisons);

      printMultiRuntimeResults(comparisons, summaries);

      const report: MultiRuntimeReport = {
        timestamp: new Date().toISOString(),
        environment: {
          python_version: pythonVersion,
          numpy_version: numpyVersion,
          numpyjs_version: packageJson.version,
          runtimes: runtimeVersions,
          machine: getMachineInfo(),
        },
        results: comparisons,
        summaries,
      };

      const jsonPath = options.output
        ? path.resolve(options.output)
        : path.join(resultsDir, `latest${modeSuffix}.json`);
      fs.writeFileSync(jsonPath, JSON.stringify(report, null, 2));
      console.log(`Results saved to: ${jsonPath}`);

      const historyDir = path.join(resultsDir, 'history');
      if (!fs.existsSync(historyDir)) fs.mkdirSync(historyDir, { recursive: true });
      const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
      fs.writeFileSync(
        path.join(historyDir, `benchmark${modeSuffix}-${timestamp}.json`),
        JSON.stringify(report, null, 2)
      );

      const htmlPath = path.join(plotsDir, `latest${modeSuffix}.html`);
      generateMultiRuntimeHTMLReport(report, htmlPath);
      console.log(`HTML report saved to: ${htmlPath}`);

      const pngPath = path.join(plotsDir, `latest${modeSuffix}.png`);
      await generateMultiRuntimePNGChart(report, pngPath);
      console.log(`PNG chart saved to: ${pngPath}`);

      console.log(`\nView report: open ${htmlPath}`);
    }
  } catch (error) {
    console.error('Benchmark failed:', error);
    process.exit(1);
  }
}

function printHelp() {
  console.log(`
NumPy vs numpy-ts Benchmark Suite

Usage:
  npm run bench [options]

Options:
  --quick              Quick benchmarks (~50 representative ops, 1 sample, 50ms/sample)
  --standard           Standard benchmarks (all ops, float64, 5 samples, 100ms/sample, default)
  --full               Full benchmarks (all ops, all dtypes, 10 samples, 200ms/sample)
  --large              Deprecated alias for --full
  --single-thread      Force NumPy to run single-threaded (OMP/MKL/OpenBLAS)
  --runtime <list>     Comma-separated runtimes to use (default: auto-detect)
                       Values: node, deno, bun  (e.g. --runtime node,bun)
  --category <name>    Run only benchmarks in specified category
  --fresh              Force re-run Python benchmarks (skip cache)
  --output <path>      Save JSON results to specified path
  --help, -h           Show this help message

Categories:
  creation             Array creation (zeros, ones, arange, etc.)
  arithmetic           Arithmetic operations (add, multiply, etc.; includes int64/uint64 in full)
  math                 Math ops (sqrt, exp, log, trig, etc.)
  linalg               Linear algebra (matmul, dot, inv, svd, etc.)
  reductions           Reductions (sum, mean, std, etc.)
  statistics           Statistics & histograms (cov, corrcoef, histogram, etc.)
  manipulation         Array manipulation (reshape, concatenate, etc.)
  sorting              Sorting & searching (sort, argsort, where, etc.)
  logic                Logic & comparison (isnan, where, logical_and, etc.)
  sets                 Set operations (unique, etc.)
  bitwise              Bitwise operations
  indexing             Indexing (take, compress, etc.)
  gradient             Gradient & differences (diff, gradient, etc.)
  random               Random number generation
  fft                  Fast Fourier Transform
  io                   IO operations (parseNpy, serializeNpy, etc.)
  complex              Complex number operations
  polynomials          Polynomial operations
  utilities            Type utilities (can_cast, result_type, etc.)

Examples:
  npm run bench                           # Run standard benchmarks (all detected runtimes)
  npm run bench:quick                     # Run quick benchmarks
  npm run bench -- --runtime node         # Node.js only (legacy behavior)
  npm run bench -- --runtime node,bun     # Node + Bun
  npm run bench:node                      # Shorthand: Node only
  npm run bench:deno                      # Shorthand: Deno only
  npm run bench:bun                       # Shorthand: Bun only
  npm run bench -- --category linalg      # Run only linalg benchmarks
`);
}

main().catch((err) => {
  console.error('Fatal error:', err);
  process.exit(1);
});
