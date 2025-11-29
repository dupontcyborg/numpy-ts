#!/usr/bin/env node
/**
 * Main benchmark orchestrator
 * Runs benchmarks for both Python NumPy and numpy-ts, compares results
 */

import * as fs from 'fs';
import * as path from 'path';
import { getBenchmarkSpecs, filterByCategory } from './specs';
import { runBenchmarks, setBenchmarkConfig } from './runner';
import { runPythonBenchmarks } from './python-runner';
import { compareResults, calculateSummary, printResults } from './analysis';
import { generateHTMLReport } from './visualization';
import { generatePNGChart } from './chart-generator';
import { validateBenchmarks } from './validation';
import type { BenchmarkOptions, BenchmarkReport } from './types';

// Read version from root package.json
const packageJson = JSON.parse(
  fs.readFileSync(path.resolve(__dirname, '../../package.json'), 'utf-8')
);

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
    } else if (arg === '--large') {
      options.mode = 'large';
    } else if (arg === '--category' && i + 1 < args.length) {
      options.category = args[++i];
    } else if (arg === '--output' && i + 1 < args.length) {
      options.output = args[++i];
    } else if (arg === '--help' || arg === '-h') {
      printHelp();
      process.exit(0);
    }
  }

  // Configure benchmark settings based on mode
  let minSampleTimeMs: number;
  let targetSamples: number;

  if (options.mode === 'quick') {
    // Quick mode: single sample, shorter sample time for fast feedback
    minSampleTimeMs = 50;
    targetSamples = 1;
    setBenchmarkConfig(minSampleTimeMs, targetSamples);
  } else if (options.mode === 'large') {
    // Large mode: fewer samples, longer sample time for large arrays
    minSampleTimeMs = 200;
    targetSamples = 3;
    setBenchmarkConfig(minSampleTimeMs, targetSamples);
  } else {
    // Standard mode: multiple samples, full sample time for accurate results
    minSampleTimeMs = 100;
    targetSamples = 5;
    setBenchmarkConfig(minSampleTimeMs, targetSamples);
  }

  console.log('🚀 NumPy vs numpy-ts Benchmark Suite\n');
  console.log(`Mode: ${options.mode}`);

  // Get benchmark specifications
  let specs = getBenchmarkSpecs(options.mode || 'standard');

  if (options.category) {
    console.log(`Category filter: ${options.category}`);
    specs = filterByCategory(specs, options.category);

    if (specs.length === 0) {
      console.error(`❌ No benchmarks found for category: ${options.category}`);
      process.exit(1);
    }
  }

  console.log(`Total benchmarks: ${specs.length}\n`);

  try {
    // Validate correctness before benchmarking (skip BigInt and IO benchmarks)
    const validatableSpecs = specs.filter(
      (spec) => spec.category !== 'bigint' && spec.category !== 'io'
    );
    if (validatableSpecs.length > 0) {
      console.log('Validating correctness against NumPy...');
      await validateBenchmarks(validatableSpecs);
      console.log('');
    }
    const skippedCount = specs.length - validatableSpecs.length;
    if (skippedCount > 0) {
      console.log(
        `⚠️  Skipping validation for ${skippedCount} benchmarks (BigInt/IO cannot be validated against Python)\n`
      );
    }

    // Run numpy-ts benchmarks
    console.log('Running numpy-ts benchmarks...');
    const numpyjsResults = await runBenchmarks(specs);

    // Run Python NumPy benchmarks
    console.log('\nRunning Python NumPy benchmarks...');
    const { results: numpyResults, pythonVersion, numpyVersion } = await runPythonBenchmarks(
      specs,
      minSampleTimeMs,
      targetSamples
    );

    // Compare results
    const comparisons = compareResults(specs, numpyResults, numpyjsResults);
    const summary = calculateSummary(comparisons);

    // Print results to console
    printResults(comparisons, summary);

    // Create report
    const report: BenchmarkReport = {
      timestamp: new Date().toISOString(),
      environment: {
        node_version: process.version,
        python_version: pythonVersion,
        numpy_version: numpyVersion,
        numpyjs_version: packageJson.version,
      },
      results: comparisons,
      summary,
    };

    // Save results
    const resultsDir = path.resolve(__dirname, '../results');
    const plotsDir = path.resolve(resultsDir, 'plots');

    // Ensure directories exist
    if (!fs.existsSync(resultsDir)) {
      fs.mkdirSync(resultsDir, { recursive: true });
    }
    if (!fs.existsSync(plotsDir)) {
      fs.mkdirSync(plotsDir, { recursive: true });
    }

    // Determine file suffix based on mode
    const modeSuffix = options.mode === 'large' ? '-large' : '';

    // Save JSON results
    const jsonPath = options.output
      ? path.resolve(options.output)
      : path.join(resultsDir, `latest${modeSuffix}.json`);
    fs.writeFileSync(jsonPath, JSON.stringify(report, null, 2));
    console.log(`Results saved to: ${jsonPath}`);

    // Save historical results
    const historyDir = path.join(resultsDir, 'history');
    if (!fs.existsSync(historyDir)) {
      fs.mkdirSync(historyDir, { recursive: true });
    }
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
    const historyPath = path.join(historyDir, `benchmark${modeSuffix}-${timestamp}.json`);
    fs.writeFileSync(historyPath, JSON.stringify(report, null, 2));

    // Generate HTML report
    const htmlPath = path.join(plotsDir, `latest${modeSuffix}.html`);
    generateHTMLReport(report, htmlPath);
    console.log(`HTML report saved to: ${htmlPath}`);

    // Generate PNG chart
    const pngPath = path.join(plotsDir, `latest${modeSuffix}.png`);
    await generatePNGChart(report, pngPath);
    console.log(`PNG chart saved to: ${pngPath}`);

    console.log(`\nView report: open ${htmlPath}`);
  } catch (error) {
    console.error('❌ Benchmark failed:', error);
    process.exit(1);
  }
}

function printHelp() {
  console.log(`
NumPy vs numpy-ts Benchmark Suite

Usage:
  npm run bench [options]

Options:
  --quick              Quick benchmarks (1 sample, 50ms/sample, ~2-3min)
                       Arrays: 1K, 100x100, 500x500
  --standard           Standard benchmarks (5 samples, 100ms/sample, ~5-10min, default)
                       Arrays: 1K, 100x100, 500x500
  --large              Large array benchmarks (3 samples, 200ms/sample, ~10-20min)
                       Arrays: 10K, 316x316 (~100K), 1000x1000 (1M)
  --category <name>    Run only benchmarks in specified category
  --output <path>      Save JSON results to specified path
  --help, -h           Show this help message

Categories:
  creation             Array creation (zeros, ones, arange, etc.)
  arithmetic           Arithmetic operations (add, multiply, etc.)
  linalg               Linear algebra (matmul, transpose)
  reductions           Reductions (sum, mean, max, min)
  reshape              Reshape operations (reshape, flatten, ravel)
  io                   IO operations (parseNpy, serializeNpy, etc.)
  bigint               BigInt (int64/uint64) operations

Examples:
  npm run bench                           # Run standard benchmarks
  npm run bench:quick                     # Run quick benchmarks
  npm run bench:large                     # Run large array benchmarks
  npm run bench -- --category linalg      # Run only linalg benchmarks
  npm run bench -- --output out.json      # Standard benchmarks, save to out.json
`);
}

main().catch((err) => {
  console.error('Fatal error:', err);
  process.exit(1);
});
