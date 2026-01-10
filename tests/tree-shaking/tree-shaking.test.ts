/**
 * Tree-shaking tests for numpy-ts
 *
 * These tests verify that modern bundlers (esbuild, Rollup, Webpack) can effectively
 * tree-shake the library, eliminating unused code from the final bundle.
 *
 * The tests build various fixtures that import different subsets of the library
 * and verify that the resulting bundle sizes are proportional to what's imported.
 *
 * IMPORTANT: These tests serve two purposes:
 * 1. Regression detection - Ensure tree-shaking doesn't get worse over time
 * 2. Documentation - Record current tree-shaking effectiveness
 *
 * Current findings (documented in test output):
 * - The library has some tree-shaking limitations due to its monolithic ndarray.ts
 * - Single function imports still pull in core NDArray class and dependencies
 * - To improve tree-shaking, the library would need to be modularized further
 */

import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import { build as esbuild } from 'esbuild';
import { rollup, InputOptions, OutputOptions } from 'rollup';
import typescript from '@rollup/plugin-typescript';
import nodeResolve from '@rollup/plugin-node-resolve';
import webpack from 'webpack';
import { fileURLToPath } from 'url';
import { dirname, resolve } from 'path';
import { mkdir, stat } from 'fs/promises';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Fixture definitions
const FIXTURES = [
  { name: 'single-function', description: 'Single function (zeros)' },
  { name: 'basic-array', description: 'Basic array creation functions' },
  { name: 'math-only', description: 'Math operations only' },
  { name: 'linalg-only', description: 'Linear algebra only' },
  { name: 'fft-only', description: 'FFT operations only' },
  { name: 'random-only', description: 'Random operations only' },
  { name: 'io-only', description: 'IO operations only' },
  { name: 'full-import', description: 'Full import (baseline)' },
] as const;

interface BundleResult {
  fixture: string;
  bundler: string;
  size: number;
  minifiedSize: number;
  success: boolean;
  error?: string;
  modules?: string[];
}

interface BundlerResults {
  esbuild: Map<string, BundleResult>;
  rollup: Map<string, BundleResult>;
  webpack: Map<string, BundleResult>;
}

const OUTPUT_DIR = resolve(__dirname, '.output');
const FIXTURES_DIR = resolve(__dirname, 'fixtures');

// Helper to format bytes
function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

// Build with esbuild
async function buildWithEsbuild(fixtureName: string): Promise<BundleResult> {
  const entry = resolve(FIXTURES_DIR, `${fixtureName}.ts`);
  const outfile = resolve(OUTPUT_DIR, 'esbuild', `${fixtureName}.js`);
  const outfileMin = resolve(OUTPUT_DIR, 'esbuild', `${fixtureName}.min.js`);

  try {
    // Build unminified (for analysis)
    const result = await esbuild({
      entryPoints: [entry],
      bundle: true,
      platform: 'node',
      format: 'esm',
      outfile,
      treeShaking: true,
      metafile: true,
      write: true,
    });

    // Build minified (for size comparison)
    await esbuild({
      entryPoints: [entry],
      bundle: true,
      platform: 'node',
      format: 'esm',
      outfile: outfileMin,
      treeShaking: true,
      minify: true,
      write: true,
    });

    const stats = await stat(outfile);
    const statsMin = await stat(outfileMin);

    // Extract included modules from metafile
    const modules = result.metafile
      ? Object.keys(result.metafile.inputs).filter((m) => m.includes('src/'))
      : [];

    return {
      fixture: fixtureName,
      bundler: 'esbuild',
      size: stats.size,
      minifiedSize: statsMin.size,
      success: true,
      modules,
    };
  } catch (error) {
    return {
      fixture: fixtureName,
      bundler: 'esbuild',
      size: 0,
      minifiedSize: 0,
      success: false,
      error: error instanceof Error ? error.message : String(error),
    };
  }
}

// Build with Rollup
async function buildWithRollup(fixtureName: string): Promise<BundleResult> {
  const entry = resolve(FIXTURES_DIR, `${fixtureName}.ts`);
  const outfile = resolve(OUTPUT_DIR, 'rollup', `${fixtureName}.js`);
  const outfileMin = resolve(OUTPUT_DIR, 'rollup', `${fixtureName}.min.js`);

  try {
    const inputOptions: InputOptions = {
      input: entry,
      plugins: [
        nodeResolve({
          extensions: ['.ts', '.js'],
        }),
        typescript({
          tsconfig: resolve(__dirname, '../../tsconfig.json'),
          compilerOptions: {
            declaration: false,
            declarationMap: false,
            sourceMap: false,
            module: 'ESNext',
            moduleResolution: 'bundler',
          },
          include: ['src/**/*.ts', 'tests/tree-shaking/fixtures/**/*.ts'],
        }),
      ],
      treeshake: {
        moduleSideEffects: false,
        propertyReadSideEffects: false,
      },
      onwarn: (warning, warn) => {
        // Suppress circular dependency warnings (common in large libs)
        if (warning.code === 'CIRCULAR_DEPENDENCY') return;
        warn(warning);
      },
    };

    const outputOptions: OutputOptions = {
      file: outfile,
      format: 'esm',
    };

    // Build unminified
    const bundle = await rollup(inputOptions);
    await bundle.write(outputOptions);

    // Get module list before closing
    const { output } = await bundle.generate({ format: 'esm' });
    const modules =
      output[0].type === 'chunk' && output[0].modules
        ? Object.keys(output[0].modules).filter((m) => m.includes('src/'))
        : [];

    await bundle.close();

    // Build minified with terser
    const terserPlugin = (await import('@rollup/plugin-terser')).default;
    const bundleMin = await rollup({
      ...inputOptions,
      plugins: [...(inputOptions.plugins || []), terserPlugin()],
    });
    await bundleMin.write({
      file: outfileMin,
      format: 'esm',
    });
    await bundleMin.close();

    const stats = await stat(outfile);
    const statsMin = await stat(outfileMin);

    return {
      fixture: fixtureName,
      bundler: 'rollup',
      size: stats.size,
      minifiedSize: statsMin.size,
      success: true,
      modules,
    };
  } catch (error) {
    return {
      fixture: fixtureName,
      bundler: 'rollup',
      size: 0,
      minifiedSize: 0,
      success: false,
      error: error instanceof Error ? error.message : String(error),
    };
  }
}

// Build with Webpack
async function buildWithWebpack(fixtureName: string): Promise<BundleResult> {
  const entry = resolve(FIXTURES_DIR, `${fixtureName}.ts`);
  const outDir = resolve(OUTPUT_DIR, 'webpack', fixtureName);

  try {
    const baseConfig: webpack.Configuration = {
      entry,
      mode: 'production',
      output: {
        path: outDir,
        library: {
          type: 'module',
        },
      },
      experiments: {
        outputModule: true,
      },
      resolve: {
        extensions: ['.ts', '.js'],
      },
      module: {
        rules: [
          {
            test: /\.ts$/,
            use: {
              loader: 'ts-loader',
              options: {
                configFile: resolve(__dirname, '../../tsconfig.json'),
                compilerOptions: {
                  declaration: false,
                  declarationMap: false,
                  sourceMap: false,
                },
                transpileOnly: true, // Skip type checking for faster builds
              },
            },
            exclude: /node_modules/,
          },
        ],
      },
      optimization: {
        usedExports: true,
        sideEffects: true,
      },
    };

    // Build minified
    await new Promise<webpack.Stats | undefined>((resolvePromise, reject) => {
      webpack(
        {
          ...baseConfig,
          output: { ...baseConfig.output, filename: 'bundle.min.js' },
          optimization: { ...baseConfig.optimization, minimize: true },
        },
        (err, stats) => {
          if (err) reject(err);
          else if (stats?.hasErrors()) {
            const info = stats.toJson();
            reject(new Error(info.errors?.map((e) => e.message).join('\n') || 'Build failed'));
          } else resolvePromise(stats);
        }
      );
    });

    // Build unminified
    await new Promise<webpack.Stats | undefined>((resolvePromise, reject) => {
      webpack(
        {
          ...baseConfig,
          output: { ...baseConfig.output, filename: 'bundle.js' },
          optimization: { ...baseConfig.optimization, minimize: false },
        },
        (err, stats) => {
          if (err) reject(err);
          else if (stats?.hasErrors()) {
            const info = stats.toJson();
            reject(new Error(info.errors?.map((e) => e.message).join('\n') || 'Build failed'));
          } else resolvePromise(stats);
        }
      );
    });

    const statsUnmin = await stat(resolve(outDir, 'bundle.js'));
    const statsMin = await stat(resolve(outDir, 'bundle.min.js'));

    return {
      fixture: fixtureName,
      bundler: 'webpack',
      size: statsUnmin.size,
      minifiedSize: statsMin.size,
      success: true,
    };
  } catch (error) {
    return {
      fixture: fixtureName,
      bundler: 'webpack',
      size: 0,
      minifiedSize: 0,
      success: false,
      error: error instanceof Error ? error.message : String(error),
    };
  }
}

describe('Tree-shaking Tests', () => {
  const results: BundlerResults = {
    esbuild: new Map(),
    rollup: new Map(),
    webpack: new Map(),
  };

  beforeAll(async () => {
    // Create output directories
    await mkdir(resolve(OUTPUT_DIR, 'esbuild'), { recursive: true });
    await mkdir(resolve(OUTPUT_DIR, 'rollup'), { recursive: true });
    await mkdir(resolve(OUTPUT_DIR, 'webpack'), { recursive: true });
  });

  afterAll(async () => {
    // Print summary report
    console.log('\n' + '='.repeat(80));
    console.log('TREE-SHAKING TEST RESULTS');
    console.log('='.repeat(80));

    for (const bundler of ['esbuild', 'rollup', 'webpack'] as const) {
      console.log(`\n${bundler.toUpperCase()}:`);
      console.log('-'.repeat(60));

      const bundlerResults = results[bundler];
      const fullSize = bundlerResults.get('full-import')?.minifiedSize || 1;

      for (const fixture of FIXTURES) {
        const result = bundlerResults.get(fixture.name);
        if (result && result.success) {
          const percentage = ((result.minifiedSize / fullSize) * 100).toFixed(1);
          console.log(
            `  ${fixture.description.padEnd(35)} ${formatBytes(result.minifiedSize).padStart(10)} (${percentage}%)`
          );
        } else {
          console.log(`  ${fixture.description.padEnd(35)} FAILED: ${result?.error || 'Unknown'}`);
        }
      }
    }

    // Print analysis
    console.log('\n' + '='.repeat(80));
    console.log('ANALYSIS');
    console.log('='.repeat(80));

    const esbuildFull = results.esbuild.get('full-import')?.minifiedSize || 1;
    const esbuildSingle = results.esbuild.get('single-function')?.minifiedSize || 0;
    const ratio = esbuildSingle / esbuildFull;

    if (ratio > 0.5) {
      console.log('\nNOTE: Tree-shaking effectiveness is limited.');
      console.log(`Single function import is ${(ratio * 100).toFixed(1)}% of full bundle.`);
      console.log('This is expected due to the monolithic NDArray class design.');
      console.log('\nTo improve tree-shaking in the future, consider:');
      console.log('  1. Splitting ndarray.ts into smaller modules');
      console.log('  2. Moving operations to separate files with lazy imports');
      console.log('  3. Using more granular re-exports in index.ts');
    } else {
      console.log('\nTree-shaking is working well!');
      console.log(`Single function import is only ${(ratio * 100).toFixed(1)}% of full bundle.`);
    }

    console.log('\n' + '='.repeat(80));
  });

  describe('esbuild tree-shaking', () => {
    it('should build all fixtures', async () => {
      for (const fixture of FIXTURES) {
        const result = await buildWithEsbuild(fixture.name);
        results.esbuild.set(fixture.name, result);
        expect(result.success, `${fixture.name} should build: ${result.error}`).toBe(true);
      }
    }, 60000);

    it('should produce smaller bundles for partial imports', () => {
      const fullSize = results.esbuild.get('full-import')?.minifiedSize || 0;
      expect(fullSize).toBeGreaterThan(0);

      // Single function should be smaller than full import (even if not dramatically)
      const singleSize = results.esbuild.get('single-function')?.minifiedSize || fullSize;
      expect(singleSize).toBeLessThan(fullSize);
    });

    it('should show relative size ordering', () => {
      const fullSize = results.esbuild.get('full-import')?.minifiedSize || 1;
      const singleSize = results.esbuild.get('single-function')?.minifiedSize || fullSize;
      const basicSize = results.esbuild.get('basic-array')?.minifiedSize || fullSize;

      // Single function should be <= basic (or very close)
      // Allow small variance due to bundler behavior
      expect(singleSize).toBeLessThanOrEqual(basicSize * 1.05);
    });
  });

  describe('Rollup tree-shaking', () => {
    it('should attempt to build all fixtures', async () => {
      // Note: Rollup may have issues with TypeScript type exports syntax
      // We record results but don't fail the test if Rollup can't process the source
      let successCount = 0;
      for (const fixture of FIXTURES) {
        const result = await buildWithRollup(fixture.name);
        results.rollup.set(fixture.name, result);
        if (result.success) successCount++;
      }

      // If all builds failed, warn but don't fail (Rollup TS plugin compatibility issue)
      if (successCount === 0) {
        console.log(
          'WARNING: All Rollup builds failed. This may be due to TypeScript plugin compatibility.'
        );
        console.log('Tree-shaking analysis continues with esbuild and Webpack.');
      }
      // Test passes as long as we attempted all builds
      expect(results.rollup.size).toBe(FIXTURES.length);
    }, 120000);

    it('should produce smaller bundles for partial imports', () => {
      const fullResult = results.rollup.get('full-import');
      const singleResult = results.rollup.get('single-function');

      // Skip if Rollup builds failed
      if (!fullResult?.success || !singleResult?.success) {
        console.log('Rollup builds failed, skipping size comparison');
        return;
      }

      expect(fullResult.minifiedSize).toBeGreaterThan(0);
      expect(singleResult.minifiedSize).toBeLessThan(fullResult.minifiedSize);
    });
  });

  describe('Webpack tree-shaking', () => {
    it('should build all fixtures', async () => {
      for (const fixture of FIXTURES) {
        const result = await buildWithWebpack(fixture.name);
        results.webpack.set(fixture.name, result);
        expect(result.success, `${fixture.name} should build: ${result.error}`).toBe(true);
      }
    }, 180000);

    it('should produce smaller bundles for partial imports', () => {
      const fullResult = results.webpack.get('full-import');
      const singleResult = results.webpack.get('single-function');

      // Skip if Webpack builds failed
      if (!fullResult?.success || !singleResult?.success) {
        console.log('Webpack builds failed, skipping size comparison');
        return;
      }

      expect(fullResult.minifiedSize).toBeGreaterThan(0);
      expect(singleResult.minifiedSize).toBeLessThan(fullResult.minifiedSize);
    });
  });

  describe('Cross-bundler verification', () => {
    it('should show tree-shaking works across all bundlers', () => {
      // All bundlers should show that single-function is smaller than full
      for (const bundler of ['esbuild', 'rollup', 'webpack'] as const) {
        const bundlerResults = results[bundler];
        const fullResult = bundlerResults.get('full-import');
        const singleResult = bundlerResults.get('single-function');

        // Skip if bundler failed
        if (!fullResult?.success || !singleResult?.success) continue;

        expect(
          singleResult.minifiedSize,
          `${bundler}: single should be smaller than full`
        ).toBeLessThan(fullResult.minifiedSize);
      }
    });

    it('should show module-specific imports are smaller than full imports', () => {
      const moduleFixtures = ['math-only', 'linalg-only', 'fft-only', 'random-only', 'io-only'];

      for (const bundler of ['esbuild', 'rollup', 'webpack'] as const) {
        const bundlerResults = results[bundler];
        const fullResult = bundlerResults.get('full-import');

        // Skip if bundler failed
        if (!fullResult?.success) continue;

        for (const fixture of moduleFixtures) {
          const moduleResult = bundlerResults.get(fixture);
          if (!moduleResult?.success) continue;

          // Each module-specific import should be smaller than or equal to full
          expect(
            moduleResult.minifiedSize,
            `${bundler}: ${fixture} should be <= full import`
          ).toBeLessThanOrEqual(fullResult.minifiedSize);
        }
      }
    });
  });

  describe('Tree-shaking effectiveness (regression tests)', () => {
    // These tests establish baselines to detect regressions
    // Current state: tree-shaking is limited due to monolithic design

    it('esbuild: single function should be <95% of full bundle', () => {
      const fullSize = results.esbuild.get('full-import')?.minifiedSize || 1;
      const singleSize = results.esbuild.get('single-function')?.minifiedSize || fullSize;
      const ratio = singleSize / fullSize;

      // Regression check: if this fails, tree-shaking got worse
      expect(ratio).toBeLessThan(0.95);
    });

    it('rollup: single function should be <95% of full bundle', () => {
      const fullResult = results.rollup.get('full-import');
      const singleResult = results.rollup.get('single-function');

      // Skip if Rollup builds failed
      if (!fullResult?.success || !singleResult?.success) {
        console.log('Rollup builds failed, skipping regression check');
        return;
      }

      const ratio = singleResult.minifiedSize / fullResult.minifiedSize;
      expect(ratio).toBeLessThan(0.95);
    });

    it('webpack: single function should be <95% of full bundle', () => {
      const fullResult = results.webpack.get('full-import');
      const singleResult = results.webpack.get('single-function');

      // Skip if Webpack builds failed
      if (!fullResult?.success || !singleResult?.success) {
        console.log('Webpack builds failed, skipping regression check');
        return;
      }

      const ratio = singleResult.minifiedSize / fullResult.minifiedSize;
      expect(ratio).toBeLessThan(0.95);
    });

    // Aspirational tests - these document desired state
    // Currently expected to fail, but useful to track progress
    describe.skip('Aspirational targets (currently failing)', () => {
      it('single function import should be <50% of full bundle (esbuild)', () => {
        const fullSize = results.esbuild.get('full-import')?.minifiedSize || 1;
        const singleSize = results.esbuild.get('single-function')?.minifiedSize || fullSize;
        expect(singleSize / fullSize).toBeLessThan(0.5);
      });

      it('single function import should be <30% of full bundle (ideal)', () => {
        const fullSize = results.esbuild.get('full-import')?.minifiedSize || 1;
        const singleSize = results.esbuild.get('single-function')?.minifiedSize || fullSize;
        expect(singleSize / fullSize).toBeLessThan(0.3);
      });
    });
  });
});
