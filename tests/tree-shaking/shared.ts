/**
 * Shared fixtures, types, and build functions for tree-shaking tests.
 * Split into separate files per bundler for parallel execution.
 */

import { build as esbuildBuild } from 'esbuild';
import { rollup, InputOptions, OutputOptions } from 'rollup';
import nodeResolve from '@rollup/plugin-node-resolve';
import alias from '@rollup/plugin-alias';
import typescript from '@rollup/plugin-typescript';
import webpack from 'webpack';
import { fileURLToPath } from 'url';
import { dirname, resolve } from 'path';
import { stat, readFile } from 'fs/promises';
import { gzipSync } from 'zlib';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Path to the built ESM modules (tree-shakeable)
export const NUMPY_TS_ESM = resolve(__dirname, '../../dist/esm/index.js');
export const NUMPY_TS_CORE = resolve(__dirname, '../../dist/esm/core.js');

// Main entry point fixtures (numpy-ts — method chaining, limited tree-shaking)
export const FULL_FIXTURES = [
  { name: 'full-import', description: 'Full import (baseline)' },
  { name: 'full-single', description: 'Full: single function (zeros)' },
  { name: 'full-math', description: 'Full: math operations' },
  { name: 'full-linalg', description: 'Full: linear algebra' },
  { name: 'full-fft', description: 'Full: FFT operations' },
] as const;

// Core entry point fixtures (numpy-ts/core — fully tree-shakeable)
export const CORE_FIXTURES = [
  { name: 'core-single', description: 'Core: single function (zeros)' },
  { name: 'core-math', description: 'Core: math operations' },
  { name: 'core-linalg', description: 'Core: linear algebra' },
  { name: 'core-fft', description: 'Core: FFT operations' },
] as const;

// All fixtures combined
export const ALL_FIXTURES = [...FULL_FIXTURES, ...CORE_FIXTURES] as const;

export interface BundleResult {
  fixture: string;
  bundler: string;
  size: number;
  minifiedSize: number;
  gzipSize?: number;
  success: boolean;
  error?: string;
  modules?: string[];
}

export const OUTPUT_DIR = resolve(__dirname, '.output');
export const FIXTURES_DIR = resolve(__dirname, 'fixtures');

/**
 * Tree-shaking effectiveness thresholds.
 *
 * All values are percentages of the full-import bundle size.
 * The main entry point (numpy-ts) has limited tree-shaking because NDArray's
 * method chaining pulls in most of the library. The core entry point
 * (numpy-ts/core) is fully tree-shakeable.
 *
 * Current baselines (2026-03, esbuild):
 *   Main entry:  single-function = 84%, math-only = 87%, fft-only = 92%
 *   Core entry:  standalone-single = 2%, standalone-math = 8%, standalone-linalg = 20%
 */
export const THRESHOLDS = {
  // --- Main entry point (numpy-ts) ---
  // These are high because method chaining prevents effective tree-shaking.
  // We still test them to catch catastrophic regressions (e.g. duplicated code).

  /** main: single function import (% of full) — currently ~84-91% across bundlers */
  mainSingleFunction: 95,

  // --- Core entry point (numpy-ts/core) ---
  // This is where tree-shaking actually works. Keep these tight.

  /** core: single function like zeros() (% of full) — currently ~2% */
  coreSingleFunction: 5,
  /** core: math operations bundle (% of full) — currently ~8% */
  coreMath: 15,
  /** core: linalg operations bundle (% of full) — currently ~20% */
  coreLinalg: 30,
  /** core: FFT operations bundle (% of full) — TBD, first run */
  coreFft: 35,
};

export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

export function logBundleSizes(bundler: string, results: Map<string, BundleResult>): void {
  const fullSize = results.get('full-import')?.minifiedSize || 1;
  const hasGzip = [...results.values()].some((r) => r.gzipSize !== undefined);
  const sorted = [...results.entries()]
    .filter(([, r]) => r.success)
    .sort(([, a], [, b]) => a.minifiedSize - b.minifiedSize);

  const header = hasGzip ? `${bundler} bundle sizes (minified + gzip):` : `${bundler} bundle sizes (minified):`;
  console.log(`\n${header}`);
  for (const [name, r] of sorted) {
    const pct = ((r.minifiedSize / fullSize) * 100).toFixed(1);
    const gz = r.gzipSize !== undefined ? `  gzip: ${formatBytes(r.gzipSize).padStart(10)}` : '';
    console.log(`  ${name.padEnd(20)} ${formatBytes(r.minifiedSize).padStart(10)}  (${pct.padStart(5)}%)${gz}`);
  }
}

// Build with esbuild
export async function buildWithEsbuild(fixtureName: string): Promise<BundleResult> {
  const entry = resolve(FIXTURES_DIR, `${fixtureName}.ts`);
  const outfile = resolve(OUTPUT_DIR, 'esbuild', `${fixtureName}.js`);
  const outfileMin = resolve(OUTPUT_DIR, 'esbuild', `${fixtureName}.min.js`);

  try {
    const result = await esbuildBuild({
      entryPoints: [entry],
      bundle: true,
      platform: 'node',
      format: 'esm',
      outfile,
      treeShaking: true,
      metafile: true,
      write: true,
      alias: {
        'numpy-ts/core': NUMPY_TS_CORE,
        'numpy-ts': NUMPY_TS_ESM,
      },
    });

    await esbuildBuild({
      entryPoints: [entry],
      bundle: true,
      platform: 'node',
      format: 'esm',
      outfile: outfileMin,
      treeShaking: true,
      minify: true,
      write: true,
      alias: {
        'numpy-ts/core': NUMPY_TS_CORE,
        'numpy-ts': NUMPY_TS_ESM,
      },
    });

    const stats = await stat(outfile);
    const statsMin = await stat(outfileMin);
    const minifiedContent = await readFile(outfileMin);
    const gzipSize = gzipSync(minifiedContent).length;

    const modules = result.metafile
      ? Object.keys(result.metafile.inputs).filter((m) => m.includes('src/'))
      : [];

    return {
      fixture: fixtureName,
      bundler: 'esbuild',
      size: stats.size,
      minifiedSize: statsMin.size,
      gzipSize,
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
export async function buildWithRollup(fixtureName: string): Promise<BundleResult> {
  const entry = resolve(FIXTURES_DIR, `${fixtureName}.ts`);
  const outfile = resolve(OUTPUT_DIR, 'rollup', `${fixtureName}.js`);
  const outfileMin = resolve(OUTPUT_DIR, 'rollup', `${fixtureName}.min.js`);

  try {
    const inputOptions: InputOptions = {
      input: entry,
      plugins: [
        alias({
          entries: [
            { find: 'numpy-ts/core', replacement: NUMPY_TS_CORE },
            { find: 'numpy-ts', replacement: NUMPY_TS_ESM },
          ],
        }),
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
            outDir: resolve(OUTPUT_DIR, 'rollup'),
          },
          include: ['tests/tree-shaking/fixtures/**/*.ts'],
        }),
      ],
      treeshake: {
        moduleSideEffects: false,
        propertyReadSideEffects: false,
      },
      onwarn: (warning, warn) => {
        if (warning.code === 'CIRCULAR_DEPENDENCY') return;
        warn(warning);
      },
    };

    const outputOptions: OutputOptions = {
      file: outfile,
      format: 'esm',
    };

    const bundle = await rollup(inputOptions);
    await bundle.write(outputOptions);

    const { output } = await bundle.generate({ format: 'esm' });
    const modules =
      output[0].type === 'chunk' && output[0].modules
        ? Object.keys(output[0].modules).filter((m) => m.includes('src/'))
        : [];

    await bundle.close();

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
export async function buildWithWebpack(fixtureName: string): Promise<BundleResult> {
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
        alias: {
          'numpy-ts/core': NUMPY_TS_CORE,
          'numpy-ts': NUMPY_TS_ESM,
        },
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
                transpileOnly: true,
              },
            },
            exclude: /node_modules/,
          },
          {
            test: /\.js$/,
            resolve: {
              fullySpecified: false,
            },
          },
        ],
      },
      externals: [/^node:/],
      optimization: {
        usedExports: true,
        sideEffects: true,
      },
    };

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
