/**
 * Bundler dispose-protocol tests
 *
 * Verifies that the Symbol.dispose polyfill survives tree-shaking and that
 * transpiled `using` statements work correctly through esbuild, Rollup, and
 * Webpack. Each bundler builds the dispose-using fixture with a downlevel
 * target (ES2020, which lacks native `using`), then we execute the bundle
 * and assert it exits cleanly.
 *
 * A failure here reproduces the Safari "JS object not disposable" bug that
 * occurs when the polyfill is missing and a transpiled helper can't find
 * [Symbol.dispose].
 */

import { describe, it, expect, beforeAll } from 'vitest';
import { build as esbuildBuild } from 'esbuild';
import { rollup } from 'rollup';
import nodeResolve from '@rollup/plugin-node-resolve';
import alias from '@rollup/plugin-alias';
import webpack from 'webpack';
import { mkdir, readFile, stat } from 'fs/promises';
import { resolve, dirname } from 'path';
import { fileURLToPath } from 'url';
import { execFileSync } from 'child_process';

import { NUMPY_TS_ESM, NUMPY_TS_CORE } from './shared';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const FIXTURE = resolve(__dirname, 'fixtures/dispose-using.ts');
const BROWSER_FIXTURE = resolve(__dirname, 'fixtures/dispose-using-browser.ts');
const OUT_DIR = resolve(__dirname, '.output/dispose');
const BROWSER_OUT_DIR = resolve(__dirname, '.output/dispose/browser');
const NUMPY_TS_BROWSER = resolve(__dirname, '../../dist/numpy-ts.browser.js');

/** Run a bundled JS file with Node and return stdout. Throws on non-zero exit. */
function execBundle(file: string): string {
  return execFileSync('node', [file], {
    encoding: 'utf-8',
    timeout: 15000,
    env: { ...process.env, NODE_NO_WARNINGS: '1' },
  }).trim();
}

describe('esbuild dispose protocol', () => {
  const outfile = resolve(OUT_DIR, 'esbuild-dispose.mjs');

  beforeAll(async () => {
    await mkdir(resolve(OUT_DIR), { recursive: true });
  });

  it('builds dispose-using fixture with downlevel target', async () => {
    await esbuildBuild({
      entryPoints: [FIXTURE],
      bundle: true,
      platform: 'node',
      format: 'esm',
      // ES2020 forces `using` to be downleveled — the helper will look up
      // Symbol.dispose || Symbol.for("Symbol.dispose").
      target: 'es2020',
      outfile,
      write: true,
      alias: {
        'numpy-ts/core': NUMPY_TS_CORE,
        'numpy-ts': NUMPY_TS_ESM,
      },
    });

    const s = await stat(outfile);
    expect(s.size).toBeGreaterThan(0);
  }, 60000);

  it('bundled output executes without "not disposable" error', () => {
    const out = execBundle(outfile);
    expect(out).toContain('PASS');
  });

  it('bundle contains the polyfill', async () => {
    const code = await readFile(outfile, 'utf-8');
    expect(code).toContain('Symbol.dispose');
  });

  it('bundle does not contain raw `using` declarations', async () => {
    const code = await readFile(outfile, 'utf-8');
    expect(code).not.toMatch(/\busing\s+\w+\s*=/);
  });
});

describe('Rollup dispose protocol', () => {
  const preTranspiled = resolve(OUT_DIR, 'rollup-dispose-pre.mjs');
  const outfile = resolve(OUT_DIR, 'rollup-dispose.mjs');

  beforeAll(async () => {
    await mkdir(resolve(OUT_DIR), { recursive: true });
  });

  it('builds dispose-using fixture with downlevel target', async () => {
    // Step 1: Pre-transpile the TS fixture to ES2020 JS with esbuild.
    // @rollup/plugin-typescript cannot downlevel `using` reliably, but
    // esbuild can. This mirrors real-world setups where Rollup delegates
    // transpilation to esbuild.
    await esbuildBuild({
      entryPoints: [FIXTURE],
      bundle: false,
      platform: 'node',
      format: 'esm',
      target: 'es2020',
      outfile: preTranspiled,
      write: true,
    });

    // Step 2: Bundle the pre-transpiled JS with Rollup.
    const bundle = await rollup({
      input: preTranspiled,
      plugins: [
        alias({
          entries: [
            { find: 'numpy-ts/core', replacement: NUMPY_TS_CORE },
            { find: 'numpy-ts', replacement: NUMPY_TS_ESM },
          ],
        }),
        nodeResolve({ extensions: ['.mjs', '.js'] }),
      ],
      treeshake: {
        moduleSideEffects: true,
      },
      onwarn: (warning, warn) => {
        if (warning.code === 'CIRCULAR_DEPENDENCY') return;
        warn(warning);
      },
    });

    await bundle.write({ file: outfile, format: 'esm' });
    await bundle.close();

    const s = await stat(outfile);
    expect(s.size).toBeGreaterThan(0);
  }, 120000);

  it('bundled output executes without "not disposable" error', () => {
    const out = execBundle(outfile);
    expect(out).toContain('PASS');
  });

  it('bundle contains the polyfill', async () => {
    const code = await readFile(outfile, 'utf-8');
    expect(code).toContain('Symbol.dispose');
  });

  it('bundle does not contain raw `using` declarations', async () => {
    const code = await readFile(outfile, 'utf-8');
    expect(code).not.toMatch(/\busing\s+\w+\s*=/);
  });
});

describe('Webpack dispose protocol', () => {
  const outDir = resolve(OUT_DIR, 'webpack-dispose');
  const outfile = resolve(outDir, 'bundle.mjs');

  beforeAll(async () => {
    await mkdir(outDir, { recursive: true });
  });

  it('builds dispose-using fixture with downlevel target', async () => {
    await new Promise<void>((resolveP, reject) => {
      webpack(
        {
          entry: FIXTURE,
          mode: 'production',
          output: {
            path: outDir,
            filename: 'bundle.mjs',
            library: { type: 'module' },
          },
          experiments: { outputModule: true },
          target: 'es2020',
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
                      target: 'ES2020',
                    },
                    transpileOnly: true,
                  },
                },
                exclude: /node_modules/,
              },
              {
                test: /\.js$/,
                resolve: { fullySpecified: false },
              },
            ],
          },
          externals: [/^node:/],
          optimization: {
            usedExports: true,
            sideEffects: true,
            minimize: false,
          },
        },
        (err, stats) => {
          if (err) reject(err);
          else if (stats?.hasErrors()) {
            const info = stats.toJson();
            reject(new Error(info.errors?.map((e) => e.message).join('\n') || 'Build failed'));
          } else resolveP();
        }
      );
    });

    const s = await stat(outfile);
    expect(s.size).toBeGreaterThan(0);
  }, 120000);

  it('bundled output executes without "not disposable" error', () => {
    const out = execBundle(outfile);
    expect(out).toContain('PASS');
  });

  it('bundle contains the polyfill', async () => {
    const code = await readFile(outfile, 'utf-8');
    expect(code).toContain('Symbol.dispose');
  });

  it('bundle does not contain raw `using` declarations', async () => {
    const code = await readFile(outfile, 'utf-8');
    expect(code).not.toMatch(/\busing\s+\w+\s*=/);
  });
});

// ---------------------------------------------------------------------------
// Browser-targeted builds
//
// These bundles are consumed by the browser-dispose vitest project (Playwright).
// They use the browser fixture (exports run()) and alias numpy-ts to the
// browser bundle so WASM loading works in browser environments.
// ---------------------------------------------------------------------------

describe('browser builds', () => {
  beforeAll(async () => {
    await mkdir(BROWSER_OUT_DIR, { recursive: true });
  });

  describe('esbuild browser build', () => {
    const browserOutfile = resolve(BROWSER_OUT_DIR, 'esbuild.mjs');

    it('builds browser fixture with downlevel target', async () => {
      await esbuildBuild({
        entryPoints: [BROWSER_FIXTURE],
        bundle: true,
        platform: 'browser',
        format: 'esm',
        target: 'es2020',
        outfile: browserOutfile,
        write: true,
        external: ['node:fs', 'node:fs/promises', 'node:module'],
        alias: {
          'numpy-ts/core': NUMPY_TS_BROWSER,
          'numpy-ts': NUMPY_TS_BROWSER,
        },
      });

      const s = await stat(browserOutfile);
      expect(s.size).toBeGreaterThan(0);
    }, 60000);

    it('exports run() that passes on Node', async () => {
      const mod = await import(browserOutfile);
      const result = mod.run();
      expect(result.passed, result.error).toBe(true);
      expect(result.leak).toBe(0);
    });

    it('no raw using in bundle', async () => {
      const code = await readFile(browserOutfile, 'utf-8');
      expect(code).not.toMatch(/\busing\s+\w+\s*=/);
    });
  });

  describe('Rollup browser build', () => {
    // @rollup/plugin-typescript does not downlevel `using` (confirmed bug).
    // Real Rollup users use rollup-plugin-esbuild instead — our pre-transpile
    // approach is functionally equivalent.
    const browserPreTranspiled = resolve(BROWSER_OUT_DIR, 'rollup-pre.mjs');
    const browserOutfile = resolve(BROWSER_OUT_DIR, 'rollup.mjs');

    it('builds browser fixture with downlevel target', async () => {
      await esbuildBuild({
        entryPoints: [BROWSER_FIXTURE],
        bundle: false,
        platform: 'browser',
        format: 'esm',
        target: 'es2020',
        outfile: browserPreTranspiled,
        write: true,
      });

      const bundle = await rollup({
        input: browserPreTranspiled,
        plugins: [
          alias({
            entries: [
              { find: 'numpy-ts/core', replacement: NUMPY_TS_BROWSER },
              { find: 'numpy-ts', replacement: NUMPY_TS_BROWSER },
            ],
          }),
          nodeResolve({ extensions: ['.mjs', '.js'] }),
        ],
        treeshake: { moduleSideEffects: true },
        onwarn: (warning, warn) => {
          if (warning.code === 'CIRCULAR_DEPENDENCY') return;
          warn(warning);
        },
      });

      await bundle.write({ file: browserOutfile, format: 'esm' });
      await bundle.close();

      const s = await stat(browserOutfile);
      expect(s.size).toBeGreaterThan(0);
    }, 120000);

    it('exports run() that passes on Node', async () => {
      const mod = await import(browserOutfile);
      const result = mod.run();
      expect(result.passed, result.error).toBe(true);
      expect(result.leak).toBe(0);
    });

    it('no raw using in bundle', async () => {
      const code = await readFile(browserOutfile, 'utf-8');
      expect(code).not.toMatch(/\busing\s+\w+\s*=/);
    });
  });

  describe('Webpack browser build', () => {
    const webpackBrowserDir = resolve(BROWSER_OUT_DIR, 'webpack');
    const browserOutfile = resolve(webpackBrowserDir, 'bundle.mjs');

    it('builds browser fixture with downlevel target', async () => {
      await mkdir(webpackBrowserDir, { recursive: true });

      await new Promise<void>((resolveP, reject) => {
        webpack(
          {
            entry: BROWSER_FIXTURE,
            mode: 'production',
            output: {
              path: webpackBrowserDir,
              filename: 'bundle.mjs',
              library: { type: 'module' },
            },
            experiments: { outputModule: true },
            target: ['web', 'es2020'],
            resolve: {
              extensions: ['.ts', '.js'],
              alias: {
                'numpy-ts/core': NUMPY_TS_BROWSER,
                'numpy-ts': NUMPY_TS_BROWSER,
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
                        target: 'ES2020',
                      },
                      transpileOnly: true,
                    },
                  },
                  exclude: /node_modules/,
                },
                {
                  test: /\.js$/,
                  resolve: { fullySpecified: false },
                },
              ],
            },
            externals: [/^node:/],
            optimization: {
              usedExports: true,
              sideEffects: true,
              minimize: false,
            },
          },
          (err, stats) => {
            if (err) reject(err);
            else if (stats?.hasErrors()) {
              const info = stats.toJson();
              reject(new Error(info.errors?.map((e) => e.message).join('\n') || 'Build failed'));
            } else resolveP();
          }
        );
      });

      const s = await stat(browserOutfile);
      expect(s.size).toBeGreaterThan(0);
    }, 120000);

    it('exports run() that passes on Node', async () => {
      const mod = await import(browserOutfile);
      const result = mod.run();
      expect(result.passed, result.error).toBe(true);
      expect(result.leak).toBe(0);
    });

    it('no raw using in bundle', async () => {
      const code = await readFile(browserOutfile, 'utf-8');
      expect(code).not.toMatch(/\busing\s+\w+\s*=/);
    });
  });
});
