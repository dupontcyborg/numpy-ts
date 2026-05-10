/**
 * Browser dispose-protocol test — bundled path
 *
 * Tests the real end-to-end user path: user writes `using arr = np.zeros(...)`,
 * a bundler (esbuild/rollup/webpack) transpiles `using` to a helper that looks
 * up Symbol.dispose, and the polyfill ensures the helper finds the method.
 *
 * Architecture:
 *   1. The tree-shaking project (groupOrder 2) builds browser-targeted bundles
 *      from dispose-using-browser.ts with each bundler (target: es2020).
 *   2. This test (groupOrder 3) loads those pre-built bundles in Playwright
 *      and calls run() to verify the dispose protocol works.
 *
 * This file contains ZERO `using` syntax — it is safe to load in WebKit,
 * which cannot parse `using`. The `using` keyword only exists inside the
 * pre-built bundles where it has been transpiled to helper code.
 *
 * Test matrix covered:
 *   esbuild  × Chromium, Firefox, WebKit
 *   Rollup   × Chromium, Firefox, WebKit
 *   Webpack  × Chromium, Firefox, WebKit
 */

import { describe, expect, test } from 'vitest';

const BUNDLERS = [
  { name: 'esbuild', path: '/tests/tree-shaking/.output/dispose/browser/esbuild.mjs' },
  { name: 'Rollup', path: '/tests/tree-shaking/.output/dispose/browser/rollup.mjs' },
  { name: 'Webpack', path: '/tests/tree-shaking/.output/dispose/browser/webpack/bundle.mjs' },
] as const;

describe.each(BUNDLERS)('$name dispose in browser', ({ name, path }) => {
  test(`${name}: run() passes (using + wasmFreeBytes lifecycle)`, async () => {
    const mod = await import(/* @vite-ignore */ path);
    const result = mod.run();

    expect(result.passed, `${name}: ${result.error}`).toBe(true);
    expect(result.leak, `${name}: leaked ${result.leak} bytes`).toBe(0);
  });
});
