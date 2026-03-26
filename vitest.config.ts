import { defineConfig, defineProject } from 'vitest/config';
import { playwright } from '@vitest/browser-playwright';

export default defineConfig({
  test: {
    globals: true,
    environment: 'node',
    reporters: ['default'],
    coverage: {
      provider: 'v8',
      reporter: ['text', 'text-summary', 'json', 'json-summary', 'html'],
      reportsDirectory: './coverage',
      exclude: [
        'node_modules/**',
        'dist/**',
        'benchmarks/**',
        '**/*.d.ts',
        '**/*.config.*',
        '**/test/**',
        '**/tests/**',
        'src/common/wasm/bins/**',
      ],
      thresholds: {
        statements: 80,
        branches: 75,
        functions: 95,
        lines: 80,
      },
    },
    // Projects configuration (replaces workspace in Vitest 4+)
    projects: [
      // Unit tests (fast, no Python needed)
      defineProject({
        test: {
          name: 'unit',
          include: ['tests/unit/**'],
          exclude: ['**/node_modules/**'],
          environment: 'node',
        },
      }),
      // Validation tests (requires Python + NumPy)
      defineProject({
        test: {
          name: 'validation',
          include: ['tests/validation/**'],
          exclude: ['**/node_modules/**', '**/*.md', '**/numpy-oracle.ts', '**/_helpers.ts'],
          environment: 'node',
        },
      }),
      // Runtime file IO tests (runs under Node, Bun, and Deno)
      defineProject({
        test: {
          name: 'runtime-io',
          include: ['tests/runtime/**/*.test.ts'],
          exclude: ['**/node_modules/**'],
          environment: 'node',
        },
      }),
      // ESM bundle smoke test (tests dist/esm/ with file IO)
      defineProject({
        test: {
          name: 'bundle-esm',
          include: ['tests/bundles/node.test.ts'],
          environment: 'node',
        },
      }),
      // Browser IIFE bundle test (runs in real browser)
      defineProject({
        test: {
          name: 'bundle-browser',
          include: ['tests/bundles/browser.test.ts'],
          browser: {
            enabled: true,
            headless: true,
            provider: playwright({
              launchOptions: {
                headless: true,
              },
            }),
            instances: [{ browser: 'chromium' }],
          },
        },
      }),
      // Full unit test suite in the browser (Playwright + Chromium)
      // Validates that all array operations work in a real browser environment
      defineProject({
        test: {
          name: 'browser-unit',
          include: ['tests/unit/**'],
          exclude: ['**/node_modules/**', '**/__screenshots__/**'],
          browser: {
            enabled: true,
            headless: true,
            screenshotFailures: false,
            provider: playwright({
              launchOptions: {
                headless: true,
              },
            }),
            instances: [{ browser: 'chromium' }],
          },
        },
      }),
      // Tree-shaking tests (tests with multiple bundlers)
      // Runs after all other projects (groupOrder: 1) so the beforeAll can
      // rebuild with production (ReleaseFast) WASM without clobbering dist/
      // while bundle tests are still reading from it.
      defineProject({
        test: {
          name: 'tree-shaking',
          include: ['tests/tree-shaking/**/*.test.ts'],
          exclude: ['**/node_modules/**', '**/fixtures/**'],
          environment: 'node',
          testTimeout: 300000, // 5 minutes - bundling can take time
          sequence: { groupOrder: 1 },
          globalSetup: ['tests/tree-shaking/global-setup.ts'],
        },
      }),
    ],
  },
});
