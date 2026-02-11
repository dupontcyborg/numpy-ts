import { defineConfig, defineProject } from 'vitest/config';
import { playwright } from '@vitest/browser-playwright';

export default defineConfig({
  test: {
    globals: true,
    environment: 'node',
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
      // Unit tests only
      defineProject({
        test: {
          name: 'unit',
          include: ['tests/unit/**'],
          exclude: ['**/node_modules/**'],
          environment: 'node',
        },
      }),
      // Validation tests only
      defineProject({
        test: {
          name: 'validation',
          include: ['tests/validation/**'],
          exclude: ['**/node_modules/**', '**/*.md', '**/numpy-oracle.ts'],
          environment: 'node',
        },
      }),
      // Validation tests without slow exports test (for coverage)
      defineProject({
        test: {
          name: 'validation-quick',
          include: ['tests/validation/**'],
          exclude: [
            '**/node_modules/**',
            '**/*.md',
            '**/numpy-oracle.ts',
            '**/exports.numpy.test.ts',
          ],
          environment: 'node',
        },
      }),
      // Quick tests (unit + validation except slow exports test)
      defineProject({
        test: {
          name: 'quick',
          include: ['tests/unit/**', 'tests/validation/**'],
          exclude: [
            '**/node_modules/**',
            '**/*.md',
            '**/numpy-oracle.ts',
            '**/exports.numpy.test.ts',
          ],
          environment: 'node',
        },
      }),
      // Node.js CJS bundle test
      defineProject({
        test: {
          name: 'bundle-node',
          include: ['tests/bundles/node.test.ts'],
          environment: 'node',
        },
      }),
      // ESM bundle test
      defineProject({
        test: {
          name: 'bundle-esm',
          include: ['tests/bundles/esm.test.mjs'],
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
      // Tree-shaking tests (tests with multiple bundlers)
      defineProject({
        test: {
          name: 'tree-shaking',
          include: ['tests/tree-shaking/**/*.test.ts'],
          exclude: ['**/node_modules/**', '**/fixtures/**'],
          environment: 'node',
          testTimeout: 300000, // 5 minutes - bundling can take time
        },
      }),
    ],
  },
});
