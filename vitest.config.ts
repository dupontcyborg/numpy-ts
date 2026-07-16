import { defineConfig, defineProject } from 'vitest/config';
import { playwright } from '@vitest/browser-playwright';

export default defineConfig({
  test: {
    globalSetup: ['tests/tree-shaking/global-setup.ts'],
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
        'src/core/index.ts',
        'src/full/index.ts',
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
          // Inject __VERSION_PLACEHOLDER__ for source-imported __version__.
          setupFiles: ['tests/setup-version.ts'],
        },
      }),
      // Validation tests (requires Python + NumPy)
      defineProject({
        test: {
          name: 'validation',
          include: ['tests/validation/**'],
          exclude: ['**/node_modules/**', '**/*.md', '**/numpy-oracle.ts', '**/_helpers.ts', '**/_dtype-matrix.ts', '**/_*.ts'],
          environment: 'node',
          setupFiles: ['tests/setup-version.ts', 'tests/validation/_oracle-teardown.ts'],
        },
      }),
      // Runtime file IO tests (runs under Node, Bun, and Deno)
      defineProject({
        test: {
          name: 'runtime-io',
          include: ['tests/runtime/**/*.test.ts'],
          exclude: ['**/node_modules/**'],
          environment: 'node',
          setupFiles: ['tests/setup-version.ts'],
        },
      }),
      // ESM bundle smoke test (tests dist/esm/ with file IO)
      // Runs in groupOrder: 1 so the multi-file ESM dynamic import isn't
      // starved by the parallel browser-unit projects (3 engines × all unit
      // tests) — on CI, that contention pushed the import past 30s.
      defineProject({
        test: {
          name: 'bundle-esm',
          include: ['tests/bundles/node.test.ts'],
          environment: 'node',
          sequence: { groupOrder: 1 },
        },
      }),
      // Browser ESM bundle test (runs in real browser — all major engines)
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
            instances: [
              { browser: 'chromium' },
              { browser: 'firefox' },
              { browser: 'webkit' },
            ],
          },
        },
      }),
      // Full unit test suite in the browser (Playwright — all major engines)
      // Validates that all array operations work in a real browser environment
      defineProject({
        test: {
          name: 'browser-unit',
          include: ['tests/unit/**'],
          // Exclude dispose-wasm-lifecycle — it uses `using` syntax which WebKit can't parse.
          // That file is covered by browser-dispose-vanilla (Chromium+Firefox only).
          exclude: ['**/node_modules/**', '**/__screenshots__/**', '**/dispose-wasm-lifecycle*', '**/dtype-promotion.test.ts'],
          setupFiles: ['tests/setup-version.ts'],
          browser: {
            enabled: true,
            headless: true,
            screenshotFailures: false,
            provider: playwright({
              launchOptions: {
                headless: true,
              },
            }),
            instances: [
              { browser: 'chromium' },
              { browser: 'firefox' },
              { browser: 'webkit' },
            ],
          },
        },
      }),
      // WASM memory edge-case tests (OOM fallback, double-free, refcount, mixed inputs)
      defineProject({
        test: {
          name: 'wasm-memory',
          include: ['tests/wasm-memory/**/*.test.ts'],
          exclude: ['**/node_modules/**'],
          environment: 'node',
          testTimeout: 60000,
          setupFiles: ['tests/setup-version.ts'],
        },
      }),
      // WASM memory leak tests (validates operations don't leak WASM heap)
      defineProject({
        test: {
          name: 'leaks',
          include: ['tests/leaks/**/*.test.ts'],
          exclude: ['**/node_modules/**'],
          environment: 'node',
          testTimeout: 300000, // 5 minutes — runs many specs
          setupFiles: ['tests/setup-version.ts'],
        },
      }),
      // Tree-shaking tests (tests with multiple bundlers)
      // Runs last (groupOrder: 2) so the setupFile can rebuild with production
      // (ReleaseFast) WASM without clobbering dist/ while bundle tests
      // (bundle-esm at groupOrder: 1, bundle-browser at 0) are still reading
      // from it.
      // Uses setupFiles (not globalSetup) so the build respects project ordering.
      defineProject({
        test: {
          name: 'tree-shaking',
          include: ['tests/tree-shaking/**/*.test.ts'],
          exclude: ['**/node_modules/**', '**/fixtures/**'],
          environment: 'node',
          testTimeout: 300000, // 5 minutes - bundling can take time
          sequence: { groupOrder: 2 },
          setupFiles: ['tests/tree-shaking/setup.ts'],
        },
      }),
      // Browser dispose tests — bundled path (esbuild/rollup/webpack × all engines)
      // Runs at groupOrder 3 so the tree-shaking project (groupOrder 2) has
      // already built the browser bundles in .output/dispose/browser/.
      // The test file contains NO `using` syntax — safe for WebKit.
      defineProject({
        test: {
          name: 'browser-dispose',
          include: ['tests/bundles/browser-dispose-bundled.test.ts'],
          testTimeout: 60000,
          sequence: { groupOrder: 3 },
          browser: {
            enabled: true,
            headless: true,
            provider: playwright({ launchOptions: { headless: true } }),
            instances: [
              { browser: 'chromium' },
              { browser: 'firefox' },
              { browser: 'webkit' },
            ],
          },
        },
      }),
      // Browser dispose tests — vanilla path (native `using`, no transpiler)
      // Chromium and Firefox only — WebKit cannot parse `using` syntax.
      defineProject({
        test: {
          name: 'browser-dispose-vanilla',
          include: ['tests/bundles/browser-dispose.test.ts'],
          testTimeout: 60000,
          browser: {
            enabled: true,
            headless: true,
            provider: playwright({ launchOptions: { headless: true } }),
            instances: [
              { browser: 'chromium' },
              { browser: 'firefox' },
              // NO webkit — cannot parse `using` syntax
            ],
          },
        },
      }),
    ],
  },
});
