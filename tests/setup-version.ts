/**
 * Inject the package.json version onto globalThis as __VERSION_PLACEHOLDER__.
 *
 * The built bundle (dist/esm/) has this baked in by esbuild's `define` (see
 * build.ts). For source-import test runs (no build step), src/index.ts's
 * `typeof __VERSION_PLACEHOLDER__` check resolves to the value injected here.
 *
 * Wired into each source-importing project as `setupFiles` in vitest.config.ts.
 * Imports package.json directly so the value is inlined at transform time —
 * works in both Node and browser (vite handles JSON imports natively).
 *
 * The bundle smoke test (tests/bundles/node.test.ts) separately asserts the
 * built artifact reports a real semver, not the 'dev' fallback.
 */
import pkg from '../package.json';

(globalThis as { __VERSION_PLACEHOLDER__?: string }).__VERSION_PLACEHOLDER__ = pkg.version;
