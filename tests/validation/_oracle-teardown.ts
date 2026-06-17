/**
 * Validation-project setup file: kill the persistent NumPy oracle worker after
 * each test file completes. Without this, the worker (blocked reading stdin)
 * keeps its pipes open and prevents the vitest pool from shutting down cleanly.
 *
 * Per-file teardown still keeps the worker alive across every assertion within a
 * file, so `import numpy` is paid once per file instead of once per assertion.
 *
 * Named with a leading underscore so the validation project's glob excludes it
 * from test collection; it is wired in explicitly via `setupFiles`.
 */
import { afterAll } from 'vitest';
import { killNumpyWorker } from './numpy-oracle';

afterAll(() => {
  killNumpyWorker();
});
