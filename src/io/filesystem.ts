/// <reference types="node" />

/**
 * Runtime filesystem abstraction for cross-platform file IO.
 *
 * Pre-loads `node:fs` and `node:fs/promises` asynchronously at module load
 * in Node-like environments. Both getFs() and getFsSync() read from this cache.
 * In browsers, all IO functions throw a helpful error.
 */

type FsPromises = typeof import('node:fs/promises');
type FsSync = typeof import('node:fs');

let cachedFsPromises: FsPromises | undefined;
let cachedFsSync: FsSync | undefined;

// Declare globals that exist in Deno/Bun but not in standard TS lib
declare const Deno: unknown;
declare const Bun: unknown;

/**
 * Detect whether we're running in a Node-like environment (Node, Bun, Deno).
 */
function isNodeLike(): boolean {
  return !!(
    (typeof globalThis !== 'undefined' &&
      globalThis.process &&
      globalThis.process.versions &&
      globalThis.process.versions.node) ||
    typeof Deno !== 'undefined' ||
    typeof Bun !== 'undefined'
  );
}

function throwBrowserError(): never {
  throw new Error(
    'File IO requires Node.js, Bun, or Deno. In the browser, use parseNpy(buffer) / parseTxt(text) with fetch() instead.',
  );
}

/**
 * Extract the default export or the module itself from a dynamic import result.
 * Node built-in dynamic imports may return { default: module } or the module directly.
 */
function unwrapModule<T>(mod: T & { default?: T }): T {
  return mod.default ?? mod;
}

// In Node-like environments, eagerly resolve fs modules at import time.
// Built-in module imports resolve in the next microtask — well before any
// user code can call getFs/getFsSync.
if (isNodeLike()) {
  void Promise.all([import('node:fs'), import('node:fs/promises')])
    .then(([fs, fsp]) => {
      cachedFsSync = unwrapModule(fs) as FsSync;
      cachedFsPromises = unwrapModule(fsp) as FsPromises;
    })
    .catch(() => {
      // ignore — will surface as an error when getFs/getFsSync is called
    });
}

/**
 * Get the async fs module (`node:fs/promises`).
 * Caches the result after the first successful resolution.
 */
export async function getFs(): Promise<FsPromises> {
  if (cachedFsPromises) return cachedFsPromises;
  if (!isNodeLike()) throwBrowserError();
  const [fsp, fs] = await Promise.all([import('node:fs/promises'), import('node:fs')]);
  cachedFsPromises = unwrapModule(fsp) as FsPromises;
  cachedFsSync = unwrapModule(fs) as FsSync;
  return cachedFsPromises;
}

/**
 * Get the sync fs module (`node:fs`).
 * Reads from the cache populated at module load time or by a prior getFs() call.
 */
export function getFsSync(): FsSync {
  if (cachedFsSync) return cachedFsSync;
  if (!isNodeLike()) throwBrowserError();
  throw new Error(
    'node:fs has not been resolved yet. Call any async IO function first ' +
      '(e.g., await loadNpy(...)) or use the async variants directly.',
  );
}
