/**
 * Smoke test for Node.js ESM bundle
 * Tests the actual distributed tree-shakeable ESM from dist/esm/
 */

import { describe, test, expect, beforeAll, afterAll } from 'vitest';
import { resolve, join } from 'path';
import { fileURLToPath } from 'url';
import { dirname } from 'path';
import { mkdtempSync, rmSync } from 'fs';
import { tmpdir } from 'os';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

describe('Node.js ESM Bundle Smoke Test', () => {
  let np: any;
  let tmpDir: string;

  beforeAll(async () => {
    const bundlePath = resolve(__dirname, '../../dist/esm/index.js');
    np = await import(bundlePath);
    tmpDir = mkdtempSync(join(tmpdir(), 'numpy-ts-node-'));
  }, 30_000);

  afterAll(() => {
    rmSync(tmpDir, { recursive: true, force: true });
  });

  test('should export main functions', () => {
    expect(np.array).toBeDefined();
    expect(np.zeros).toBeDefined();
    expect(np.ones).toBeDefined();
    expect(np.arange).toBeDefined();
  });

  test('should create arrays', () => {
    const arr = np.array([1, 2, 3, 4]);
    expect(arr.shape).toEqual([4]);
    expect(arr.toArray()).toEqual([1, 2, 3, 4]);
  });

  test('should perform basic math', () => {
    const a = np.array([1, 2, 3]);
    const b = np.array([4, 5, 6]);
    const result = a.add(b);
    expect(result.toArray()).toEqual([5, 7, 9]);
  });

  test('should handle matrix operations', () => {
    const a = np.array([
      [1, 2],
      [3, 4],
    ]);
    const b = np.array([
      [5, 6],
      [7, 8],
    ]);
    const result = a.matmul(b);
    expect(result.shape).toEqual([2, 2]);
    expect(result.toArray()).toEqual([
      [19, 22],
      [43, 50],
    ]);
  });

  test('should create zeros and ones', () => {
    const z = np.zeros([2, 3]);
    expect(z.shape).toEqual([2, 3]);
    expect(z.toArray()).toEqual([
      [0, 0, 0],
      [0, 0, 0],
    ]);

    const o = np.ones([3, 2]);
    expect(o.shape).toEqual([3, 2]);
    expect(o.toArray()).toEqual([
      [1, 1],
      [1, 1],
      [1, 1],
    ]);
  });

  // ── File IO ─────────────────────────────────────────────────────────

  test('should export file IO functions', () => {
    expect(typeof np.loadNpy).toBe('function');
    expect(typeof np.loadNpySync).toBe('function');
    expect(typeof np.saveNpy).toBe('function');
    expect(typeof np.saveNpySync).toBe('function');
    expect(typeof np.loadtxt).toBe('function');
    expect(typeof np.savetxt).toBe('function');
    expect(typeof np.savez).toBe('function');
    expect(typeof np.load).toBe('function');
    expect(typeof np.save).toBe('function');
  });

  test('saveNpy + loadNpy async roundtrip', async () => {
    const arr = np.array([
      [1, 2],
      [3, 4],
    ]);
    const p = join(tmpDir, 'node-async.npy');
    await np.saveNpy(p, arr);
    const loaded = await np.loadNpy(p);
    expect(loaded.tolist()).toEqual([
      [1, 2],
      [3, 4],
    ]);
  });

  test('saveNpySync + loadNpySync sync roundtrip', () => {
    const arr = np.array([10, 20, 30]);
    const p = join(tmpDir, 'node-sync.npy');
    np.saveNpySync(p, arr);
    const loaded = np.loadNpySync(p);
    expect(loaded.tolist()).toEqual([10, 20, 30]);
  });
});
