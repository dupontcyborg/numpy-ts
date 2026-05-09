/**
 * File IO tests — verifies loadNpy/saveNpy/loadtxt/savetxt roundtrips.
 *
 * This test file is included in the `runtime-io` vitest project and is
 * designed to run under Node, Bun, and Deno (via vitest) to confirm that
 * the lazy fs resolution in src/io/filesystem.ts works on each runtime.
 */

import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { afterAll, beforeAll, describe, expect, test } from 'vitest';

import {
  array,
  load,
  loadNpy,
  loadNpySync,
  loadNpzFile,
  loadNpzFileSync,
  loadSync,
  loadtxt,
  loadtxtSync,
  save,
  saveNpy,
  saveNpySync,
  saveNpzFile,
  saveNpzFileSync,
  saveSync,
  savetxt,
  savetxtSync,
  savez,
  savez_compressed,
} from '../../src';

describe('File IO roundtrips', () => {
  let dir: string;

  beforeAll(() => {
    dir = mkdtempSync(join(tmpdir(), 'numpy-ts-io-'));
  });

  afterAll(() => {
    rmSync(dir, { recursive: true, force: true });
  });

  // ── NPY ───────────────────────────────────────────────────────────────

  test('saveNpy + loadNpy async roundtrip', async () => {
    const arr = array([1, 2, 3, 4, 5]);
    const p = join(dir, 'async.npy');
    await saveNpy(p, arr);
    const loaded = await loadNpy(p);
    expect(loaded.tolist()).toEqual([1, 2, 3, 4, 5]);
    expect(loaded.shape).toEqual([5]);
    expect(loaded.dtype).toBe('float64');
  });

  test('saveNpySync + loadNpySync roundtrip', () => {
    const arr = array([
      [1, 2],
      [3, 4],
    ]);
    const p = join(dir, 'sync.npy');
    saveNpySync(p, arr);
    const loaded = loadNpySync(p);
    expect(loaded.tolist()).toEqual([
      [1, 2],
      [3, 4],
    ]);
    expect(loaded.shape).toEqual([2, 2]);
  });

  // ── NPZ ───────────────────────────────────────────────────────────────

  test('saveNpzFile + loadNpzFile async roundtrip', async () => {
    const x = array([10, 20, 30]);
    const y = array([1.5, 2.5]);
    const p = join(dir, 'multi.npz');
    await saveNpzFile(p, { x, y });
    const result = await loadNpzFile(p);
    expect(result.arrays.get('x')!.tolist()).toEqual([10, 20, 30]);
    expect(result.arrays.get('y')!.tolist()).toEqual([1.5, 2.5]);
  });

  test('saveNpzFileSync + loadNpzFileSync roundtrip', () => {
    const a = array([7, 8, 9]);
    const p = join(dir, 'sync-multi.npz');
    saveNpzFileSync(p, { a });
    const result = loadNpzFileSync(p);
    expect(result.arrays.get('a')!.tolist()).toEqual([7, 8, 9]);
  });

  test('savez appends .npz extension', async () => {
    const arr = array([1, 2]);
    const p = join(dir, 'savez-test');
    await savez(p, [arr]);
    const result = await loadNpzFile(p + '.npz');
    expect(result.arrays.get('arr_0')!.tolist()).toEqual([1, 2]);
  });

  test('savez_compressed roundtrip', async () => {
    const arr = array([100, 200, 300]);
    const p = join(dir, 'compressed.npz');
    await savez_compressed(p, { data: arr });
    const result = await loadNpzFile(p);
    expect(result.arrays.get('data')!.tolist()).toEqual([100, 200, 300]);
  });

  // ── Auto-detect (load/save) ───────────────────────────────────────────

  test('save + load auto-detect .npy', async () => {
    const arr = array([42]);
    const p = join(dir, 'auto.npy');
    await save(p, arr);
    const loaded = await load(p);
    expect((loaded as any).tolist()).toEqual([42]);
  });

  test('saveSync + loadSync auto-detect .npy', () => {
    const arr = array([99]);
    const p = join(dir, 'auto-sync.npy');
    saveSync(p, arr);
    const loaded = loadSync(p);
    expect((loaded as any).tolist()).toEqual([99]);
  });

  test('load auto-detects .npz', async () => {
    const arr = array([1, 2]);
    const p = join(dir, 'auto.npz');
    await saveNpzFile(p, { arr });
    const result = (await load(p)) as any;
    expect(result.arrays.get('arr')!.tolist()).toEqual([1, 2]);
  });

  // ── Text ──────────────────────────────────────────────────────────────

  test('savetxt + loadtxt async roundtrip', async () => {
    const arr = array([
      [1, 2, 3],
      [4, 5, 6],
    ]);
    const p = join(dir, 'data.csv');
    await savetxt(p, arr, { delimiter: ',' });
    const loaded = await loadtxt(p, { delimiter: ',' });
    expect(loaded.shape).toEqual([2, 3]);
    expect(loaded.tolist()).toEqual([
      [1, 2, 3],
      [4, 5, 6],
    ]);
  });

  test('savetxtSync + loadtxtSync roundtrip', () => {
    const arr = array([
      [10, 20],
      [30, 40],
    ]);
    const p = join(dir, 'sync.txt');
    savetxtSync(p, arr);
    const loaded = loadtxtSync(p);
    expect(loaded.shape).toEqual([2, 2]);
    expect(loaded.tolist()).toEqual([
      [10, 20],
      [30, 40],
    ]);
  });

  // ── Error cases ───────────────────────────────────────────────────────

  test('load rejects unknown extension', async () => {
    await expect(load(join(dir, 'test.xyz'))).rejects.toThrow('Unknown file extension');
  });

  test('save rejects non-.npy extension', async () => {
    await expect(save(join(dir, 'test.npz'), array([1]))).rejects.toThrow('Use saveNpz');
  });
});
