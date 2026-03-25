import { describe, it, expect, beforeAll } from 'vitest';
import { mkdir } from 'fs/promises';
import {
  ALL_FIXTURES,
  OUTPUT_DIR,
  THRESHOLDS,
  BundleResult,
  logBundleSizes,
  buildWithEsbuild,
} from './shared';
import { resolve } from 'path';

const results = new Map<string, BundleResult>();

describe('esbuild tree-shaking', () => {
  beforeAll(async () => {
    await mkdir(resolve(OUTPUT_DIR, 'esbuild'), { recursive: true });
  });

  it('should build all fixtures', async () => {
    const builds = await Promise.all(
      ALL_FIXTURES.map(async (fixture) => {
        const result = await buildWithEsbuild(fixture.name);
        results.set(fixture.name, result);
        return result;
      })
    );
    for (const result of builds) {
      expect(result.success, `${result.fixture} should build: ${result.error}`).toBe(true);
    }

    logBundleSizes('esbuild', results);
  }, 60000);

  it('tree-shaking effectiveness should not regress', () => {
    const full = results.get('full-import');
    if (!full?.success) return;
    const fullSize = full.minifiedSize;

    const checks: [string, number][] = [
      ['full-single', THRESHOLDS.mainSingleFunction],
      ['core-single', THRESHOLDS.coreSingleFunction],
      ['core-math', THRESHOLDS.coreMath],
      ['core-linalg', THRESHOLDS.coreLinalg],
      ['core-fft', THRESHOLDS.coreFft],
    ];

    for (const [name, threshold] of checks) {
      const r = results.get(name);
      if (!r?.success) continue;
      const pct = (r.minifiedSize / fullSize) * 100;
      expect(pct, `${name} is ${pct.toFixed(1)}% of full (threshold: ${threshold}%)`).toBeLessThan(
        threshold
      );
    }
  });
});
