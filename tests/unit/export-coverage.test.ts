/**
 * Tests that standalone exports cover all main library function exports
 *
 * This test ensures that numpy-ts/standalone exports all the same functions
 * as the main numpy-ts entry point (excluding intentional differences like
 * NDArray class and namespace objects).
 */

import { describe, it, expect } from 'vitest';
import * as main from '../../src/index';
import * as standalone from '../../src/standalone';

describe('Export Coverage', () => {
  // Exports that are intentionally different between main and standalone
  const INTENTIONAL_EXCLUSIONS = new Set([
    // NDArray class (standalone uses NDArrayCore)
    'NDArray',

    // Namespace objects (these contain methods, not tree-shakeable)
    'random',
    'fft',

    // Version placeholder
    '__version__',
  ]);

  // Type-only exports (not functions, just TypeScript types)
  const TYPE_ONLY_EXPORTS = new Set([
    'NpyHeader',
    'NpyMetadata',
    'NpyVersion',
    'NpzParseOptions',
    'NpzParseResult',
    'NpzSerializeOptions',
    'ComplexInput',
  ]);

  it('should list all main exports', () => {
    const mainExports = Object.keys(main).sort();
    console.log('\n=== MAIN LIBRARY EXPORTS ===');
    console.log(`Total: ${mainExports.length} exports`);
    console.log(mainExports.join(', '));
  });

  it('should list all standalone exports', () => {
    const standaloneExports = Object.keys(standalone).sort();
    console.log('\n=== STANDALONE EXPORTS ===');
    console.log(`Total: ${standaloneExports.length} exports`);
    console.log(standaloneExports.join(', '));
  });

  it('should identify missing exports in standalone', () => {
    const mainExports = new Set(Object.keys(main));
    const standaloneExports = new Set(Object.keys(standalone));

    const missingInStandalone: string[] = [];

    for (const exp of mainExports) {
      if (INTENTIONAL_EXCLUSIONS.has(exp)) continue;
      if (TYPE_ONLY_EXPORTS.has(exp)) continue;
      if (!standaloneExports.has(exp)) {
        missingInStandalone.push(exp);
      }
    }

    console.log('\n=== MISSING IN STANDALONE ===');
    if (missingInStandalone.length === 0) {
      console.log('None! Standalone has full coverage.');
    } else {
      console.log(`Missing ${missingInStandalone.length} exports:`);
      missingInStandalone.sort().forEach((exp) => console.log(`  - ${exp}`));
    }

    // This is the actual test - standalone should export everything main does
    // (except intentional exclusions)
    expect(
      missingInStandalone,
      `Standalone is missing these exports: ${missingInStandalone.join(', ')}`
    ).toEqual([]);
  });

  it('should identify extra exports in standalone (not in main)', () => {
    const mainExports = new Set(Object.keys(main));
    const standaloneExports = new Set(Object.keys(standalone));

    const extraInStandalone: string[] = [];

    for (const exp of standaloneExports) {
      if (!mainExports.has(exp)) {
        extraInStandalone.push(exp);
      }
    }

    console.log('\n=== EXTRA IN STANDALONE (not in main) ===');
    if (extraInStandalone.length === 0) {
      console.log('None.');
    } else {
      console.log(`Extra ${extraInStandalone.length} exports:`);
      extraInStandalone.sort().forEach((exp) => console.log(`  - ${exp}`));
    }

    // Extra exports in standalone are fine (they're aliases), just log them
  });

  it('should show coverage statistics', () => {
    const mainExports = new Set(Object.keys(main));
    const standaloneExports = new Set(Object.keys(standalone));

    // Remove intentional exclusions and type-only from main count
    let comparableMainCount = 0;
    for (const exp of mainExports) {
      if (!INTENTIONAL_EXCLUSIONS.has(exp) && !TYPE_ONLY_EXPORTS.has(exp)) {
        comparableMainCount++;
      }
    }

    // Count how many of those are in standalone
    let coveredCount = 0;
    for (const exp of mainExports) {
      if (INTENTIONAL_EXCLUSIONS.has(exp)) continue;
      if (TYPE_ONLY_EXPORTS.has(exp)) continue;
      if (standaloneExports.has(exp)) {
        coveredCount++;
      }
    }

    const coverage = ((coveredCount / comparableMainCount) * 100).toFixed(1);

    console.log('\n=== COVERAGE STATISTICS ===');
    console.log(`Main exports (total): ${mainExports.size}`);
    console.log(`Main exports (comparable): ${comparableMainCount}`);
    console.log(`Standalone exports: ${standaloneExports.size}`);
    console.log(`Coverage: ${coveredCount}/${comparableMainCount} (${coverage}%)`);

    // Require 100% coverage
    expect(Number(coverage)).toBe(100);
  });
});
