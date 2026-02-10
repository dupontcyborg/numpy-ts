/**
 * Tests that core exports cover all main library function exports
 *
 * This test ensures that numpy-ts/core exports all the same functions
 * as the main numpy-ts entry point (excluding intentional differences like
 * NDArray class and namespace objects).
 */

import { describe, it, expect } from 'vitest';
import * as main from '../../src/index';
import * as core from '../../src/core';

describe('Export Coverage', () => {
  // Exports that are intentionally different between main and core
  const INTENTIONAL_EXCLUSIONS = new Set([
    // NDArray class (core uses NDArrayCore)
    'NDArray',

    // Namespace objects (these contain methods, not tree-shakeable)
    'random',
    'fft',

    // Version placeholder
    '__version__',

    // IO functions (available via numpy-ts/io, not included in core)
    'parseNpy',
    'serializeNpy',
    'parseNpyHeader',
    'parseNpyData',
    'UnsupportedDTypeError',
    'InvalidNpyError',
    'SUPPORTED_DTYPES',
    'DTYPE_TO_DESCR',
    'parseNpz',
    'parseNpzSync',
    'loadNpz',
    'loadNpzSync',
    'serializeNpz',
    'serializeNpzSync',
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

  it('should list all core exports', () => {
    const coreExports = Object.keys(core).sort();
    console.log('\n=== CORE EXPORTS ===');
    console.log(`Total: ${coreExports.length} exports`);
    console.log(coreExports.join(', '));
  });

  it('should identify missing exports in core', () => {
    const mainExports = new Set(Object.keys(main));
    const coreExports = new Set(Object.keys(core));

    const missingInStandalone: string[] = [];

    for (const exp of mainExports) {
      if (INTENTIONAL_EXCLUSIONS.has(exp)) continue;
      if (TYPE_ONLY_EXPORTS.has(exp)) continue;
      if (!coreExports.has(exp)) {
        missingInStandalone.push(exp);
      }
    }

    console.log('\n=== MISSING IN CORE ===');
    if (missingInStandalone.length === 0) {
      console.log('None! Core has full coverage.');
    } else {
      console.log(`Missing ${missingInStandalone.length} exports:`);
      missingInStandalone.sort().forEach((exp) => console.log(`  - ${exp}`));
    }

    // This is the actual test - core should export everything main does
    // (except intentional exclusions)
    expect(
      missingInStandalone,
      `Core is missing these exports: ${missingInStandalone.join(', ')}`
    ).toEqual([]);
  });

  it('should identify extra exports in core (not in main)', () => {
    const mainExports = new Set(Object.keys(main));
    const coreExports = new Set(Object.keys(core));

    const extraInStandalone: string[] = [];

    for (const exp of coreExports) {
      if (!mainExports.has(exp)) {
        extraInStandalone.push(exp);
      }
    }

    console.log('\n=== EXTRA IN CORE (not in main) ===');
    if (extraInStandalone.length === 0) {
      console.log('None.');
    } else {
      console.log(`Extra ${extraInStandalone.length} exports:`);
      extraInStandalone.sort().forEach((exp) => console.log(`  - ${exp}`));
    }

    // Extra exports in core are fine (they're aliases), just log them
  });

  it('should show coverage statistics', () => {
    const mainExports = new Set(Object.keys(main));
    const coreExports = new Set(Object.keys(core));

    // Remove intentional exclusions and type-only from main count
    let comparableMainCount = 0;
    for (const exp of mainExports) {
      if (!INTENTIONAL_EXCLUSIONS.has(exp) && !TYPE_ONLY_EXPORTS.has(exp)) {
        comparableMainCount++;
      }
    }

    // Count how many of those are in core
    let coveredCount = 0;
    for (const exp of mainExports) {
      if (INTENTIONAL_EXCLUSIONS.has(exp)) continue;
      if (TYPE_ONLY_EXPORTS.has(exp)) continue;
      if (coreExports.has(exp)) {
        coveredCount++;
      }
    }

    const coverage = ((coveredCount / comparableMainCount) * 100).toFixed(1);

    console.log('\n=== COVERAGE STATISTICS ===');
    console.log(`Main exports (total): ${mainExports.size}`);
    console.log(`Main exports (comparable): ${comparableMainCount}`);
    console.log(`Core exports: ${coreExports.size}`);
    console.log(`Coverage: ${coveredCount}/${comparableMainCount} (${coverage}%)`);

    // Require 100% coverage
    expect(Number(coverage)).toBe(100);
  });
});
