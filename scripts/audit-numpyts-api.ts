#!/usr/bin/env ts-node
/**
 * Audit numpy-ts API to compare against NumPy.
 *
 * This script:
 * 1. Extracts all exported functions from numpy-ts
 * 2. Extracts all methods from NDArray class
 * 3. Exports to JSON for comparison
 */

import * as fs from 'fs';
import * as path from 'path';
import { fileURLToPath } from 'url';
import { dirname } from 'path';

// Import everything from numpy-ts (main entry point)
import * as np from '../src/index.js';

// Import from node entry point to get I/O functions (load, save, savez, savez_compressed)
import * as npNode from '../src/node.js';

// ES module equivalent of __dirname
const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

function getNumpyTsTopLevelFunctions(): Record<string, any> {
  const functions: Record<string, any> = {};

  // Scan main entry point
  for (const name of Object.keys(np)) {
    const obj = (np as any)[name];

    if (typeof obj === 'function') {
      functions[name] = {
        type: 'function',
        module: 'numpy-ts',
      };
    } else if (typeof obj === 'object' && obj !== null) {
      // Check if it's a namespace (like linalg, random)
      const subFuncs = Object.keys(obj).filter((k) => typeof obj[k] === 'function');
      if (subFuncs.length > 0) {
        functions[name] = {
          type: 'namespace',
          functions: subFuncs,
          module: `numpy-ts.${name}`,
        };
      }
    }
  }

  // Scan node entry point for I/O functions that are only exported from /node
  // These include binary I/O (load, save, savez, savez_compressed)
  // and text I/O (loadtxt, savetxt, genfromtxt, fromregex)
  const nodeOnlyFunctions = [
    'load',
    'save',
    'savez',
    'savez_compressed',
    'loadtxt',
    'savetxt',
    'genfromtxt',
    'fromregex',
  ];
  for (const name of nodeOnlyFunctions) {
    if (name in npNode && typeof (npNode as any)[name] === 'function') {
      functions[name] = {
        type: 'function',
        module: 'numpy-ts/node',
      };
    }
  }

  return functions;
}

function getNDArrayMethods(): Record<string, any> {
  const methods: Record<string, any> = {};

  // Get NDArray class
  const NDArray = (np as any).NDArray;

  if (!NDArray) {
    console.error('Could not find NDArray class');
    return methods;
  }

  // Get prototype methods
  const prototype = NDArray.prototype;
  const propertyNames = Object.getOwnPropertyNames(prototype);

  for (const name of propertyNames) {
    // Skip constructor and private methods
    if (name === 'constructor' || name.startsWith('_')) {
      continue;
    }

    const descriptor = Object.getOwnPropertyDescriptor(prototype, name);

    if (descriptor) {
      if (typeof descriptor.value === 'function') {
        methods[name] = {
          type: 'method',
          class: 'NDArray',
        };
      } else if (descriptor.get || descriptor.set) {
        methods[name] = {
          type: 'property',
          class: 'NDArray',
          getter: !!descriptor.get,
          setter: !!descriptor.set,
        };
      }
    }
  }

  return methods;
}

function flattenNamespaces(functions: Record<string, any>): Record<string, any> {
  const flattened: Record<string, any> = {};

  for (const [name, obj] of Object.entries(functions)) {
    if (obj.type === 'namespace' && obj.functions) {
      // Add namespace functions as "namespace.function"
      for (const func of obj.functions) {
        flattened[`${name}.${func}`] = {
          type: 'function',
          module: obj.module,
        };
      }
    } else if (obj.type === 'function') {
      flattened[name] = obj;
    }
  }

  return flattened;
}

function categorizeByType(functions: Record<string, any>): Record<string, string[]> {
  const categories: Record<string, string[]> = {
    'Top-level Functions': [],
    'Linear Algebra (linalg)': [],
    'Random (random)': [],
    'FFT (fft)': [],
  };

  for (const [name, obj] of Object.entries(functions)) {
    if (name.startsWith('linalg.')) {
      categories['Linear Algebra (linalg)'].push(name);
    } else if (name.startsWith('random.')) {
      categories['Random (random)'].push(name);
    } else if (name.startsWith('fft.')) {
      categories['FFT (fft)'].push(name);
    } else {
      categories['Top-level Functions'].push(name);
    }
  }

  return categories;
}

function main() {
  console.log('Auditing numpy-ts API...\n');

  // Get top-level functions
  console.log('1. Extracting numpy-ts top-level exports...');
  const topLevel = getNumpyTsTopLevelFunctions();
  const topLevelCount = Object.keys(topLevel).length;
  console.log(`   Found ${topLevelCount} exports`);

  // Flatten namespaces
  console.log('2. Flattening namespaces...');
  const flattened = flattenNamespaces(topLevel);
  const flattenedCount = Object.keys(flattened).length;
  console.log(`   Total functions: ${flattenedCount}`);

  // Get NDArray methods
  console.log('3. Extracting NDArray methods...');
  const methods = getNDArrayMethods();
  const methodsCount = Object.keys(methods).length;
  console.log(`   Found ${methodsCount} methods/properties`);

  // Categorize
  console.log('4. Categorizing...');
  const categorized = categorizeByType(flattened);

  // Build output
  const output = {
    top_level_exports: topLevel,
    all_functions: flattened,
    ndarray_methods: methods,
    categorized,
    stats: {
      total_exports: topLevelCount,
      total_functions: flattenedCount,
      total_ndarray_methods: methodsCount,
      by_category: Object.fromEntries(Object.entries(categorized).map(([k, v]) => [k, v.length])),
    },
  };

  // Write to JSON
  const outputPath = path.join(__dirname, 'numpyts-api-audit.json');
  fs.writeFileSync(outputPath, JSON.stringify(output, null, 2));
  console.log(`\n5. Exported to ${outputPath}`);

  // Print summary
  console.log('\n' + '='.repeat(60));
  console.log('SUMMARY');
  console.log('='.repeat(60));
  console.log(`Total exports:           ${topLevelCount}`);
  console.log(`Total functions:         ${flattenedCount}`);
  console.log(`NDArray methods:         ${methodsCount}`);
  console.log();

  for (const [category, funcs] of Object.entries(categorized)) {
    console.log(`${category.padEnd(30)}: ${funcs.length} functions`);
  }

  console.log('\n' + '='.repeat(60));

  // List all functions
  console.log('\nAll implemented functions:');
  const allFuncs = Object.keys(flattened).sort();
  for (let i = 0; i < Math.min(20, allFuncs.length); i++) {
    console.log(`  - ${allFuncs[i]}`);
  }
  if (allFuncs.length > 20) {
    console.log(`  ... and ${allFuncs.length - 20} more`);
  }

  console.log('\nDone!');
}

main();
