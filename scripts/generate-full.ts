/**
 * Code Generator for full/ module
 *
 * This script generates full/index.ts from core/ module functions.
 * It transforms NDArrayCore return types to NDArray and wraps functions
 * to upgrade results.
 *
 * Run with: npx ts-node scripts/generate-full.ts
 */

import { Project, FunctionDeclaration, SourceFile, SyntaxKind } from 'ts-morph';
import * as fs from 'fs';
import * as path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const CORE_DIR = path.join(__dirname, '../src/core');
const FULL_DIR = path.join(__dirname, '../src/full');
const OUTPUT_FILE = path.join(FULL_DIR, 'index.ts');

// Files to skip (not function modules)
const SKIP_FILES = new Set(['index.ts', 'types.ts']);

// Functions that should NOT be wrapped (they don't return NDArrayCore)
const NO_WRAP_FUNCTIONS = new Set([
  // Functions returning primitives
  'ndim', 'size', 'item', 'tolist', 'tobytes', 'tofile', 'fill', 'byteswap', 'view',
  'can_cast', 'common_type', 'result_type', 'min_scalar_type', 'issubdtype', 'typename', 'mintypecode',
  'iscomplexobj', 'isrealobj', 'isfortran', 'isscalar', 'iterable', 'isdtype', 'promote_types',
  'count_nonzero',
  // Functions returning other types
  'set_printoptions', 'get_printoptions', 'printoptions',
  'format_float_positional', 'format_float_scientific', 'base_repr', 'binary_repr',
  'array2string', 'array_repr', 'array_str',
  'geterr', 'seterr',
  'linalg', // namespace object
  'einsum_path', // returns tuple
  'may_share_memory', 'shares_memory', // return boolean
  'array_equal', 'array_equiv', // return boolean
  'allclose', // returns boolean
  // Functions returning complex objects with NDArrayCore fields
  'unique', 'unique_all', 'unique_counts', 'unique_inverse',
  'histogram', 'histogram2d', 'histogramdd', 'histogram_bin_edges',
  // Functions returning union of single and array (NDArrayCore | NDArrayCore[])
  'gradient', 'atleast_1d', 'atleast_2d', 'atleast_3d', 'searchsorted',
  // Functions returning NDArrayCore[] with overloads
  'split', 'dsplit', 'hsplit', 'vsplit', 'array_split',
  // Functions with optional array parameters returning NDArrayCore[] | undefined
  'broadcast_arrays', 'meshgrid', 'einsum',
  // Functions with rest parameters or union returns needing special handling
  'ix_', 'where',
  // meshgrid is re-exported from ./ndarray (needs NDArray-aware implementation)
  'meshgrid',
]);

// Map function names when calling core (for reserved keywords)
const CORE_NAME_MAP: Record<string, string> = {
  'delete_': 'delete',
};

// Custom wrapper overrides for functions that need hand-written logic.
// Key = function name, value = { returnType, body }.
// The body can reference: core, up, NDArray, NDArrayCore, Complex,
// ArrayStorage, promoteDTypes, getTypedArrayConstructor, isComplexDType, DType.
const CUSTOM_WRAPPERS: Record<string, { returnType: string; body: string }> = {
  // cross returns NDArrayCore | number | Complex from core, but full/ should
  // always return NDArray by wrapping scalar results into 0-d arrays (NumPy behavior)
  cross: {
    returnType: 'NDArray',
    body: `const r = core.cross(a, b, axisa, axisb, axisc, axis);
  const dtype = promoteDTypes(a.dtype as DType, b.dtype as DType);
  if (r instanceof Complex) {
    const baseDtype = dtype === 'complex64' ? 'float32' : 'float64';
    const Ctor = getTypedArrayConstructor(baseDtype)!;
    const data = new Ctor(2);
    data[0] = r.re;
    data[1] = r.im;
    return NDArray.fromStorage(ArrayStorage.fromData(data, [], dtype));
  }
  if (typeof r === 'number') {
    if (isComplexDType(dtype)) {
      const baseDtype = dtype === 'complex64' ? 'float32' : 'float64';
      const Ctor = getTypedArrayConstructor(baseDtype)!;
      const data = new Ctor(2);
      data[0] = r;
      data[1] = 0;
      return NDArray.fromStorage(ArrayStorage.fromData(data, [], dtype));
    }
    const Ctor = getTypedArrayConstructor(dtype)!;
    const data = new Ctor(1);
    data[0] = r as never;
    return NDArray.fromStorage(ArrayStorage.fromData(data, [], dtype));
  }
  return up(r);`,
  },
  // apply_over_axes: callback receives NDArrayCore from core, but user expects NDArray
  apply_over_axes: {
    returnType: 'NDArray',
    body: `const wrappedFunc = (arr: NDArrayCore, axis: number): NDArrayCore => {
    return func(up(arr), axis);
  };
  return up(core.apply_over_axes(wrappedFunc, a, axes));`,
  },
  // apply_along_axis: callback receives NDArrayCore from core, but user expects NDArray
  apply_along_axis: {
    returnType: 'NDArray',
    body: `const wrappedFunc1d = (arr: NDArrayCore): NDArrayCore | number => {
    return func1d(up(arr));
  };
  return up(core.apply_along_axis(wrappedFunc1d, axis, arr));`,
  },
};

// Functions returning arrays (should be wrapped)
const RETURNS_ARRAY_FUNCTIONS = new Set([
  // Most core functions return arrays
]);

interface FunctionInfo {
  name: string;
  jsDoc: string;
  params: string;
  paramNames: string[];  // Store parameter names separately for reliable function calls
  returnType: string;
  isExported: boolean;
}

function extractFunctionInfo(func: FunctionDeclaration): FunctionInfo | null {
  const name = func.getName();
  if (!name) return null;

  const jsDocNodes = func.getJsDocs();
  const jsDoc = jsDocNodes.map(doc => doc.getFullText()).join('\n');

  // Extract parameter names directly using ts-morph (reliable, handles complex types)
  const paramNames: string[] = [];

  const params = func.getParameters().map(p => {
    const pName = p.getName();
    paramNames.push(pName);  // Store the parameter name

    const typeNode = p.getTypeNode();
    const type = typeNode ? typeNode.getText() : 'any';
    const isOptional = p.isOptional() || p.hasInitializer();
    const initializer = p.getInitializer()?.getText();

    if (initializer) {
      return `${pName}: ${type} = ${initializer}`;
    } else if (isOptional) {
      return `${pName}?: ${type}`;
    } else {
      return `${pName}: ${type}`;
    }
  }).join(', ');

  const returnTypeNode = func.getReturnTypeNode();
  const returnType = returnTypeNode ? returnTypeNode.getText() : 'any';

  return {
    name,
    jsDoc,
    params,
    paramNames,
    returnType,
    isExported: func.isExported(),
  };
}

function shouldWrapFunction(info: FunctionInfo): boolean {
  if (NO_WRAP_FUNCTIONS.has(info.name)) {
    return false;
  }

  // Check if return type includes NDArrayCore
  if (info.returnType.includes('NDArrayCore')) {
    return true;
  }

  // Check if it returns an array of NDArrayCore
  if (info.returnType.includes('NDArrayCore[]')) {
    return true;
  }

  return false;
}

function transformReturnType(returnType: string): string {
  return returnType
    .replace(/\[NDArrayCore,\s*NDArrayCore\]/g, '[NDArray, NDArray]')
    .replace(/NDArrayCore\[\]/g, 'NDArray[]')
    .replace(/NDArrayCore/g, 'NDArray');
}

function generateWrapperFunction(info: FunctionInfo, moduleName: string): string {
  // Check for custom wrapper override
  if (CUSTOM_WRAPPERS[info.name]) {
    const custom = CUSTOM_WRAPPERS[info.name];
    return `${info.jsDoc}
export function ${info.name}(${info.params}): ${custom.returnType} {
  ${custom.body}
}`;
  }

  const transformedReturnType = transformReturnType(info.returnType);

  // Use pre-extracted parameter names (reliable, handles complex types like [number[], number[]])
  const paramNamesStr = info.paramNames.join(', ');
  const rt = info.returnType;

  // Map function name for core call (e.g., delete_ -> delete for reserved keywords)
  const coreFuncName = CORE_NAME_MAP[info.name] || info.name;

  let body: string;

  // Detect return type patterns
  const isTupleOfArrays = /^\[NDArrayCore,\s*NDArrayCore\]$/.test(rt.replace(/\s/g, ''));
  const isArrayOfArrays = rt.includes('NDArrayCore[]') && !rt.includes('|');
  const isOptionalArray = /NDArrayCore\[\]\s*\|\s*undefined/.test(rt) || /undefined\s*\|\s*NDArrayCore\[\]/.test(rt);
  const isUnionWithPrimitive = rt.includes('NDArrayCore') && (rt.includes('number') || rt.includes('bigint') || rt.includes('Complex') || rt.includes('boolean'));
  const isOptionalSingle = /NDArrayCore\s*\|\s*undefined/.test(rt) || /undefined\s*\|\s*NDArrayCore/.test(rt);
  const isPureSingle = rt === 'NDArrayCore';

  if (isTupleOfArrays) {
    // [NDArrayCore, NDArrayCore] → [up(r[0]), up(r[1])] as [NDArray, NDArray]
    body = `const r = core.${coreFuncName}(${paramNamesStr}); return [up(r[0]), up(r[1])] as [NDArray, NDArray];`;
  } else if (isOptionalArray) {
    // NDArrayCore[] | undefined → result?.map(up)
    body = `const r = core.${coreFuncName}(${paramNamesStr}); return r?.map(up);`;
  } else if (isArrayOfArrays) {
    // NDArrayCore[] → result.map(up)
    body = `return core.${coreFuncName}(${paramNamesStr}).map(up);`;
  } else if (isUnionWithPrimitive) {
    // number | NDArrayCore | ... → conditional wrap
    body = `const r = core.${coreFuncName}(${paramNamesStr}); return r instanceof NDArrayCore ? up(r) : r;`;
  } else if (isOptionalSingle) {
    // NDArrayCore | undefined → result ? up(result) : undefined
    body = `const r = core.${coreFuncName}(${paramNamesStr}); return r ? up(r) : undefined;`;
  } else if (isPureSingle) {
    // Pure NDArrayCore → up(result)
    body = `return up(core.${coreFuncName}(${paramNamesStr}));`;
  } else {
    // Default: try simple wrap (will fail for complex cases)
    body = `return up(core.${coreFuncName}(${paramNamesStr}));`;
  }

  return `${info.jsDoc}
export function ${info.name}(${info.params}): ${transformedReturnType} {
  ${body}
}`;
}

function generateReexport(info: FunctionInfo): string {
  return `export { ${info.name} } from '../core/${getModuleName(info.name)}';`;
}

function getModuleName(funcName: string): string {
  // This would need to be tracked properly, for now we'll use 'index'
  return 'index';
}

async function main() {
  console.log('Generating full/index.ts from core/ modules...\n');

  const project = new Project({
    tsConfigFilePath: path.join(__dirname, '../tsconfig.json'),
  });

  // Collect all function info from core modules
  const allFunctions: Map<string, FunctionInfo> = new Map();
  const wrappedFunctions: Set<string> = new Set();
  const reexportedFunctions: Set<string> = new Set();

  const coreFiles = fs.readdirSync(CORE_DIR)
    .filter(f => f.endsWith('.ts') && !SKIP_FILES.has(f));

  console.log(`Processing ${coreFiles.length} core modules...`);

  // Track const exports separately
  const constExports: Set<string> = new Set();

  for (const file of coreFiles) {
    const filePath = path.join(CORE_DIR, file);
    const sourceFile = project.addSourceFileAtPath(filePath);

    const functions = sourceFile.getFunctions()
      .filter(f => f.isExported())
      .map(extractFunctionInfo)
      .filter((f): f is FunctionInfo => f !== null);

    for (const func of functions) {
      if (shouldWrapFunction(func)) {
        wrappedFunctions.add(func.name);
      } else {
        reexportedFunctions.add(func.name);
      }
      allFunctions.set(func.name, func);
    }

    // Also collect exported variable declarations (const exports, aliases)
    const varStatements = sourceFile.getVariableStatements()
      .filter(v => v.isExported());

    for (const varStatement of varStatements) {
      for (const decl of varStatement.getDeclarations()) {
        const name = decl.getName();
        constExports.add(name);
        reexportedFunctions.add(name);
      }
    }

    console.log(`  ${file}: ${functions.length} functions (${functions.filter(f => shouldWrapFunction(f)).length} wrapped), ${varStatements.length} const exports`);
  }

  // Generate the output file
  const output: string[] = [];

  output.push(`// AUTO-GENERATED - DO NOT EDIT
// Run \`npm run generate\` to regenerate this file from core/ modules

/**
 * Full Module - NDArray with method chaining
 *
 * This module wraps core/ functions to return NDArray (with methods)
 * instead of NDArrayCore (minimal). Use this for method chaining.
 *
 * @example
 * \`\`\`typescript
 * import { array } from 'numpy-ts/full';
 *
 * const result = array([1, 2, 3]).add(10).reshape([3, 1]);
 * \`\`\`
 */

import * as core from '../core';
import { NDArray } from './ndarray';
import { NDArrayCore } from '../common/ndarray-core';
import { ArrayStorage } from '../common/storage';
import { Complex } from '../common/complex';
import type { DType, TypedArray } from '../core/types';
import {
  DEFAULT_DTYPE,
  getTypedArrayConstructor,
  promoteDTypes,
  isComplexDType,
} from '../common/dtype';

// Helper to upgrade NDArrayCore to NDArray (zero-copy via shared storage)
// If already an NDArray, return as-is to preserve identity
// Also preserves base reference for views
const up = (x: NDArrayCore): NDArray => {
  if (x instanceof NDArray) return x;
  const base = x.base ? up(x.base) : undefined;
  return NDArray.fromStorage(x.storage, base);
};

// Re-export types
export type { DType, TypedArray } from '../core/types';
export { NDArray, meshgrid } from './ndarray';
export { NDArrayCore } from '../common/ndarray-core';
export { Complex } from '../common/complex';
`);

  // Generate wrapped functions
  output.push('\n// ============================================================');
  output.push('// Wrapped Functions (return NDArray)');
  output.push('// ============================================================\n');

  for (const [name, info] of allFunctions) {
    if (wrappedFunctions.has(name)) {
      output.push(generateWrapperFunction(info, ''));
      output.push('');
    }
  }

  // Re-export non-wrapped functions
  output.push('\n// ============================================================');
  output.push('// Re-exported Functions (no wrapping needed)');
  output.push('// ============================================================\n');

  // Names already exported in the header (from ./ndarray or elsewhere)
  const HEADER_EXPORTS = new Set(['NDArray', 'NDArrayCore', 'Complex', 'meshgrid']);

  const reexports: string[] = [];
  for (const name of reexportedFunctions) {
    if (!HEADER_EXPORTS.has(name)) {
      reexports.push(name);
    }
  }

  if (reexports.length > 0) {
    output.push(`export {`);
    output.push(`  ${reexports.join(',\n  ')},`);
    output.push(`} from '../core';`);
  }

  // Write output
  const outputContent = output.join('\n');
  fs.writeFileSync(OUTPUT_FILE, outputContent);

  console.log(`\nGenerated ${OUTPUT_FILE}`);
  console.log(`  - ${wrappedFunctions.size} wrapped functions`);
  console.log(`  - ${reexportedFunctions.size} re-exported functions`);
}

main().catch(console.error);
