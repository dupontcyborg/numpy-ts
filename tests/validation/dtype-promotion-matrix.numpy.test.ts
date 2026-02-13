/**
 * NumPy validation tests for complete dtype promotion matrix
 * Validates all 11×11 dtype combinations against actual NumPy behavior
 */

import { describe, it, expect, beforeAll } from 'vitest';
import { promoteDTypes } from '../../src/common/dtype';
import { ones } from '../../src';
import type { DType } from '../../src/common/dtype';
import { runNumPy, checkNumPyAvailable } from './numpy-oracle';

describe('NumPy Validation: Complete DType Promotion Matrix', () => {
  beforeAll(() => {
    if (!checkNumPyAvailable()) {
      throw new Error('Python NumPy not available');
    }
  });

  // All supported dtypes (11 types, no complex)
  const allDTypes: DType[] = [
    'bool',
    'int8',
    'int16',
    'int32',
    'int64',
    'uint8',
    'uint16',
    'uint32',
    'uint64',
    'float32',
    'float64',
  ];

  // Map our dtypes to NumPy dtypes
  const dtypeMap: Record<DType, string> = {
    bool: 'bool_',
    int8: 'int8',
    int16: 'int16',
    int32: 'int32',
    int64: 'int64',
    uint8: 'uint8',
    uint16: 'uint16',
    uint32: 'uint32',
    uint64: 'uint64',
    float32: 'float32',
    float64: 'float64',
  };

  // Map NumPy dtype strings back to our DType
  const reverseMap: Record<string, DType> = {
    bool: 'bool',
    int8: 'int8',
    int16: 'int16',
    int32: 'int32',
    int64: 'int64',
    uint8: 'uint8',
    uint16: 'uint16',
    uint32: 'uint32',
    uint64: 'uint64',
    float32: 'float32',
    float64: 'float64',
  };

  describe('Complete promotion matrix - all 121 combinations', () => {
    // Batch compute ALL 121 combinations in a single Python call for speed
    let npyPromotionMatrix: Record<string, Record<string, string>>;

    beforeAll(() => {
      // Generate Python code to compute all combinations at once
      const pythonCode = `
dtypes = {
  'bool': np.bool_,
  'int8': np.int8,
  'int16': np.int16,
  'int32': np.int32,
  'int64': np.int64,
  'uint8': np.uint8,
  'uint16': np.uint16,
  'uint32': np.uint32,
  'uint64': np.uint64,
  'float32': np.float32,
  'float64': np.float64,
}

dtype_names = list(dtypes.keys())
matrix = {}

for dtype1_name in dtype_names:
  matrix[dtype1_name] = {}
  for dtype2_name in dtype_names:
    a = np.array([1], dtype=dtypes[dtype1_name])
    b = np.array([1], dtype=dtypes[dtype2_name])
    r = a + b
    matrix[dtype1_name][dtype2_name] = str(r.dtype)

# Assign to result for runNumPy to capture
result = matrix
      `;

      const pyResult = runNumPy(pythonCode);
      // The matrix is returned in pyResult.value
      npyPromotionMatrix = pyResult.value as Record<string, Record<string, string>>;
    });

    for (const dtype1 of allDTypes) {
      for (const dtype2 of allDTypes) {
        it(`validates ${dtype1} + ${dtype2} matches NumPy`, () => {
          // Get our promotion result
          const jsResult = promoteDTypes(dtype1, dtype2);

          // Get NumPy's result from pre-computed matrix
          const npyDtype = reverseMap[npyPromotionMatrix[dtype1][dtype2]];

          // Our promotion should match NumPy's
          expect(jsResult).toBe(npyDtype);
        });
      }
    }
  });

  describe('Array arithmetic promotion matches NumPy', () => {
    // Test a representative subset to avoid excessive NumPy calls
    const testPairs: Array<[DType, DType]> = [
      // Bool with everything
      ['bool', 'bool'],
      ['bool', 'int32'],
      ['bool', 'float64'],

      // Same signedness, different sizes
      ['int8', 'int16'],
      ['int8', 'int32'],
      ['int16', 'int32'],
      ['int32', 'int64'],
      ['uint8', 'uint16'],
      ['uint8', 'uint32'],
      ['uint16', 'uint32'],
      ['uint32', 'uint64'],

      // Mixed signedness, same size
      ['int8', 'uint8'],
      ['int16', 'uint16'],
      ['int32', 'uint32'],
      ['int64', 'uint64'],

      // Mixed signedness, different sizes
      ['int8', 'uint16'],
      ['int8', 'uint32'],
      ['int16', 'uint32'],
      ['int32', 'uint64'],

      // Integer + float
      ['int8', 'float32'],
      ['int8', 'float64'],
      ['int32', 'float32'],
      ['int32', 'float64'],
      ['int64', 'float32'],
      ['int64', 'float64'],
      ['uint8', 'float32'],
      ['uint16', 'float32'],
      ['uint32', 'float64'],

      // Float types
      ['float32', 'float32'],
      ['float32', 'float64'],
      ['float64', 'float64'],
    ];

    testPairs.forEach(([dtype1, dtype2]) => {
      it(`${dtype1} array + ${dtype2} array matches NumPy`, () => {
        const a = ones([2], dtype1);
        const b = ones([2], dtype2);
        const jsResult = a.add(b);

        const pyResult = runNumPy(`
import numpy as np
a = np.ones(2, dtype=np.${dtypeMap[dtype1]})
b = np.ones(2, dtype=np.${dtypeMap[dtype2]})
result = a + b
        `);

        const npyDtype = reverseMap[pyResult.dtype];
        expect(jsResult.dtype).toBe(npyDtype);
      });
    });
  });

  describe('Commutative property verified against NumPy', () => {
    // Test commutativity for important type pairs
    const testPairs: Array<[DType, DType]> = [
      ['int32', 'float64'],
      ['int64', 'uint64'],
      ['float32', 'int8'],
      ['int8', 'uint16'],
      ['uint32', 'int16'],
    ];

    testPairs.forEach(([dtype1, dtype2]) => {
      it(`${dtype1} + ${dtype2} === ${dtype2} + ${dtype1} (matches NumPy)`, () => {
        // Our results
        const jsResult1 = promoteDTypes(dtype1, dtype2);
        const jsResult2 = promoteDTypes(dtype2, dtype1);

        // NumPy results
        const pyResult1 = runNumPy(`
import numpy as np
a = np.array([1], dtype=np.${dtypeMap[dtype1]})
b = np.array([1], dtype=np.${dtypeMap[dtype2]})
result = a + b
        `);

        const pyResult2 = runNumPy(`
import numpy as np
a = np.array([1], dtype=np.${dtypeMap[dtype2]})
b = np.array([1], dtype=np.${dtypeMap[dtype1]})
result = a + b
        `);

        // Check commutativity
        expect(jsResult1).toBe(jsResult2);
        expect(pyResult1.dtype).toBe(pyResult2.dtype);

        // Check we match NumPy
        const npyDtype = reverseMap[pyResult1.dtype];
        expect(jsResult1).toBe(npyDtype);
      });
    });
  });

  describe('Specific promotion rules match NumPy', () => {
    it('bool promotes to other type', () => {
      const testTypes: DType[] = ['int8', 'uint16', 'float32', 'int64'];

      for (const dtype of testTypes) {
        const jsResult = promoteDTypes('bool', dtype);

        const pyResult = runNumPy(`
import numpy as np
a = np.array([True], dtype=np.bool_)
b = np.array([1], dtype=np.${dtypeMap[dtype]})
result = a + b
        `);

        const npyDtype = reverseMap[pyResult.dtype];
        expect(jsResult).toBe(npyDtype);
        expect(jsResult).toBe(dtype); // bool should promote to the other type
      }
    });

    it('signed + unsigned of same size matches NumPy promotion', () => {
      const pairs: Array<[DType, DType, DType]> = [
        ['int8', 'uint8', 'int16'],
        ['int16', 'uint16', 'int32'],
        ['int32', 'uint32', 'int64'],
      ];

      for (const [signed, unsigned, expected] of pairs) {
        const jsResult = promoteDTypes(signed, unsigned);

        const pyResult = runNumPy(`
import numpy as np
a = np.array([1], dtype=np.${dtypeMap[signed]})
b = np.array([1], dtype=np.${dtypeMap[unsigned]})
result = a + b
        `);

        const npyDtype = reverseMap[pyResult.dtype];
        expect(jsResult).toBe(expected);
        expect(jsResult).toBe(npyDtype);
      }
    });

    it('int64 + uint64 promotes to float64 (matches NumPy)', () => {
      const jsResult = promoteDTypes('int64', 'uint64');

      const pyResult = runNumPy(`
import numpy as np
a = np.array([1], dtype=np.int64)
b = np.array([1], dtype=np.uint64)
result = a + b
      `);

      const npyDtype = reverseMap[pyResult.dtype];
      expect(jsResult).toBe('float64');
      expect(jsResult).toBe(npyDtype);
    });

    it('small integer + float32 promotes to float32, large integer + float32 promotes to float64 (matches NumPy)', () => {
      // Batch all dtype checks into a single Python call for performance
      const pyResult = runNumPy(`
import numpy as np
results = {}
for dtype_name in ['int8', 'int16', 'uint8', 'uint16', 'int32', 'int64', 'uint32', 'uint64']:
    dtype = getattr(np, dtype_name)
    a = np.array([1], dtype=dtype)
    b = np.array([1.0], dtype=np.float32)
    results[dtype_name] = str((a + b).dtype)
result = results
      `);

      const pyResults = pyResult.value as Record<string, string>;

      // Small integers (8, 16 bit) + float32 → float32
      const smallIntTypes: DType[] = ['int8', 'int16', 'uint8', 'uint16'];
      for (const intType of smallIntTypes) {
        const jsResult = promoteDTypes(intType, 'float32');
        const npyDtype = reverseMap[pyResults[intType]];
        expect(jsResult).toBe('float32');
        expect(jsResult).toBe(npyDtype);
      }

      // Large integers (32, 64 bit) + float32 → float64 (precision safety)
      const largeIntTypes: DType[] = ['int32', 'int64', 'uint32', 'uint64'];
      for (const intType of largeIntTypes) {
        const jsResult = promoteDTypes(intType, 'float32');
        const npyDtype = reverseMap[pyResults[intType]];
        expect(jsResult).toBe('float64');
        expect(jsResult).toBe(npyDtype);
      }
    });

    it('float32 + float32 stays float32 (matches NumPy)', () => {
      const jsResult = promoteDTypes('float32', 'float32');

      const pyResult = runNumPy(`
import numpy as np
a = np.array([1.0], dtype=np.float32)
b = np.array([1.0], dtype=np.float32)
result = a + b
      `);

      const npyDtype = reverseMap[pyResult.dtype];
      expect(jsResult).toBe('float32');
      expect(jsResult).toBe(npyDtype);
    });
  });

  describe('BigInt dtypes match NumPy behavior', () => {
    it('int64 operations match NumPy', () => {
      const testWith: DType[] = ['int64', 'int32', 'uint32', 'float64'];

      for (const dtype of testWith) {
        const jsResult = promoteDTypes('int64', dtype);

        const pyResult = runNumPy(`
import numpy as np
a = np.array([1], dtype=np.int64)
b = np.array([1], dtype=np.${dtypeMap[dtype]})
result = a + b
        `);

        const npyDtype = reverseMap[pyResult.dtype];
        expect(jsResult).toBe(npyDtype);
      }
    });

    it('uint64 operations match NumPy', () => {
      const testWith: DType[] = ['uint64', 'uint32', 'int64', 'float64'];

      for (const dtype of testWith) {
        const jsResult = promoteDTypes('uint64', dtype);

        const pyResult = runNumPy(`
import numpy as np
a = np.array([1], dtype=np.uint64)
b = np.array([1], dtype=np.${dtypeMap[dtype]})
result = a + b
        `);

        const npyDtype = reverseMap[pyResult.dtype];
        expect(jsResult).toBe(npyDtype);
      }
    });
  });

  describe('Edge cases match NumPy', () => {
    it('extreme type combinations match NumPy', () => {
      const extremePairs: Array<[DType, DType]> = [
        ['bool', 'float64'],
        ['int8', 'uint64'],
        ['uint8', 'int64'],
        ['float32', 'uint64'],
      ];

      for (const [dtype1, dtype2] of extremePairs) {
        const jsResult = promoteDTypes(dtype1, dtype2);

        const pyResult = runNumPy(`
import numpy as np
a = np.array([1], dtype=np.${dtypeMap[dtype1]})
b = np.array([1], dtype=np.${dtypeMap[dtype2]})
result = a + b
        `);

        const npyDtype = reverseMap[pyResult.dtype];
        expect(jsResult).toBe(npyDtype);
      }
    });
  });
});
