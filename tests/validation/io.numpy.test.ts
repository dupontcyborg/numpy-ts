/**
 * NPY/NPZ Cross-Language Validation Tests
 *
 * Tests that:
 * 1. NPY/NPZ files created by Python can be read correctly by numpy-ts
 * 2. NPY/NPZ files created by numpy-ts can be read correctly by Python
 */

import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import { execSync } from 'child_process';
import { writeFileSync, readFileSync, unlinkSync, mkdtempSync, rmSync } from 'fs';
import { tmpdir } from 'os';
import { join } from 'path';

import { parseNpy, serializeNpy } from '../../src/io/npy';
import { parseNpz, serializeNpz, serializeNpzSync } from '../../src/io/npz';
import { array, arange } from '../../src/core/ndarray';
import type { DType } from '../../src/core/dtype';

// Get Python command from environment or use default
const PYTHON_CMD = process.env.NUMPY_PYTHON || 'python3';

// Check if NumPy is available
let hasNumPy = false;
try {
  execSync(`${PYTHON_CMD} -c "import numpy"`, { stdio: 'pipe' });
  hasNumPy = true;
} catch {
  hasNumPy = false;
}

const describeIfNumPy = hasNumPy ? describe : describe.skip;

// Temp directory for test files
let tempDir: string;

describeIfNumPy('NPY/NPZ Cross-Language Validation', () => {
  beforeAll(() => {
    tempDir = mkdtempSync(join(tmpdir(), 'numpy-ts-validation-'));
  });

  afterAll(() => {
    try {
      rmSync(tempDir, { recursive: true, force: true });
    } catch {
      // Ignore cleanup errors
    }
  });

  describe('Python → TypeScript (read files created by NumPy)', () => {
    const dtypeTests: Array<{
      dtype: string;
      npDtype: string;
      values: string;
      expected: number[] | bigint[];
    }> = [
      {
        dtype: 'float64',
        npDtype: 'float64',
        values: '[1.5, 2.5, 3.5]',
        expected: [1.5, 2.5, 3.5],
      },
      {
        dtype: 'float32',
        npDtype: 'float32',
        values: '[1.5, 2.5, 3.5]',
        expected: [1.5, 2.5, 3.5],
      },
      {
        dtype: 'int64',
        npDtype: 'int64',
        values: '[1, 2, 3]',
        expected: [BigInt(1), BigInt(2), BigInt(3)],
      },
      { dtype: 'int32', npDtype: 'int32', values: '[1, 2, 3]', expected: [1, 2, 3] },
      { dtype: 'int16', npDtype: 'int16', values: '[1, 2, 3]', expected: [1, 2, 3] },
      { dtype: 'int8', npDtype: 'int8', values: '[1, 2, 3]', expected: [1, 2, 3] },
      {
        dtype: 'uint64',
        npDtype: 'uint64',
        values: '[1, 2, 3]',
        expected: [BigInt(1), BigInt(2), BigInt(3)],
      },
      { dtype: 'uint32', npDtype: 'uint32', values: '[1, 2, 3]', expected: [1, 2, 3] },
      { dtype: 'uint16', npDtype: 'uint16', values: '[1, 2, 3]', expected: [1, 2, 3] },
      { dtype: 'uint8', npDtype: 'uint8', values: '[1, 2, 3]', expected: [1, 2, 3] },
      { dtype: 'bool', npDtype: 'bool', values: '[True, False, True]', expected: [1, 0, 1] },
    ];

    for (const { dtype, npDtype, values, expected } of dtypeTests) {
      it(`reads ${dtype} NPY file from NumPy`, () => {
        const npyPath = join(tempDir, `test_${dtype}.npy`);

        // Create NPY file with Python
        const pythonCode = `
import numpy as np
arr = np.array(${values}, dtype=np.${npDtype})
np.save('${npyPath}', arr)
`;
        execSync(`${PYTHON_CMD} -c "${pythonCode}"`, { stdio: 'pipe' });

        // Read with numpy-ts
        const npyBytes = readFileSync(npyPath);
        const arr = parseNpy(npyBytes);

        expect(arr.dtype).toBe(dtype);
        expect(arr.shape).toEqual([3]);
        expect(arr.toArray()).toEqual(expected);

        unlinkSync(npyPath);
      });
    }

    it('reads 2D array from NumPy', () => {
      const npyPath = join(tempDir, 'test_2d.npy');

      const pythonCode = `
import numpy as np
arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
np.save('${npyPath}', arr)
`;
      execSync(`${PYTHON_CMD} -c "${pythonCode}"`, { stdio: 'pipe' });

      const npyBytes = readFileSync(npyPath);
      const arr = parseNpy(npyBytes);

      expect(arr.shape).toEqual([2, 3]);
      expect(arr.toArray()).toEqual([
        [1, 2, 3],
        [4, 5, 6],
      ]);

      unlinkSync(npyPath);
    });

    it('reads 3D array from NumPy', () => {
      const npyPath = join(tempDir, 'test_3d.npy');

      const pythonCode = `
import numpy as np
arr = np.arange(24).reshape(2, 3, 4).astype(np.int32)
np.save('${npyPath}', arr)
`;
      execSync(`${PYTHON_CMD} -c "${pythonCode}"`, { stdio: 'pipe' });

      const npyBytes = readFileSync(npyPath);
      const arr = parseNpy(npyBytes);

      expect(arr.dtype).toBe('int32');
      expect(arr.shape).toEqual([2, 3, 4]);
      expect(arr.get([0, 0, 0])).toBe(0);
      expect(arr.get([1, 2, 3])).toBe(23);

      unlinkSync(npyPath);
    });

    it('reads NPZ file with multiple arrays from NumPy', async () => {
      const npzPath = join(tempDir, 'test_multi.npz');

      const pythonCode = `
import numpy as np
a = np.array([1, 2, 3], dtype=np.float64)
b = np.array([[4, 5], [6, 7]], dtype=np.int32)
c = np.array([True, False, True], dtype=np.bool_)
np.savez('${npzPath}', a=a, b=b, c=c)
`;
      execSync(`${PYTHON_CMD} -c "${pythonCode}"`, { stdio: 'pipe' });

      const npzBytes = readFileSync(npzPath);
      const result = await parseNpz(npzBytes);

      expect(result.arrays.size).toBe(3);
      expect(result.arrays.get('a')!.toArray()).toEqual([1, 2, 3]);
      expect(result.arrays.get('b')!.toArray()).toEqual([
        [4, 5],
        [6, 7],
      ]);
      expect(result.arrays.get('c')!.toArray()).toEqual([1, 0, 1]);

      unlinkSync(npzPath);
    });

    it('reads compressed NPZ file from NumPy', async () => {
      const npzPath = join(tempDir, 'test_compressed.npz');

      const pythonCode = `
import numpy as np
arr = np.arange(1000, dtype=np.float64)
np.savez_compressed('${npzPath}', data=arr)
`;
      execSync(`${PYTHON_CMD} -c "${pythonCode}"`, { stdio: 'pipe' });

      const npzBytes = readFileSync(npzPath);
      const result = await parseNpz(npzBytes);

      const loaded = result.arrays.get('data')!;
      expect(loaded.shape).toEqual([1000]);
      expect(loaded.get([0])).toBe(0);
      expect(loaded.get([999])).toBe(999);

      unlinkSync(npzPath);
    });

    it('reads NPZ with positional args from NumPy', async () => {
      const npzPath = join(tempDir, 'test_positional.npz');

      // Use int32 explicitly to avoid int64 -> BigInt conversion
      const pythonCode = `
import numpy as np
a = np.array([1, 2, 3], dtype=np.int32)
b = np.array([4, 5, 6], dtype=np.int32)
np.savez('${npzPath}', a, b)
`;
      execSync(`${PYTHON_CMD} -c "${pythonCode}"`, { stdio: 'pipe' });

      const npzBytes = readFileSync(npzPath);
      const result = await parseNpz(npzBytes);

      expect(result.arrays.get('arr_0')!.toArray()).toEqual([1, 2, 3]);
      expect(result.arrays.get('arr_1')!.toArray()).toEqual([4, 5, 6]);

      unlinkSync(npzPath);
    });
  });

  describe('TypeScript → Python (validate files created by numpy-ts)', () => {
    const dtypeTests: Array<{
      dtype: DType;
      values: number[] | bigint[];
    }> = [
      { dtype: 'float64', values: [1.5, 2.5, 3.5] },
      { dtype: 'float32', values: [1.5, 2.5, 3.5] },
      { dtype: 'int64', values: [BigInt(1), BigInt(2), BigInt(3)] },
      { dtype: 'int32', values: [1, 2, 3] },
      { dtype: 'int16', values: [1, 2, 3] },
      { dtype: 'int8', values: [1, 2, 3] },
      { dtype: 'uint64', values: [BigInt(1), BigInt(2), BigInt(3)] },
      { dtype: 'uint32', values: [1, 2, 3] },
      { dtype: 'uint16', values: [1, 2, 3] },
      { dtype: 'uint8', values: [1, 2, 3] },
      { dtype: 'bool', values: [1, 0, 1] },
    ];

    for (const { dtype, values } of dtypeTests) {
      it(`validates ${dtype} NPY file with NumPy`, () => {
        const npyPath = join(tempDir, `ts_${dtype}.npy`);

        // Create NPY file with numpy-ts
        const arr = array(values, dtype);
        const npyBytes = serializeNpy(arr);
        writeFileSync(npyPath, npyBytes);

        // Validate with Python
        const pythonCode = `
import numpy as np
import json
arr = np.load('${npyPath}')
expected_values = [1, 2, 3] if '${dtype}' != 'bool' else [True, False, True]
if '${dtype}' in ['float32', 'float64']:
    expected_values = [1.5, 2.5, 3.5]
result = {
    'dtype': str(arr.dtype),
    'shape': list(arr.shape),
    'values_match': np.allclose(arr, expected_values) if arr.dtype.kind == 'f' else (arr.tolist() == expected_values)
}
print(json.dumps(result))
`;

        const output = execSync(`${PYTHON_CMD} -c "${pythonCode}"`, { encoding: 'utf-8' });
        const result = JSON.parse(output);

        expect(result.shape).toEqual([3]);
        expect(result.values_match).toBe(true);

        unlinkSync(npyPath);
      });
    }

    it('validates 2D array with NumPy', () => {
      const npyPath = join(tempDir, 'ts_2d.npy');

      const arr = array(
        [
          [1, 2, 3],
          [4, 5, 6],
        ],
        'float64'
      );
      const npyBytes = serializeNpy(arr);
      writeFileSync(npyPath, npyBytes);

      const pythonCode = `
import numpy as np
import json
arr = np.load('${npyPath}')
expected = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
result = {
    'shape': list(arr.shape),
    'values_match': np.allclose(arr, expected)
}
print(json.dumps(result))
`;

      const output = execSync(`${PYTHON_CMD} -c "${pythonCode}"`, { encoding: 'utf-8' });
      const result = JSON.parse(output);

      expect(result.shape).toEqual([2, 3]);
      expect(result.values_match).toBe(true);

      unlinkSync(npyPath);
    });

    it('validates NPZ file with NumPy', async () => {
      const npzPath = join(tempDir, 'ts_multi.npz');

      const a = array([1, 2, 3], 'float64');
      const b = array(
        [
          [4, 5],
          [6, 7],
        ],
        'int32'
      );

      const npzBytes = await serializeNpz({ a, b });
      writeFileSync(npzPath, npzBytes);

      const pythonCode = `
import numpy as np
import json
data = np.load('${npzPath}')
result = {
    'keys': sorted(list(data.keys())),
    'a_shape': list(data['a'].shape),
    'b_shape': list(data['b'].shape),
    'a_values_match': np.allclose(data['a'], [1, 2, 3]),
    'b_values_match': np.array_equal(data['b'], [[4, 5], [6, 7]])
}
print(json.dumps(result))
`;

      const output = execSync(`${PYTHON_CMD} -c "${pythonCode}"`, { encoding: 'utf-8' });
      const result = JSON.parse(output);

      expect(result.keys).toEqual(['a', 'b']);
      expect(result.a_shape).toEqual([3]);
      expect(result.b_shape).toEqual([2, 2]);
      expect(result.a_values_match).toBe(true);
      expect(result.b_values_match).toBe(true);

      unlinkSync(npzPath);
    });

    it('validates positional arrays in NPZ with NumPy', () => {
      const npzPath = join(tempDir, 'ts_positional.npz');

      const arr1 = array([1, 2, 3]);
      const arr2 = array([4, 5, 6]);
      const arr3 = array([7, 8, 9]);

      const npzBytes = serializeNpzSync([arr1, arr2, arr3]);
      writeFileSync(npzPath, npzBytes);

      const pythonCode = `
import numpy as np
import json
data = np.load('${npzPath}')
result = {
    'keys': sorted(list(data.keys())),
    'arr_0_match': np.allclose(data['arr_0'], [1, 2, 3]),
    'arr_1_match': np.allclose(data['arr_1'], [4, 5, 6]),
    'arr_2_match': np.allclose(data['arr_2'], [7, 8, 9])
}
print(json.dumps(result))
`;

      const output = execSync(`${PYTHON_CMD} -c "${pythonCode}"`, { encoding: 'utf-8' });
      const result = JSON.parse(output);

      expect(result.keys).toEqual(['arr_0', 'arr_1', 'arr_2']);
      expect(result.arr_0_match).toBe(true);
      expect(result.arr_1_match).toBe(true);
      expect(result.arr_2_match).toBe(true);

      unlinkSync(npzPath);
    });
  });

  describe('Round-trip validation (TS → Python → TS)', () => {
    it('round-trips large array through Python', async () => {
      const npyPath = join(tempDir, 'roundtrip.npy');

      // Create with numpy-ts
      const original = arange(1000);
      const npyBytes = serializeNpy(original);
      writeFileSync(npyPath, npyBytes);

      // Modify with Python (multiply by 2)
      const pythonCode = `
import numpy as np
arr = np.load('${npyPath}')
arr = arr * 2
np.save('${npyPath}', arr)
`;
      execSync(`${PYTHON_CMD} -c "${pythonCode}"`, { stdio: 'pipe' });

      // Read back with numpy-ts
      const modifiedBytes = readFileSync(npyPath);
      const modified = parseNpy(modifiedBytes);

      expect(modified.shape).toEqual([1000]);
      expect(modified.get([0])).toBe(0);
      expect(modified.get([1])).toBe(2);
      expect(modified.get([999])).toBe(1998);

      unlinkSync(npyPath);
    });
  });
});
