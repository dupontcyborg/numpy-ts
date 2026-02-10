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
import { parseTxt, serializeTxt, genfromtxt, fromregex } from '../../src/io/txt';
import { array, arange } from '../../src';
import { Complex } from '../../src/common/complex';
import type { DType } from '../../src/common/dtype';

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

    it('reads complex128 NPY file from NumPy', () => {
      const npyPath = join(tempDir, 'test_complex128.npy');

      const pythonCode = `
import numpy as np
arr = np.array([1+2j, 3+4j, 5+6j], dtype=np.complex128)
np.save('${npyPath}', arr)
`;
      execSync(`${PYTHON_CMD} -c "${pythonCode}"`, { stdio: 'pipe' });

      const npyBytes = readFileSync(npyPath);
      const arr = parseNpy(npyBytes);

      expect(arr.dtype).toBe('complex128');
      expect(arr.shape).toEqual([3]);
      const values = arr.toArray() as Complex[];
      expect(values[0].re).toBeCloseTo(1, 10);
      expect(values[0].im).toBeCloseTo(2, 10);
      expect(values[1].re).toBeCloseTo(3, 10);
      expect(values[1].im).toBeCloseTo(4, 10);
      expect(values[2].re).toBeCloseTo(5, 10);
      expect(values[2].im).toBeCloseTo(6, 10);

      unlinkSync(npyPath);
    });

    it('reads complex64 NPY file from NumPy', () => {
      const npyPath = join(tempDir, 'test_complex64.npy');

      const pythonCode = `
import numpy as np
arr = np.array([1+2j, 3+4j, 5+6j], dtype=np.complex64)
np.save('${npyPath}', arr)
`;
      execSync(`${PYTHON_CMD} -c "${pythonCode}"`, { stdio: 'pipe' });

      const npyBytes = readFileSync(npyPath);
      const arr = parseNpy(npyBytes);

      expect(arr.dtype).toBe('complex64');
      expect(arr.shape).toEqual([3]);
      const values = arr.toArray() as Complex[];
      expect(values[0].re).toBeCloseTo(1, 5);
      expect(values[0].im).toBeCloseTo(2, 5);
      expect(values[1].re).toBeCloseTo(3, 5);
      expect(values[1].im).toBeCloseTo(4, 5);
      expect(values[2].re).toBeCloseTo(5, 5);
      expect(values[2].im).toBeCloseTo(6, 5);

      unlinkSync(npyPath);
    });

    it('reads 2D complex128 array from NumPy', () => {
      const npyPath = join(tempDir, 'test_complex128_2d.npy');

      const pythonCode = `
import numpy as np
arr = np.array([[1+1j, 2+2j], [3+3j, 4+4j]], dtype=np.complex128)
np.save('${npyPath}', arr)
`;
      execSync(`${PYTHON_CMD} -c "${pythonCode}"`, { stdio: 'pipe' });

      const npyBytes = readFileSync(npyPath);
      const arr = parseNpy(npyBytes);

      expect(arr.dtype).toBe('complex128');
      expect(arr.shape).toEqual([2, 2]);
      const val00 = arr.get([0, 0]) as Complex;
      const val11 = arr.get([1, 1]) as Complex;
      expect(val00.re).toBeCloseTo(1, 10);
      expect(val00.im).toBeCloseTo(1, 10);
      expect(val11.re).toBeCloseTo(4, 10);
      expect(val11.im).toBeCloseTo(4, 10);

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

    it('validates complex128 NPY file with NumPy', () => {
      const npyPath = join(tempDir, 'ts_complex128.npy');

      const arr = array([new Complex(1, 2), new Complex(3, 4), new Complex(5, 6)], 'complex128');
      const npyBytes = serializeNpy(arr);
      writeFileSync(npyPath, npyBytes);

      const pythonCode = `
import numpy as np
import json
arr = np.load('${npyPath}')
expected = np.array([1+2j, 3+4j, 5+6j], dtype=np.complex128)
result = {
    'dtype': str(arr.dtype),
    'shape': list(arr.shape),
    'values_match': np.allclose(arr, expected)
}
print(json.dumps(result))
`;

      const output = execSync(`${PYTHON_CMD} -c "${pythonCode}"`, { encoding: 'utf-8' });
      const result = JSON.parse(output);

      expect(result.dtype).toBe('complex128');
      expect(result.shape).toEqual([3]);
      expect(result.values_match).toBe(true);

      unlinkSync(npyPath);
    });

    it('validates complex64 NPY file with NumPy', () => {
      const npyPath = join(tempDir, 'ts_complex64.npy');

      const arr = array([new Complex(1, 2), new Complex(3, 4), new Complex(5, 6)], 'complex64');
      const npyBytes = serializeNpy(arr);
      writeFileSync(npyPath, npyBytes);

      const pythonCode = `
import numpy as np
import json
arr = np.load('${npyPath}')
expected = np.array([1+2j, 3+4j, 5+6j], dtype=np.complex64)
result = {
    'dtype': str(arr.dtype),
    'shape': list(arr.shape),
    'values_match': np.allclose(arr, expected)
}
print(json.dumps(result))
`;

      const output = execSync(`${PYTHON_CMD} -c "${pythonCode}"`, { encoding: 'utf-8' });
      const result = JSON.parse(output);

      expect(result.dtype).toBe('complex64');
      expect(result.shape).toEqual([3]);
      expect(result.values_match).toBe(true);

      unlinkSync(npyPath);
    });

    it('validates 2D complex128 array with NumPy', () => {
      const npyPath = join(tempDir, 'ts_complex128_2d.npy');

      const arr = array(
        [
          [new Complex(1, 1), new Complex(2, 2)],
          [new Complex(3, 3), new Complex(4, 4)],
        ],
        'complex128'
      );
      const npyBytes = serializeNpy(arr);
      writeFileSync(npyPath, npyBytes);

      const pythonCode = `
import numpy as np
import json
arr = np.load('${npyPath}')
expected = np.array([[1+1j, 2+2j], [3+3j, 4+4j]], dtype=np.complex128)
result = {
    'dtype': str(arr.dtype),
    'shape': list(arr.shape),
    'values_match': np.allclose(arr, expected)
}
print(json.dumps(result))
`;

      const output = execSync(`${PYTHON_CMD} -c "${pythonCode}"`, { encoding: 'utf-8' });
      const result = JSON.parse(output);

      expect(result.dtype).toBe('complex128');
      expect(result.shape).toEqual([2, 2]);
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

  describe('Text I/O Cross-Language Validation', () => {
    describe('Python → TypeScript (read text files created by NumPy)', () => {
      it('reads loadtxt output from NumPy (whitespace)', () => {
        const txtPath = join(tempDir, 'test_loadtxt.txt');

        const pythonCode = `
import numpy as np
arr = np.array([[1.5, 2.5, 3.5], [4.5, 5.5, 6.5]])
np.savetxt('${txtPath}', arr)
`;
        execSync(`${PYTHON_CMD} -c "${pythonCode}"`, { stdio: 'pipe' });

        const content = readFileSync(txtPath, 'utf-8');
        const arr = parseTxt(content);

        expect(arr.shape).toEqual([2, 3]);
        expect(arr.get([0, 0])).toBeCloseTo(1.5, 5);
        expect(arr.get([1, 2])).toBeCloseTo(6.5, 5);

        unlinkSync(txtPath);
      });

      it('reads loadtxt output from NumPy (CSV)', () => {
        const txtPath = join(tempDir, 'test_csv.txt');

        const pythonCode = `
import numpy as np
arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
np.savetxt('${txtPath}', arr, delimiter=',', fmt='%d')
`;
        execSync(`${PYTHON_CMD} -c "${pythonCode}"`, { stdio: 'pipe' });

        const content = readFileSync(txtPath, 'utf-8');
        const arr = parseTxt(content, { delimiter: ',' });

        expect(arr.shape).toEqual([3, 3]);
        expect(arr.toArray()).toEqual([
          [1, 2, 3],
          [4, 5, 6],
          [7, 8, 9],
        ]);

        unlinkSync(txtPath);
      });

      it('reads file with header from NumPy', () => {
        const txtPath = join(tempDir, 'test_header.txt');

        const pythonCode = `
import numpy as np
arr = np.array([[1.0, 2.0], [3.0, 4.0]])
np.savetxt('${txtPath}', arr, header='x y', fmt='%.1f')
`;
        execSync(`${PYTHON_CMD} -c "${pythonCode}"`, { stdio: 'pipe' });

        const content = readFileSync(txtPath, 'utf-8');
        const arr = parseTxt(content);

        expect(arr.shape).toEqual([2, 2]);
        expect(arr.toArray()).toEqual([
          [1, 2],
          [3, 4],
        ]);

        unlinkSync(txtPath);
      });

      it('reads 1D array from NumPy', () => {
        const txtPath = join(tempDir, 'test_1d.txt');

        const pythonCode = `
import numpy as np
arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
np.savetxt('${txtPath}', arr, fmt='%.1f')
`;
        execSync(`${PYTHON_CMD} -c "${pythonCode}"`, { stdio: 'pipe' });

        const content = readFileSync(txtPath, 'utf-8');
        const arr = parseTxt(content);

        expect(arr.shape).toEqual([5]);
        expect(arr.toArray()).toEqual([1, 2, 3, 4, 5]);

        unlinkSync(txtPath);
      });
    });

    describe('TypeScript → Python (validate text files created by numpy-ts)', () => {
      it('validates savetxt output with NumPy loadtxt', () => {
        const txtPath = join(tempDir, 'ts_savetxt.txt');

        const arr = array([
          [1.5, 2.5, 3.5],
          [4.5, 5.5, 6.5],
        ]);
        const content = serializeTxt(arr, { fmt: '%.6f' });
        writeFileSync(txtPath, content);

        const pythonCode = `
import numpy as np
import json
arr = np.loadtxt('${txtPath}')
result = {
    'shape': list(arr.shape),
    'values_match': np.allclose(arr, [[1.5, 2.5, 3.5], [4.5, 5.5, 6.5]])
}
print(json.dumps(result))
`;

        const output = execSync(`${PYTHON_CMD} -c "${pythonCode}"`, { encoding: 'utf-8' });
        const result = JSON.parse(output);

        expect(result.shape).toEqual([2, 3]);
        expect(result.values_match).toBe(true);

        unlinkSync(txtPath);
      });

      it('validates CSV output with NumPy loadtxt', () => {
        const txtPath = join(tempDir, 'ts_csv.txt');

        const arr = array([
          [1, 2, 3],
          [4, 5, 6],
        ]);
        const content = serializeTxt(arr, { fmt: '%d', delimiter: ',' });
        writeFileSync(txtPath, content);

        const pythonCode = `
import numpy as np
import json
arr = np.loadtxt('${txtPath}', delimiter=',')
result = {
    'shape': list(arr.shape),
    'values_match': np.array_equal(arr, [[1, 2, 3], [4, 5, 6]])
}
print(json.dumps(result))
`;

        const output = execSync(`${PYTHON_CMD} -c "${pythonCode}"`, { encoding: 'utf-8' });
        const result = JSON.parse(output);

        expect(result.shape).toEqual([2, 3]);
        expect(result.values_match).toBe(true);

        unlinkSync(txtPath);
      });

      it('validates 1D array output with NumPy', () => {
        const txtPath = join(tempDir, 'ts_1d.txt');

        const arr = array([10, 20, 30, 40, 50]);
        const content = serializeTxt(arr, { fmt: '%d' });
        writeFileSync(txtPath, content);

        const pythonCode = `
import numpy as np
import json
arr = np.loadtxt('${txtPath}')
result = {
    'shape': list(arr.shape),
    'values': arr.tolist()
}
print(json.dumps(result))
`;

        const output = execSync(`${PYTHON_CMD} -c "${pythonCode}"`, { encoding: 'utf-8' });
        const result = JSON.parse(output);

        expect(result.shape).toEqual([5]);
        expect(result.values).toEqual([10, 20, 30, 40, 50]);

        unlinkSync(txtPath);
      });

      it('validates scientific notation output', () => {
        const txtPath = join(tempDir, 'ts_scientific.txt');

        const arr = array([1e-10, 2e5, 3.14159]);
        const content = serializeTxt(arr, { fmt: '%.6e' });
        writeFileSync(txtPath, content);

        const pythonCode = `
import numpy as np
import json
arr = np.loadtxt('${txtPath}')
expected = [1e-10, 2e5, 3.14159]
result = {
    'shape': list(arr.shape),
    'values_match': np.allclose(arr, expected)
}
print(json.dumps(result))
`;

        const output = execSync(`${PYTHON_CMD} -c "${pythonCode}"`, { encoding: 'utf-8' });
        const result = JSON.parse(output);

        expect(result.values_match).toBe(true);

        unlinkSync(txtPath);
      });
    });

    describe('genfromtxt validation', () => {
      it('handles missing values like NumPy', () => {
        const txtPath = join(tempDir, 'test_missing.txt');

        // Create file with missing values
        writeFileSync(txtPath, '1,2,3\n4,,6\n7,8,9\n');

        // Compare with NumPy's genfromtxt
        const pythonCode = `
import numpy as np
import json
arr = np.genfromtxt('${txtPath}', delimiter=',')
result = {
    'shape': list(arr.shape),
    'has_nan': bool(np.isnan(arr[1, 1])),
    'valid_values': [arr[0, 0], arr[2, 2]]
}
print(json.dumps(result))
`;

        const output = execSync(`${PYTHON_CMD} -c "${pythonCode}"`, { encoding: 'utf-8' });
        const npResult = JSON.parse(output);

        // Now test numpy-ts
        const content = readFileSync(txtPath, 'utf-8');
        const arr = genfromtxt(content, { delimiter: ',' });

        expect(arr.shape).toEqual(npResult.shape);
        expect(Number.isNaN(arr.get([1, 1]) as number)).toBe(npResult.has_nan);
        expect(arr.get([0, 0])).toBe(npResult.valid_values[0]);
        expect(arr.get([2, 2])).toBe(npResult.valid_values[1]);

        unlinkSync(txtPath);
      });
    });

    describe('fromregex validation', () => {
      it('extracts values correctly with regex', () => {
        const txtPath = join(tempDir, 'test_regex.txt');

        writeFileSync(txtPath, 'Point: x=1.5, y=2.5\nPoint: x=3.5, y=4.5\nPoint: x=5.5, y=6.5\n');

        // Note: NumPy's fromregex requires structured dtype, so we validate
        // against expected values directly rather than comparing to NumPy

        const content = readFileSync(txtPath, 'utf-8');
        const arr = fromregex(content, /x=([\d.]+), y=([\d.]+)/);

        expect(arr.shape).toEqual([3, 2]);
        expect(arr.toArray()).toEqual([
          [1.5, 2.5],
          [3.5, 4.5],
          [5.5, 6.5],
        ]);

        unlinkSync(txtPath);
      });

      it('extracts single column values', () => {
        const txtPath = join(tempDir, 'test_regex_single.txt');

        writeFileSync(txtPath, 'value: 10\nvalue: 20\nvalue: 30\n');

        const content = readFileSync(txtPath, 'utf-8');
        const arr = fromregex(content, /value: (\d+)/);

        expect(arr.shape).toEqual([3]);
        expect(arr.toArray()).toEqual([10, 20, 30]);

        unlinkSync(txtPath);
      });
    });
  });
});
