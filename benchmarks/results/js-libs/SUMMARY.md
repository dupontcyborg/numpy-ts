# numpy-ts vs JS numerical libraries — full sweep summary

Dtypes covered: float64, float32, float16, int8, int16, int32, int64, uint8, uint16, uint32, uint64, bool, complex64, complex128
Headline = geomean ops/sec (higher = faster); ratio = lib / numpy-ts.

## Per-dtype geomean ops/sec

| dtype | specs | numpy-ts | numpy(pyodide) | jax-js | mathjs | numeric | ml-matrix | numjs | stdlib | tfjs |
|-|-|-|-|-|-|-|-|-|-|-|
| float64 | 93 | 123K | 72K | 16K | 4K | 53K | 16K | 42K | 21K | · |
| float32 | 93 | 147K | 76K | 18K | · | · | · | 45K | 14K | 49K |
| float16 | 93 | 62K | 31K | · | · | · | · | · | 24K | · |
| int8 | 93 | 176K | 74K | · | · | · | · | 61K | 13K | · |
| int16 | 93 | 146K | 88K | · | · | · | · | 56K | 13K | · |
| int32 | 93 | 131K | 83K | 17K | · | · | · | 55K | 4K | 42K |
| int64 | 93 | 84K | 81K | · | · | · | · | · | · | · |
| uint8 | 93 | 150K | 77K | · | · | · | · | 62K | 4K | · |
| uint16 | 93 | 136K | 91K | · | · | · | · | 56K | 4K | · |
| uint32 | 93 | 118K | 86K | 17K | · | · | · | 54K | 4K | · |
| uint64 | 93 | 78K | 82K | · | · | · | · | · | · | · |
| bool | 21 | 33K | 35K | 33K | · | · | · | · | · | 7K |
| complex64 | 82 | 18K | 42K | · | · | · | · | · | · | · |
| complex128 | 82 | 16K | 37K | · | · | · | · | · | · | · |

## Coverage (specs run / total)

| dtype | numpy(pyodide) | jax-js | mathjs | numeric | ml-matrix | numjs | stdlib | tfjs |
|-|-|-|-|-|-|-|-|-|-|
| float64 | 90/93 | 86/93 | 56/93 | 31/93 | 38/93 | 37/93 | 43/93 | · |
| float32 | 90/93 | 86/93 | · | · | · | 37/93 | 42/93 | 64/93 |
| float16 | 90/93 | · | · | · | · | · | 7/93 | · |
| int8 | 90/93 | · | · | · | · | 37/93 | 42/93 | · |
| int16 | 90/93 | · | · | · | · | 37/93 | 42/93 | · |
| int32 | 90/93 | 61/93 | · | · | · | 37/93 | 42/93 | 55/93 |
| int64 | 90/93 | · | · | · | · | · | · | · |
| uint8 | 90/93 | · | · | · | · | 37/93 | 42/93 | · |
| uint16 | 90/93 | · | · | · | · | 37/93 | 42/93 | · |
| uint32 | 90/93 | 61/93 | · | · | · | 37/93 | 42/93 | · |
| uint64 | 90/93 | · | · | · | · | · | · | · |
| bool | 21/21 | 6/21 | · | · | · | · | · | 3/21 |
| complex64 | 79/82 | · | · | · | · | · | · | · |
| complex128 | 79/82 | · | · | · | · | · | · | · |

## Geomean ratio vs numpy-ts (×; <1 = faster than numpy-ts)

| dtype | numpy(pyodide) | jax-js | mathjs | numeric | ml-matrix | numjs | stdlib | tfjs |
|-|-|-|-|-|-|-|-|-|-|
| float64 | 1.66 | 7.57 | 27.44 | 1.95 | 5.61 | 2.38 | 4.79 | · |
| float32 | 1.89 | 8.03 | · | · | · | 2.92 | 8.58 | 3.23 |
| float16 | 1.87 | · | · | · | · | · | 2.58 | · |
| int8 | 2.23 | · | · | · | · | 2.43 | 8.64 | · |
| int16 | 1.61 | · | · | · | · | 2.35 | 8.09 | · |
| int32 | 1.52 | 13.82 | · | · | · | 2.11 | 22.21 | 3.86 |
| int64 | 1.00 | · | · | · | · | · | · | · |
| uint8 | 1.82 | · | · | · | · | 2.05 | 24.93 | · |
| uint16 | 1.44 | · | · | · | · | 2.35 | 26.15 | · |
| uint32 | 1.33 | 12.28 | · | · | · | 1.99 | 22.80 | · |
| uint64 | 0.91 | · | · | · | · | · | · | · |
| bool | 0.96 | 3.93 | · | · | · | · | · | 16.88 |
| complex64 | 0.49 | · | · | · | · | · | · | · |
| complex128 | 0.47 | · | · | · | · | · | · | · |

## Specs where a competitor beats numpy-ts (ratio < 1)

numpy-ts is fastest on 75.5% of all 2356 head-to-heads.

- **numpy(pyodide)**: 392 specs — e.g. inner 2D · 2D [100x100] [complex128] 158.8×; inner 2D · 2D [100x100] [complex64] 155.5×; subtract [100x100] - scalar [complex64] 149.2×; subtract [100x100] - scalar [complex128] 138.2×
- **numjs**: 77 specs — e.g. transpose [100x100] [uint8] 139.2×; transpose [100x100] [int16] 88.5×; transpose [100x100] [uint16] 83.5×; mod [100x100] % scalar [int32] 67.1×
- **jax-js**: 47 specs — e.g. remainder [100x100] % scalar [int32] 28.7×; remainder [100x100] % scalar [uint32] 28.6×; mod [100x100] % scalar [uint32] 28.0×; mod [100x100] % scalar [int32] 27.4×
- **stdlib**: 27 specs — e.g. transpose [100x100] [int16] 12.9×; dot 1D · 1D [1000] [float16] 11.6×; transpose [100x100] [float64] 6.2×; transpose [100x100] [float32] 4.0×
- **tfjs**: 25 specs — e.g. transpose [100x100] [int32] 26.7×; mod [100x100] % [100x100] [int32] 16.0×; arcsinh [100x100] [int32] 13.2×; floor_divide [100x100] // [100x100] [int32] 8.5×
- **numeric**: 6 specs — e.g. transpose [100x100] [float64] 6.2×; mod [100x100] % scalar [float64] 4.1×; exp [100x100] [float64] 2.5×; log [100x100] [float64] 2.0×
- **ml-matrix**: 2 specs — e.g. transpose [100x100] [float64] 1.9×; trace [100x100] [float64] 1.6×
- **mathjs**: 1 specs — e.g. transpose [100x100] [float64] 2.1×

## numpy-ts vs NumPy (Pyodide) — same-input, same-timer

| dtype | numpy-ts ops/sec | numpy(pyodide) ops/sec | numpy-ts vs NumPy |
|-|-|-|-|
| float64 | 123K | 72K | 1.66× faster |
| float32 | 147K | 76K | 1.89× faster |
| float16 | 62K | 31K | 1.87× faster |
| int8 | 176K | 74K | 2.23× faster |
| int16 | 146K | 88K | 1.61× faster |
| int32 | 131K | 83K | 1.52× faster |
| int64 | 84K | 81K | 1.00× faster |
| uint8 | 150K | 77K | 1.82× faster |
| uint16 | 136K | 91K | 1.44× faster |
| uint32 | 118K | 86K | 1.33× faster |
| uint64 | 78K | 82K | 1.10× slower |
| bool | 33K | 35K | 1.04× slower |
| complex64 | 18K | 42K | 2.04× slower |
| complex128 | 16K | 37K | 2.12× slower |
