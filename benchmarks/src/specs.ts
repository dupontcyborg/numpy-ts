/**
 * Benchmark specifications
 * Defines all benchmarks to run
 */

import type { BenchmarkCase, BenchmarkMode } from './types';

export function getBenchmarkSpecs(mode: BenchmarkMode = 'standard'): BenchmarkCase[] {
  // Array sizes vary by mode
  // Auto-calibration handles the iteration count
  const sizeConfig = {
    quick: {
      small: 1000,
      medium: [100, 100] as [number, number],
      large: [500, 500] as [number, number],
    },
    standard: {
      small: 1000,
      medium: [100, 100] as [number, number],
      large: [500, 500] as [number, number],
    },
    large: {
      small: 10000,
      medium: [316, 316] as [number, number], // ~100K elements
      large: [1000, 1000] as [number, number], // 1M elements
    },
  };

  const sizes = sizeConfig[mode] || sizeConfig.standard;

  const warmupConfig = {
    quick: {
      iterations: 1, // Not used with auto-calibration, kept for compatibility
      warmup: 3, // Less warmup for faster feedback
    },
    standard: {
      iterations: 1, // Not used with auto-calibration, kept for compatibility
      warmup: 10, // More warmup for stable results
    },
    large: {
      iterations: 1, // Not used with auto-calibration, kept for compatibility
      warmup: 5, // Moderate warmup for large arrays
    },
  };

  const config = warmupConfig[mode] || warmupConfig.standard;

  const { iterations, warmup } = config;
  const specs: BenchmarkCase[] = [];

  // ========================================
  // Array Creation Benchmarks
  // ========================================

  specs.push({
    name: `zeros [${sizes.small}]`,
    category: 'creation',
    operation: 'zeros',
    setup: {
      shape: { shape: [sizes.small] },
    },
    iterations,
    warmup,
  });

  if (Array.isArray(sizes.medium)) {
    specs.push({
      name: `zeros [${sizes.medium.join('x')}]`,
      category: 'creation',
      operation: 'zeros',
      setup: {
        shape: { shape: sizes.medium },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `ones [${sizes.medium.join('x')}]`,
      category: 'creation',
      operation: 'ones',
      setup: {
        shape: { shape: sizes.medium },
      },
      iterations,
      warmup,
    });
  }

  specs.push({
    name: `arange(${sizes.small})`,
    category: 'creation',
    operation: 'arange',
    setup: {
      n: { shape: [sizes.small] },
    },
    iterations,
    warmup,
  });

  specs.push({
    name: `linspace(0, 100, ${sizes.small})`,
    category: 'creation',
    operation: 'linspace',
    setup: {
      n: { shape: [sizes.small] },
    },
    iterations,
    warmup,
  });

  specs.push({
    name: `logspace(0, 3, ${sizes.small})`,
    category: 'creation',
    operation: 'logspace',
    setup: {
      n: { shape: [sizes.small] },
    },
    iterations,
    warmup,
  });

  specs.push({
    name: `geomspace(1, 1000, ${sizes.small})`,
    category: 'creation',
    operation: 'geomspace',
    setup: {
      n: { shape: [sizes.small] },
    },
    iterations,
    warmup,
  });

  if (Array.isArray(sizes.medium)) {
    const eyeSize = sizes.medium[0]!;
    specs.push({
      name: `eye(${eyeSize})`,
      category: 'creation',
      operation: 'eye',
      setup: {
        n: { shape: [eyeSize] },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `identity(${eyeSize})`,
      category: 'creation',
      operation: 'identity',
      setup: {
        n: { shape: [eyeSize] },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `empty [${sizes.medium.join('x')}]`,
      category: 'creation',
      operation: 'empty',
      setup: {
        shape: { shape: sizes.medium },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `full [${sizes.medium.join('x')}]`,
      category: 'creation',
      operation: 'full',
      setup: {
        shape: { shape: sizes.medium },
        fill_value: { shape: [7] },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `copy [${sizes.medium.join('x')}]`,
      category: 'creation',
      operation: 'copy',
      setup: {
        a: { shape: sizes.medium, fill: 'ones' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `zeros_like [${sizes.medium.join('x')}]`,
      category: 'creation',
      operation: 'zeros_like',
      setup: {
        a: { shape: sizes.medium, fill: 'ones' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `asarray_chkfinite [${sizes.medium.join('x')}]`,
      category: 'creation',
      operation: 'asarray_chkfinite',
      setup: {
        a: { shape: sizes.medium, fill: 'arange' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `require [${sizes.medium.join('x')}]`,
      category: 'creation',
      operation: 'require',
      setup: {
        a: { shape: sizes.medium, fill: 'arange' },
      },
      iterations,
      warmup,
    });
  }

  // ========================================
  // Arithmetic Benchmarks
  // ========================================

  if (Array.isArray(sizes.medium)) {
    specs.push({
      name: `add [${sizes.medium.join('x')}] + scalar`,
      category: 'arithmetic',
      operation: 'add',
      setup: {
        a: { shape: sizes.medium, fill: 'ones' },
        b: { shape: [1], fill: 'ones' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `add [${sizes.medium.join('x')}] + [${sizes.medium.join('x')}]`,
      category: 'arithmetic',
      operation: 'add',
      setup: {
        a: { shape: sizes.medium, fill: 'ones' },
        b: { shape: sizes.medium, fill: 'ones' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `multiply [${sizes.medium.join('x')}] * scalar`,
      category: 'arithmetic',
      operation: 'multiply',
      setup: {
        a: { shape: sizes.medium, fill: 'ones' },
        b: { shape: [1], value: 2 },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `multiply [${sizes.medium.join('x')}] * [${sizes.medium.join('x')}]`,
      category: 'arithmetic',
      operation: 'multiply',
      setup: {
        a: { shape: sizes.medium, fill: 'arange' },
        b: { shape: sizes.medium, fill: 'arange' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `mod [${sizes.medium.join('x')}] % scalar`,
      category: 'arithmetic',
      operation: 'mod',
      setup: {
        a: { shape: sizes.medium, fill: 'arange' }, // arange data
        b: { shape: [1], value: 7 }, // Scalar 7
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `floor_divide [${sizes.medium.join('x')}] // scalar`,
      category: 'arithmetic',
      operation: 'floor_divide',
      setup: {
        a: { shape: sizes.medium, fill: 'arange' }, // arange data
        b: { shape: [1], value: 3 }, // Scalar 3 (avoid zeros)
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `reciprocal [${sizes.medium.join('x')}]`,
      category: 'arithmetic',
      operation: 'reciprocal',
      setup: {
        a: { shape: sizes.medium, fill: 'ones' }, // Use ones to avoid 1/0
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `cbrt [${sizes.medium.join('x')}]`,
      category: 'arithmetic',
      operation: 'cbrt',
      setup: {
        a: { shape: sizes.medium, fill: 'arange', value: 1 },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `fabs [${sizes.medium.join('x')}]`,
      category: 'arithmetic',
      operation: 'fabs',
      setup: {
        a: { shape: sizes.medium, fill: 'arange', value: -100 },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `divmod [${sizes.medium.join('x')}] % scalar`,
      category: 'arithmetic',
      operation: 'divmod',
      setup: {
        a: { shape: sizes.medium, fill: 'arange' },
        b: { shape: [1], value: 7 },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `gcd [${sizes.medium.join('x')}] scalar`,
      category: 'arithmetic',
      operation: 'gcd',
      setup: {
        a: { shape: sizes.medium, fill: 'arange', value: 1, dtype: 'int32' },
        b: { shape: [1], value: 6, dtype: 'int32' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `lcm [${sizes.medium.join('x')}] scalar`,
      category: 'arithmetic',
      operation: 'lcm',
      setup: {
        a: { shape: sizes.medium, fill: 'arange', value: 1, dtype: 'int32' },
        b: { shape: [1], value: 6, dtype: 'int32' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `float_power [${sizes.medium.join('x')}] ** scalar`,
      category: 'arithmetic',
      operation: 'float_power',
      setup: {
        a: { shape: sizes.medium, fill: 'arange', value: 1 },
        b: { shape: [1], value: 2 },
      },
      iterations,
      warmup,
    });
  }

  // ========================================
  // Mathematical Operations Benchmarks
  // ========================================

  if (Array.isArray(sizes.medium)) {
    specs.push({
      name: `sqrt [${sizes.medium.join('x')}]`,
      category: 'math',
      operation: 'sqrt',
      setup: {
        a: { shape: sizes.medium, fill: 'arange', value: 1 },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `power [${sizes.medium.join('x')}] ** 2`,
      category: 'math',
      operation: 'power',
      setup: {
        a: { shape: sizes.medium, fill: 'arange', value: 1 },
        b: { shape: [1], value: 2 },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `absolute [${sizes.medium.join('x')}]`,
      category: 'math',
      operation: 'absolute',
      setup: {
        a: { shape: sizes.medium, fill: 'arange', value: -100 },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `negative [${sizes.medium.join('x')}]`,
      category: 'math',
      operation: 'negative',
      setup: {
        a: { shape: sizes.medium, fill: 'ones' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `sign [${sizes.medium.join('x')}]`,
      category: 'math',
      operation: 'sign',
      setup: {
        a: { shape: sizes.medium, fill: 'arange', value: -100 },
      },
      iterations,
      warmup,
    });

    // Trigonometric functions
    specs.push({
      name: `sin [${sizes.medium.join('x')}]`,
      category: 'math',
      operation: 'sin',
      setup: {
        a: { shape: sizes.medium, fill: 'arange' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `cos [${sizes.medium.join('x')}]`,
      category: 'math',
      operation: 'cos',
      setup: {
        a: { shape: sizes.medium, fill: 'arange' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `tan [${sizes.medium.join('x')}]`,
      category: 'math',
      operation: 'tan',
      setup: {
        a: { shape: sizes.medium, fill: 'arange' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `arctan2 [${sizes.medium.join('x')}]`,
      category: 'math',
      operation: 'arctan2',
      setup: {
        a: { shape: sizes.medium, fill: 'arange', value: 1 },
        b: { shape: sizes.medium, fill: 'arange', value: 1 },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `hypot [${sizes.medium.join('x')}]`,
      category: 'math',
      operation: 'hypot',
      setup: {
        a: { shape: sizes.medium, fill: 'arange', value: 1 },
        b: { shape: sizes.medium, fill: 'arange', value: 1 },
      },
      iterations,
      warmup,
    });

    // Hyperbolic functions
    // Use scaled values to avoid overflow (sinh/cosh overflow around 710)
    specs.push({
      name: `sinh [${sizes.medium.join('x')}]`,
      category: 'math',
      operation: 'sinh',
      setup: {
        a: { shape: sizes.medium, fill: 'ones' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `cosh [${sizes.medium.join('x')}]`,
      category: 'math',
      operation: 'cosh',
      setup: {
        a: { shape: sizes.medium, fill: 'ones' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `tanh [${sizes.medium.join('x')}]`,
      category: 'math',
      operation: 'tanh',
      setup: {
        a: { shape: sizes.medium, fill: 'arange' },
      },
      iterations,
      warmup,
    });

    // Exponential functions
    specs.push({
      name: `exp [${sizes.medium.join('x')}]`,
      category: 'math',
      operation: 'exp',
      setup: {
        a: { shape: sizes.medium, fill: 'ones' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `exp2 [${sizes.medium.join('x')}]`,
      category: 'math',
      operation: 'exp2',
      setup: {
        a: { shape: sizes.medium, fill: 'ones' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `log [${sizes.medium.join('x')}]`,
      category: 'math',
      operation: 'log',
      setup: {
        a: { shape: sizes.medium, fill: 'ones' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `log2 [${sizes.medium.join('x')}]`,
      category: 'math',
      operation: 'log2',
      setup: {
        a: { shape: sizes.medium, fill: 'ones' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `log10 [${sizes.medium.join('x')}]`,
      category: 'math',
      operation: 'log10',
      setup: {
        a: { shape: sizes.medium, fill: 'ones' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `logaddexp [${sizes.medium.join('x')}]`,
      category: 'math',
      operation: 'logaddexp',
      setup: {
        a: { shape: sizes.medium, fill: 'arange' },
        b: { shape: sizes.medium, fill: 'arange' },
      },
      iterations,
      warmup,
    });

    // Gradient functions
    specs.push({
      name: `diff [${sizes.medium.join('x')}]`,
      category: 'gradient',
      operation: 'diff',
      setup: {
        a: { shape: sizes.medium, fill: 'arange' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `gradient [${sizes.small}]`,
      category: 'gradient',
      operation: 'gradient',
      setup: {
        a: { shape: [sizes.small], fill: 'arange' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `cross [${sizes.small}x3]`,
      category: 'gradient',
      operation: 'cross',
      setup: {
        a: { shape: [sizes.small, 3], fill: 'arange' },
        b: { shape: [sizes.small, 3], fill: 'ones' },
      },
      iterations,
      warmup,
    });
  }

  // ========================================
  // Linear Algebra Benchmarks
  // ========================================

  if (Array.isArray(sizes.medium)) {
    const [m, n] = sizes.medium;

    // Dot product benchmarks
    specs.push({
      name: `dot 1D · 1D [${sizes.small}]`,
      category: 'linalg',
      operation: 'dot',
      setup: {
        a: { shape: [sizes.small], fill: 'arange' },
        b: { shape: [sizes.small], fill: 'arange' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `dot 2D · 1D [${m}x${n}] · [${n}]`,
      category: 'linalg',
      operation: 'dot',
      setup: {
        a: { shape: [m!, n!], fill: 'arange' },
        b: { shape: [n!], fill: 'arange' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `dot 2D · 2D [${m}x${n}] · [${n}x${m}]`,
      category: 'linalg',
      operation: 'dot',
      setup: {
        a: { shape: [m!, n!], fill: 'arange', dtype: 'float64' },
        b: { shape: [n!, m!], fill: 'arange', dtype: 'float64' },
      },
      iterations,
      warmup,
    });

    // Matrix multiplication
    specs.push({
      name: `matmul [${m}x${n}] @ [${n}x${m}]`,
      category: 'linalg',
      operation: 'matmul',
      setup: {
        a: { shape: [m!, n!], fill: 'arange', dtype: 'float64' },
        b: { shape: [n!, m!], fill: 'arange', dtype: 'float64' },
      },
      iterations,
      warmup,
    });

    // Trace
    specs.push({
      name: `trace [${m}x${n}]`,
      category: 'linalg',
      operation: 'trace',
      setup: {
        a: { shape: [m!, n!], fill: 'arange' },
      },
      iterations,
      warmup,
    });

    // Inner product
    specs.push({
      name: `inner 1D · 1D [${sizes.small}]`,
      category: 'linalg',
      operation: 'inner',
      setup: {
        a: { shape: [sizes.small], fill: 'arange' },
        b: { shape: [sizes.small], fill: 'arange' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `inner 2D · 2D [${m}x${n}]`,
      category: 'linalg',
      operation: 'inner',
      setup: {
        a: { shape: [m!, n!], fill: 'arange' },
        b: { shape: [m!, n!], fill: 'arange' },
      },
      iterations,
      warmup,
    });

    // Outer product
    specs.push({
      name: `outer [${sizes.small}] x [${sizes.small}]`,
      category: 'linalg',
      operation: 'outer',
      setup: {
        a: { shape: [sizes.small], fill: 'arange' },
        b: { shape: [sizes.small], fill: 'arange' },
      },
      iterations,
      warmup,
    });

    // Trace
    specs.push({
      name: `trace [${m}x${n}]`,
      category: 'linalg',
      operation: 'trace',
      setup: {
        a: { shape: [m!, n!], fill: 'arange' },
      },
      iterations,
      warmup,
    });

    // Transpose
    specs.push({
      name: `transpose [${m}x${n}]`,
      category: 'linalg',
      operation: 'transpose',
      setup: {
        a: { shape: [m!, n!], fill: 'arange' },
      },
      iterations,
      warmup,
    });
  }

  // Larger matmul if not in quick mode
  if (mode !== 'quick' && Array.isArray(sizes.large)) {
    const [m, n] = sizes.large;
    specs.push({
      name: `matmul [${m}x${n}] @ [${n}x${m}]`,
      category: 'linalg',
      operation: 'matmul',
      setup: {
        a: { shape: [m!, n!], fill: 'arange', dtype: 'float64' },
        b: { shape: [n!, m!], fill: 'arange', dtype: 'float64' },
      },
      iterations: Math.floor(iterations / 2), // Fewer iterations for large
      warmup: Math.floor(warmup / 2),
    });
  }

  // ========================================
  // Reduction Benchmarks
  // ========================================

  if (Array.isArray(sizes.medium)) {
    specs.push({
      name: `sum [${sizes.medium.join('x')}]`,
      category: 'reductions',
      operation: 'sum',
      setup: {
        a: { shape: sizes.medium, fill: 'arange' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `sum [${sizes.medium.join('x')}] axis=0`,
      category: 'reductions',
      operation: 'sum',
      setup: {
        a: { shape: sizes.medium, fill: 'arange' },
        axis: { shape: [0] },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `mean [${sizes.medium.join('x')}]`,
      category: 'reductions',
      operation: 'mean',
      setup: {
        a: { shape: sizes.medium, fill: 'arange' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `max [${sizes.medium.join('x')}]`,
      category: 'reductions',
      operation: 'max',
      setup: {
        a: { shape: sizes.medium, fill: 'arange' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `min [${sizes.medium.join('x')}]`,
      category: 'reductions',
      operation: 'min',
      setup: {
        a: { shape: sizes.medium, fill: 'arange' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `prod [${sizes.medium.join('x')}]`,
      category: 'reductions',
      operation: 'prod',
      setup: {
        a: { shape: sizes.medium, fill: 'ones' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `argmin [${sizes.medium.join('x')}]`,
      category: 'reductions',
      operation: 'argmin',
      setup: {
        a: { shape: sizes.medium, fill: 'arange' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `argmax [${sizes.medium.join('x')}]`,
      category: 'reductions',
      operation: 'argmax',
      setup: {
        a: { shape: sizes.medium, fill: 'arange' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `var [${sizes.medium.join('x')}]`,
      category: 'reductions',
      operation: 'var',
      setup: {
        a: { shape: sizes.medium, fill: 'arange' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `std [${sizes.medium.join('x')}]`,
      category: 'reductions',
      operation: 'std',
      setup: {
        a: { shape: sizes.medium, fill: 'arange' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `all [${sizes.medium.join('x')}]`,
      category: 'reductions',
      operation: 'all',
      setup: {
        a: { shape: sizes.medium, fill: 'ones' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `any [${sizes.medium.join('x')}]`,
      category: 'reductions',
      operation: 'any',
      setup: {
        a: { shape: sizes.medium, fill: 'zeros' },
      },
      iterations,
      warmup,
    });

    // New reduction functions
    specs.push({
      name: `cumsum [${sizes.medium.join('x')}]`,
      category: 'reductions',
      operation: 'cumsum',
      setup: {
        a: { shape: sizes.medium, fill: 'arange' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `cumprod [${sizes.medium.join('x')}]`,
      category: 'reductions',
      operation: 'cumprod',
      setup: {
        a: { shape: sizes.medium, fill: 'ones' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `ptp [${sizes.medium.join('x')}]`,
      category: 'reductions',
      operation: 'ptp',
      setup: {
        a: { shape: sizes.medium, fill: 'arange' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `median [${sizes.medium.join('x')}]`,
      category: 'reductions',
      operation: 'median',
      setup: {
        a: { shape: sizes.medium, fill: 'arange' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `percentile [${sizes.medium.join('x')}]`,
      category: 'reductions',
      operation: 'percentile',
      setup: {
        a: { shape: sizes.medium, fill: 'arange' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `quantile [${sizes.medium.join('x')}]`,
      category: 'reductions',
      operation: 'quantile',
      setup: {
        a: { shape: sizes.medium, fill: 'arange' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `average [${sizes.medium.join('x')}]`,
      category: 'reductions',
      operation: 'average',
      setup: {
        a: { shape: sizes.medium, fill: 'arange' },
      },
      iterations,
      warmup,
    });

    // NaN-aware reduction functions
    specs.push({
      name: `nansum [${sizes.medium.join('x')}]`,
      category: 'reductions',
      operation: 'nansum',
      setup: {
        a: { shape: sizes.medium, fill: 'arange' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `nanmean [${sizes.medium.join('x')}]`,
      category: 'reductions',
      operation: 'nanmean',
      setup: {
        a: { shape: sizes.medium, fill: 'arange' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `nanmin [${sizes.medium.join('x')}]`,
      category: 'reductions',
      operation: 'nanmin',
      setup: {
        a: { shape: sizes.medium, fill: 'arange' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `nanmax [${sizes.medium.join('x')}]`,
      category: 'reductions',
      operation: 'nanmax',
      setup: {
        a: { shape: sizes.medium, fill: 'arange' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `nanquantile [${sizes.medium.join('x')}]`,
      category: 'reductions',
      operation: 'nanquantile',
      setup: {
        a: { shape: sizes.medium, fill: 'arange' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `nanpercentile [${sizes.medium.join('x')}]`,
      category: 'reductions',
      operation: 'nanpercentile',
      setup: {
        a: { shape: sizes.medium, fill: 'arange' },
      },
      iterations,
      warmup,
    });
  }

  // ========================================
  // Reshape Benchmarks
  // ========================================

  if (Array.isArray(sizes.medium)) {
    const [m, n] = sizes.medium;

    specs.push({
      name: `reshape [${m}x${n}] -> [${n}x${m}] (contiguous)`,
      category: 'reshape',
      operation: 'reshape',
      setup: {
        a: { shape: [m!, n!], fill: 'arange' },
        new_shape: { shape: [n!, m!] },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `flatten [${m}x${n}]`,
      category: 'reshape',
      operation: 'flatten',
      setup: {
        a: { shape: [m!, n!], fill: 'arange' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `ravel [${m}x${n}]`,
      category: 'reshape',
      operation: 'ravel',
      setup: {
        a: { shape: [m!, n!], fill: 'arange' },
      },
      iterations,
      warmup,
    });

    // ========================================
    // Array Manipulation Benchmarks
    // ========================================

    specs.push({
      name: `swapaxes [${m}x${n}]`,
      category: 'manipulation',
      operation: 'swapaxes',
      setup: {
        a: { shape: [m!, n!], fill: 'arange' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `concatenate [${m}x${n}] + [${m}x${n}]`,
      category: 'manipulation',
      operation: 'concatenate',
      setup: {
        a: { shape: [m!, n!], fill: 'arange' },
        b: { shape: [m!, n!], fill: 'arange' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `stack [${m}x${n}] + [${m}x${n}]`,
      category: 'manipulation',
      operation: 'stack',
      setup: {
        a: { shape: [m!, n!], fill: 'arange' },
        b: { shape: [m!, n!], fill: 'arange' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `vstack [${m}x${n}]`,
      category: 'manipulation',
      operation: 'vstack',
      setup: {
        a: { shape: [m!, n!], fill: 'arange' },
        b: { shape: [m!, n!], fill: 'arange' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `hstack [${m}x${n}]`,
      category: 'manipulation',
      operation: 'hstack',
      setup: {
        a: { shape: [m!, n!], fill: 'arange' },
        b: { shape: [m!, n!], fill: 'arange' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `tile [${m}x${n}] x [2,2]`,
      category: 'manipulation',
      operation: 'tile',
      setup: {
        a: { shape: [m!, n!], fill: 'arange' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `repeat [${m}x${n}] x 2`,
      category: 'manipulation',
      operation: 'repeat',
      setup: {
        a: { shape: [m!, n!], fill: 'arange' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `broadcast_to [${n}] -> [${m}x${n}]`,
      category: 'manipulation',
      operation: 'broadcast_to',
      setup: {
        a: { shape: [n!], fill: 'arange' },
        target_shape: { shape: [m!, n!] },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `take [${m}x${n}] 100 indices`,
      category: 'manipulation',
      operation: 'take',
      setup: {
        a: { shape: [m!, n!], fill: 'arange' },
        indices: { shape: Array.from({ length: 100 }, (_, i) => i % (m! * n!)) },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `concat [${m}x${n}] + [${m}x${n}]`,
      category: 'manipulation',
      operation: 'concat',
      setup: {
        a: { shape: [m!, n!], fill: 'arange' },
        b: { shape: [m!, n!], fill: 'arange' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `unstack [${m}x${n}]`,
      category: 'manipulation',
      operation: 'unstack',
      setup: {
        a: { shape: [m!, n!], fill: 'arange' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `block [${m}x${n}] + [${m}x${n}]`,
      category: 'manipulation',
      operation: 'block',
      setup: {
        a: { shape: [m!, n!], fill: 'arange' },
        b: { shape: [m!, n!], fill: 'arange' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `item [${m}x${n}]`,
      category: 'manipulation',
      operation: 'item',
      setup: {
        a: { shape: [m!, n!], fill: 'arange' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `tolist [${m}x${n}]`,
      category: 'manipulation',
      operation: 'tolist',
      setup: {
        a: { shape: [m!, n!], fill: 'arange' },
      },
      iterations,
      warmup,
    });
  }

  // ========================================
  // IO Benchmarks (NPY/NPZ parsing and serialization)
  // ========================================

  if (Array.isArray(sizes.medium)) {
    const [m, n] = sizes.medium;
    const ioSize = m! * n!; // Total elements for IO benchmarks

    // NPY serialization
    specs.push({
      name: `serializeNpy [${m}x${n}] float64`,
      category: 'io',
      operation: 'serializeNpy',
      setup: {
        a: { shape: [m!, n!], fill: 'arange', dtype: 'float64' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `serializeNpy [${m}x${n}] int32`,
      category: 'io',
      operation: 'serializeNpy',
      setup: {
        a: { shape: [m!, n!], fill: 'arange', dtype: 'int32' },
      },
      iterations,
      warmup,
    });

    // NPY parsing (uses pre-serialized bytes)
    specs.push({
      name: `parseNpy [${m}x${n}] float64`,
      category: 'io',
      operation: 'parseNpy',
      setup: {
        a: { shape: [m!, n!], fill: 'arange', dtype: 'float64' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `parseNpy [${m}x${n}] int32`,
      category: 'io',
      operation: 'parseNpy',
      setup: {
        a: { shape: [m!, n!], fill: 'arange', dtype: 'int32' },
      },
      iterations,
      warmup,
    });

    // NPZ serialization (sync, no compression)
    specs.push({
      name: `serializeNpzSync {a, b} [${m}x${n}]`,
      category: 'io',
      operation: 'serializeNpzSync',
      setup: {
        a: { shape: [m!, n!], fill: 'arange', dtype: 'float64' },
        b: { shape: [m!, n!], fill: 'ones', dtype: 'float64' },
      },
      iterations,
      warmup,
    });

    // NPZ parsing (sync, no compression)
    specs.push({
      name: `parseNpzSync {a, b} [${m}x${n}]`,
      category: 'io',
      operation: 'parseNpzSync',
      setup: {
        a: { shape: [m!, n!], fill: 'arange', dtype: 'float64' },
        b: { shape: [m!, n!], fill: 'ones', dtype: 'float64' },
      },
      iterations,
      warmup,
    });
  }

  // Larger IO benchmarks for non-quick mode
  if (mode !== 'quick' && Array.isArray(sizes.large)) {
    const [m, n] = sizes.large;

    specs.push({
      name: `serializeNpy [${m}x${n}] float64`,
      category: 'io',
      operation: 'serializeNpy',
      setup: {
        a: { shape: [m!, n!], fill: 'arange', dtype: 'float64' },
      },
      iterations: Math.floor(iterations / 2),
      warmup: Math.floor(warmup / 2),
    });

    specs.push({
      name: `parseNpy [${m}x${n}] float64`,
      category: 'io',
      operation: 'parseNpy',
      setup: {
        a: { shape: [m!, n!], fill: 'arange', dtype: 'float64' },
      },
      iterations: Math.floor(iterations / 2),
      warmup: Math.floor(warmup / 2),
    });
  }

  // ========================================
  // BigInt (64-bit) Benchmarks
  // ========================================
  // Tests representative operations with int64/uint64 dtypes
  // to compare BigInt performance vs standard numeric types

  if (Array.isArray(sizes.medium)) {
    // Creation with BigInt
    specs.push({
      name: `zeros [${sizes.medium.join('x')}] (int64)`,
      category: 'bigint',
      operation: 'zeros',
      setup: {
        shape: { shape: sizes.medium, dtype: 'int64' },
      },
      iterations,
      warmup,
    });

    // Arithmetic with BigInt
    specs.push({
      name: `add [${sizes.medium.join('x')}] + [${sizes.medium.join('x')}] (int64)`,
      category: 'bigint',
      operation: 'add',
      setup: {
        a: { shape: sizes.medium, fill: 'arange', dtype: 'int64' },
        b: { shape: sizes.medium, fill: 'ones', dtype: 'int64' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `multiply [${sizes.medium.join('x')}] * scalar (int64)`,
      category: 'bigint',
      operation: 'multiply',
      setup: {
        a: { shape: sizes.medium, fill: 'arange', dtype: 'int64' },
        b: { shape: [1], value: 2, dtype: 'int64' },
      },
      iterations,
      warmup,
    });

    // Math operations with BigInt
    specs.push({
      name: `absolute [${sizes.medium.join('x')}] (int64)`,
      category: 'bigint',
      operation: 'absolute',
      setup: {
        a: { shape: sizes.medium, fill: 'arange', value: -100, dtype: 'int64' },
      },
      iterations,
      warmup,
    });

    // Reductions with BigInt
    specs.push({
      name: `sum [${sizes.medium.join('x')}] (int64)`,
      category: 'bigint',
      operation: 'sum',
      setup: {
        a: { shape: sizes.medium, fill: 'arange', dtype: 'int64' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `max [${sizes.medium.join('x')}] (int64)`,
      category: 'bigint',
      operation: 'max',
      setup: {
        a: { shape: sizes.medium, fill: 'arange', dtype: 'int64' },
      },
      iterations,
      warmup,
    });

    // Unsigned BigInt test
    specs.push({
      name: `add [${sizes.medium.join('x')}] + [${sizes.medium.join('x')}] (uint64)`,
      category: 'bigint',
      operation: 'add',
      setup: {
        a: { shape: sizes.medium, fill: 'arange', dtype: 'uint64' },
        b: { shape: sizes.medium, fill: 'ones', dtype: 'uint64' },
      },
      iterations,
      warmup,
    });
  }

  // New functions benchmarks
  // Trigonometric conversions
  specs.push({
    name: `deg2rad [${sizes.small}]`,
    category: 'trig',
    operation: 'deg2rad',
    setup: {
      a: { shape: [sizes.small], fill: 'arange', value: 0 },
    },
    iterations,
    warmup,
  });

  specs.push({
    name: `rad2deg [${sizes.small}]`,
    category: 'trig',
    operation: 'rad2deg',
    setup: {
      a: { shape: [sizes.small], fill: 'arange', value: 0 },
    },
    iterations,
    warmup,
  });

  if (Array.isArray(sizes.medium)) {
    // Linear algebra operations
    specs.push({
      name: `diagonal [${sizes.medium.join('x')}]`,
      category: 'linalg',
      operation: 'diagonal',
      setup: {
        a: { shape: sizes.medium, fill: 'arange' },
      },
      iterations,
      warmup,
    });

    // Kron produces large outputs, so use smaller inputs (10x10 -> 100x100 output)
    const kronSize = [10, 10] as [number, number];
    specs.push({
      name: `kron [${kronSize.join('x')}] ⊗ [${kronSize.join('x')}]`,
      category: 'linalg',
      operation: 'kron',
      setup: {
        a: { shape: kronSize, fill: 'arange' },
        b: { shape: kronSize, fill: 'ones' },
      },
      iterations,
      warmup,
    });

    // New array creation benchmarks
    specs.push({
      name: `diag [${sizes.medium[0]}]`,
      category: 'creation',
      operation: 'diag',
      setup: {
        a: { shape: [sizes.medium[0]!], fill: 'arange' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `tri [${sizes.medium.join('x')}]`,
      category: 'creation',
      operation: 'tri',
      setup: {
        shape: { shape: sizes.medium },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `tril [${sizes.medium.join('x')}]`,
      category: 'creation',
      operation: 'tril',
      setup: {
        a: { shape: sizes.medium, fill: 'arange' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `triu [${sizes.medium.join('x')}]`,
      category: 'creation',
      operation: 'triu',
      setup: {
        a: { shape: sizes.medium, fill: 'arange' },
      },
      iterations,
      warmup,
    });

    // New array manipulation benchmarks
    specs.push({
      name: `flip [${sizes.medium.join('x')}]`,
      category: 'manipulation',
      operation: 'flip',
      setup: {
        a: { shape: sizes.medium, fill: 'arange' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `rot90 [${sizes.medium.join('x')}]`,
      category: 'manipulation',
      operation: 'rot90',
      setup: {
        a: { shape: sizes.medium, fill: 'arange' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `roll [${sizes.medium.join('x')}] shift=10`,
      category: 'manipulation',
      operation: 'roll',
      setup: {
        a: { shape: sizes.medium, fill: 'arange' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `pad [${sizes.medium.join('x')}] width=2`,
      category: 'manipulation',
      operation: 'pad',
      setup: {
        a: { shape: sizes.medium, fill: 'arange' },
      },
      iterations,
      warmup,
    });

    // einsum - matrix multiplication
    const einsumSize = [50, 50] as [number, number];
    specs.push({
      name: `einsum matmul [${einsumSize.join('x')}]`,
      category: 'linalg',
      operation: 'einsum',
      setup: {
        subscripts: { shape: [], value: 'ij,jk->ik' },
        a: { shape: einsumSize, fill: 'arange' },
        b: { shape: einsumSize, fill: 'ones' },
      },
      iterations,
      warmup,
    });

    // ========================================
    // numpy.linalg Module Benchmarks
    // ========================================

    // Use smaller sizes for O(n³) operations
    // These benchmarks use special 'invertible' fill mode handled in runner
    const linalgSize = [50, 50] as [number, number];
    const linalgN = linalgSize[0]!;

    specs.push({
      name: `linalg.det [${linalgN}x${linalgN}]`,
      category: 'linalg',
      operation: 'linalg_det',
      setup: {
        a: { shape: linalgSize, fill: 'invertible', dtype: 'float64' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `linalg.inv [${linalgN}x${linalgN}]`,
      category: 'linalg',
      operation: 'linalg_inv',
      setup: {
        a: { shape: linalgSize, fill: 'invertible', dtype: 'float64' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `linalg.solve [${linalgN}x${linalgN}]`,
      category: 'linalg',
      operation: 'linalg_solve',
      setup: {
        a: { shape: linalgSize, fill: 'invertible', dtype: 'float64' },
        b: { shape: [linalgN], fill: 'ones', dtype: 'float64' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `linalg.qr [${linalgN}x${linalgN}]`,
      category: 'linalg',
      operation: 'linalg_qr',
      setup: {
        a: { shape: linalgSize, fill: 'arange', dtype: 'float64' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `linalg.cholesky [${linalgN}x${linalgN}]`,
      category: 'linalg',
      operation: 'linalg_cholesky',
      setup: {
        // Setup will create a positive definite matrix in runner
        a: { shape: linalgSize, fill: 'arange', dtype: 'float64' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `linalg.svd [${linalgN}x${linalgN}]`,
      category: 'linalg',
      operation: 'linalg_svd',
      setup: {
        a: { shape: linalgSize, fill: 'arange', dtype: 'float64' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `linalg.eigh [${linalgN}x${linalgN}]`,
      category: 'linalg',
      operation: 'linalg_eigh',
      setup: {
        // Will be made symmetric in runner (eigh is for symmetric matrices)
        a: { shape: linalgSize, fill: 'arange', dtype: 'float64' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `linalg.norm [${linalgN}x${linalgN}]`,
      category: 'linalg',
      operation: 'linalg_norm',
      setup: {
        a: { shape: linalgSize, fill: 'arange', dtype: 'float64' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `linalg.matrix_rank [${linalgN}x${linalgN}]`,
      category: 'linalg',
      operation: 'linalg_matrix_rank',
      setup: {
        a: { shape: linalgSize, fill: 'arange', dtype: 'float64' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `linalg.pinv [${linalgN}x${linalgN}]`,
      category: 'linalg',
      operation: 'linalg_pinv',
      setup: {
        a: { shape: linalgSize, fill: 'arange', dtype: 'float64' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `linalg.cond [${linalgN}x${linalgN}]`,
      category: 'linalg',
      operation: 'linalg_cond',
      setup: {
        a: { shape: linalgSize, fill: 'arange', dtype: 'float64' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `linalg.matrix_power [${linalgN}x${linalgN}] n=3`,
      category: 'linalg',
      operation: 'linalg_matrix_power',
      setup: {
        a: { shape: linalgSize, fill: 'arange', dtype: 'float64' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `linalg.lstsq [${linalgN}x${linalgN}]`,
      category: 'linalg',
      operation: 'linalg_lstsq',
      setup: {
        a: { shape: linalgSize, fill: 'arange', dtype: 'float64' },
        b: { shape: [linalgN], fill: 'ones', dtype: 'float64' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `linalg.cross [3]`,
      category: 'linalg',
      operation: 'linalg_cross',
      setup: {
        a: { shape: [3], fill: 'arange', dtype: 'float64' },
        b: { shape: [3], fill: 'ones', dtype: 'float64' },
      },
      iterations,
      warmup,
    });

    // New linalg functions
    specs.push({
      name: `linalg.slogdet [${linalgN}x${linalgN}]`,
      category: 'linalg',
      operation: 'linalg_slogdet',
      setup: {
        a: { shape: linalgSize, fill: 'arange', dtype: 'float64' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `linalg.svdvals [${linalgN}x${linalgN}]`,
      category: 'linalg',
      operation: 'linalg_svdvals',
      setup: {
        a: { shape: linalgSize, fill: 'arange', dtype: 'float64' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `linalg.multi_dot [${linalgN}x${linalgN}] x3`,
      category: 'linalg',
      operation: 'linalg_multi_dot',
      setup: {
        a: { shape: linalgSize, fill: 'arange', dtype: 'float64' },
        b: { shape: linalgSize, fill: 'arange', dtype: 'float64' },
        c: { shape: linalgSize, fill: 'arange', dtype: 'float64' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `vdot [1000]`,
      category: 'linalg',
      operation: 'vdot',
      setup: {
        a: { shape: [1000], fill: 'arange', dtype: 'float64' },
        b: { shape: [1000], fill: 'ones', dtype: 'float64' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `vecdot [${sizes.medium.join('x')}]`,
      category: 'linalg',
      operation: 'vecdot',
      setup: {
        a: { shape: sizes.medium, fill: 'arange', dtype: 'float64' },
        b: { shape: sizes.medium, fill: 'ones', dtype: 'float64' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `matrix_transpose [${sizes.medium.join('x')}]`,
      category: 'linalg',
      operation: 'matrix_transpose',
      setup: {
        a: { shape: sizes.medium, fill: 'arange', dtype: 'float64' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `matvec [${linalgN}x${linalgN}] · [${linalgN}]`,
      category: 'linalg',
      operation: 'matvec',
      setup: {
        a: { shape: linalgSize, fill: 'arange', dtype: 'float64' },
        b: { shape: [linalgN], fill: 'ones', dtype: 'float64' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `vecmat [${linalgN}] · [${linalgN}x${linalgN}]`,
      category: 'linalg',
      operation: 'vecmat',
      setup: {
        a: { shape: [linalgN], fill: 'arange', dtype: 'float64' },
        b: { shape: linalgSize, fill: 'ones', dtype: 'float64' },
      },
      iterations,
      warmup,
    });

    // Indexing benchmarks
    specs.push({
      name: `take_along_axis [${sizes.medium.join('x')}]`,
      category: 'indexing',
      operation: 'take_along_axis',
      setup: {
        a: { shape: sizes.medium, fill: 'arange' },
        // indices array with same shape
        b: { shape: sizes.medium, fill: 'zeros', dtype: 'int32' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `compress [${sizes.medium.join('x')}]`,
      category: 'indexing',
      operation: 'compress',
      setup: {
        a: { shape: sizes.medium, fill: 'arange' },
        b: { shape: [sizes.medium[0]!], fill: 'ones', dtype: 'int32' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `diag_indices n=${sizes.medium[0]}`,
      category: 'indexing',
      operation: 'diag_indices',
      setup: {
        n: { shape: [sizes.medium[0]!] },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `tril_indices n=${sizes.medium[0]}`,
      category: 'indexing',
      operation: 'tril_indices',
      setup: {
        n: { shape: [sizes.medium[0]!] },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `triu_indices n=${sizes.medium[0]}`,
      category: 'indexing',
      operation: 'triu_indices',
      setup: {
        n: { shape: [sizes.medium[0]!] },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `indices [${sizes.medium.join('x')}]`,
      category: 'indexing',
      operation: 'indices',
      setup: {
        shape: { shape: sizes.medium },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `ravel_multi_index [${sizes.small}]`,
      category: 'indexing',
      operation: 'ravel_multi_index',
      setup: {
        // 1D index arrays
        a: { shape: [sizes.small], fill: 'zeros', dtype: 'int32' },
        b: { shape: [sizes.small], fill: 'zeros', dtype: 'int32' },
        dims: { shape: [100, 100] },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `unravel_index [${sizes.small}]`,
      category: 'indexing',
      operation: 'unravel_index',
      setup: {
        a: { shape: [sizes.small], fill: 'arange', dtype: 'int32' },
        dims: { shape: [100, 100] },
      },
      iterations,
      warmup,
    });

    // ========================================
    // Bitwise Operations Benchmarks
    // ========================================

    specs.push({
      name: `bitwise_and [${sizes.medium.join('x')}] & [${sizes.medium.join('x')}]`,
      category: 'bitwise',
      operation: 'bitwise_and',
      setup: {
        a: { shape: sizes.medium, fill: 'arange', dtype: 'int32' },
        b: { shape: sizes.medium, fill: 'arange', dtype: 'int32' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `bitwise_or [${sizes.medium.join('x')}] | [${sizes.medium.join('x')}]`,
      category: 'bitwise',
      operation: 'bitwise_or',
      setup: {
        a: { shape: sizes.medium, fill: 'arange', dtype: 'int32' },
        b: { shape: sizes.medium, fill: 'arange', dtype: 'int32' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `bitwise_xor [${sizes.medium.join('x')}] ^ [${sizes.medium.join('x')}]`,
      category: 'bitwise',
      operation: 'bitwise_xor',
      setup: {
        a: { shape: sizes.medium, fill: 'arange', dtype: 'int32' },
        b: { shape: sizes.medium, fill: 'arange', dtype: 'int32' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `bitwise_not [${sizes.medium.join('x')}]`,
      category: 'bitwise',
      operation: 'bitwise_not',
      setup: {
        a: { shape: sizes.medium, fill: 'arange', dtype: 'int32' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `invert [${sizes.medium.join('x')}]`,
      category: 'bitwise',
      operation: 'invert',
      setup: {
        a: { shape: sizes.medium, fill: 'arange', dtype: 'int32' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `left_shift [${sizes.medium.join('x')}] << 2`,
      category: 'bitwise',
      operation: 'left_shift',
      setup: {
        a: { shape: sizes.medium, fill: 'arange', dtype: 'int32' },
        b: { shape: [1], value: 2, dtype: 'int32' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `right_shift [${sizes.medium.join('x')}] >> 2`,
      category: 'bitwise',
      operation: 'right_shift',
      setup: {
        a: { shape: sizes.medium, fill: 'arange', dtype: 'int32' },
        b: { shape: [1], value: 2, dtype: 'int32' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `packbits [${sizes.small}]`,
      category: 'bitwise',
      operation: 'packbits',
      setup: {
        a: { shape: [sizes.small], fill: 'arange', dtype: 'uint8' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `unpackbits [${Math.ceil(sizes.small / 8)}]`,
      category: 'bitwise',
      operation: 'unpackbits',
      setup: {
        a: { shape: [Math.ceil(sizes.small / 8)], fill: 'arange', dtype: 'uint8' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `bitwise_count [${sizes.medium.join('x')}]`,
      category: 'bitwise',
      operation: 'bitwise_count',
      setup: {
        a: { shape: sizes.medium, fill: 'arange', dtype: 'uint32' },
      },
      iterations,
      warmup,
    });

    // ========================================
    // Sorting Benchmarks
    // ========================================

    specs.push({
      name: `sort [${sizes.medium.join('x')}]`,
      category: 'sorting',
      operation: 'sort',
      setup: {
        a: { shape: sizes.medium, fill: 'arange' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `argsort [${sizes.medium.join('x')}]`,
      category: 'sorting',
      operation: 'argsort',
      setup: {
        a: { shape: sizes.medium, fill: 'arange' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `partition [${sizes.medium.join('x')}] kth=${Math.floor(sizes.medium[0]! / 2)}`,
      category: 'sorting',
      operation: 'partition',
      setup: {
        a: { shape: sizes.medium, fill: 'arange' },
        kth: { shape: [Math.floor(sizes.medium[0]! / 2)] },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `argpartition [${sizes.medium.join('x')}] kth=${Math.floor(sizes.medium[0]! / 2)}`,
      category: 'sorting',
      operation: 'argpartition',
      setup: {
        a: { shape: sizes.medium, fill: 'arange' },
        kth: { shape: [Math.floor(sizes.medium[0]! / 2)] },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `lexsort [${sizes.small}] x 2 keys`,
      category: 'sorting',
      operation: 'lexsort',
      setup: {
        a: { shape: [sizes.small], fill: 'arange' },
        b: { shape: [sizes.small], fill: 'arange' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `sort_complex [${sizes.small}]`,
      category: 'sorting',
      operation: 'sort_complex',
      setup: {
        a: { shape: [sizes.small], fill: 'arange' },
      },
      iterations,
      warmup,
    });

    // ========================================
    // Searching Benchmarks
    // ========================================

    specs.push({
      name: `nonzero [${sizes.medium.join('x')}]`,
      category: 'searching',
      operation: 'nonzero',
      setup: {
        a: { shape: sizes.medium, fill: 'arange' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `argwhere [${sizes.medium.join('x')}]`,
      category: 'searching',
      operation: 'argwhere',
      setup: {
        a: { shape: sizes.medium, fill: 'arange' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `flatnonzero [${sizes.medium.join('x')}]`,
      category: 'searching',
      operation: 'flatnonzero',
      setup: {
        a: { shape: sizes.medium, fill: 'arange' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `where [${sizes.medium.join('x')}] with x,y`,
      category: 'searching',
      operation: 'where',
      setup: {
        a: { shape: sizes.medium, fill: 'ones', dtype: 'int32' },
        b: { shape: sizes.medium, fill: 'arange' },
        c: { shape: sizes.medium, fill: 'zeros' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `searchsorted [${sizes.small}]`,
      category: 'searching',
      operation: 'searchsorted',
      setup: {
        a: { shape: [sizes.small], fill: 'arange' },
        b: { shape: [100], fill: 'arange' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `extract [${sizes.medium.join('x')}]`,
      category: 'searching',
      operation: 'extract',
      setup: {
        condition: { shape: sizes.medium, fill: 'ones', dtype: 'int32' },
        a: { shape: sizes.medium, fill: 'arange' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `count_nonzero [${sizes.medium.join('x')}]`,
      category: 'searching',
      operation: 'count_nonzero',
      setup: {
        a: { shape: sizes.medium, fill: 'arange' },
      },
      iterations,
      warmup,
    });

    // ========================================
    // Statistics Benchmarks
    // ========================================

    specs.push({
      name: `bincount [${sizes.small}]`,
      category: 'statistics',
      operation: 'bincount',
      setup: {
        a: { shape: [sizes.small], fill: 'arange', dtype: 'int32' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `digitize [${sizes.small}]`,
      category: 'statistics',
      operation: 'digitize',
      setup: {
        a: { shape: [sizes.small], fill: 'arange' },
        b: { shape: [100], fill: 'arange' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `histogram [${sizes.small}]`,
      category: 'statistics',
      operation: 'histogram',
      setup: {
        a: { shape: [sizes.small], fill: 'arange' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `histogram2d [${sizes.small}]`,
      category: 'statistics',
      operation: 'histogram2d',
      setup: {
        a: { shape: [sizes.small], fill: 'arange' },
        b: { shape: [sizes.small], fill: 'arange' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `correlate [${sizes.small}]`,
      category: 'statistics',
      operation: 'correlate',
      setup: {
        a: { shape: [sizes.small], fill: 'arange' },
        b: { shape: [100], fill: 'ones' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `convolve [${sizes.small}]`,
      category: 'statistics',
      operation: 'convolve',
      setup: {
        a: { shape: [sizes.small], fill: 'arange' },
        b: { shape: [100], fill: 'ones' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `cov [${sizes.medium.join('x')}]`,
      category: 'statistics',
      operation: 'cov',
      setup: {
        a: { shape: sizes.medium, fill: 'arange' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `corrcoef [${sizes.medium.join('x')}]`,
      category: 'statistics',
      operation: 'corrcoef',
      setup: {
        a: { shape: sizes.medium, fill: 'arange' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `histogram_bin_edges [${sizes.small}]`,
      category: 'statistics',
      operation: 'histogram_bin_edges',
      setup: {
        a: { shape: [sizes.small], fill: 'arange' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `trapezoid [${sizes.small}]`,
      category: 'statistics',
      operation: 'trapezoid',
      setup: {
        a: { shape: [sizes.small], fill: 'arange' },
      },
      iterations,
      warmup,
    });

    // Set operation benchmarks
    specs.push({
      name: `trim_zeros [${sizes.small}]`,
      category: 'sets',
      operation: 'trim_zeros',
      setup: {
        a: { shape: [sizes.small], fill: 'arange' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `unique_values [${sizes.small}]`,
      category: 'sets',
      operation: 'unique_values',
      setup: {
        a: { shape: [sizes.small], fill: 'arange' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `unique_counts [${sizes.small}]`,
      category: 'sets',
      operation: 'unique_counts',
      setup: {
        a: { shape: [sizes.small], fill: 'arange' },
      },
      iterations,
      warmup,
    });

    // ========================================
    // Logic Benchmarks
    // ========================================

    specs.push({
      name: `logical_and [${sizes.medium.join('x')}] & scalar`,
      category: 'logic',
      operation: 'logical_and',
      setup: {
        a: { shape: sizes.medium, fill: 'arange' },
        scalar: { shape: [1], value: 1 },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `logical_and [${sizes.medium.join('x')}] & [${sizes.medium.join('x')}]`,
      category: 'logic',
      operation: 'logical_and',
      setup: {
        a: { shape: sizes.medium, fill: 'arange' },
        b: { shape: sizes.medium, fill: 'ones' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `logical_or [${sizes.medium.join('x')}] | scalar`,
      category: 'logic',
      operation: 'logical_or',
      setup: {
        a: { shape: sizes.medium, fill: 'arange' },
        scalar: { shape: [1], value: 0 },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `logical_not [${sizes.medium.join('x')}]`,
      category: 'logic',
      operation: 'logical_not',
      setup: {
        a: { shape: sizes.medium, fill: 'arange' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `logical_xor [${sizes.medium.join('x')}] ^ scalar`,
      category: 'logic',
      operation: 'logical_xor',
      setup: {
        a: { shape: sizes.medium, fill: 'arange' },
        scalar: { shape: [1], value: 1 },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `isfinite [${sizes.medium.join('x')}]`,
      category: 'logic',
      operation: 'isfinite',
      setup: {
        a: { shape: sizes.medium, fill: 'arange' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `isnan [${sizes.medium.join('x')}]`,
      category: 'logic',
      operation: 'isnan',
      setup: {
        a: { shape: sizes.medium, fill: 'arange' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `signbit [${sizes.medium.join('x')}]`,
      category: 'logic',
      operation: 'signbit',
      setup: {
        a: { shape: sizes.medium, fill: 'arange', value: -50 },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `copysign [${sizes.medium.join('x')}] scalar`,
      category: 'logic',
      operation: 'copysign',
      setup: {
        a: { shape: sizes.medium, fill: 'arange' },
        scalar: { shape: [1], value: -1 },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `isneginf [${sizes.medium.join('x')}]`,
      category: 'logic',
      operation: 'isneginf',
      setup: {
        a: { shape: sizes.medium, fill: 'arange' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `isposinf [${sizes.medium.join('x')}]`,
      category: 'logic',
      operation: 'isposinf',
      setup: {
        a: { shape: sizes.medium, fill: 'arange' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `isreal [${sizes.medium.join('x')}]`,
      category: 'logic',
      operation: 'isreal',
      setup: {
        a: { shape: sizes.medium, fill: 'arange' },
      },
      iterations,
      warmup,
    });

    // ========================================
    // Random Benchmarks
    // ========================================

    specs.push({
      name: `random.random [${sizes.medium.join('x')}]`,
      category: 'random',
      operation: 'random_random',
      setup: {
        shape: { shape: sizes.medium },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `random.rand [${sizes.medium.join('x')}]`,
      category: 'random',
      operation: 'random_rand',
      setup: {
        shape: { shape: sizes.medium },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `random.randn [${sizes.medium.join('x')}]`,
      category: 'random',
      operation: 'random_randn',
      setup: {
        shape: { shape: sizes.medium },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `random.randint [${sizes.medium.join('x')}]`,
      category: 'random',
      operation: 'random_randint',
      setup: {
        shape: { shape: sizes.medium },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `random.uniform [${sizes.medium.join('x')}]`,
      category: 'random',
      operation: 'random_uniform',
      setup: {
        shape: { shape: sizes.medium },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `random.normal [${sizes.medium.join('x')}]`,
      category: 'random',
      operation: 'random_normal',
      setup: {
        shape: { shape: sizes.medium },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `random.standard_normal [${sizes.medium.join('x')}]`,
      category: 'random',
      operation: 'random_standard_normal',
      setup: {
        shape: { shape: sizes.medium },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `random.exponential [${sizes.medium.join('x')}]`,
      category: 'random',
      operation: 'random_exponential',
      setup: {
        shape: { shape: sizes.medium },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `random.poisson [${sizes.medium.join('x')}]`,
      category: 'random',
      operation: 'random_poisson',
      setup: {
        shape: { shape: sizes.medium },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `random.binomial [${sizes.medium.join('x')}]`,
      category: 'random',
      operation: 'random_binomial',
      setup: {
        shape: { shape: sizes.medium },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `random.choice [${sizes.small}]`,
      category: 'random',
      operation: 'random_choice',
      setup: {
        n: { shape: [sizes.small] },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `random.permutation [${sizes.small}]`,
      category: 'random',
      operation: 'random_permutation',
      setup: {
        n: { shape: [sizes.small] },
      },
      iterations,
      warmup,
    });

    // ========================================
    // Complex Number Benchmarks
    // ========================================

    specs.push({
      name: `zeros [${sizes.medium.join('x')}] (complex128)`,
      category: 'complex',
      operation: 'complex_zeros',
      setup: {
        shape: { shape: sizes.medium },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `ones [${sizes.medium.join('x')}] (complex128)`,
      category: 'complex',
      operation: 'complex_ones',
      setup: {
        shape: { shape: sizes.medium },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `add [${sizes.medium.join('x')}] + [${sizes.medium.join('x')}] (complex128)`,
      category: 'complex',
      operation: 'complex_add',
      setup: {
        a: { shape: sizes.medium, fill: 'complex' },
        b: { shape: sizes.medium, fill: 'complex' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `multiply [${sizes.medium.join('x')}] * [${sizes.medium.join('x')}] (complex128)`,
      category: 'complex',
      operation: 'complex_multiply',
      setup: {
        a: { shape: sizes.medium, fill: 'complex' },
        b: { shape: sizes.medium, fill: 'complex' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `divide [${sizes.medium.join('x')}] / [${sizes.medium.join('x')}] (complex128)`,
      category: 'complex',
      operation: 'complex_divide',
      setup: {
        a: { shape: sizes.medium, fill: 'complex' },
        b: { shape: sizes.medium, fill: 'complex' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `real [${sizes.medium.join('x')}] (complex128)`,
      category: 'complex',
      operation: 'complex_real',
      setup: {
        a: { shape: sizes.medium, fill: 'complex' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `imag [${sizes.medium.join('x')}] (complex128)`,
      category: 'complex',
      operation: 'complex_imag',
      setup: {
        a: { shape: sizes.medium, fill: 'complex' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `conj [${sizes.medium.join('x')}] (complex128)`,
      category: 'complex',
      operation: 'complex_conj',
      setup: {
        a: { shape: sizes.medium, fill: 'complex' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `angle [${sizes.medium.join('x')}] (complex128)`,
      category: 'complex',
      operation: 'complex_angle',
      setup: {
        a: { shape: sizes.medium, fill: 'complex' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `abs [${sizes.medium.join('x')}] (complex128)`,
      category: 'complex',
      operation: 'complex_abs',
      setup: {
        a: { shape: sizes.medium, fill: 'complex' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `sqrt [${sizes.medium.join('x')}] (complex128)`,
      category: 'complex',
      operation: 'complex_sqrt',
      setup: {
        a: { shape: sizes.medium, fill: 'complex' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `sum [${sizes.medium.join('x')}] (complex128)`,
      category: 'complex',
      operation: 'complex_sum',
      setup: {
        a: { shape: sizes.medium, fill: 'complex' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `mean [${sizes.medium.join('x')}] (complex128)`,
      category: 'complex',
      operation: 'complex_mean',
      setup: {
        a: { shape: sizes.medium, fill: 'complex' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `prod [${sizes.medium.join('x')}] (complex128)`,
      category: 'complex',
      operation: 'complex_prod',
      setup: {
        a: { shape: sizes.medium, fill: 'complex' },
      },
      iterations,
      warmup,
    });

    // ========================================
    // Polynomial Benchmarks
    // ========================================

    specs.push({
      name: `poly [10 roots]`,
      category: 'polynomials',
      operation: 'poly',
      setup: {
        a: { shape: [10], fill: 'arange' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `polyadd [10] + [10]`,
      category: 'polynomials',
      operation: 'polyadd',
      setup: {
        a: { shape: [10], fill: 'arange' },
        b: { shape: [10], fill: 'ones' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `polyder [10]`,
      category: 'polynomials',
      operation: 'polyder',
      setup: {
        a: { shape: [10], fill: 'arange' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `polydiv [10] / [5]`,
      category: 'polynomials',
      operation: 'polydiv',
      setup: {
        a: { shape: [10], fill: 'ones' },
        b: { shape: [5], fill: 'ones' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `polyfit [100] deg=5`,
      category: 'polynomials',
      operation: 'polyfit',
      setup: {
        a: { shape: [100], fill: 'arange' },
        b: { shape: [100], fill: 'arange' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `polyint [10]`,
      category: 'polynomials',
      operation: 'polyint',
      setup: {
        a: { shape: [10], fill: 'arange' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `polymul [10] * [10]`,
      category: 'polynomials',
      operation: 'polymul',
      setup: {
        a: { shape: [10], fill: 'ones' },
        b: { shape: [10], fill: 'ones' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `polysub [10] - [10]`,
      category: 'polynomials',
      operation: 'polysub',
      setup: {
        a: { shape: [10], fill: 'arange' },
        b: { shape: [10], fill: 'ones' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `polyval [10] at [100 points]`,
      category: 'polynomials',
      operation: 'polyval',
      setup: {
        a: { shape: [10], fill: 'arange' },
        b: { shape: [100], fill: 'arange' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `roots [5]`,
      category: 'polynomials',
      operation: 'roots',
      setup: {
        a: { shape: [5], fill: 'ones' },
      },
      iterations,
      warmup,
    });

    // ========================================
    // Type Checking Benchmarks
    // ========================================

    specs.push({
      name: `can_cast`,
      category: 'type_checking',
      operation: 'can_cast',
      setup: {},
      iterations,
      warmup,
    });

    specs.push({
      name: `result_type`,
      category: 'type_checking',
      operation: 'result_type',
      setup: {},
      iterations,
      warmup,
    });

    specs.push({
      name: `min_scalar_type`,
      category: 'type_checking',
      operation: 'min_scalar_type',
      setup: {},
      iterations,
      warmup,
    });

    specs.push({
      name: `issubdtype`,
      category: 'type_checking',
      operation: 'issubdtype',
      setup: {},
      iterations,
      warmup,
    });
  }

  // ========================================
  // Other Math Benchmarks
  // ========================================

  if (Array.isArray(sizes.medium)) {
    specs.push({
      name: `clip [${sizes.medium.join('x')}]`,
      category: 'other_math',
      operation: 'clip',
      setup: {
        a: { shape: sizes.medium, fill: 'arange' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `maximum [${sizes.medium.join('x')}]`,
      category: 'other_math',
      operation: 'maximum',
      setup: {
        a: { shape: sizes.medium, fill: 'random' },
        b: { shape: sizes.medium, fill: 'random' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `minimum [${sizes.medium.join('x')}]`,
      category: 'other_math',
      operation: 'minimum',
      setup: {
        a: { shape: sizes.medium, fill: 'random' },
        b: { shape: sizes.medium, fill: 'random' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `fmax [${sizes.medium.join('x')}]`,
      category: 'other_math',
      operation: 'fmax',
      setup: {
        a: { shape: sizes.medium, fill: 'random' },
        b: { shape: sizes.medium, fill: 'random' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `fmin [${sizes.medium.join('x')}]`,
      category: 'other_math',
      operation: 'fmin',
      setup: {
        a: { shape: sizes.medium, fill: 'random' },
        b: { shape: sizes.medium, fill: 'random' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `nan_to_num [${sizes.medium.join('x')}]`,
      category: 'other_math',
      operation: 'nan_to_num',
      setup: {
        a: { shape: sizes.medium, fill: 'random' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `sinc [${sizes.medium.join('x')}]`,
      category: 'other_math',
      operation: 'sinc',
      setup: {
        a: { shape: sizes.medium, fill: 'random' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `i0 [${sizes.medium.join('x')}]`,
      category: 'other_math',
      operation: 'i0',
      setup: {
        a: { shape: sizes.medium, fill: 'random' },
      },
      iterations,
      warmup,
    });
  }

  specs.push({
    name: `interp [${sizes.small}]`,
    category: 'other_math',
    operation: 'interp',
    setup: {
      x: { shape: [sizes.small], fill: 'arange' },
      xp: { shape: [100], fill: 'arange' },
      fp: { shape: [100], fill: 'arange' },
    },
    iterations,
    warmup,
  });

  specs.push({
    name: `unwrap [${sizes.small}]`,
    category: 'other_math',
    operation: 'unwrap',
    setup: {
      a: { shape: [sizes.small], fill: 'random' },
    },
    iterations,
    warmup,
  });

  // ========================================
  // FFT Benchmarks
  // ========================================

  // 1D FFT operations
  specs.push({
    name: `fft [${sizes.small}]`,
    category: 'fft',
    operation: 'fft',
    setup: {
      a: { shape: [sizes.small], fill: 'arange' },
    },
    iterations,
    warmup,
  });

  specs.push({
    name: `ifft [${sizes.small}]`,
    category: 'fft',
    operation: 'ifft',
    setup: {
      a: { shape: [sizes.small], fill: 'complex' },
    },
    iterations,
    warmup,
  });

  specs.push({
    name: `rfft [${sizes.small}]`,
    category: 'fft',
    operation: 'rfft',
    setup: {
      a: { shape: [sizes.small], fill: 'arange' },
    },
    iterations,
    warmup,
  });

  specs.push({
    name: `irfft [${sizes.small / 2 + 1}]`,
    category: 'fft',
    operation: 'irfft',
    setup: {
      a: { shape: [Math.floor(sizes.small / 2) + 1], fill: 'complex' },
    },
    iterations,
    warmup,
  });

  specs.push({
    name: `hfft [${sizes.small / 2 + 1}]`,
    category: 'fft',
    operation: 'hfft',
    setup: {
      a: { shape: [Math.floor(sizes.small / 2) + 1], fill: 'complex' },
    },
    iterations,
    warmup,
  });

  specs.push({
    name: `ihfft [${sizes.small}]`,
    category: 'fft',
    operation: 'ihfft',
    setup: {
      a: { shape: [sizes.small], fill: 'arange' },
    },
    iterations,
    warmup,
  });

  // 2D FFT operations
  if (Array.isArray(sizes.medium)) {
    specs.push({
      name: `fft2 [${sizes.medium.join('x')}]`,
      category: 'fft',
      operation: 'fft2',
      setup: {
        a: { shape: sizes.medium, fill: 'arange' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `ifft2 [${sizes.medium.join('x')}]`,
      category: 'fft',
      operation: 'ifft2',
      setup: {
        a: { shape: sizes.medium, fill: 'complex' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `rfft2 [${sizes.medium.join('x')}]`,
      category: 'fft',
      operation: 'rfft2',
      setup: {
        a: { shape: sizes.medium, fill: 'arange' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `irfft2 [${sizes.medium[0]}x${Math.floor(sizes.medium[1] / 2) + 1}]`,
      category: 'fft',
      operation: 'irfft2',
      setup: {
        a: { shape: [sizes.medium[0], Math.floor(sizes.medium[1] / 2) + 1], fill: 'complex' },
      },
      iterations,
      warmup,
    });
  }

  // N-D FFT operations (use 3D)
  const fft3dSize = [32, 32, 32];
  specs.push({
    name: `fftn [${fft3dSize.join('x')}]`,
    category: 'fft',
    operation: 'fftn',
    setup: {
      a: { shape: fft3dSize, fill: 'arange' },
    },
    iterations,
    warmup,
  });

  specs.push({
    name: `ifftn [${fft3dSize.join('x')}]`,
    category: 'fft',
    operation: 'ifftn',
    setup: {
      a: { shape: fft3dSize, fill: 'complex' },
    },
    iterations,
    warmup,
  });

  specs.push({
    name: `rfftn [${fft3dSize.join('x')}]`,
    category: 'fft',
    operation: 'rfftn',
    setup: {
      a: { shape: fft3dSize, fill: 'arange' },
    },
    iterations,
    warmup,
  });

  specs.push({
    name: `irfftn [${fft3dSize[0]}x${fft3dSize[1]}x${Math.floor(fft3dSize[2] / 2) + 1}]`,
    category: 'fft',
    operation: 'irfftn',
    setup: {
      a: { shape: [fft3dSize[0], fft3dSize[1], Math.floor(fft3dSize[2] / 2) + 1], fill: 'complex' },
    },
    iterations,
    warmup,
  });

  // FFT utility functions
  specs.push({
    name: `fftfreq(${sizes.small})`,
    category: 'fft',
    operation: 'fftfreq',
    setup: {
      n: { shape: [sizes.small] },
    },
    iterations,
    warmup,
  });

  specs.push({
    name: `rfftfreq(${sizes.small})`,
    category: 'fft',
    operation: 'rfftfreq',
    setup: {
      n: { shape: [sizes.small] },
    },
    iterations,
    warmup,
  });

  specs.push({
    name: `fftshift [${sizes.small}]`,
    category: 'fft',
    operation: 'fftshift',
    setup: {
      a: { shape: [sizes.small], fill: 'arange' },
    },
    iterations,
    warmup,
  });

  specs.push({
    name: `ifftshift [${sizes.small}]`,
    category: 'fft',
    operation: 'ifftshift',
    setup: {
      a: { shape: [sizes.small], fill: 'arange' },
    },
    iterations,
    warmup,
  });

  return specs;
}

export function filterByCategory(specs: BenchmarkCase[], category: string): BenchmarkCase[] {
  return specs.filter((spec) => spec.category === category);
}

export function getCategories(specs: BenchmarkCase[]): string[] {
  const categories = new Set(specs.map((spec) => spec.category));
  return Array.from(categories).sort();
}
