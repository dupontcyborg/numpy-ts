/**
 * Complex number class for numpy-ts
 *
 * Represents complex numbers in JavaScript, similar to Python's complex type.
 * Used when converting complex arrays to JavaScript values via toArray().
 */

/**
 * Represents a complex number with real and imaginary parts.
 *
 * @example
 * ```typescript
 * const z = new Complex(1, 2);  // 1 + 2i
 * console.log(z.re);  // 1
 * console.log(z.im);  // 2
 * console.log(z.toString());  // "(1+2j)"
 * ```
 */
export class Complex {
  /** Real part */
  readonly re: number;
  /** Imaginary part */
  readonly im: number;

  constructor(re: number, im: number = 0) {
    this.re = re;
    this.im = im;
  }

  /**
   * Returns the magnitude (absolute value) of the complex number.
   * |z| = sqrt(re² + im²)
   */
  abs(): number {
    return Math.sqrt(this.re * this.re + this.im * this.im);
  }

  /**
   * Returns the phase angle (argument) of the complex number in radians.
   * arg(z) = atan2(im, re)
   */
  angle(): number {
    return Math.atan2(this.im, this.re);
  }

  /**
   * Returns the complex conjugate.
   * conj(a + bi) = a - bi
   */
  conj(): Complex {
    return new Complex(this.re, -this.im);
  }

  /**
   * Add another complex number or real number.
   */
  add(other: Complex | number): Complex {
    if (typeof other === 'number') {
      return new Complex(this.re + other, this.im);
    }
    return new Complex(this.re + other.re, this.im + other.im);
  }

  /**
   * Subtract another complex number or real number.
   */
  sub(other: Complex | number): Complex {
    if (typeof other === 'number') {
      return new Complex(this.re - other, this.im);
    }
    return new Complex(this.re - other.re, this.im - other.im);
  }

  /**
   * Multiply by another complex number or real number.
   * (a + bi)(c + di) = (ac - bd) + (ad + bc)i
   */
  mul(other: Complex | number): Complex {
    if (typeof other === 'number') {
      return new Complex(this.re * other, this.im * other);
    }
    return new Complex(
      this.re * other.re - this.im * other.im,
      this.re * other.im + this.im * other.re
    );
  }

  /**
   * Divide by another complex number or real number.
   * (a + bi) / (c + di) = ((ac + bd) + (bc - ad)i) / (c² + d²)
   */
  div(other: Complex | number): Complex {
    if (typeof other === 'number') {
      return new Complex(this.re / other, this.im / other);
    }
    const denom = other.re * other.re + other.im * other.im;
    return new Complex(
      (this.re * other.re + this.im * other.im) / denom,
      (this.im * other.re - this.re * other.im) / denom
    );
  }

  /**
   * Returns the negation of this complex number.
   */
  neg(): Complex {
    return new Complex(-this.re, -this.im);
  }

  /**
   * Check equality with another complex number.
   */
  equals(other: Complex): boolean {
    return this.re === other.re && this.im === other.im;
  }

  /**
   * String representation matching NumPy/Python format: "(a+bj)"
   */
  toString(): string {
    if (this.im === 0) {
      return `(${this.re}+0j)`;
    }
    if (this.im < 0) {
      return `(${this.re}${this.im}j)`;
    }
    return `(${this.re}+${this.im}j)`;
  }

  /**
   * Create a Complex from various input formats.
   * Accepts:
   * - Complex instance
   * - {re, im} object
   * - [re, im] array
   * - number (creates re + 0i)
   */
  static from(value: ComplexInput): Complex {
    if (value instanceof Complex) {
      return value;
    }
    if (typeof value === 'number') {
      return new Complex(value, 0);
    }
    if (Array.isArray(value)) {
      return new Complex(value[0] ?? 0, value[1] ?? 0);
    }
    if (typeof value === 'object' && value !== null && 're' in value) {
      return new Complex(value.re ?? 0, value.im ?? 0);
    }
    throw new Error(`Cannot convert ${value} to Complex`);
  }

  /**
   * Check if a value is a complex number representation.
   */
  static isComplex(value: unknown): value is ComplexInput {
    if (value instanceof Complex) return true;
    if (typeof value === 'object' && value !== null && 're' in value && 'im' in value) {
      return true;
    }
    return false;
  }
}

/**
 * Input types that can be converted to Complex.
 */
export type ComplexInput = Complex | { re: number; im?: number } | [number, number] | number;

/**
 * Helper to check if a value looks like a complex number input.
 */
export function isComplexLike(value: unknown): value is ComplexInput {
  if (value instanceof Complex) return true;
  if (typeof value === 'object' && value !== null && 're' in value) {
    return true;
  }
  return false;
}
