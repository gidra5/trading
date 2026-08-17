export function circularConvolutionPower(
  probabilities: Float64Array,
  exponent: number,
): Float64Array {
  if (!Number.isInteger(exponent) || exponent < 1) {
    throw new Error(`Convolution exponent must be a positive integer, got ${exponent}.`);
  }
  if (!isPowerOfTwo(probabilities.length)) {
    throw new Error(`FFT length must be a power of two, got ${probabilities.length}.`);
  }
  const real = probabilities.slice();
  const imaginary = new Float64Array(probabilities.length);
  fftInPlace(real, imaginary, false);
  for (let index = 0; index < real.length; index += 1) {
    const magnitude = Math.min(1, Math.hypot(real[index]!, imaginary[index]!));
    const poweredMagnitude = magnitude === 0
      ? 0
      : Math.exp(exponent * Math.log(magnitude));
    const angle = exponent * Math.atan2(imaginary[index]!, real[index]!);
    real[index] = poweredMagnitude * Math.cos(angle);
    imaginary[index] = poweredMagnitude * Math.sin(angle);
  }
  fftInPlace(real, imaginary, true);
  let sum = 0;
  for (let index = 0; index < real.length; index += 1) {
    real[index] = Math.max(0, real[index]!);
    sum += real[index]!;
  }
  if (!(sum > 0)) throw new Error("Convolution produced no probability mass.");
  for (let index = 0; index < real.length; index += 1) real[index] /= sum;
  return real;
}

export function compoundCircularConvolution(
  markProbabilities: Float64Array,
  countProbabilities: readonly number[],
): Float64Array {
  if (!isPowerOfTwo(markProbabilities.length)) {
    throw new Error(`FFT length must be a power of two, got ${markProbabilities.length}.`);
  }
  if (countProbabilities.length < 1
    || countProbabilities.some((probability) => !Number.isFinite(probability) || probability < 0)) {
    throw new Error("Count probabilities must be a non-empty non-negative distribution.");
  }
  const countMass = countProbabilities.reduce((sum, probability) => sum + probability, 0);
  if (Math.abs(countMass - 1) > 1e-9) {
    throw new Error(`Count probabilities contain ${countMass} mass instead of one.`);
  }
  const real = markProbabilities.slice();
  const imaginary = new Float64Array(markProbabilities.length);
  fftInPlace(real, imaginary, false);
  for (let index = 0; index < real.length; index += 1) {
    const markReal = real[index]!;
    const markImaginary = imaginary[index]!;
    let powerReal = 1;
    let powerImaginary = 0;
    let mixtureReal = 0;
    let mixtureImaginary = 0;
    for (const probability of countProbabilities) {
      mixtureReal += probability * powerReal;
      mixtureImaginary += probability * powerImaginary;
      const nextPowerReal = powerReal * markReal - powerImaginary * markImaginary;
      powerImaginary = powerReal * markImaginary + powerImaginary * markReal;
      powerReal = nextPowerReal;
    }
    real[index] = mixtureReal;
    imaginary[index] = mixtureImaginary;
  }
  fftInPlace(real, imaginary, true);
  let mass = 0;
  for (let index = 0; index < real.length; index += 1) {
    real[index] = Math.max(0, real[index]!);
    mass += real[index]!;
  }
  if (!(mass > 0)) throw new Error("Compound convolution produced no probability mass.");
  for (let index = 0; index < real.length; index += 1) real[index] /= mass;
  return real;
}

export function fftInPlace(
  real: Float64Array,
  imaginary: Float64Array,
  inverse: boolean,
): void {
  const size = real.length;
  if (imaginary.length !== size || !isPowerOfTwo(size)) {
    throw new Error("FFT arrays must have equal power-of-two lengths.");
  }
  for (let index = 1, reversed = 0; index < size; index += 1) {
    let bit = size >> 1;
    while (reversed & bit) {
      reversed ^= bit;
      bit >>= 1;
    }
    reversed ^= bit;
    if (index < reversed) {
      [real[index], real[reversed]] = [real[reversed]!, real[index]!];
      [imaginary[index], imaginary[reversed]] = [imaginary[reversed]!, imaginary[index]!];
    }
  }
  for (let length = 2; length <= size; length <<= 1) {
    const angle = (inverse ? 2 : -2) * Math.PI / length;
    const stepReal = Math.cos(angle);
    const stepImaginary = Math.sin(angle);
    for (let start = 0; start < size; start += length) {
      let twiddleReal = 1;
      let twiddleImaginary = 0;
      const half = length >> 1;
      for (let offset = 0; offset < half; offset += 1) {
        const evenIndex = start + offset;
        const oddIndex = evenIndex + half;
        const oddReal = real[oddIndex]! * twiddleReal
          - imaginary[oddIndex]! * twiddleImaginary;
        const oddImaginary = real[oddIndex]! * twiddleImaginary
          + imaginary[oddIndex]! * twiddleReal;
        const evenReal = real[evenIndex]!;
        const evenImaginary = imaginary[evenIndex]!;
        real[evenIndex] = evenReal + oddReal;
        imaginary[evenIndex] = evenImaginary + oddImaginary;
        real[oddIndex] = evenReal - oddReal;
        imaginary[oddIndex] = evenImaginary - oddImaginary;
        const nextTwiddleReal = twiddleReal * stepReal - twiddleImaginary * stepImaginary;
        twiddleImaginary = twiddleReal * stepImaginary + twiddleImaginary * stepReal;
        twiddleReal = nextTwiddleReal;
      }
    }
  }
  if (inverse) {
    for (let index = 0; index < size; index += 1) {
      real[index] /= size;
      imaginary[index] /= size;
    }
  }
}

function isPowerOfTwo(value: number): boolean {
  return value > 0 && (value & (value - 1)) === 0;
}
