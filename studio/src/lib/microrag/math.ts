/** Distance kernels ported from micro-rag `math.rs` (4-wide unroll, cosine in [-1, 1]). */

export class VectorError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "VectorError";
  }
}

export function validateDimensions(a: number[], b: number[]): void {
  if (a.length === 0 || b.length === 0) {
    throw new VectorError("Vector cannot be empty (zero dimensions)");
  }
  if (a.length !== b.length) {
    throw new VectorError(`Dimension mismatch: expected ${a.length}, got ${b.length}`);
  }
}

export function dotProduct(a: number[], b: number[]): number {
  validateDimensions(a, b);
  const len = a.length;
  const chunks = Math.floor(len / 4);
  let sum0 = 0;
  let sum1 = 0;
  let sum2 = 0;
  let sum3 = 0;
  for (let i = 0; i < chunks; i++) {
    const idx = i * 4;
    sum0 += a[idx] * b[idx];
    sum1 += a[idx + 1] * b[idx + 1];
    sum2 += a[idx + 2] * b[idx + 2];
    sum3 += a[idx + 3] * b[idx + 3];
  }
  let total = sum0 + sum1 + (sum2 + sum3);
  for (let i = chunks * 4; i < len; i++) total += a[i] * b[i];
  return total;
}

export function squaredEuclidean(a: number[], b: number[]): number {
  validateDimensions(a, b);
  const len = a.length;
  const chunks = Math.floor(len / 4);
  let sum0 = 0;
  let sum1 = 0;
  let sum2 = 0;
  let sum3 = 0;
  for (let i = 0; i < chunks; i++) {
    const idx = i * 4;
    const d0 = a[idx] - b[idx];
    const d1 = a[idx + 1] - b[idx + 1];
    const d2 = a[idx + 2] - b[idx + 2];
    const d3 = a[idx + 3] - b[idx + 3];
    sum0 += d0 * d0;
    sum1 += d1 * d1;
    sum2 += d2 * d2;
    sum3 += d3 * d3;
  }
  let total = sum0 + sum1 + (sum2 + sum3);
  for (let i = chunks * 4; i < len; i++) {
    const d = a[i] - b[i];
    total += d * d;
  }
  return total;
}

export function cosineSimilarity(a: number[], b: number[]): number {
  validateDimensions(a, b);
  let dot = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  if (normA <= Number.EPSILON || normB <= Number.EPSILON) {
    throw new VectorError("Zero-magnitude vector encountered during cosine similarity");
  }
  const similarity = dot / (Math.sqrt(normA) * Math.sqrt(normB));
  return Math.max(-1, Math.min(1, similarity));
}

export function l2Normalize(v: number[]): void {
  if (v.length === 0) throw new VectorError("Vector cannot be empty (zero dimensions)");
  let normSq = 0;
  for (let i = 0; i < v.length; i++) normSq += v[i] * v[i];
  if (normSq <= Number.EPSILON) {
    throw new VectorError("Zero-magnitude vector encountered during normalization");
  }
  const inv = 1 / Math.sqrt(normSq);
  for (let i = 0; i < v.length; i++) v[i] *= inv;
}

export function magnitudeSq(v: number[]): number {
  let s = 0;
  for (let i = 0; i < v.length; i++) s += v[i] * v[i];
  return s;
}
