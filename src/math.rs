use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Error, Debug, PartialEq)]
pub enum VectorError {
    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },
    #[error("Vector cannot be empty (zero dimensions)")]
    EmptyVector,
    #[error("Zero-magnitude vector encountered during normalization or cosine similarity")]
    ZeroMagnitudeVector,
    #[error("Vector contains non-finite value (NaN or infinity)")]
    NonFiniteValue,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Metric {
    Cosine,
    DotProduct,
    Euclidean,
}

impl Metric {
    /// Calculates the raw similarity or distance between two vectors.
    /// Higher score always means "more similar" in our unified search ranking,
    /// so Euclidean distance is negated (`-dist`) for uniform max-heap ranking,
    /// while raw distance functions remain exposed for exact geometric bounds.
    #[inline]
    pub fn score(&self, a: &[f32], b: &[f32]) -> Result<f32, VectorError> {
        match self {
            Metric::Cosine => cosine_similarity(a, b),
            Metric::DotProduct => dot_product(a, b),
            Metric::Euclidean => Ok(-euclidean_distance(a, b)?),
        }
    }
}

#[inline]
pub fn validate_dimensions(a: &[f32], b: &[f32]) -> Result<(), VectorError> {
    if a.is_empty() || b.is_empty() {
        return Err(VectorError::EmptyVector);
    }
    if a.len() != b.len() {
        return Err(VectorError::DimensionMismatch {
            expected: a.len(),
            actual: b.len(),
        });
    }
    Ok(())
}

#[inline]
pub fn validate_finite(v: &[f32]) -> Result<(), VectorError> {
    if !v.iter().all(|x| x.is_finite()) {
        return Err(VectorError::NonFiniteValue);
    }
    Ok(())
}

/// Computes the inner dot product of two slices using 4-way loop unrolling
/// to encourage LLVM auto-vectorization (SIMD).
#[inline]
pub fn dot_product(a: &[f32], b: &[f32]) -> Result<f32, VectorError> {
    validate_dimensions(a, b)?;
    validate_finite(a)?;
    validate_finite(b)?;

    let len = a.len();
    let chunks = len / 4;
    let mut sum0 = 0.0_f32;
    let mut sum1 = 0.0_f32;
    let mut sum2 = 0.0_f32;
    let mut sum3 = 0.0_f32;

    for i in 0..chunks {
        let idx = i * 4;
        sum0 += a[idx] * b[idx];
        sum1 += a[idx + 1] * b[idx + 1];
        sum2 += a[idx + 2] * b[idx + 2];
        sum3 += a[idx + 3] * b[idx + 3];
    }

    let mut total = (sum0 + sum1) + (sum2 + sum3);
    for i in (chunks * 4)..len {
        total += a[i] * b[i];
    }

    Ok(total)
}

/// Computes the squared L2 (Euclidean) distance between two vectors.
#[inline]
pub fn squared_euclidean_distance(a: &[f32], b: &[f32]) -> Result<f32, VectorError> {
    validate_dimensions(a, b)?;
    validate_finite(a)?;
    validate_finite(b)?;

    let len = a.len();
    let chunks = len / 4;
    let mut sum0 = 0.0_f32;
    let mut sum1 = 0.0_f32;
    let mut sum2 = 0.0_f32;
    let mut sum3 = 0.0_f32;

    for i in 0..chunks {
        let idx = i * 4;
        let d0 = a[idx] - b[idx];
        let d1 = a[idx + 1] - b[idx + 1];
        let d2 = a[idx + 2] - b[idx + 2];
        let d3 = a[idx + 3] - b[idx + 3];
        sum0 += d0 * d0;
        sum1 += d1 * d1;
        sum2 += d2 * d2;
        sum3 += d3 * d3;
    }

    let mut total = (sum0 + sum1) + (sum2 + sum3);
    for i in (chunks * 4)..len {
        let d = a[i] - b[i];
        total += d * d;
    }

    Ok(total)
}

/// Computes the exact L2 (Euclidean) distance between two vectors.
#[inline]
pub fn euclidean_distance(a: &[f32], b: &[f32]) -> Result<f32, VectorError> {
    Ok(squared_euclidean_distance(a, b)?.sqrt())
}

/// Computes the cosine similarity in `[-1.0, 1.0]`.
/// Returns `Err(VectorError::ZeroMagnitudeVector)` if either vector has L2 norm == 0.
#[inline]
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> Result<f32, VectorError> {
    validate_dimensions(a, b)?;
    validate_finite(a)?;
    validate_finite(b)?;

    let mut dot = 0.0_f32;
    let mut norm_a = 0.0_f32;
    let mut norm_b = 0.0_f32;

    for (&x, &y) in a.iter().zip(b.iter()) {
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }

    if norm_a <= f32::EPSILON || norm_b <= f32::EPSILON {
        return Err(VectorError::ZeroMagnitudeVector);
    }

    let similarity = dot / (norm_a.sqrt() * norm_b.sqrt());
    Ok(similarity.clamp(-1.0, 1.0))
}

/// Normalizes a vector in-place to unit L2 length.
pub fn l2_normalize(v: &mut [f32]) -> Result<(), VectorError> {
    if v.is_empty() {
        return Err(VectorError::EmptyVector);
    }
    validate_finite(v)?;
    let norm_sq: f32 = v.iter().map(|&x| x * x).sum();
    if norm_sq <= f32::EPSILON {
        return Err(VectorError::ZeroMagnitudeVector);
    }
    let inv_norm = 1.0 / norm_sq.sqrt();
    for x in v.iter_mut() {
        *x *= inv_norm;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dot_product_exact() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![5.0, 4.0, 3.0, 2.0, 1.0];
        let res = dot_product(&a, &b).unwrap();
        assert!((res - 35.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_orthogonal_and_collinear() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 5.0, 0.0];
        let c = vec![2.0, 0.0, 0.0];
        let d = vec![-3.0, 0.0, 0.0];

        assert!((cosine_similarity(&a, &b).unwrap() - 0.0).abs() < 1e-6);
        assert!((cosine_similarity(&a, &c).unwrap() - 1.0).abs() < 1e-6);
        assert!((cosine_similarity(&a, &d).unwrap() - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn test_euclidean_distance() {
        let a = vec![0.0, 0.0, 0.0];
        let b = vec![3.0, 4.0, 0.0];
        assert!((euclidean_distance(&a, &b).unwrap() - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_edge_cases_zero_vector_and_mismatch() {
        let zero = vec![0.0, 0.0, 0.0];
        let unit = vec![1.0, 0.0, 0.0];
        let short = vec![1.0, 2.0];
        let empty: Vec<f32> = vec![];

        assert_eq!(
            cosine_similarity(&zero, &unit),
            Err(VectorError::ZeroMagnitudeVector)
        );
        assert_eq!(
            dot_product(&unit, &short),
            Err(VectorError::DimensionMismatch {
                expected: 3,
                actual: 2
            })
        );
        assert_eq!(dot_product(&empty, &empty), Err(VectorError::EmptyVector));
    }

    #[test]
    fn test_rejects_nan_and_infinity() {
        let valid = vec![1.0, 2.0, 3.0];
        let invalid_cases = [
            vec![f32::NAN, 1.0, 0.0],
            vec![1.0, f32::INFINITY, 0.0],
            vec![0.0, 1.0, f32::NEG_INFINITY],
        ];

        for bad in &invalid_cases {
            assert_eq!(validate_finite(bad), Err(VectorError::NonFiniteValue));
            assert_eq!(dot_product(bad, &valid), Err(VectorError::NonFiniteValue));
            assert_eq!(dot_product(&valid, bad), Err(VectorError::NonFiniteValue));
            assert_eq!(
                squared_euclidean_distance(bad, &valid),
                Err(VectorError::NonFiniteValue)
            );
            assert_eq!(
                squared_euclidean_distance(&valid, bad),
                Err(VectorError::NonFiniteValue)
            );
            assert_eq!(
                euclidean_distance(bad, &valid),
                Err(VectorError::NonFiniteValue)
            );
            assert_eq!(
                cosine_similarity(bad, &valid),
                Err(VectorError::NonFiniteValue)
            );
            assert_eq!(
                cosine_similarity(&valid, bad),
                Err(VectorError::NonFiniteValue)
            );
            assert_eq!(
                Metric::Cosine.score(bad, &valid),
                Err(VectorError::NonFiniteValue)
            );
            assert_eq!(
                Metric::DotProduct.score(bad, &valid),
                Err(VectorError::NonFiniteValue)
            );
            assert_eq!(
                Metric::Euclidean.score(bad, &valid),
                Err(VectorError::NonFiniteValue)
            );

            let mut to_norm = bad.clone();
            assert_eq!(l2_normalize(&mut to_norm), Err(VectorError::NonFiniteValue));
        }
    }
}
