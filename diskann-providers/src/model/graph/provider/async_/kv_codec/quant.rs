/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Quant vector store validation helpers shared by graph providers.

use diskann::{ANNError, ANNResult};

/// Validate the inputs to a PQ-compress + set call.
///
/// * `i < total`              — id must be in range
/// * `vf32_len == full_dim`   — pre-compression vector dim matches
#[inline]
pub fn validate_set(i: usize, total: usize, vf32_len: usize, full_dim: usize) -> ANNResult<()> {
    if i >= total {
        return Err(ANNError::log_index_error(
            "Vector id is out of boundary in the dataset.",
        ));
    }
    if vf32_len != full_dim {
        return Err(ANNError::log_index_error(
            "Vector f32 dimension is not equal to the expected dimension.",
        ));
    }
    Ok(())
}

/// Validate the inputs to a raw quant-vector set call.
#[inline]
pub fn validate_set_quant(i: usize, total: usize, v_len: usize, pq_chunks: usize) -> ANNResult<()> {
    if i >= total {
        return Err(ANNError::log_index_error(
            "Vector id is out of boundary in the dataset.",
        ));
    }
    if v_len != pq_chunks {
        return Err(ANNError::log_index_error(
            "Vector dimension is not equal to the expected dimension.",
        ));
    }
    Ok(())
}

/// Validate the size of bytes returned by the backend matches the expected
/// `pq_chunks` byte count.
#[inline]
pub fn validate_read_size(
    backend: &'static str,
    i: usize,
    actual: usize,
    expected: usize,
) -> ANNResult<()> {
    if actual != expected {
        return Err(ANNError::log_index_error(format!(
            "The {} entry for vector id {} has size {} instead of the expected size {}",
            backend, i, actual, expected,
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn set_validation_ok() {
        validate_set(0, 10, 4, 4).unwrap();
        validate_set_quant(0, 10, 2, 2).unwrap();
        validate_read_size("test", 0, 16, 16).unwrap();
    }

    #[test]
    fn set_id_out_of_range() {
        assert!(validate_set(10, 10, 4, 4).is_err());
        assert!(validate_set_quant(10, 10, 2, 2).is_err());
    }

    #[test]
    fn set_dim_mismatch() {
        assert!(validate_set(0, 10, 3, 4).is_err());
        assert!(validate_set_quant(0, 10, 3, 2).is_err());
    }
}
