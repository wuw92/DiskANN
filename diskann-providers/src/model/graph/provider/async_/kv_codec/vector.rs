/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Vector store validation helpers shared by graph providers.
//!
//! The bytes themselves are written / read via `bytemuck` cast; only the
//! pre/post validation lives here.

use diskann::{ANNError, ANNResult};

/// Validate inputs to a `set_vector` call.
///
/// * `i < total`            — id must be in range
/// * `v_len == expected_dim` — vector dimension must match
#[inline]
pub fn validate_set(i: usize, total: usize, v_len: usize, expected_dim: usize) -> ANNResult<()> {
    if v_len != expected_dim {
        return Err(ANNError::log_index_error(
            "Vector dimension is not equal to the expected dimension.",
        ));
    }
    if i >= total {
        return Err(ANNError::log_index_error(
            "Vector id is out of boundary in the dataset.",
        ));
    }
    Ok(())
}

/// Validate the size of bytes returned by the backend matches the expected
/// `dim * sizeof(T)`. `backend` is included in the error message to keep
/// per-provider diagnostics readable.
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
    fn validate_set_ok() {
        validate_set(0, 10, 4, 4).unwrap();
    }

    #[test]
    fn validate_set_dim_mismatch() {
        assert!(validate_set(0, 10, 3, 4).is_err());
    }

    #[test]
    fn validate_set_id_out_of_range() {
        assert!(validate_set(10, 10, 4, 4).is_err());
    }

    #[test]
    fn validate_read_size_ok() {
        validate_read_size("test", 0, 16, 16).unwrap();
    }

    #[test]
    fn validate_read_size_mismatch() {
        let err = validate_read_size("test", 7, 12, 16).unwrap_err();
        assert!(format!("{}", err).contains("vector id 7"));
    }
}
