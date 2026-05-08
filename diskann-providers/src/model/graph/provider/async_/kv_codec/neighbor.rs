/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Adjacency list serialization shared by graph providers.
//!
//! Layout written / read by both bf-tree and rocksdb backends:
//!
//! `|VectorId|VectorId|...|Invalid|...|VectorId(list length)|`
//!
//! - the first `nbr_count` slots are the actual neighbor IDs
//! - intermediate slots may be uninitialized ("Invalid") padding
//! - the last meaningful slot is the list length, encoded as a `VectorId`
//!
//! The two providers diverge only in how they get/put bytes; everything else
//! (validation, length extraction, value assembly) lives here.

use bytemuck::{bytes_of, cast_slice};
use diskann::{
    ANNError, ANNResult,
    utils::{TryIntoVectorId, VectorId},
};

/// Serialize a neighbor list into the on-disk layout
/// `|VectorId|...|VectorId(len)|`.
///
/// Caller has already verified `neighbors.len() <= max_degree`.
#[inline]
#[allow(clippy::expect_used)]
pub fn serialize<I: VectorId>(neighbors: &[I]) -> Vec<u8> {
    let edges = cast_slice::<I, u8>(neighbors);
    let len = neighbors
        .len()
        .try_into_vector_id()
        .expect("Fail to convert #neighbors as neighbor vec Id");
    let len_bytes = bytes_of::<I>(&len);
    [edges, len_bytes].concat()
}

/// Validate the bytes returned by a backend and decode the neighbor count.
///
/// `read_size`  — number of bytes returned by the backend (0 = empty list).
/// `dim`        — `1 + max_degree`; the capacity in vector-id slots.
/// `slots`      — the writable buffer the bytes were written into; this is
///                read at index `read_size / sizeof(I) - 1` to recover the
///                length prefix.
///
/// Returns the decoded neighbor count, or 0 if `read_size == 0`.
#[inline]
pub fn validate_and_count<I: VectorId>(
    read_size: usize,
    dim: usize,
    slots: &[I],
) -> ANNResult<usize> {
    let id_size = std::mem::size_of::<I>();

    if read_size == 0 {
        return Ok(0);
    }

    if read_size < id_size {
        return Err(ANNError::log_index_error(
            "Retrieved neighbor list is shorter than a single VectorID",
        ));
    }

    if read_size > id_size * dim {
        return Err(ANNError::log_index_error(
            "Retrieved neighbor list is longer than the max degree",
        ));
    }

    if !read_size.is_multiple_of(id_size) {
        return Err(ANNError::log_index_error(
            "Retrieved neighbor list length is not in the multiple of VectorID",
        ));
    }

    let nbr_count = slots[read_size / id_size - 1].into_usize();

    if read_size < id_size * (nbr_count + 1) {
        return Err(ANNError::log_index_error(
            "The length of the retrieved neighbor list is shorter than the specified length",
        ));
    }

    Ok(nbr_count)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serialize_then_count_round_trip() {
        let neighbors: Vec<u32> = vec![1, 2, 3];
        let bytes = serialize(&neighbors);

        // Result is 4 IDs: 1, 2, 3, len=3, => 16 bytes for u32.
        assert_eq!(bytes.len(), 4 * std::mem::size_of::<u32>());

        let mut slots = [0u32; 8];
        bytemuck::cast_slice_mut::<u32, u8>(&mut slots)[..bytes.len()].copy_from_slice(&bytes);

        let nbr_count = validate_and_count::<u32>(bytes.len(), 8, &slots).unwrap();
        assert_eq!(nbr_count, 3);
        assert_eq!(&slots[..nbr_count], &[1, 2, 3]);
    }

    #[test]
    fn empty_read_returns_zero() {
        let slots = [0u32; 4];
        assert_eq!(validate_and_count::<u32>(0, 4, &slots).unwrap(), 0);
    }

    #[test]
    fn shorter_than_single_id_errors() {
        let slots = [0u32; 4];
        assert!(validate_and_count::<u32>(2, 4, &slots).is_err());
    }

    #[test]
    fn longer_than_max_degree_errors() {
        let slots = [0u32; 4];
        // dim=4, id_size=4, so max bytes = 16; supply 20.
        assert!(validate_and_count::<u32>(20, 4, &slots).is_err());
    }

    #[test]
    fn non_multiple_size_errors() {
        let slots = [0u32; 4];
        assert!(validate_and_count::<u32>(7, 4, &slots).is_err());
    }

    #[test]
    fn declared_length_greater_than_payload_errors() {
        // read_size = 8 bytes (= 2 u32 slots): one neighbor + one length
        // length declared = 5 (impossible because we only have 1 actual neighbor)
        let slots = [9u32, 5, 0, 0];
        assert!(validate_and_count::<u32>(8, 8, &slots).is_err());
    }
}
