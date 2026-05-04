/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Serialization/deserialization for [`TableDeleteProviderAsync`] delete bitmaps.
//!
//! The on-disk format is a sequence of little-endian `u32` words, each holding 32 deletion
//! bits. Bit `b` of word `w` corresponds to vector ID `w * 32 + b`. The number of words
//! is `ceil(max_size / 32)`.

use diskann_providers::model::graph::provider::async_::TableDeleteProviderAsync;

/// Serialize the delete bitmap to bytes (little-endian `u32` words).
///
/// Reconstructs each word by querying the public `is_deleted` API so that no
/// access to private fields is required.
pub(crate) fn delete_bitmap_to_bytes(provider: &TableDeleteProviderAsync) -> Vec<u8> {
    let num_words = provider.max_size.div_ceil(32);
    let mut bytes = Vec::with_capacity(num_words * 4);

    for slot in 0..num_words {
        let mut word: u32 = 0;
        for bit in 0..32 {
            let id = slot * 32 + bit;
            if id < provider.max_size && provider.is_deleted(id) {
                word |= 1 << (bit as u32);
            }
        }
        bytes.extend_from_slice(&word.to_le_bytes());
    }

    bytes
}

/// Create a [`TableDeleteProviderAsync`] from serialized bytes (little-endian `u32` words).
///
/// Returns an error if the byte length does not match the expected word count or is not
/// a multiple of 4. Non-zero padding bits in the final word (above `max_size`) are rejected.
pub(crate) fn delete_bitmap_from_bytes(
    bytes: &[u8],
    max_size: usize,
) -> Result<TableDeleteProviderAsync, String> {
    let expected_words = max_size.div_ceil(32);

    let (chunks, remainder) = bytes.as_chunks::<{ std::mem::size_of::<u32>() }>();
    if chunks.len() != expected_words {
        return Err(format!(
            "Delete bitmap size mismatch: expected {} u32 values, got {}",
            expected_words,
            chunks.len()
        ));
    }
    if !remainder.is_empty() {
        return Err("Length of bytes is not a multiple of 4".to_string());
    }

    // Reject non-zero padding bits in the final word.
    if expected_words > 0 {
        let used_bits_in_last_word = max_size % 32;
        if used_bits_in_last_word != 0 {
            let last_word = u32::from_le_bytes(chunks[expected_words - 1]);
            let padding_mask = !((1u32 << used_bits_in_last_word) - 1);
            if last_word & padding_mask != 0 {
                return Err(format!(
                    "Non-zero padding bits in final word: {:#010x} (mask {:#010x})",
                    last_word, padding_mask
                ));
            }
        }
    }

    let provider = TableDeleteProviderAsync::new(max_size);
    for (slot, chunk) in chunks.iter().enumerate() {
        let word = u32::from_le_bytes(*chunk);
        for bit in 0..32 {
            let id = slot * 32 + bit;
            if id < max_size && (word & (1 << bit)) != 0 {
                provider.delete(id);
            }
        }
    }

    Ok(provider)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_provider(max_size: usize, deleted_ids: &[usize]) -> TableDeleteProviderAsync {
        let provider = TableDeleteProviderAsync::new(max_size);
        for &id in deleted_ids {
            provider.delete(id);
        }
        provider
    }

    #[test]
    fn roundtrip() {
        let original = make_provider(50, &[0, 5, 20, 34, 48]);
        let bytes = delete_bitmap_to_bytes(&original);
        let loaded = delete_bitmap_from_bytes(&bytes, 50).unwrap();

        for i in 0..50 {
            assert_eq!(
                original.is_deleted(i),
                loaded.is_deleted(i),
                "mismatch at id {i}"
            );
        }
    }

    #[test]
    fn size_mismatch() {
        let bytes = vec![0u8; 4]; // 1 u32 word, but max_size=50 needs 2
        let result = delete_bitmap_from_bytes(&bytes, 50);
        assert!(result.is_err());
    }

    #[test]
    fn not_multiple_of_4() {
        let bytes = vec![0u8; 9]; // 9 bytes is not a multiple of 4
        let result = delete_bitmap_from_bytes(&bytes, 50);
        assert!(result.is_err());
    }

    #[test]
    fn known_fixture() {
        // max_size=3, ids 0 and 2 deleted → word = 0b101 = 5
        let expected_bytes: Vec<u8> = 5u32.to_le_bytes().to_vec();
        let provider = make_provider(3, &[0, 2]);
        let bytes = delete_bitmap_to_bytes(&provider);
        assert_eq!(bytes, expected_bytes);

        // Reverse direction
        let loaded = delete_bitmap_from_bytes(&expected_bytes, 3).unwrap();
        assert!(loaded.is_deleted(0));
        assert!(!loaded.is_deleted(1));
        assert!(loaded.is_deleted(2));
    }

    #[test]
    fn rejects_padding_bits() {
        // max_size=3 → only bits 0..2 valid in one u32 word
        // Set bit 3 (padding) → should be rejected
        let bad_word: u32 = 0b1000; // bit 3 set, which is out of range
        let bytes = bad_word.to_le_bytes().to_vec();
        let result = delete_bitmap_from_bytes(&bytes, 3);
        assert!(result.is_err());
        assert!(result.err().unwrap().contains("padding"));
    }

    #[test]
    fn empty_bitmap() {
        let provider = make_provider(64, &[]);
        let bytes = delete_bitmap_to_bytes(&provider);
        assert_eq!(bytes, vec![0u8; 8]); // 2 u32 words, all zeros
        let loaded = delete_bitmap_from_bytes(&bytes, 64).unwrap();
        for i in 0..64 {
            assert!(!loaded.is_deleted(i));
        }
    }

    #[test]
    fn exact_word_boundary() {
        // max_size=32 → exactly 1 word, no padding bits
        let provider = make_provider(32, &[0, 15, 31]);
        let bytes = delete_bitmap_to_bytes(&provider);
        let loaded = delete_bitmap_from_bytes(&bytes, 32).unwrap();
        assert!(loaded.is_deleted(0));
        assert!(loaded.is_deleted(15));
        assert!(loaded.is_deleted(31));
        assert!(!loaded.is_deleted(1));
    }
}
