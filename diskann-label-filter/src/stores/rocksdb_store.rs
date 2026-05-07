/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! RocksDB-backed key-value store implementation.
//!
//! Mirrors the `BfTreeStore` shape so that the inverted index can be parameterized
//! over either backend at compile time via the `KvStore` trait.

use crate::traits::kv_store_traits::{KeyRange, KvIterator, KvStore, RangeBound};
use rocksdb::{Direction, IteratorMode, Options, ReadOptions, WriteBatch, DB};
use std::path::Path;
use std::sync::Arc;

/// A persistent key-value store backed by RocksDB.
///
/// # Properties
///
/// - **Persistence**: on-disk LSM-tree
/// - **Concurrency**: `Arc<DB>`; RocksDB handles internal synchronization
/// - **Range scans**: implemented via RocksDB iterator with iterate bounds
/// - **Atomic batches**: `WriteBatch` provides atomic batch_set / batch_del,
///   stronger than `BfTreeStore` whose batches are sequential and non-atomic
///
/// # Notes
///
/// RocksDB has no first-class in-memory mode; `temp()` (test-only) uses a
/// `tempfile::TempDir` as the storage path, which is sufficient for unit tests
/// but is still disk-backed (subject to filesystem caching).
#[derive(Clone)]
pub struct RocksdbStore {
    db: Arc<DB>,
}

#[derive(Debug)]
pub struct RocksdbStoreError(pub String);

impl std::fmt::Display for RocksdbStoreError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "RocksDB store error: {}", self.0)
    }
}

impl std::error::Error for RocksdbStoreError {}

impl From<rocksdb::Error> for RocksdbStoreError {
    fn from(e: rocksdb::Error) -> Self {
        RocksdbStoreError(e.to_string())
    }
}

impl From<String> for RocksdbStoreError {
    fn from(s: String) -> Self {
        RocksdbStoreError(s)
    }
}

impl From<&str> for RocksdbStoreError {
    fn from(s: &str) -> Self {
        RocksdbStoreError(s.to_string())
    }
}

impl RocksdbStore {
    /// Opens a RocksDB at `path` with sensible defaults (`create_if_missing = true`).
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, RocksdbStoreError> {
        let mut opts = Options::default();
        opts.create_if_missing(true);
        Self::open_with_options(path, opts)
    }

    /// Opens a RocksDB with caller-provided `Options`.
    pub fn open_with_options<P: AsRef<Path>>(
        path: P,
        opts: Options,
    ) -> Result<Self, RocksdbStoreError> {
        let db = DB::open(&opts, path)?;
        Ok(Self { db: Arc::new(db) })
    }

    /// Creates a temporary RocksDB rooted in a `TempDir` for testing. The
    /// returned `TempDir` must outlive the store; dropping it removes the data.
    #[cfg(test)]
    pub fn temp() -> Result<(Self, tempfile::TempDir), RocksdbStoreError> {
        let dir = tempfile::tempdir().map_err(|e| RocksdbStoreError(e.to_string()))?;
        let store = Self::open(dir.path())?;
        Ok((store, dir))
    }
}

impl KvStore for RocksdbStore {
    type Error = RocksdbStoreError;

    fn get(&self, key: &[u8]) -> Result<Option<Vec<u8>>, Self::Error> {
        Ok(self.db.get(key)?)
    }

    fn set(&self, key: &[u8], value: &[u8]) -> Result<(), Self::Error> {
        self.db.put(key, value)?;
        Ok(())
    }

    fn del(&self, key: &[u8]) -> Result<(), Self::Error> {
        self.db.delete(key)?;
        Ok(())
    }

    fn range<R>(&self, range: R) -> Result<KvIterator<'_, Self::Error>, Self::Error>
    where
        R: Into<KeyRange>,
    {
        let key_range = range.into();
        let mut read_opts = ReadOptions::default();

        // Conservative iterate bounds; final inclusion is decided by `KeyRange::contains`.
        // RocksDB's iterate_lower_bound is inclusive, iterate_upper_bound is exclusive.
        let lower_seek = match &key_range.start {
            RangeBound::Unbounded => None,
            RangeBound::Included(s) | RangeBound::Excluded(s) => {
                read_opts.set_iterate_lower_bound(s.clone());
                Some(s.clone())
            }
        };
        if let RangeBound::Excluded(e) = &key_range.end {
            read_opts.set_iterate_upper_bound(e.clone());
        }

        let mode = match &lower_seek {
            Some(s) => IteratorMode::From(s, Direction::Forward),
            None => IteratorMode::Start,
        };
        let iter = self.db.iterator_opt(mode, read_opts);

        let filtered = iter.filter_map(move |result| match result {
            Ok((k, v)) => {
                if key_range.contains(&k) {
                    Some(Ok((k.into_vec(), v.into_vec())))
                } else {
                    None
                }
            }
            Err(e) => Some(Err(RocksdbStoreError::from(e))),
        });

        Ok(Box::new(filtered))
    }

    fn batch_get(&self, keys: &[&[u8]]) -> Result<Vec<Option<Vec<u8>>>, Self::Error> {
        self.db
            .multi_get(keys)
            .into_iter()
            .map(|r| r.map_err(RocksdbStoreError::from))
            .collect()
    }

    fn batch_set(&self, entries: &[(&[u8], &[u8])]) -> Result<(), Self::Error> {
        let mut batch = WriteBatch::default();
        for (k, v) in entries {
            batch.put(k, v);
        }
        self.db.write(batch)?;
        Ok(())
    }

    fn batch_del(&self, keys: &[&[u8]]) -> Result<(), Self::Error> {
        let mut batch = WriteBatch::default();
        for k in keys {
            batch.delete(k);
        }
        self.db.write(batch)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn small_key_small_value() {
        let (store, _dir) = RocksdbStore::temp().unwrap();
        store.set(b"key", b"v").unwrap();
        assert_eq!(store.get(b"key").unwrap(), Some(b"v".to_vec()));
    }

    #[test]
    fn larger_key_larger_value() {
        let (store, _dir) = RocksdbStore::temp().unwrap();
        let key = &[b'k'; 1000];
        let value = &[b'v'; 1000];
        store.set(key, value).unwrap();
        assert_eq!(store.get(key).unwrap(), Some(value.to_vec()));
    }

    #[test]
    fn test_basic_operations() {
        let (store, _dir) = RocksdbStore::temp().unwrap();

        store.set(b"key1", b"value1").unwrap();
        assert_eq!(store.get(b"key1").unwrap(), Some(b"value1".to_vec()));

        store.set(b"key1", b"value2").unwrap();
        assert_eq!(store.get(b"key1").unwrap(), Some(b"value2".to_vec()));

        store.del(b"key1").unwrap();
        assert_eq!(store.get(b"key1").unwrap(), None);

        assert_eq!(store.get(b"missing").unwrap(), None);
    }

    #[test]
    fn test_batch_operations_atomic() {
        let (store, _dir) = RocksdbStore::temp().unwrap();

        store
            .batch_set(&[
                (b"k1".as_slice(), b"v1".as_slice()),
                (b"k2".as_slice(), b"v2".as_slice()),
                (b"k3".as_slice(), b"v3".as_slice()),
            ])
            .unwrap();

        let results = store.batch_get(&[b"k1", b"k2", b"k3", b"k4"]).unwrap();
        assert_eq!(results[0], Some(b"v1".to_vec()));
        assert_eq!(results[1], Some(b"v2".to_vec()));
        assert_eq!(results[2], Some(b"v3".to_vec()));
        assert_eq!(results[3], None);

        store.batch_del(&[b"k1", b"k3"]).unwrap();
        assert_eq!(store.get(b"k1").unwrap(), None);
        assert_eq!(store.get(b"k2").unwrap(), Some(b"v2".to_vec()));
        assert_eq!(store.get(b"k3").unwrap(), None);
    }

    #[test]
    fn test_range_scan_exclusive() {
        let (store, _dir) = RocksdbStore::temp().unwrap();
        for (k, v) in [
            (b"a".as_slice(), b"1".as_slice()),
            (b"b", b"2"),
            (b"c", b"3"),
            (b"d", b"4"),
            (b"e", b"5"),
        ] {
            store.set(k, v).unwrap();
        }

        let results: Vec<_> = store
            .range(b"b"..b"d")
            .unwrap()
            .collect::<Result<Vec<_>, _>>()
            .unwrap();

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, b"b");
        assert_eq!(results[1].0, b"c");
    }

    #[test]
    fn test_range_scan_inclusive() {
        let (store, _dir) = RocksdbStore::temp().unwrap();
        for (k, v) in [
            (b"a".as_slice(), b"1".as_slice()),
            (b"b", b"2"),
            (b"c", b"3"),
            (b"d", b"4"),
        ] {
            store.set(k, v).unwrap();
        }

        let results: Vec<_> = store
            .range(b"b"..=b"d")
            .unwrap()
            .collect::<Result<Vec<_>, _>>()
            .unwrap();

        assert_eq!(results.len(), 3);
        assert_eq!(results[0].0, b"b");
        assert_eq!(results[2].0, b"d");
    }

    #[test]
    fn test_full_scan() {
        let (store, _dir) = RocksdbStore::temp().unwrap();
        store.set(b"a", b"1").unwrap();
        store.set(b"b", b"2").unwrap();
        store.set(b"c", b"3").unwrap();

        let results: Vec<_> = store
            .range(..)
            .unwrap()
            .collect::<Result<Vec<_>, _>>()
            .unwrap();

        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_idempotent_delete() {
        let (store, _dir) = RocksdbStore::temp().unwrap();
        store.set(b"key", b"value").unwrap();
        store.del(b"key").unwrap();
        store.del(b"key").unwrap();
        assert_eq!(store.get(b"key").unwrap(), None);
    }

    #[test]
    fn test_concurrent_access() {
        use std::thread;

        let (store, _dir) = RocksdbStore::temp().unwrap();
        let store = Arc::new(store);
        let mut handles = vec![];

        for i in 0..10 {
            let store_clone = Arc::clone(&store);
            handles.push(thread::spawn(move || {
                let key = format!("key_{}", i);
                let value = format!("value_{}", i);
                store_clone.set(key.as_bytes(), value.as_bytes()).unwrap();
            }));
        }
        for h in handles {
            h.join().unwrap();
        }

        for i in 0..10 {
            let key = format!("key_{}", i);
            let expected = format!("value_{}", i);
            assert_eq!(
                store.get(key.as_bytes()).unwrap(),
                Some(expected.into_bytes())
            );
        }
    }
}
