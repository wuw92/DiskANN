/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

#![allow(dead_code)]

//! redb-backed graph data provider (pure-Rust B+ tree KV).
//!
//! Structurally mirrors `super::rocksdb`. Reuses the shared `kv_codec`
//! module for byte layouts and validation.

mod neighbor_provider;
mod provider;
mod quant_vector_provider;
mod vector_provider;

pub use provider::{
    AsVectorDtype, CreateQuantProvider, FullAccessor, GraphParams, Hidden, Index, QuantAccessor,
    QuantIndex, RedbPaths, RedbProvider, RedbProviderParameters, StartPoint, VectorDtype,
};

use std::path::PathBuf;
use std::sync::Arc;

use diskann::ANNError;
use redb::{Database, TableDefinition};

/// Single KV table used by every redb-backed inner provider. Each inner
/// provider owns its own `Database`, so the table name doesn't need to be
/// disambiguated.
pub(crate) const KV_TABLE: TableDefinition<&[u8], &[u8]> = TableDefinition::new("kv");

/// Configuration for opening a redb-backed store.
///
/// redb persists to a single file (vs RocksDB's directory).
/// For in-memory mode we put the file inside a `TempDir` held in
/// `Config` so the file outlives the resulting `Database`.
#[derive(Clone)]
pub struct Config {
    pub(crate) path: PathBuf,
    pub(crate) is_memory: bool,
    pub(crate) temp_dir: Option<Arc<tempfile::TempDir>>,
}

impl std::fmt::Debug for Config {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Config")
            .field("path", &self.path)
            .field("is_memory", &self.is_memory)
            .finish()
    }
}

impl Default for Config {
    #[allow(clippy::expect_used)]
    fn default() -> Self {
        Self::in_memory().expect("failed to create in-memory redb config")
    }
}

impl Config {
    /// Create a config pointing at an on-disk redb file.
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self {
            path: path.into(),
            is_memory: false,
            temp_dir: None,
        }
    }

    /// Create a config backed by a tempdir (redb has no native in-memory mode).
    pub fn in_memory() -> std::io::Result<Self> {
        let dir = tempfile::tempdir()?;
        // redb stores its data in a single file; put it inside the tempdir.
        let path = dir.path().join("redb.db");
        Ok(Self {
            path,
            is_memory: true,
            temp_dir: Some(Arc::new(dir)),
        })
    }

    pub fn is_memory_backend(&self) -> bool {
        self.is_memory
    }

    pub fn path(&self) -> &std::path::Path {
        &self.path
    }
}

/// Open (or create) the redb database at the path described by `config`.
pub(crate) fn open_db(config: &Config) -> Result<Database, ConfigError> {
    Database::create(&config.path).map_err(|e| ConfigError(e.to_string()))
}

/// Write a single `(key, value)` pair with durability disabled.
///
/// `redb::Error` is a large enum (clippy::result_large_err); we box it
/// to keep the `Ok` path small.
pub(crate) fn put_bytes(
    db: &Database,
    key: &[u8],
    value: &[u8],
) -> Result<(), Box<redb::Error>> {
    let mut tx = db.begin_write().map_err(|e| Box::new(e.into()))?;
    tx.set_durability(redb::Durability::None);
    {
        let mut table = tx
            .open_table(KV_TABLE)
            .map_err(|e| Box::new(e.into()))?;
        table.insert(key, value).map_err(|e| Box::new(e.into()))?;
    }
    tx.commit().map_err(|e| Box::new(e.into()))?;
    Ok(())
}

/// Delete a single key.
pub(crate) fn delete_key(db: &Database, key: &[u8]) -> Result<(), Box<redb::Error>> {
    let mut tx = db.begin_write().map_err(|e| Box::new(e.into()))?;
    tx.set_durability(redb::Durability::None);
    {
        let mut table = tx
            .open_table(KV_TABLE)
            .map_err(|e| Box::new(e.into()))?;
        table.remove(key).map_err(|e| Box::new(e.into()))?;
    }
    tx.commit().map_err(|e| Box::new(e.into()))?;
    Ok(())
}

/// Wrapper preserving the same shape as `super::rocksdb::ConfigError`.
#[derive(Debug, Clone)]
pub struct ConfigError(pub String);

impl std::fmt::Display for ConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "redb configuration error: {}", self.0)
    }
}

impl std::error::Error for ConfigError {}

impl From<ConfigError> for ANNError {
    #[track_caller]
    #[inline(never)]
    fn from(error: ConfigError) -> ANNError {
        ANNError::new(diskann::ANNErrorKind::IndexError, error)
    }
}
