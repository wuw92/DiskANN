/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

// Phase 1 keeps persistence-related types (RocksdbParams, RocksdbPaths,
// SavedParams, db()/config() accessors) for Phase 2 to wire into save/load.
// They're currently unreferenced.
#![allow(dead_code)]

//! RocksDB-backed graph data provider.
//!
//! This module mirrors `super::bf_tree` structurally (see ROCKSDB_PROVIDER_PLAN.md
//! at the worktree root). Phase 1 accepts duplication; Phase 3 will extract the
//! shared serialization layer.

mod neighbor_provider;
mod provider;
mod quant_vector_provider;
mod vector_provider;

pub use provider::{
    AsVectorDtype, CreateQuantProvider, FullAccessor, GraphParams, Hidden, Index, QuantAccessor,
    QuantIndex, RocksdbPaths, RocksdbProvider, RocksdbProviderParameters, StartPoint, VectorDtype,
};

use std::path::PathBuf;
use std::sync::Arc;

use diskann::ANNError;
use rocksdb::DB;

/// Configuration for opening a RocksDB-backed store.
///
/// Mirrors the role of `bf_tree::Config` for the bf-tree provider, but only
/// carries the bits this provider actually exercises:
/// - target `path` on disk
/// - whether the backend should be in-memory (i.e. backed by a `TempDir`)
///
/// The `Options` used internally are constructed from defaults at open time.
#[derive(Clone)]
pub struct Config {
    pub(crate) path: PathBuf,
    pub(crate) is_memory: bool,
    /// When `is_memory` is true, this `TempDir` is held alive (via Arc so
    /// `Config` remains `Clone`) for the lifetime of the resulting `DB`.
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
    /// Default config opens an in-memory backend (mirrors bf_tree's default).
    /// Panics on tempdir creation failure, which is essentially impossible on
    /// any sane system; an `expect` here is acceptable as `Default` can't
    /// return `Result`.
    #[allow(clippy::expect_used)]
    fn default() -> Self {
        Self::in_memory().expect("failed to create in-memory rocksdb config")
    }
}

impl Config {
    /// Create a config pointing at an on-disk RocksDB.
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self {
            path: path.into(),
            is_memory: false,
            temp_dir: None,
        }
    }

    /// Create a config backed by a tempdir (rocksdb has no native in-memory mode).
    /// Returns `Err` on tempdir creation failure.
    pub fn in_memory() -> std::io::Result<Self> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().to_path_buf();
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

/// Open a RocksDB at the path described by `config`.
pub(crate) fn open_db(config: &Config) -> Result<DB, ConfigError> {
    let mut opts = rocksdb::Options::default();
    opts.create_if_missing(true);
    DB::open(&opts, &config.path).map_err(ConfigError)
}

/// Wrapper around [`rocksdb::Error`] that implements [`std::error::Error`]
/// in a shape compatible with `super::bf_tree::ConfigError`.
#[derive(Debug, Clone)]
pub struct ConfigError(pub rocksdb::Error);

impl std::fmt::Display for ConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "RocksDB configuration error: {}", self.0)
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
