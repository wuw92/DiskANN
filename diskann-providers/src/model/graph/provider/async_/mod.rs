/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */
pub mod experimental;

pub mod common;
pub use common::{PrefetchCacheLineLevel, StartPoints, VectorGuard};

// Backend-agnostic codec helpers shared by `bf_tree` / `rocksdb` / `redb` providers.
#[cfg(any(
    feature = "bf_tree",
    feature = "rocksdb_provider",
    feature = "redb_provider"
))]
pub mod kv_codec;

pub(crate) mod postprocess;

pub mod distances;

pub mod memory_vector_provider;
pub use memory_vector_provider::MemoryVectorProviderAsync;

pub mod memory_quant_vector_provider;
pub use memory_quant_vector_provider::MemoryQuantVectorProviderAsync;

pub mod simple_neighbor_provider;
pub use simple_neighbor_provider::SimpleNeighborProviderAsync;

pub mod table_delete_provider;
pub use table_delete_provider::TableDeleteProviderAsync;

pub mod fast_memory_vector_provider;
pub use fast_memory_vector_provider::FastMemoryVectorProviderAsync;

pub mod fast_memory_quant_vector_provider;
pub use fast_memory_quant_vector_provider::FastMemoryQuantVectorProviderAsync;

// The default `inmem` data provider for the async index.
pub mod inmem;

// Bf-tree based data provider for the async index
#[cfg(feature = "bf_tree")]
pub mod bf_tree;

// Caching proxy provider to accelerate slow providers.
#[cfg(feature = "bf_tree")]
pub mod caching;

// RocksDB based data provider for the async index (parallel to bf_tree).
#[cfg(feature = "rocksdb_provider")]
pub mod rocksdb;

// redb based data provider (pure-Rust B+ tree KV).
#[cfg(feature = "redb_provider")]
pub mod redb;
