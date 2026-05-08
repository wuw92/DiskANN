/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Backend-agnostic codec helpers shared by `bf_tree` and `rocksdb` graph
//! providers (Phase 3 of the rocksdb integration plan).
//!
//! These helpers operate on raw bytes and avoid touching backend I/O — each
//! provider keeps its own thin wrapper around the actual KV store calls
//! (`BfTree::read/insert/delete` vs `DB::get/put/delete`) and delegates the
//! validation / serialization bits to this module.

pub mod neighbor;
pub mod quant;
pub mod vector;
