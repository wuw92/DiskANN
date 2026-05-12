/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! redb vector provider (parallel to `super::rocksdb::vector_provider`).

use std::marker::PhantomData;

use bytemuck::{bytes_of, cast_slice};
use diskann::{
    ANNError, ANNErrorKind, ANNResult,
    utils::{ErrorToVectorId, TryIntoVectorId, VectorId, VectorRepr},
};
use redb::Database;
use thiserror::Error;

use super::super::common::TestCallCount;
use super::super::kv_codec::vector as vector_codec;
use super::{Config, KV_TABLE, open_db, put_bytes};

pub struct VectorProvider<T: VectorRepr, I: VectorId = u32> {
    dim: usize,
    pub max_vectors: usize,
    pub num_start_points: usize,
    vector_index: Database,
    config: Config,
    pub(super) num_get_calls: TestCallCount,
    _phantom: PhantomData<(T, I)>,
}

impl<T: VectorRepr, I: VectorId> VectorProvider<T, I> {
    pub fn new_with_config(
        max_vectors: usize,
        dim: usize,
        num_start_points: usize,
        config: Config,
    ) -> ANNResult<Self> {
        let vector_index = open_db(&config)?;

        Ok(Self {
            dim,
            max_vectors,
            num_start_points,
            vector_index,
            config,
            num_get_calls: TestCallCount::default(),
            _phantom: PhantomData,
        })
    }

    #[inline(always)]
    pub fn new_from_db(
        max_vectors: usize,
        dim: usize,
        num_start_points: usize,
        vector_index: Database,
        config: Config,
    ) -> Self {
        Self {
            dim,
            max_vectors,
            num_start_points,
            vector_index,
            config,
            num_get_calls: TestCallCount::default(),
            _phantom: PhantomData,
        }
    }

    #[inline(always)]
    pub fn total(&self) -> usize {
        self.max_vectors + self.num_start_points
    }

    #[inline(always)]
    pub fn dim(&self) -> usize {
        self.dim
    }

    #[inline(always)]
    pub fn starting_points(&self) -> Result<Vec<I>, ErrorToVectorId<usize, I>> {
        (self.max_vectors..self.total())
            .map(|i| i.try_into_vector_id())
            .collect()
    }

    pub(crate) fn config(&self) -> &Config {
        &self.config
    }

    pub(crate) fn db(&self) -> &Database {
        &self.vector_index
    }

    #[inline(always)]
    pub(crate) fn set_vector_sync(&self, i: usize, v: &[T]) -> ANNResult<()> {
        vector_codec::validate_set(i, self.total(), v.len(), self.dim)?;

        let key = bytes_of::<usize>(&i);
        let value = cast_slice::<T, u8>(v);

        put_bytes(&self.vector_index, key, value)
            .map_err(|e| ANNError::log_index_error(format!("redb put failed: {}", e)))?;

        Ok(())
    }

    pub(crate) fn get_vector_into(&self, i: usize, buffer: &mut [T]) -> ANNResult<()> {
        if buffer.len() != self.dim {
            #[derive(Debug, Error)]
            #[error("expected a buffer with dim {0}, instead got {1}")]
            struct WrongDim(usize, usize);

            return Err(ANNError::new(
                ANNErrorKind::IndexError,
                WrongDim(self.dim(), buffer.len()),
            ));
        }

        self.num_get_calls.increment();

        // redb requires a transaction + table scope per read. The AccessGuard
        // returned by `table.get` borrows zero-copy from the page cache.
        let tx = self
            .vector_index
            .begin_read()
            .map_err(|e| ANNError::log_index_error(format!("redb begin_read failed: {}", e)))?;
        let table = tx
            .open_table(KV_TABLE)
            .map_err(|e| ANNError::log_index_error(format!("redb open_table failed: {}", e)))?;
        let guard = table
            .get(&bytes_of(&i)[..])
            .map_err(|e| ANNError::log_index_error(format!("redb get failed: {}", e)))?;

        let g = match guard {
            Some(g) => g,
            None => {
                return Err(ANNError::log_index_error(format!(
                    "The redb entry for vector id {} is not found",
                    i
                )));
            }
        };

        let bytes = g.value();
        let expected = std::mem::size_of::<T>() * self.dim;
        vector_codec::validate_read_size("redb", i, bytes.len(), expected)?;
        bytemuck::must_cast_slice_mut::<_, u8>(buffer).copy_from_slice(bytes);

        Ok(())
    }

    #[inline(always)]
    pub(crate) fn get_vector_sync(&self, i: usize) -> ANNResult<Vec<T>> {
        let mut vector = vec![T::default(); self.dim];
        self.get_vector_into(i, &mut vector)?;
        Ok(vector)
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use diskann::utils::vecid_from_usize;
    use tokio::task::JoinSet;

    use super::*;

    #[tokio::test(flavor = "multi_thread", worker_threads = 5)]
    async fn test_parallel_tree_traversal() {
        let num_points = 100;
        let config = Config::default();
        let vector_provider =
            Arc::new(VectorProvider::<f32>::new_with_config(num_points, 3, 2, config).unwrap());
        let mut set = JoinSet::new();
        for i in 0..num_points {
            let vector = vec![i as f32, (i + 1) as f32, (i + 2) as f32];
            let vector_provider_clone = Arc::clone(&vector_provider);
            set.spawn(async move {
                vector_provider_clone
                    .set_vector_sync(vecid_from_usize(i).unwrap(), &vector)
                    .unwrap()
            });
        }

        while let Some(res) = set.join_next().await {
            res.unwrap();
        }

        for i in 0..num_points {
            let vector = vector_provider
                .get_vector_sync(vecid_from_usize(i).unwrap())
                .unwrap();
            assert_eq!(&vector, &vec![(i as f32), (i + 1) as f32, (i + 2) as f32]);
        }
        assert_eq!(vector_provider.num_get_calls.get(), num_points);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 5)]
    async fn test_parallel_vector_access() {
        let num_points = 3;
        let frozen_points = 2;
        let dim = 3;
        let config = Config::default();

        let provider = Arc::new(
            VectorProvider::<f32>::new_with_config(num_points, dim, frozen_points, config).unwrap(),
        );

        let mut set = JoinSet::new();
        for _ in 0..5 {
            let provider_ref = Arc::clone(&provider);
            set.spawn(async move {
                provider_ref.set_vector_sync(0, &[0.0, 0.0, 0.0]).unwrap();
                provider_ref.set_vector_sync(1, &[1.0, 1.0, 1.0]).unwrap();
                provider_ref.set_vector_sync(2, &[2.0, 2.0, 2.0]).unwrap();
                provider_ref.set_vector_sync(3, &[3.0, 3.0, 3.0]).unwrap();
                provider_ref.set_vector_sync(4, &[4.0, 4.0, 4.0]).unwrap();

                assert_eq!(provider_ref.get_vector_sync(4).unwrap(), &[4.0, 4.0, 4.0]);
                assert_eq!(provider_ref.get_vector_sync(3).unwrap(), &[3.0, 3.0, 3.0]);
                assert_eq!(provider_ref.get_vector_sync(2).unwrap(), &[2.0, 2.0, 2.0]);
                assert_eq!(provider_ref.get_vector_sync(1).unwrap(), &[1.0, 1.0, 1.0]);
                assert_eq!(provider_ref.get_vector_sync(0).unwrap(), &[0.0, 0.0, 0.0]);

                assert!(provider_ref.set_vector_sync(5, &[0.0, 0.0, 0.0]).is_err());
                assert!(provider_ref.set_vector_sync(2, &[0.0, 0.0]).is_err());
            });
        }

        while let Some(res) = set.join_next().await {
            res.unwrap();
        }
    }
}
