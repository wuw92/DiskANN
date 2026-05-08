/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! RocksDB vector provider (parallel to `super::bf_tree::vector_provider`).

use std::marker::PhantomData;

use bytemuck::{bytes_of, cast_slice};
use diskann::{
    ANNError, ANNErrorKind, ANNResult,
    utils::{ErrorToVectorId, TryIntoVectorId, VectorId, VectorRepr},
};
use rocksdb::DB;
use thiserror::Error;

use super::super::common::TestCallCount;
use super::{Config, open_db};

pub struct VectorProvider<T: VectorRepr, I: VectorId = u32> {
    dim: usize,
    pub max_vectors: usize,
    pub num_start_points: usize,
    vector_index: DB,
    config: Config,
    pub(super) num_get_calls: TestCallCount,
    _phantom: PhantomData<(T, I)>,
}

impl<T: VectorRepr, I: VectorId> VectorProvider<T, I> {
    /// Create a new instance from a rocksdb config.
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

    /// Create a new instance from an existing DB handle (for snapshot reload).
    #[inline(always)]
    pub fn new_from_db(
        max_vectors: usize,
        dim: usize,
        num_start_points: usize,
        vector_index: DB,
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

    /// Return the total number of points including starting points.
    #[inline(always)]
    pub fn total(&self) -> usize {
        self.max_vectors + self.num_start_points
    }

    /// Return the vector dimension.
    #[inline(always)]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Return a vector of vector Ids of the starting points.
    #[inline(always)]
    pub fn starting_points(&self) -> Result<Vec<I>, ErrorToVectorId<usize, I>> {
        (self.max_vectors..self.total())
            .map(|i| i.try_into_vector_id())
            .collect()
    }

    /// Access the rocksdb config.
    pub(crate) fn config(&self) -> &Config {
        &self.config
    }

    /// Access the underlying RocksDB handle.
    pub(crate) fn db(&self) -> &DB {
        &self.vector_index
    }

    /// Set vector with Id `i` to `v`.
    ///
    /// Errors if:
    /// * `i >= self.total()`: id out of bounds.
    /// * `v.len() != self.dim()`: wrong dimension.
    #[inline(always)]
    pub(crate) fn set_vector_sync(&self, i: usize, v: &[T]) -> ANNResult<()> {
        if v.len() != self.dim {
            return Err(ANNError::log_index_error(
                "Vector dimension is not equal to the expected dimension.",
            ));
        }
        if i >= self.total() {
            return Err(ANNError::log_index_error(
                "Vector id is out of boundary in the dataset.",
            ));
        }

        let key = bytes_of::<usize>(&i);
        let value = cast_slice::<T, u8>(v);

        self.vector_index
            .put(key, value)
            .map_err(|e| ANNError::log_index_error(format!("rocksdb put failed: {}", e)))?;

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
        let value = self
            .vector_index
            .get(bytes_of(&i))
            .map_err(|e| ANNError::log_index_error(format!("rocksdb get failed: {}", e)))?;

        let bytes = match value {
            Some(b) => b,
            None => {
                return Err(ANNError::log_index_error(format!(
                    "The rocksdb entry for vector id {} is not found",
                    i
                )));
            }
        };

        let vector_size = std::mem::size_of::<T>() * self.dim;
        if bytes.len() != vector_size {
            return Err(ANNError::log_index_error(format!(
                "The rocksdb entry for vector id {} has size {} instead of the expected size {}",
                i,
                bytes.len(),
                vector_size,
            )));
        }
        bytemuck::must_cast_slice_mut::<_, u8>(buffer).copy_from_slice(&bytes);

        Ok(())
    }

    /// Return the vector at index `i`.
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
