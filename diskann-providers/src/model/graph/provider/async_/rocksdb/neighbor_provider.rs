/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! RocksDB neighbor list provider (parallel to `super::bf_tree::neighbor_provider`).

use std::marker::PhantomData;

use bytemuck::{bytes_of, cast_slice, cast_slice_mut};
use diskann::{
    ANNError, ANNResult,
    graph::AdjacencyList,
    provider::HasId,
    utils::{IntoUsize, VectorId},
};
use rocksdb::DB;

use super::super::common::TestCallCount;
use super::super::kv_codec::neighbor as neighbor_codec;
use super::{Config, open_db};

pub struct NeighborProvider<I: VectorId> {
    adjacency_list_index: DB,
    config: Config,
    dim: usize, // Max number of neighbors in a neighbor list + 1 for the neighbor count
    pub num_get_calls: TestCallCount,
    _phantom: PhantomData<I>,
}

impl<I: VectorId> HasId for NeighborProvider<I> {
    type Id = I;
}

impl<I: VectorId> NeighborProvider<I> {
    /// Create a new instance from a rocksdb config.
    pub fn new_with_config(max_degree: u32, config: Config) -> ANNResult<Self> {
        let adj_list_index = open_db(&config)?;
        Ok(Self::new(max_degree, adj_list_index, config))
    }

    fn new(max_degree: u32, adjacency_list_index: DB, config: Config) -> Self {
        Self {
            adjacency_list_index,
            config,
            dim: 1 + max_degree.into_usize(),
            num_get_calls: TestCallCount::default(),
            _phantom: PhantomData,
        }
    }

    /// Access the rocksdb config.
    pub(crate) fn config(&self) -> &Config {
        &self.config
    }

    /// Access the underlying RocksDB handle.
    pub(crate) fn db(&self) -> &DB {
        &self.adjacency_list_index
    }

    /// Return the maximum degree (number of neighbors per vector).
    pub fn max_degree(&self) -> u32 {
        (self.dim - 1) as u32
    }

    /// Create a new instance from an existing DB handle (for snapshot reload).
    pub(crate) fn new_from_db(max_degree: u32, adjacency_list_index: DB, config: Config) -> Self {
        Self::new(max_degree, adjacency_list_index, config)
    }

    /// Retrieve the neighbor list of a vector.
    ///
    /// `neighbors` is cleared first upon each invocation.
    /// One data copy is involved which copies the data from rocksdb to `neighbors`.
    pub fn get_neighbors(&self, vector_id: I, neighbors: &mut AdjacencyList<I>) -> ANNResult<()> {
        #[cfg(test)]
        self.num_get_calls.increment();

        // Resize 'neighbors' to hold any full-size neighbor list.
        let mut guard = neighbors.resize(self.dim);

        let i = vector_id.into_usize();
        let key = bytes_of::<usize>(&i);

        let value = self
            .adjacency_list_index
            .get(key)
            .map_err(|e| ANNError::log_index_error(format!("rocksdb get failed: {}", e)))?;

        let bytes = match value {
            Some(b) => b,
            None => {
                return Err(ANNError::log_index_error(
                    "The rocksdb entry for the vector key is not found",
                ));
            }
        };

        let read_size = bytes.len();
        if read_size > 0 {
            // Copy the bytes into the guard buffer so the codec can read the
            // length suffix in-place.
            let dest = cast_slice_mut::<I, u8>(&mut guard);
            dest[..read_size].copy_from_slice(&bytes);

            let nbr_count = neighbor_codec::validate_and_count::<I>(read_size, self.dim, &guard)?;
            guard.finish(nbr_count);
        }

        Ok(())
    }

    /// Insert a neighbor list of a vector in rocksdb as a (K, V) pair.
    /// K: vector id
    /// V: |VectorId|VectorId|...|Invalid|Invalid|VectorId (list length)|
    /// Where list length is the full list length and 'Invalid' indicates unfilled empty slots in the list.
    /// Note: assuming all neighbors in the input list, 'neighbors', are valid.
    /// Two data copies are involved: 1) Copy from the immutable `neighbors` to the proper byte array with neighbor length;
    /// 2) Copy from the byte array to rocksdb.
    pub fn set_neighbors(&self, vector_id: I, neighbors: &[I]) -> ANNResult<()> {
        #[cfg(test)]
        self.num_get_calls.increment();

        if neighbors.len() > self.dim - 1 {
            return Err(ANNError::log_index_error(
                "The provided neighbor list is longer than the max degree",
            ));
        }

        let i = vector_id.into_usize();
        let key = bytes_of::<usize>(&i);
        let value = neighbor_codec::serialize(neighbors);

        self.adjacency_list_index
            .put(key, &value)
            .map_err(|e| ANNError::log_index_error(format!("rocksdb put failed: {}", e)))?;

        Ok(())
    }

    /// Append unique vectors into a neighbor list.
    /// The newly appended neighbor list will always be extended to 'dim' long to avoid frequent mem copy in rocksdb.
    /// Note: assuming all neighbors in the input list, 'new_neighbor_ids', are valid.
    /// Three data copies: 1) get_neighbors; 2) copy new neighbors to the neighbor list; 3) copy the new neighbor list to rocksdb.
    #[allow(clippy::expect_used)]
    pub fn append_vector(&self, vector_id: I, new_neighbor_ids: &[I]) -> ANNResult<()> {
        let mut neighbor_list = AdjacencyList::with_capacity(self.dim);
        self.get_neighbors(vector_id, &mut neighbor_list)?;

        let mut new_neighbor_added = false;
        for new_neighbor_id in new_neighbor_ids {
            if neighbor_list.len() == self.dim - 1 {
                break;
            }
            new_neighbor_added |= neighbor_list.push(*new_neighbor_id);
        }

        if new_neighbor_added {
            let nbr_count = neighbor_list.len();
            let mut neighbor_list: Vec<_> = neighbor_list.into();
            neighbor_list.resize(self.dim, I::default());
            neighbor_list[self.dim - 1] =
                I::from_usize(nbr_count).expect("Fails to cast usize as VectorId");

            let i = vector_id.into_usize();
            let key = bytes_of::<usize>(&i);
            let value = cast_slice::<I, u8>(&neighbor_list);
            self.adjacency_list_index
                .put(key, value)
                .map_err(|e| ANNError::log_index_error(format!("rocksdb put failed: {}", e)))?;
        }

        Ok(())
    }

    pub fn delete_vector(&self, vector_id: I) -> ANNResult<()> {
        let i = vector_id.into_usize();
        let key = bytes_of::<usize>(&i);

        self.adjacency_list_index
            .delete(key)
            .map_err(|e| ANNError::log_index_error(format!("rocksdb delete failed: {}", e)))?;
        Ok(())
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use tokio::task::JoinSet;

    use super::*;

    #[tokio::test]
    async fn test_neighbor_accessors() {
        let config = Config::default();
        let neighbor_provider = NeighborProvider::<u32>::new_with_config(6, config).unwrap();

        let adj_list = vec![1, 2, 3];
        neighbor_provider.set_neighbors(1, &adj_list).unwrap();

        let mut result = AdjacencyList::with_capacity(10);
        neighbor_provider.get_neighbors(1, &mut result).unwrap();
        assert_eq!(&*adj_list, &*result);

        let mut new_neighbors = vec![9, 2, 9];
        neighbor_provider.append_vector(1, &new_neighbors).unwrap();

        neighbor_provider.get_neighbors(1, &mut result).unwrap();

        let mut adj_list_new = vec![1, 2, 3, 9];
        assert_eq!(&*adj_list_new, &*result);

        new_neighbors = vec![5, 6, 7];
        neighbor_provider.append_vector(1, &new_neighbors).unwrap();

        neighbor_provider.get_neighbors(1, &mut result).unwrap();

        adj_list_new = vec![1, 2, 3, 9, 5, 6];
        assert_eq!(&*adj_list_new, &*result);

        new_neighbors = vec![];
        neighbor_provider.set_neighbors(1, &new_neighbors).unwrap();
        neighbor_provider.get_neighbors(1, &mut result).unwrap();

        assert_eq!(&*new_neighbors, &*result);

        new_neighbors = vec![3, 4, 5];
        neighbor_provider.append_vector(1, &new_neighbors).unwrap();

        neighbor_provider.get_neighbors(1, &mut result).unwrap();

        assert_eq!(&*new_neighbors, &*result);

        neighbor_provider.delete_vector(1).unwrap();

        assert!(neighbor_provider.get_neighbors(1, &mut result).is_err());

        new_neighbors = vec![];
        assert_eq!(&*new_neighbors, &*result);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 5)]
    async fn test_parallel_tree_traversal() {
        let config = Config::default();
        let neighbor_provider =
            Arc::new(NeighborProvider::<u32>::new_with_config(120, config).unwrap());

        let mut set = JoinSet::new();
        for i in 0..100 {
            let neighbor_list = vec![i as u32, (i + 1) as u32, (i + 2) as u32];
            let neighbor_provider_clone = Arc::clone(&neighbor_provider);
            set.spawn(async move {
                neighbor_provider_clone
                    .set_neighbors(i as u32, &neighbor_list)
                    .unwrap()
            });
        }

        while let Some(res) = set.join_next().await {
            res.unwrap();
        }

        let mut result = AdjacencyList::with_capacity(neighbor_provider.dim);
        for i in 0..100 {
            neighbor_provider
                .get_neighbors(i as u32, &mut result)
                .unwrap();

            let neighbor_list = vec![i as u32, (i + 1) as u32, (i + 2) as u32];
            assert_eq!(&*neighbor_list, &*result);
        }
    }

    #[tokio::test]
    async fn test_max_degree() {
        let neighbor_provider =
            NeighborProvider::<u32>::new_with_config(6, Config::default()).unwrap();
        assert_eq!(neighbor_provider.max_degree(), 6);

        let neighbor_provider =
            NeighborProvider::<u32>::new_with_config(120, Config::default()).unwrap();
        assert_eq!(neighbor_provider.max_degree(), 120);

        let neighbor_provider =
            NeighborProvider::<u32>::new_with_config(1, Config::default()).unwrap();
        assert_eq!(neighbor_provider.max_degree(), 1);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 5)]
    async fn test_parallel_neighbor_access() {
        let config = Config::default();
        let neighbor_provider =
            Arc::new(NeighborProvider::<u32>::new_with_config(120, config).unwrap());

        let mut set = JoinSet::new();
        for _ in 0..5 {
            let neighbor_provider_clone = Arc::clone(&neighbor_provider);
            set.spawn(async move {
                for i in 0..5 {
                    neighbor_provider_clone
                        .set_neighbors(i as u32, &[1, 2, 3, 4, 5])
                        .unwrap();
                }

                let mut result = AdjacencyList::with_capacity(neighbor_provider_clone.dim);
                for i in 0..5 {
                    neighbor_provider_clone
                        .get_neighbors(i as u32, &mut result)
                        .unwrap();

                    assert_eq!(&[1, 2, 3, 4, 5], &*result);
                }
            });
        }

        while let Some(res) = set.join_next().await {
            res.unwrap();
        }
    }
}
