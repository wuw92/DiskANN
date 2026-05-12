/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! redb quant vector provider (parallel to `super::rocksdb::quant_vector_provider`).

use std::sync::Arc;

use bytemuck::bytes_of;
use diskann::{ANNError, ANNErrorKind, ANNResult, error::IntoANNResult, utils::VectorRepr};
use diskann_quantization::CompressInto;
use diskann_utils::object_pool::ObjectPool;
use diskann_vector::distance::Metric;
use redb::Database;
use thiserror::Error;

use super::super::common::TestCallCount;
use super::super::kv_codec::quant as quant_codec;
use super::{Config, KV_TABLE, open_db, put_bytes};
use crate::{
    model::{
        distance::common::distance_table_pool,
        pq::{self, FixedChunkPQTable},
    },
    utils::BridgeErr,
};

pub struct QuantVectorProvider {
    quant_vector_index: Database,
    config: Config,
    max_vectors: usize,
    num_start_points: usize,
    pub pq_chunk_table: Arc<FixedChunkPQTable>,
    metric: Metric,
    pub(super) num_get_calls: TestCallCount,

    vec_pool: Arc<ObjectPool<Vec<f32>>>,
}

type DistanceComputer = pq::distance::DistanceComputer<Arc<FixedChunkPQTable>>;
type QueryComputer = pq::distance::QueryComputer<Arc<FixedChunkPQTable>>;

impl QuantVectorProvider {
    pub fn new_with_config(
        dist_metric: Metric,
        max_vectors: usize,
        num_start_points: usize,
        pq_chunk_table: FixedChunkPQTable,
        config: Config,
    ) -> ANNResult<Self> {
        let quant_vector_index = open_db(&config)?;
        let vec_pool = Arc::new(distance_table_pool(&pq_chunk_table));

        Ok(Self {
            max_vectors,
            num_start_points,
            quant_vector_index,
            config,
            pq_chunk_table: Arc::new(pq_chunk_table),
            metric: dist_metric,
            num_get_calls: TestCallCount::default(),
            vec_pool,
        })
    }

    pub(crate) fn metric(&self) -> Metric {
        self.metric
    }

    pub(crate) fn config(&self) -> &Config {
        &self.config
    }

    pub(crate) fn db(&self) -> &Database {
        &self.quant_vector_index
    }

    pub(crate) fn new_from_db(
        dist_metric: Metric,
        max_vectors: usize,
        num_start_points: usize,
        pq_chunk_table: FixedChunkPQTable,
        quant_vector_index: Database,
        config: Config,
    ) -> Self {
        let vec_pool = Arc::new(distance_table_pool(&pq_chunk_table));
        Self {
            max_vectors,
            num_start_points,
            quant_vector_index,
            config,
            pq_chunk_table: Arc::new(pq_chunk_table),
            metric: dist_metric,
            num_get_calls: TestCallCount::default(),
            vec_pool,
        }
    }

    #[inline(always)]
    pub fn total(&self) -> usize {
        self.max_vectors + self.num_start_points
    }

    pub fn full_dim(&self) -> usize {
        self.pq_chunk_table.get_dim()
    }

    pub fn pq_chunks(&self) -> usize {
        self.pq_chunk_table.get_num_chunks()
    }

    pub fn query_computer<T>(&self, query: &[T]) -> ANNResult<QueryComputer>
    where
        T: Copy + VectorRepr,
    {
        QueryComputer::new(
            self.pq_chunk_table.clone(),
            self.metric,
            &T::as_f32(query).into_ann_result()?,
            Some(self.vec_pool.clone()),
        )
    }

    pub fn distance_computer(&self) -> DistanceComputer {
        DistanceComputer::new(self.pq_chunk_table.clone(), self.metric)
    }

    pub(crate) fn get_vector_into(&self, i: usize, buffer: &mut [u8]) -> ANNResult<()> {
        let expected = buffer.len();
        if buffer.len() != expected {
            #[derive(Debug, Error)]
            #[error("expected a buffer with dim {0}, instead got {1}")]
            struct WrongDim(usize, usize);

            return Err(ANNError::new(
                ANNErrorKind::IndexError,
                WrongDim(expected, buffer.len()),
            ));
        }

        self.num_get_calls.increment();

        let tx = self
            .quant_vector_index
            .begin_read()
            .map_err(|e| ANNError::log_index_error(format!("redb begin_read failed: {}", e)))?;
        let table = tx
            .open_table(KV_TABLE)
            .map_err(|e| ANNError::log_index_error(format!("redb open_table failed: {}", e)))?;
        let value = table
            .get(&bytes_of(&i)[..])
            .map_err(|e| ANNError::log_index_error(format!("redb get failed: {}", e)))?;

        let g = match value {
            Some(g) => g,
            None => {
                return ANNResult::Err(ANNError::log_index_error(format!(
                    "The redb entry for vector id {} is not found",
                    i,
                )));
            }
        };

        let bytes = g.value();
        quant_codec::validate_read_size("redb", i, bytes.len(), expected)?;

        buffer.copy_from_slice(bytes);
        Ok(())
    }

    pub(crate) fn get_vector_sync(&self, i: usize) -> ANNResult<Vec<u8>> {
        let mut value = vec![0u8; self.pq_chunks()];
        self.get_vector_into(i, &mut value)?;
        Ok(value)
    }

    pub(crate) fn set_vector_sync<T>(&self, i: usize, v: &[T]) -> ANNResult<()>
    where
        T: Copy + VectorRepr,
    {
        let vf32: &[f32] = &T::as_f32(v).into_ann_result()?;
        quant_codec::validate_set(i, self.total(), vf32.len(), self.full_dim())?;

        let key = bytes_of::<usize>(&i);

        let dim = self.pq_chunk_table.get_num_chunks();
        let quant_vector = &mut vec![0u8; dim];

        self.pq_chunk_table
            .compress_into(vf32, quant_vector)
            .bridge_err()?;

        put_bytes(&self.quant_vector_index, key, &*quant_vector)
            .map_err(|e| ANNError::log_index_error(format!("redb put failed: {}", e)))?;

        Ok(())
    }

    #[cfg(test)]
    pub(crate) fn set_quant_vector(&self, i: usize, v: &[u8]) -> ANNResult<()> {
        quant_codec::validate_set_quant(i, self.total(), v.len(), self.pq_chunks())?;

        let key = bytes_of::<usize>(&i);

        put_bytes(&self.quant_vector_index, key, v)
            .map_err(|e| ANNError::log_index_error(format!("redb put failed: {}", e)))?;

        Ok(())
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use diskann::ANNErrorKind;
    use diskann_vector::{DistanceFunction, PreprocessedDistanceFunction, distance::Metric};
    use tokio::task::JoinSet;

    use super::*;

    #[tokio::test]
    async fn common_errors() {
        let dim = 5;
        let centroid = vec![0.0; dim];
        let offsets = vec![0, dim];
        let full_pivot_data = vec![0.0; 256 * dim];

        let pq_chunk_table =
            FixedChunkPQTable::new(dim, full_pivot_data.into(), centroid.into(), offsets.into())
                .unwrap();

        let config = Config::default();
        let provider =
            QuantVectorProvider::new_with_config(Metric::L2, 10, 1, pq_chunk_table, config)
                .unwrap();

        let result = provider.set_quant_vector(20, &[]).unwrap_err();
        assert_eq!(result.kind(), ANNErrorKind::IndexError);

        let result = provider.set_vector_sync::<f32>(20, &[]).unwrap_err();
        assert_eq!(result.kind(), ANNErrorKind::IndexError);

        let result = provider.set_quant_vector(0, &[]).unwrap_err();
        assert_eq!(result.kind(), ANNErrorKind::IndexError);
    }

    fn create_test_provider() -> QuantVectorProvider {
        let num_points = 3;
        let frozen_points = 2;
        let dim = 2;

        let table = FixedChunkPQTable::new(
            dim,
            Box::new([0.0, 0.0, 1.0, 1.0, 2.0, 2.0]),
            Box::new([0.0, 0.0]),
            Box::new([0, dim]),
        )
        .unwrap();

        let config = Config::default();
        let provider = QuantVectorProvider::new_with_config(
            Metric::L2,
            num_points,
            frozen_points,
            table,
            config,
        )
        .unwrap();

        assert_eq!(provider.total(), num_points + frozen_points);
        assert_eq!(provider.full_dim(), dim);

        provider.set_vector_sync(0, &[-1.5, -1.5]).unwrap();
        provider.set_vector_sync(1, &[-0.5, -0.5]).unwrap();
        provider.set_vector_sync(2, &[0.5, 0.5]).unwrap();
        provider.set_vector_sync(3, &[1.5, 1.5]).unwrap();
        provider.set_vector_sync(4, &[2.5, 2.5]).unwrap();
        provider
    }

    #[tokio::test]
    async fn test_similarity_function() {
        let provider = create_test_provider();

        assert_eq!(provider.get_vector_sync(0).unwrap(), &[0]);
        assert_eq!(provider.get_vector_sync(1).unwrap(), &[0]);
        assert_eq!(provider.get_vector_sync(2).unwrap(), &[0]);
        assert_eq!(provider.get_vector_sync(3).unwrap(), &[1]);
        assert_eq!(provider.get_vector_sync(4).unwrap(), &[2]);

        assert!(provider.set_vector_sync(5, &[0.0, 0.0]).is_err());
        assert!(provider.set_vector_sync(2, &[0.0]).is_err());

        let c = provider.query_computer(&[-0.5, -0.5]).unwrap();
        let expected: f32 = 1.5 * 1.5 * 2.0;
        assert_eq!(
            c.evaluate_similarity(&provider.get_vector_sync(3).unwrap()),
            expected
        );

        let d = provider.distance_computer();
        assert_eq!(
            d.evaluate_similarity(
                provider.get_vector_sync(0).unwrap().as_slice(),
                provider.get_vector_sync(3).unwrap().as_slice(),
            ),
            2.0
        );

        let slice: &[f32] = &[-0.5, -0.5];
        assert_eq!(
            d.evaluate_similarity(slice, &provider.get_vector_sync(3).unwrap()),
            expected,
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 5)]
    async fn test_parallel_tree_traversal() {
        let dim = 2;
        let centroid = vec![0.0; dim];
        let offsets = vec![0, dim];
        let full_pivot_data = vec![0.0; 256 * dim];
        let pq_chunk_table =
            FixedChunkPQTable::new(dim, full_pivot_data.into(), centroid.into(), offsets.into())
                .unwrap();

        let config = Config::default();
        let provider = Arc::new(
            QuantVectorProvider::new_with_config(Metric::L2, 10, 1, pq_chunk_table, config)
                .unwrap(),
        );
        let mut set = JoinSet::new();
        for i in 0..11 {
            let vector = vec![i as f32, (i + 1) as f32];
            let provider_clone = Arc::clone(&provider);
            set.spawn(async move { provider_clone.set_vector_sync(i as usize, &vector).unwrap() });
        }

        while let Some(res) = set.join_next().await {
            res.unwrap();
        }

        let dim = provider.pq_chunk_table.get_num_chunks();
        let mut quant_vector: Vec<u8> = vec![0; dim];
        let quant_vector_ref: &mut [u8] = &mut quant_vector;

        for i in 0..11 {
            let quant_vector = provider.get_vector_sync(i as usize).unwrap();
            provider
                .pq_chunk_table
                .compress_into(&[(i as f32), (i + 1) as f32], quant_vector_ref)
                .unwrap();
            assert_eq!(&quant_vector_ref, &quant_vector);
        }
    }
}
