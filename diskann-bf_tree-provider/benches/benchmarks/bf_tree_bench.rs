/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */
use std::{num::NonZeroUsize, sync::Arc, time::Duration};

use bf_tree::Config;
use criterion::Criterion;
use diskann::{
    graph::{self, DiskANNIndex, search::Knn, search_output_buffer},
    provider::DefaultContext,
};
use diskann_bf_tree_provider::provider::{BfTreeProvider, BfTreeProviderParameters};
use diskann_providers::{
    index::diskann_async,
    model::graph::provider::async_::common::{FullPrecision, NoDeletes},
    storage::{FileStorageProvider, StorageReadProvider},
    utils::{VectorDataIterator, create_thread_pool_for_bench},
};
use diskann_utils::{io::read_bin, views::MatrixView};
use diskann_vector::distance::Metric;
use tokio::runtime::Runtime;

pub fn benchmark_bf_tree_insert(c: &mut Criterion) {
    let mut group = c.benchmark_group("bf-tree-async-insert");
    group
        .measurement_time(Duration::from_secs(3))
        .sample_size(10);
    let rt = Runtime::new().unwrap();
    group.bench_function("BfTree insert", |f| {
        f.iter(|| {
            rt.block_on(async {
                bf_tree_insert_sift_256().await;
            });
        });
    });
}

pub fn benchmark_bf_tree_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("bf-tree-async-search");
    group
        .measurement_time(Duration::from_secs(3))
        .sample_size(10);
    let rt = Runtime::new().unwrap();

    // Build the index once outside the measured loop.
    let (index, queries) = rt.block_on(async { bf_tree_build_sift_256().await });

    group.bench_function("BfTree search", |f| {
        f.iter(|| {
            rt.block_on(async {
                bf_tree_search(&index, &queries).await;
            });
        });
    });
}

async fn bf_tree_build_sift_256() -> (Arc<DiskANNIndex<BfTreeProvider<f32>>>, Vec<Vec<f32>>) {
    let index = bf_tree_setup_index().await;

    let storage_provider = FileStorageProvider;
    let dataset_iterator = VectorDataIterator::<FileStorageProvider, f32>::new(
        get_test_file_path("test_data/sift/siftsmall_learn_256pts.fbin").as_str(),
        Option::None,
        &storage_provider,
    )
    .unwrap();

    for (pos, (vector, _associated_data)) in dataset_iterator.enumerate() {
        index
            .insert(FullPrecision, &DefaultContext, &(pos as u32), &vector)
            .await
            .unwrap();
    }

    let train_data = read_bin::<f32>(
        &mut storage_provider
            .open_reader(
                get_test_file_path("test_data/sift/siftsmall_learn_256pts.fbin").as_str(),
            )
            .unwrap(),
    )
    .unwrap();

    let queries: Vec<Vec<f32>> = (0..10).map(|i| train_data.row(i).to_vec()).collect();

    (index, queries)
}

async fn bf_tree_search(index: &DiskANNIndex<BfTreeProvider<f32>>, queries: &[Vec<f32>]) {
    let top_k = 10;
    let search_l = 20;
    let mut ids = vec![0u32; top_k];
    let mut distances = vec![0.0f32; top_k];

    for query in queries {
        let mut output = search_output_buffer::IdDistance::new(&mut ids, &mut distances);
        let search_params = Knn::new_default(top_k, search_l).unwrap();
        index
            .search(
                search_params,
                &FullPrecision,
                &DefaultContext,
                query.as_slice(),
                &mut output,
            )
            .await
            .unwrap();
    }
}

async fn bf_tree_insert_sift_256() {
    let index = bf_tree_setup_index().await;

    let storage_provider = FileStorageProvider;
    let dataset_iterator = VectorDataIterator::<FileStorageProvider, f32>::new(
        get_test_file_path("test_data/sift/siftsmall_learn_256pts.fbin").as_str(),
        Option::None,
        &storage_provider,
    )
    .unwrap();

    for (pos, (vector, _associated_data)) in dataset_iterator.enumerate() {
        index
            .insert(FullPrecision, &DefaultContext, &(pos as u32), &vector)
            .await
            .unwrap();
    }
}

async fn bf_tree_setup_index() -> Arc<DiskANNIndex<BfTreeProvider<f32>>> {
    let l = 10;
    let target_degree = 32;
    let file_path = "test_data/sift/siftsmall_learn_256pts.fbin";

    let storage_provider = FileStorageProvider;
    let train_data = read_bin::<f32>(
        &mut storage_provider
            .open_reader(get_test_file_path(file_path).as_str())
            .unwrap(),
    )
    .unwrap();

    let dim = train_data.ncols();
    let num_points = train_data.nrows();

    let pool = create_thread_pool_for_bench();
    let pq_chunk_table = diskann_async::train_pq(
        train_data.as_view(),
        32,
        &mut diskann_providers::utils::create_rnd_in_tests(),
        pool.as_ref(),
    )
    .unwrap();

    let conf = graph::config::Builder::new(
        target_degree,
        graph::config::MaxDegree::default_slack(),
        l,
        (Metric::L2).into(),
    )
    .build()
    .unwrap();

    let start_point_data = vec![0.0f32; dim];
    let start_points = MatrixView::try_from(start_point_data.as_slice(), 1, dim).unwrap();

    let params = BfTreeProviderParameters {
        max_points: num_points,
        num_start_points: NonZeroUsize::new(1).unwrap(),
        dim,
        metric: Metric::L2,
        max_fp_vecs_per_fill: None,
        max_degree: conf.max_degree_u32().get(),
        vector_provider_config: Config::default(),
        quant_vector_provider_config: Config::default(),
        neighbor_list_provider_config: Config::default(),
        graph_params: None,
    };

    let provider =
        BfTreeProvider::<f32>::new(params, start_points, pq_chunk_table, NoDeletes).unwrap();

    Arc::new(DiskANNIndex::new(conf, provider, None))
}

fn get_test_file_path(relative_path: &str) -> String {
    diskann_utils::workspace_root()
        .join(relative_path)
        .to_string_lossy()
        .into_owned()
}
