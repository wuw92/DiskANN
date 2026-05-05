/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use benchmarks_iai::bf_tree_iai::{bf_tree_insert_bench_iai, bf_tree_search_bench_iai};
use iai_callgrind::{main, EventKind, LibraryBenchmarkConfig, RegressionConfig};
mod benchmarks_iai;

main!(
    config = LibraryBenchmarkConfig::default()
        .regression(
            RegressionConfig::default()
                .limits([(EventKind::Ir, 5.0), (EventKind::EstimatedCycles, 5.0)])
        );
    library_benchmark_groups =
        bf_tree_insert_bench_iai,
        bf_tree_search_bench_iai,
);
