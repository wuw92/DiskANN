/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use benchmarks::bf_tree_bench::{benchmark_bf_tree_insert, benchmark_bf_tree_search};
use criterion::{criterion_group, criterion_main};
mod benchmarks;

criterion_group!(benches, benchmark_bf_tree_insert, benchmark_bf_tree_search,);

criterion_main!(benches);
