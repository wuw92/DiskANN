/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Graph-index benchmarks backed by `BfTreeProvider` (parallel to the inmem
//! `FullPrecision` benchmarks in `super::benchmarks`).

use diskann_benchmark_runner::registry::Benchmarks;

#[cfg(feature = "bf_tree_provider")]
pub(super) fn register_benchmarks(benchmarks: &mut Benchmarks) {
    use crate::backend::index::search::plugins;
    use half::f16;

    benchmarks.register(
        "graph-index-bftree-f32",
        imp::FullPrecisionBfTree::<f32>::new()
            .search(plugins::Topk)
            .search(plugins::Range),
    );
    benchmarks.register(
        "graph-index-bftree-f16",
        imp::FullPrecisionBfTree::<f16>::new().search(plugins::Topk),
    );
    benchmarks.register(
        "graph-index-bftree-u8",
        imp::FullPrecisionBfTree::<u8>::new().search(plugins::Topk),
    );
    benchmarks.register(
        "graph-index-bftree-i8",
        imp::FullPrecisionBfTree::<i8>::new().search(plugins::Topk),
    );
}

#[cfg(not(feature = "bf_tree_provider"))]
pub(super) fn register_benchmarks(_benchmarks: &mut Benchmarks) {
    // Without the `bf_tree_provider` feature there is nothing to register.
    // Users who request `graph-index-bftree-*` will get a "benchmark not found"
    // error from the runner, which is acceptable for an opt-in backend.
}

#[cfg(feature = "bf_tree_provider")]
mod imp {
    use std::{io::Write, sync::Arc};

    use diskann::{
        graph::DiskANNIndex,
        provider::{self, DataProvider},
        utils::VectorRepr,
    };
    use diskann_benchmark_runner::{
        dispatcher::{DispatchRule, FailureScore, MatchScore},
        utils::datatype,
        Benchmark, Checkpoint, Output,
    };
    use diskann_providers::model::graph::provider::async_::{
        bf_tree::{BfTreeProvider, BfTreeProviderParameters},
        common,
    };
    use diskann_utils::{future::AsyncFriendly, sampling::WithApproximateNorm, views::Matrix};

    use crate::{
        backend::index::{
            benchmarks::{run_build, QueryType, Strategy},
            build::single_or_multi_insert,
            result::BuildResult,
            search::plugins,
        },
        inputs::graph_index::{IndexOperation, IndexSource, SearchPhase},
    };

    /// `DataProvider` type used by these benchmarks. Keeps `NoStore`/`NoDeletes`
    /// (full-precision, no quant, no deletes) to mirror the inmem
    /// `FullPrecision` benchmarks.
    type BfTreeFull<T> = BfTreeProvider<T, common::NoStore, common::NoDeletes>;

    impl<T> QueryType for BfTreeFull<T>
    where
        T: VectorRepr,
    {
        type Element = T;
    }

    pub(super) struct FullPrecisionBfTree<T>
    where
        T: VectorRepr,
    {
        plugins: plugins::Plugins<BfTreeFull<T>, SearchPhase, Strategy<common::FullPrecision>>,
    }

    impl<T> FullPrecisionBfTree<T>
    where
        T: VectorRepr,
    {
        pub(super) fn new() -> Self {
            Self {
                plugins: plugins::Plugins::new(),
            }
        }

        pub(super) fn search<P>(mut self, plugin: P) -> Self
        where
            P: plugins::Plugin<BfTreeFull<T>, SearchPhase, Strategy<common::FullPrecision>>
                + 'static,
        {
            self.plugins.register(plugin);
            self
        }
    }

    impl<T> Benchmark for FullPrecisionBfTree<T>
    where
        T: VectorRepr
            + WithApproximateNorm
            + diskann::graph::SampleableForStart
            + std::fmt::Debug
            + Copy
            + AsyncFriendly
            + bytemuck::Pod,
        datatype::Type<T>: DispatchRule<datatype::DataType>,
    {
        type Input = IndexOperation;
        type Output = BuildResult;

        fn try_match(&self, input: &IndexOperation) -> Result<MatchScore, FailureScore> {
            use crate::inputs::graph_index::GraphProviderKind;
            if input.source.graph_provider() != GraphProviderKind::BfTree {
                return Err(FailureScore(0));
            }
            let score = datatype::Type::<T>::try_match(input.source.data_type());
            if self.plugins.is_match(&input.search_phase) {
                score
            } else {
                match score {
                    Ok(_) => Err(FailureScore(0)),
                    Err(score) => Err(score),
                }
            }
        }

        fn description(
            &self,
            f: &mut std::fmt::Formatter<'_>,
            input: Option<&IndexOperation>,
        ) -> std::fmt::Result {
            use diskann_benchmark_runner::dispatcher::{Description, Why};

            match input {
                Some(arg) => {
                    let data_type = arg.source.data_type();
                    if datatype::Type::<T>::try_match(data_type).is_err() {
                        writeln!(
                            f,
                            "Data/Query Type: {}",
                            Why::<datatype::DataType, datatype::Type<T>>::new(data_type)
                        )?;
                    }
                    if !self.plugins.is_match(&arg.search_phase) {
                        writeln!(
                            f,
                            "Unsupported search phase: \"{}\" - expected one of {}",
                            arg.search_phase.kind(),
                            self.plugins.format_kinds(),
                        )?;
                    }
                    Ok(())
                }
                None => {
                    writeln!(
                        f,
                        "Data/Query Type: {}",
                        Description::<datatype::DataType, datatype::Type<T>>::new()
                    )?;
                    writeln!(f, "Search Kinds: {}", self.plugins.format_kinds())
                }
            }
        }

        fn run(
            &self,
            input: &IndexOperation,
            checkpoint: Checkpoint<'_>,
            mut output: &mut dyn Output,
        ) -> anyhow::Result<BuildResult> {
            writeln!(output, "{}", input)?;
            let (index, build_stats) = match &input.source {
                IndexSource::Build(build) => {
                    let (index, build_stats) = run_build(
                        build,
                        common::FullPrecision,
                        None,
                        output,
                        |data| {
                            // Compute start points per the requested strategy.
                            let start_points =
                                build.start_point_strategy.compute(data).map_err(|e| {
                                    anyhow::anyhow!("start point compute failed: {}", e)
                                })?;
                            let params: BfTreeProviderParameters =
                                build.bftree_parameters(data.nrows(), data.ncols());
                            let provider = BfTreeProvider::<T, _, _>::new(
                                params,
                                start_points.as_view(),
                                common::NoStore,
                                common::NoDeletes,
                            )?;
                            let cfg = build.try_as_config()?.build()?;
                            Ok(Arc::new(DiskANNIndex::new(cfg, provider, None)))
                        },
                        single_or_multi_insert,
                    )?;

                    if build.save_path.is_some() {
                        // Phase 1 of the rocksdb integration plan keeps
                        // save/load disabled on the bf_tree benchmark path
                        // (the only `SaveWith` impl is for the
                        // `TableDeleteProviderAsync` deletion variant, which
                        // we don't use here).
                        return Err(anyhow::anyhow!(
                            "graph-index-bftree-* does not support save_path in Phase 1",
                        ));
                    }

                    (index, Some(build_stats))
                }
                IndexSource::Load(_load) => {
                    return Err(anyhow::anyhow!(
                        "graph-index-bftree-* does not support loading from disk in Phase 1; \
                         only the Build path is implemented",
                    ));
                }
            };

            checkpoint.checkpoint(&build_stats)?;

            let search_results = self.plugins.run(
                index,
                &input.search_phase,
                &Strategy::new(common::FullPrecision),
            )?;

            let result = BuildResult::new(build_stats, search_results);
            writeln!(output, "\n\n{}", result)?;
            Ok(result)
        }
    }

    // Workaround so the unused-import lint doesn't fire against `provider` /
    // `DataProvider` when compiling this module standalone.
    #[allow(dead_code)]
    fn _trait_check<DP: DataProvider + for<'a> provider::SetElement<&'a [u8]>>(_: &DP) {}
    #[allow(dead_code)]
    fn _matrix_check<T>(_: &Matrix<T>) {}
}
