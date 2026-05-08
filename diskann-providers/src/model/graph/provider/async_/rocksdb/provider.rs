/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{fmt::Debug, future::Future, num::NonZeroUsize, sync::Arc};

use serde::{Deserialize, Serialize};

use super::Config;
use diskann::{
    ANNError, ANNResult, default_post_processor,
    graph::{
        AdjacencyList, DiskANNIndex, SearchOutputBuffer,
        glue::{
            self, Batch, DefaultPostProcessor, ExpandBeam, InplaceDeleteStrategy, InsertStrategy,
            MultiInsertStrategy, PruneStrategy, SearchExt, SearchStrategy,
        },
        workingset::{self, map},
    },
    neighbor::Neighbor,
    provider::{
        Accessor, BuildDistanceComputer, BuildQueryComputer, DataProvider, DefaultContext,
        DelegateNeighbor, Delete, ElementStatus, HasId, NeighborAccessor, NeighborAccessorMut,
        NoopGuard, SetElement,
    },
    utils::{IntoUsize, VectorRepr},
};
use diskann_utils::{future::AsyncFriendly, views::MatrixView};
use diskann_vector::{DistanceFunction, distance::Metric};

use crate::model::{
    graph::provider::async_::{
        TableDeleteProviderAsync,
        common::{CreateDeleteProvider, FullPrecision, Hybrid, NoDeletes, NoStore, Panics},
        distances,
        postprocess::{AsDeletionCheck, DeletionCheck, RemoveDeletedIdsAndCopy},
    },
    pq::{self, FixedChunkPQTable},
};

use super::{
    neighbor_provider::NeighborProvider, quant_vector_provider::QuantVectorProvider,
    vector_provider::VectorProvider,
};

/////////////////////
// RocksdbProvider //
/////////////////////

/// An Bf-tree based implementation of a [`DataProvider`] built around the idea of having up to
/// two layers of vector stores: a full-precision store and an optional quantized store.
/// This provider uses the identity mapping between external and internal vector IDs.
///
/// # Type Parameters:
///
/// * `T`: The primitive element type of the full-precision vector. This is typically some
///   type like `f32` or `half::f16`.
///
/// * `Q`: The full type of the quant vector store. This is not constrained by a trait and
///   rather relies on implementation for several concrete types, including:
///
///   - [`BfTreeQuantVectorProviderAsync`]: A Bf-Tree based PQ-based quantized vector store.
///   - [`NoStore`]: Disable quantization altogether. Note that this disables all
///     methods reached through quantization based [`Accessor`]s at compile-time.
///
/// * `D`: The type of the deleted vector store. Like the quantized store, this is also
///   not constrained by a trait and rather relies on implementation for concrete types.
///   These are:
///
///   - [`NoDeletes`]: Do not support deletion at all (this disables implementation of
///     the [`Delete`] trait.
///   - [`TableDeleteProviderAsync`]: A bitmap storing deletion information.
///
/// * `Ctx`: A parameter controlling the [`ExecutionContext`] to be associated with this
///   provider. For the majority of cases, this is [`DefaultContext`], but is left as
///   a parameter to allow extension.
///
/// # Indexing Strategies
///
/// * [`FullPrecision`]: The strategies implemented by [`FullPrecision`] only retrieve data
///   from the full-precision portion of the index. No quantized vectors are used.
///
///   During search, start points are filtered from the final results.
///
/// * [`Hybrid`]: The strategies implemented by [`Hybrid`] can use a mix of quantized
///   and full-precision vectors.
///
///   - Search: During search, quantized vectors are used with reranking applied to the
///     results before returning.
///
///   - Insertion: Quantized vectors are used during the search phase. During the pruning
///     phase, a hybrid of quantized and full-precision vectors are used.
///
///     The ratio of full-precision and quantized vectors is controlled by the
///     `max_fp_vecs_per_prune` parameter, which adjusts the implementation of [`Fill`].
///
/// # Examples
///
/// The following code demonstrates how to instantiate and use the `RocksdbProvider` in
/// a number of different scenarios.
///
/// ## Full-Precision Only - No Deletes
///
/// This example demonstrates how to create a `RocksdbProvider` that only supports
/// full-precision vectors.
/// ```ignore
/// use diskann_providers::model::graph::provider::async_::{
///     bf_tree::{
///         RocksdbProvider, RocksdbProviderParameters
///     },
///     common::{NoStore, NoDeletes},
/// };
/// use diskann_vector::distance::Metric;
/// use bf_tree::Config;
/// use std::num::NonZeroUsize;
///
/// let parameters = RocksdbProviderParameters {
///     max_points: 5,
///     num_start_points: NonZeroUsize::new(1).unwrap(),
///     dim: 4,
///     metric: Metric::L2,
///     max_fp_vecs_per_fill: None,
///     max_degree: 32,
///     vector_provider_config: Config::default(),
///     quant_vector_provider_config: Config::default(),
///     neighbor_list_provider_config: Config::default(),
///     graph_params: None,
/// };
///
/// // Create a table that supports 5 points and 1 start point.
/// let provider = RocksdbProvider::<f32, _>::new_empty(
///     parameters,
///     NoStore,
///     NoDeletes,
/// );
/// ```
///
/// ## Full-Precision and PQ - No Deletes
///
/// To create a two-level provider with a PQ-based quant vector store, a
/// [`FixedChunkPQTable`] can be supplied for the `quant_precursor` argument, as this
/// implements the [`CreateQuantProvider`] trait.
/// ```ignore
/// use diskann_providers::model::{
///     pq::FixedChunkPQTable,
///     graph::provider::async_::{
///         bf_tree::{
///             RocksdbProvider, RocksdbProviderParameters
///     },
///     common::NoDeletes,
///     },
/// };
/// use diskann_vector::distance::Metric;
/// use bf_tree::Config;
/// use std::num::NonZeroUsize;
///
/// // An example PQ table.
/// let dim = 4;
/// let table = FixedChunkPQTable::new(
///     dim,
///     Box::new([0.0, 0.0, 0.0, 0.0]),
///     Box::new([0.0, 0.0, 0.0, 0.0]),
///     Box::new([0, dim]),
/// ).unwrap();
///
/// let parameters = RocksdbProviderParameters {
///     max_points: 5,
///     num_start_points: NonZeroUsize::new(1).unwrap(),
///     dim: 4,
///     metric: Metric::L2,
///     max_fp_vecs_per_fill: None,
///     max_degree: 32,
///     vector_provider_config: Config::default(),
///     quant_vector_provider_config: Config::default(),
///     neighbor_list_provider_config: Config::default(),
///     graph_params: None,
/// };
///
/// // Create a table that supports 5 points and 1 start point.
/// let provider = RocksdbProvider::<f32>::new_empty(
///     parameters,
///     table,
///     NoDeletes,
/// );
/// ```
///
/// ## Full-Precision and PQ - With Deletes.
///
/// If deletes are desired, than the type [`TableBasedDeletes`] can be passed to the
/// constructor.
/// ```ignore
/// use diskann_providers::model::{
///     pq::FixedChunkPQTable,
///     graph::provider::async_::{
///     bf_tree::{
///         RocksdbProvider, RocksdbProviderParameters
///     },
///     common::TableBasedDeletes,
///     },
/// };
/// use diskann_vector::distance::Metric;
/// use bf_tree::Config;
/// use std::num::NonZeroUsize;
///
/// // An example PQ table.
/// let dim = 4;
/// let table = FixedChunkPQTable::new(
///     dim,
///     Box::new([0.0, 0.0, 0.0, 0.0]),
///     Box::new([0.0, 0.0, 0.0, 0.0]),
///     Box::new([0, dim]),
/// ).unwrap();
///
/// let parameters = RocksdbProviderParameters {
///     max_points: 5,
///     num_start_points: NonZeroUsize::new(1).unwrap(),
///     dim: 4,
///     metric: Metric::L2,
///     max_fp_vecs_per_fill: None,
///     max_degree: 32,
///     vector_provider_config: Config::default(),
///     quant_vector_provider_config: Config::default(),
///     neighbor_list_provider_config: Config::default(),
///     graph_params: None,
/// };
///
/// // Create a table that supports 5 points and 1 start point.
/// let provider = RocksdbProvider::<f32, _, _>::new_empty(
///     parameters,
///     table,
///     TableBasedDeletes,
/// );
/// ```
pub struct RocksdbProvider<T, Q = QuantVectorProvider, D = NoDeletes>
where
    T: VectorRepr,
{
    // The quant vector store. If `Q == NoStore`, the quantized operations are disabled.
    //
    pub(super) quant_vectors: Q,

    // The full vector store.
    //
    pub(super) full_vectors: VectorProvider<T>,

    // Provider that holds the graph structure as neighbors of vectors.
    //
    pub(crate) neighbor_provider: NeighborProvider<u32>,

    // The delete provider. If `D == NoDeletes`, then delete related operations are disabled.
    //
    pub(super) deleted: D,

    // A parameter controlling hybrid pruning, where some set of full-precision vectors are
    // fetched and the rest are quantized vectors
    //
    pub(super) max_fp_vecs_per_fill: usize,

    // The metric to use for distances
    //
    pub(super) metric: Metric,

    // Graph configuration parameters for persistence
    //
    pub(crate) graph_params: Option<GraphParams>,
}

#[derive(Debug, Clone)]
pub struct RocksdbProviderParameters {
    // The maximum number of valid points that provider can hold.
    pub max_points: usize,

    // The number of start points (frozen points) for graph search entry.
    pub num_start_points: NonZeroUsize,

    // The dimension of the full-precision vectors.
    pub dim: usize,

    // The metric to use for distance computations
    pub metric: Metric,

    // If quantization is used, this parameter controls how many full-precision
    // vectors are retrieved for each fill operation
    pub max_fp_vecs_per_fill: Option<usize>,

    // The maximum number of neighbors to store for each vector
    pub max_degree: u32,

    // bf-tree config for vector provider
    pub vector_provider_config: Config,

    // bf-tree config for quant vector provider
    pub quant_vector_provider_config: Config,

    // bf-tree config for neighbor list provider
    pub neighbor_list_provider_config: Config,

    // Optional graph configuration parameters for persistence
    pub graph_params: Option<GraphParams>,
}

pub type Index<T, D = NoDeletes> = Arc<DiskANNIndex<RocksdbProvider<T, NoStore, D>>>;
pub type QuantIndex<T, Q, D = NoDeletes> = Arc<DiskANNIndex<RocksdbProvider<T, Q, D>>>;

impl<T, Q, D> RocksdbProvider<T, Q, D>
where
    T: VectorRepr,
{
    /// Construct a new, unpopulated data provider.
    ///
    /// # Arguments
    /// * `params`: An instance of [`RocksdbProviderParameters`] collecting shared
    ///   configuration information.
    /// * `quant_precursor`: A precursor type for the quantizer layer.
    /// * `delete_precursor`: A precursor type for the delete layer.
    /// * `neighbor_precursor`: A precursor type for the neighbor layer.
    ///   or the neighbor layer
    pub fn new_empty<TQ, TD>(
        params: RocksdbProviderParameters,
        quant_precursor: TQ,
        delete_precursor: TD,
    ) -> ANNResult<Self>
    where
        TQ: CreateQuantProvider<Target = Q>,
        TD: CreateDeleteProvider<Target = D>,
    {
        let num_start_points = params.num_start_points.get();

        Ok(Self {
            quant_vectors: quant_precursor.create(
                params.max_points,
                num_start_points,
                params.metric,
                params.quant_vector_provider_config,
            )?,
            full_vectors: VectorProvider::new_with_config(
                params.max_points,
                params.dim,
                num_start_points,
                params.vector_provider_config,
            )?,
            neighbor_provider: NeighborProvider::new_with_config(
                params.max_degree,
                params.neighbor_list_provider_config,
            )?,
            deleted: delete_precursor.create(params.max_points + num_start_points),
            max_fp_vecs_per_fill: params.max_fp_vecs_per_fill.unwrap_or(usize::MAX),
            metric: params.metric,
            graph_params: params.graph_params,
        })
    }

    /// Construct a new data provider with start points initialized.
    ///
    /// This is the primary constructor for `RocksdbProvider`. It creates the provider
    /// and sets the start points in one operation.
    ///
    /// # Arguments
    /// * `params`: An instance of [`RocksdbProviderParameters`] collecting shared
    ///   configuration information.
    /// * `start_points`: A matrix view containing the start point vectors. The number
    ///   of rows must match `params.num_start_points.get()`.
    /// * `quant_precursor`: A precursor type for the quantizer layer.
    /// * `delete_precursor`: A precursor type for the delete layer.
    ///
    /// # Type Constraints
    /// * `Self: StartPoint<T>` - The provider must implement the `StartPoint` trait.
    pub fn new<TQ, TD>(
        params: RocksdbProviderParameters,
        start_points: MatrixView<'_, T>,
        quant_precursor: TQ,
        delete_precursor: TD,
    ) -> ANNResult<Self>
    where
        Self: StartPoint<T>,
        TQ: CreateQuantProvider<Target = Q>,
        TD: CreateDeleteProvider<Target = D>,
    {
        // Early validation before allocating resources
        if start_points.nrows() != params.num_start_points.get() {
            return Err(ANNError::log_async_index_error(format!(
                "start_points matrix has {} rows, but params.num_start_points is {}",
                start_points.nrows(),
                params.num_start_points.get(),
            )));
        }

        let provider = Self::new_empty(params.clone(), quant_precursor, delete_precursor)?;
        provider.set_start_points(Hidden(()), start_points)?;
        {
            // Initialize all neighborhoods to be empty lists.
            // This is a temporary solution to the problem of trying to access
            // an uninitialized neighbor list in functions `consolidate_deletes` and
            // `consolidate_simple` and getting an error. This is a stop-gap solution
            // until BF-tree API is improved to handle `exists` queries.
            for i in 0..params.max_points {
                let vector_id = i as u32;
                provider.neighbor_provider.set_neighbors(vector_id, &[])?;
            }
        }
        Ok(provider)
    }

    // /// Return a predicate that can be applied to `Iter::filter` to remove start points
    // /// from an iterator of neighbors.
    // ///
    // /// This is used during post-processing
    // ///
    // pub(crate) fn is_not_start_point(&self) -> impl Fn(&Neighbor<u32>) -> bool {
    //     let range = self.full_vectors.start_point_range();
    //     move |neighbor| !range.contains(&neighbor.id.into_usize())
    // }

    /// Return a vector of starting points.
    pub fn starting_points(&self) -> ANNResult<Vec<u32>> {
        Ok(self.full_vectors.starting_points()?)
    }

    /// An iterator over all ids including start points (even if they are deleted).
    pub fn iter(&self) -> std::ops::Range<u32> {
        0..(self.full_vectors.total() as u32)
    }

    pub fn num_start_points(&self) -> usize {
        self.full_vectors.num_start_points
    }

    /// Return the maximum number of points (excluding frozen/start points)
    pub fn max_points(&self) -> usize {
        self.full_vectors.max_vectors
    }

    /// Return the vector dimension
    pub fn dim(&self) -> usize {
        self.full_vectors.dim()
    }

    /// Return the distance metric
    pub fn metric(&self) -> Metric {
        self.metric
    }

    /// Return the maximum degree from the neighbor provider
    pub fn max_degree(&self) -> u32 {
        self.neighbor_provider.max_degree()
    }
}

impl<T, Q> RocksdbProvider<T, Q, TableDeleteProviderAsync>
where
    T: VectorRepr,
{
    /// A temporary method while development of deletion is in progress
    ///
    pub fn clear_delete_set(&self) {
        self.deleted.clear();
    }
}

impl<T, D> RocksdbProvider<T, QuantVectorProvider, D>
where
    T: VectorRepr,
{
    /// Return the number of vector reads for full-precision and quant-vectors respectively
    ///
    pub fn counts_for_get_vector(&self) -> (usize, usize) {
        (
            self.full_vectors.num_get_calls.get(),
            self.quant_vectors.num_get_calls.get(),
        )
    }
}

impl<T, D> RocksdbProvider<T, NoStore, D>
where
    T: VectorRepr,
{
    /// Return the number of vector reads for full-precision and quant-vectors respectively
    ///
    pub fn counts_for_get_vector(&self) -> (usize, usize) {
        (self.full_vectors.num_get_calls.get(), 0)
    }
}

/// Allow `&RocksdbProvider` to implement `IntoIter`
///
impl<T, Q, D> IntoIterator for &RocksdbProvider<T, Q, D>
where
    T: VectorRepr,
{
    type Item = u32;
    type IntoIter = std::ops::Range<u32>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

/// A helper trait to select the quant vector store.
///
/// This is also implemented for [`NoStore`], which explicitly disables deletion
/// related functionality
///
pub trait CreateQuantProvider {
    // The type of the created quant provider.
    //
    type Target;

    // Create a quant provider capable of tracking `max_points` with and additional
    // `frozen_points` at the end.
    //
    fn create(
        self,
        max_points: usize,
        frozen_points: usize,
        metric: Metric,
        config: Config,
    ) -> ANNResult<Self::Target>;
}

impl CreateQuantProvider for NoStore {
    type Target = NoStore;
    fn create(
        self,
        _max_points: usize,
        _frozen_points: usize,
        _metric: Metric,
        _config: Config,
    ) -> ANNResult<Self::Target> {
        Ok(self)
    }
}

/// Allow a `FixedChunkPQTable` to be promoted to full quant vector store.
///
impl CreateQuantProvider for FixedChunkPQTable {
    type Target = QuantVectorProvider;
    fn create(
        self,
        max_points: usize,
        frozen_points: usize,
        metric: Metric,
        config: Config,
    ) -> ANNResult<Self::Target> {
        QuantVectorProvider::new_with_config(metric, max_points, frozen_points, self, config)
    }
}

impl<T, Q, D> RocksdbProvider<T, Q, D>
where
    T: VectorRepr,
    Q: AsyncFriendly,
    D: AsyncFriendly,
{
    pub fn neighbors(&self) -> &NeighborProvider<u32> {
        &self.neighbor_provider
    }
}

///////////////////
// Data Provider //
///////////////////

impl<T, Q, D> DataProvider for RocksdbProvider<T, Q, D>
where
    T: VectorRepr,
    Q: AsyncFriendly,
    D: AsyncFriendly,
{
    type Context = DefaultContext;

    // The `RocksdbProvider` uses the identity map for IDs.
    //
    type InternalId = u32;

    // The `RocksdbProvider` uses the identity map for IDs.
    //
    type ExternalId = u32;

    // Use a general error type for now.
    //
    type Error = ANNError;

    // No insert-ID recovery.
    type Guard = NoopGuard<u32>;

    // Translate an external id to its corresponding internal id.
    //
    fn to_internal_id(
        &self,
        _context: &DefaultContext,
        gid: &Self::ExternalId,
    ) -> Result<Self::InternalId, Self::Error> {
        Ok(*gid)
    }

    // Translate an internal id its corresponding external id.
    //
    fn to_external_id(
        &self,
        _context: &DefaultContext,
        id: Self::InternalId,
    ) -> Result<Self::ExternalId, Self::Error> {
        Ok(id)
    }
}

impl<T, Q, D> HasId for RocksdbProvider<T, Q, D>
where
    T: VectorRepr,
    Q: AsyncFriendly,
    D: AsyncFriendly,
{
    type Id = u32;
}

impl<'a, T, Q, D> DelegateNeighbor<'a> for RocksdbProvider<T, Q, D>
where
    T: VectorRepr,
    Q: AsyncFriendly,
    D: AsyncFriendly,
{
    type Delegate = &'a NeighborProvider<u32>;

    fn delegate_neighbor(&'a mut self) -> Self::Delegate {
        self.neighbors()
    }
}

/// Support deletes when we have a valid delete provider.
///
impl<T, Q> Delete for RocksdbProvider<T, Q, TableDeleteProviderAsync>
where
    Q: AsyncFriendly,
    T: VectorRepr,
{
    fn release(
        &self,
        _: &DefaultContext,
        id: Self::InternalId,
    ) -> impl Future<Output = Result<(), Self::Error>> + Send {
        // delete the vector from bf-tree
        if let Err(e) = self.neighbor_provider.delete_vector(id) {
            return std::future::ready(Err(e));
        }
        self.deleted.undelete(id.into_usize());
        // set its neighbors to an empty list in the neighbor provider
        // self.neighbor_provider.set_neighbors(id, &[]);
        let res = self
            .neighbor_provider
            .set_neighbors(id, &[])
            .map_err(|err| err.context(format!("resetting neighbors for undeleted id {}", id)));
        std::future::ready(res)
    }

    /// Delete an item by external ID
    ///
    #[inline]
    fn delete(
        &self,
        _context: &DefaultContext,
        gid: &Self::ExternalId,
    ) -> impl Future<Output = Result<(), Self::Error>> + Send {
        self.deleted.delete(gid.into_usize());
        std::future::ready(Ok(()))
    }

    /// Check the status via external ID
    ///
    #[inline]
    fn status_by_external_id(
        &self,
        context: &DefaultContext,
        gid: &Self::ExternalId,
    ) -> impl Future<Output = Result<ElementStatus, Self::Error>> + Send {
        // NOTE: ID translation is the identity, so we can refer to `status_by_internal_id`.
        self.status_by_internal_id(context, *gid)
    }

    /// Check the status via internal ID
    ///
    #[inline]
    fn status_by_internal_id(
        &self,
        _context: &DefaultContext,
        id: Self::InternalId,
    ) -> impl Future<Output = Result<ElementStatus, Self::Error>> + Send {
        let status = if self.deleted.is_deleted(id.into_usize()) {
            ElementStatus::Deleted
        } else {
            ElementStatus::Valid
        };
        std::future::ready(Ok(status))
    }
}

impl NeighborAccessor for &NeighborProvider<u32> {
    fn get_neighbors(
        self,
        id: Self::Id,
        neighbors: &mut AdjacencyList<Self::Id>,
    ) -> impl Future<Output = ANNResult<Self>> + Send {
        std::future::ready(self.get_neighbors(id, neighbors).map(|_| self))
    }
}

impl NeighborAccessorMut for &NeighborProvider<u32> {
    fn set_neighbors(
        self,
        vector_id: u32,
        neighbors: &[u32],
    ) -> impl Future<Output = ANNResult<Self>> + Send {
        std::future::ready(self.set_neighbors(vector_id, neighbors).map(|_| self))
    }

    fn append_vector(
        self,
        vector_id: u32,
        new_neighbor_ids: &[u32],
    ) -> impl Future<Output = ANNResult<Self>> + Send {
        std::future::ready(
            self.append_vector(vector_id, new_neighbor_ids)
                .map(|_| self),
        )
    }
}

////////////////
// SetElement //
////////////////

/// Assign to both the full-precision and quant vector stores
///
impl<T, D> SetElement<&[T]> for RocksdbProvider<T, QuantVectorProvider, D>
where
    T: VectorRepr,
    D: AsyncFriendly,
{
    type SetError = ANNError;

    /// Store the provided element in both the full-precision and quant vector stores.
    ///
    /// The process of storing the element in the quant store will compress the vector
    ///
    fn set_element(
        &self,
        _context: &Self::Context,
        id: &u32,
        element: &[T],
    ) -> impl Future<Output = Result<Self::Guard, Self::SetError>> + Send {
        // First, try adding to the quant provider.
        //
        if let Err(err) = self.quant_vectors.set_vector_sync(id.into_usize(), element) {
            return std::future::ready(Err(err));
        }

        // Next, add to the full precision provider.
        //
        if let Err(err) = self.full_vectors.set_vector_sync(id.into_usize(), element) {
            return std::future::ready(Err(err));
        }

        // Success
        //
        std::future::ready(Ok(NoopGuard::new(*id)))
    }
}

/// Assign to just the full-precision store
///
impl<T, D> SetElement<&[T]> for RocksdbProvider<T, NoStore, D>
where
    T: VectorRepr,
    D: AsyncFriendly,
{
    type SetError = ANNError;

    /// Store the provided element in just the full-precision vector stores
    ///
    fn set_element(
        &self,
        _context: &Self::Context,
        id: &u32,
        element: &[T],
    ) -> impl Future<Output = Result<Self::Guard, Self::SetError>> + Send {
        // Add to the full precision provider
        //
        if let Err(err) = self.full_vectors.set_vector_sync(id.into_usize(), element) {
            return std::future::ready(Err(err));
        }

        // Success
        //
        std::future::ready(Ok(NoopGuard::new(*id)))
    }
}

//////////////////////
// StartPoint Trait //
//////////////////////

/// A struct with a private member that cannot be constructed outside of this module.
///
/// This is used to prevent users from calling internal methods directly.
pub struct Hidden(());

/// A trait for setting the start points of a RocksdbProvider.
///
/// This trait is implemented by `RocksdbProvider` variants that support setting start points.
/// The `Hidden` parameter ensures that users cannot call `set_start_points` directly;
/// they must go through the `RocksdbProvider::new` constructor which handles this internally.
pub trait StartPoint<T> {
    /// Set the start points of the provider.
    ///
    /// # Safety
    /// This method is internal and should not be called directly by users.
    /// Use `RocksdbProvider::new` instead.
    #[doc(hidden)]
    fn set_start_points(&self, hidden: Hidden, start_points: MatrixView<'_, T>) -> ANNResult<()>;
}

////////////////////
// SetStartPoints //
////////////////////

/// Set start points for the RocksdbProvider with quantization.
///
/// This implementation sets both the full-precision and quantized vectors for each
/// start point, as well as initializing empty neighbor lists.
impl<T, D> StartPoint<T> for RocksdbProvider<T, QuantVectorProvider, D>
where
    T: VectorRepr,
    D: AsyncFriendly,
{
    fn set_start_points(&self, _hidden: Hidden, start_points: MatrixView<'_, T>) -> ANNResult<()> {
        let start_point_ids = self.full_vectors.starting_points()?;
        if start_points.nrows() != start_point_ids.len() {
            return Err(ANNError::log_async_index_error(format!(
                "expected start_points to contain `{}` rows, instead it has {}",
                start_point_ids.len(),
                start_points.nrows(),
            )));
        }

        for (id, v) in std::iter::zip(start_point_ids, start_points.row_iter()) {
            // Set the full-precision vector
            self.full_vectors.set_vector_sync(id.into_usize(), v)?;
            // Set the quantized vector
            self.quant_vectors.set_vector_sync(id.into_usize(), v)?;
            // Initialize empty neighbor list
            self.neighbor_provider.set_neighbors(id, &[])?;
        }

        Ok(())
    }
}

/// Set start points for the RocksdbProvider without quantization.
///
/// This implementation sets the full-precision vectors for each start point
/// and initializes empty neighbor lists.
impl<T, D> StartPoint<T> for RocksdbProvider<T, NoStore, D>
where
    T: VectorRepr,
    D: AsyncFriendly,
{
    fn set_start_points(&self, _hidden: Hidden, start_points: MatrixView<'_, T>) -> ANNResult<()> {
        let start_point_ids = self.full_vectors.starting_points()?;
        if start_points.nrows() != start_point_ids.len() {
            return Err(ANNError::log_async_index_error(format!(
                "expected start_points to contain `{}` rows, instead it has {}",
                start_point_ids.len(),
                start_points.nrows(),
            )));
        }

        for (id, v) in std::iter::zip(start_point_ids, start_points.row_iter()) {
            // Set the full-precision vector
            self.full_vectors.set_vector_sync(id.into_usize(), v)?;
            // Initialize empty neighbor list
            self.neighbor_provider.set_neighbors(id, &[])?;
        }

        Ok(())
    }
}

//////////////////
// FullAccessor //
//////////////////

/// An accessor for retrieving full-precision vectors from the `RocksdbProvider`.
///
/// This type implements the following traits:
///
/// * [`Accessor`] for the [`RocksdbProvider`].
/// * [`ComputerAccessor`] for comparing full-precision distances.
/// * [`BuildQueryComputer`].
///
pub struct FullAccessor<'a, T, Q, D>
where
    T: VectorRepr,
    Q: AsyncFriendly,
    D: AsyncFriendly,
{
    /// The host provider.
    provider: &'a RocksdbProvider<T, Q, D>,
    /// A buffer to store retrieved elements.
    element: Box<[T]>,
}

impl<T, Q, D> HasId for FullAccessor<'_, T, Q, D>
where
    T: VectorRepr,
    Q: AsyncFriendly,
    D: AsyncFriendly,
{
    type Id = u32;
}

impl<T, Q, D> SearchExt for FullAccessor<'_, T, Q, D>
where
    T: VectorRepr,
    Q: AsyncFriendly,
    D: AsyncFriendly,
{
    fn starting_points(&self) -> impl Future<Output = ANNResult<Vec<u32>>> {
        std::future::ready(self.provider.starting_points())
    }
}

impl<'a, T, Q, D> FullAccessor<'a, T, Q, D>
where
    T: VectorRepr,
    Q: AsyncFriendly,
    D: AsyncFriendly,
{
    pub(crate) fn new(provider: &'a RocksdbProvider<T, Q, D>) -> Self {
        Self {
            provider,
            element: (0..provider.full_vectors.dim())
                .map(|_| T::default())
                .collect(),
        }
    }
}

impl<'a, T, Q, D> DelegateNeighbor<'a> for FullAccessor<'_, T, Q, D>
where
    T: VectorRepr,
    Q: AsyncFriendly,
    D: AsyncFriendly,
{
    type Delegate = &'a NeighborProvider<u32>;

    fn delegate_neighbor(&'a mut self) -> Self::Delegate {
        self.provider.neighbors()
    }
}

impl<T, Q, D> Accessor for FullAccessor<'_, T, Q, D>
where
    T: VectorRepr,
    Q: AsyncFriendly,
    D: AsyncFriendly,
{
    /// This accessor returns a reference to a local copy of the vector.
    type Element<'a>
        = &'a [T]
    where
        Self: 'a;

    /// The reference version of `Element` is the same as `Element`.
    type ElementRef<'a> = &'a [T];

    // Choose to panic on an out-of-bounds access rather than propagate an error.
    //
    type GetError = Panics;

    /// Return the full-precision vector stored at index `i`.
    ///
    /// This function always completes synchronously
    ///
    #[inline(always)]
    fn get_element(
        &mut self,
        id: Self::Id,
    ) -> impl Future<Output = Result<Self::Element<'_>, Self::GetError>> + Send {
        // SAFETY: We've decided to live with UB (undefined behavior) that can result from
        // potentially mixing unsynchronized reads and writes on the underlying memory
        //
        #[allow(clippy::expect_used)]
        self.provider
            .full_vectors
            .get_vector_into(id.into_usize(), &mut self.element)
            .expect("Full vector provider failed to retrieve element");

        std::future::ready(Ok(&*self.element))
    }
}

impl<T, Q, D> BuildDistanceComputer for FullAccessor<'_, T, Q, D>
where
    T: VectorRepr,
    Q: AsyncFriendly,
    D: AsyncFriendly,
{
    type DistanceComputerError = Panics;
    type DistanceComputer = T::Distance;

    fn build_distance_computer(
        &self,
    ) -> Result<Self::DistanceComputer, Self::DistanceComputerError> {
        Ok(T::distance(
            self.provider.metric,
            Some(self.provider.full_vectors.dim()),
        ))
    }
}

impl<T, Q, D> BuildQueryComputer<&[T]> for FullAccessor<'_, T, Q, D>
where
    T: VectorRepr,
    Q: AsyncFriendly,
    D: AsyncFriendly,
{
    type QueryComputerError = Panics;
    type QueryComputer = T::QueryDistance;

    fn build_query_computer(
        &self,
        from: &[T],
    ) -> Result<Self::QueryComputer, Self::QueryComputerError> {
        Ok(T::query_distance(from, self.provider.metric))
    }
}
impl<T, Q, D> ExpandBeam<&[T]> for FullAccessor<'_, T, Q, D>
where
    T: VectorRepr,
    Q: AsyncFriendly,
    D: AsyncFriendly,
{
}

impl<'a, T, Q, D> AsDeletionCheck for FullAccessor<'a, T, Q, D>
where
    T: VectorRepr,
    Q: AsyncFriendly,
    D: AsyncFriendly + DeletionCheck,
{
    type Checker = D;
    fn as_deletion_check(&self) -> &D {
        &self.provider.deleted
    }
}

///////////////////
// QuantAccessor //
///////////////////

/// An accessor that retrieves the quantized portion of the [`RocksdbProvider`].
///
/// This type implements the following traits:
///
/// * [`Accessor`] for the `RocksdbProvider`.
/// * [`BuildQueryComputer`].
///
pub struct QuantAccessor<'a, T, D>
where
    T: VectorRepr,
    D: AsyncFriendly,
{
    provider: &'a RocksdbProvider<T, QuantVectorProvider, D>,
    element: Box<[u8]>,
}

impl<T, D> HasId for QuantAccessor<'_, T, D>
where
    T: VectorRepr,
    D: AsyncFriendly,
{
    type Id = u32;
}

impl<T, D> SearchExt for QuantAccessor<'_, T, D>
where
    T: VectorRepr,
    D: AsyncFriendly,
{
    fn starting_points(&self) -> impl Future<Output = ANNResult<Vec<u32>>> {
        std::future::ready(self.provider.starting_points())
    }
}

impl<'a, T, D> QuantAccessor<'a, T, D>
where
    T: VectorRepr,
    D: AsyncFriendly,
{
    pub(crate) fn new(provider: &'a RocksdbProvider<T, QuantVectorProvider, D>) -> Self {
        Self {
            provider,
            element: (0..provider.quant_vectors.pq_chunks())
                .map(|_| u8::default())
                .collect(),
        }
    }
}

impl<'a, T, D> DelegateNeighbor<'a> for QuantAccessor<'_, T, D>
where
    T: VectorRepr,
    D: AsyncFriendly,
{
    type Delegate = &'a NeighborProvider<u32>;
    fn delegate_neighbor(&'a mut self) -> Self::Delegate {
        self.provider.neighbors()
    }
}

impl<T, D> Accessor for QuantAccessor<'_, T, D>
where
    T: VectorRepr,
    D: AsyncFriendly,
{
    /// This accessor returns a reference to a local copy of the element.
    type Element<'a>
        = &'a [u8]
    where
        Self: 'a;

    /// The reference version of `Element` is simply `Element`.
    type ElementRef<'a> = &'a [u8];

    // ANNError on access failures in bf-tree
    //
    type GetError = ANNError;

    /// Return the quantized vector stored at index `i`.
    ///
    /// This function always completes synchronously.
    ///
    fn get_element(
        &mut self,
        id: Self::Id,
    ) -> impl Future<Output = Result<Self::Element<'_>, Self::GetError>> + Send {
        let v = self
            .provider
            .quant_vectors
            .get_vector_into(id.into_usize(), &mut self.element)
            .map(|_: ()| &*self.element);

        std::future::ready(v)
    }

    /// Perform a bulk operation
    ///
    fn on_elements_unordered<Itr, F>(
        &mut self,
        itr: Itr,
        mut f: F,
    ) -> impl Future<Output = Result<(), Self::GetError>> + Send
    where
        Self: Sync,
        Itr: Iterator<Item = Self::Id> + Send,
        F: Send + FnMut(Self::ElementRef<'_>, Self::Id),
    {
        for i in itr {
            match self
                .provider
                .quant_vectors
                .get_vector_into(i.into_usize(), &mut self.element)
            {
                Ok(()) => f(&self.element, i),
                Err(e) => {
                    return std::future::ready(Err(e));
                }
            }
        }
        std::future::ready(Ok(()))
    }
}

impl<T, D> BuildQueryComputer<&[T]> for QuantAccessor<'_, T, D>
where
    T: VectorRepr,
    D: AsyncFriendly,
{
    type QueryComputerError = ANNError;
    type QueryComputer = pq::distance::QueryComputer<Arc<FixedChunkPQTable>>;

    fn build_query_computer(
        &self,
        from: &[T],
    ) -> Result<Self::QueryComputer, Self::QueryComputerError> {
        self.provider.quant_vectors.query_computer(from)
    }
}

impl<T, D> ExpandBeam<&[T]> for QuantAccessor<'_, T, D>
where
    T: VectorRepr,
    D: AsyncFriendly,
{
}

impl<'a, T, D> AsDeletionCheck for QuantAccessor<'a, T, D>
where
    T: VectorRepr,
    D: AsyncFriendly + DeletionCheck,
{
    type Checker = D;
    fn as_deletion_check(&self) -> &D {
        &self.provider.deleted
    }
}

/////////////////////
// Hybrid Accessor //
/////////////////////

/// A hybrid accessor that fetches a mixture of full-precision and quantized vectors during
/// pruning. This allows the application to trade full-precision fetches for accuracy.
///
/// This type implements the following traits:
///
/// * [`Accessor`] for the [`RocksdbProvider`].
/// * [`BuildDistanceComputer`] for computing distances among [`distances::pq::Hybrid`]
///   element types.
/// * [`Fill`] for populating a mixture of full-precision and quant vectors.
///
pub struct HybridAccessor<'a, T, D>
where
    T: VectorRepr,
    D: AsyncFriendly,
{
    provider: &'a RocksdbProvider<T, QuantVectorProvider, D>,
}

impl<'a, T, D> HybridAccessor<'a, T, D>
where
    T: VectorRepr,
    D: AsyncFriendly,
{
    fn new(provider: &'a RocksdbProvider<T, QuantVectorProvider, D>) -> Self {
        Self { provider }
    }
}

impl<T, D> HasId for HybridAccessor<'_, T, D>
where
    T: VectorRepr,
    D: AsyncFriendly,
{
    type Id = u32;
}

impl<'a, T, D> DelegateNeighbor<'a> for HybridAccessor<'_, T, D>
where
    T: VectorRepr,
    D: AsyncFriendly,
{
    type Delegate = &'a NeighborProvider<u32>;
    fn delegate_neighbor(&'a mut self) -> Self::Delegate {
        self.provider.neighbors()
    }
}

impl<T, D> Accessor for HybridAccessor<'_, T, D>
where
    T: VectorRepr,
    D: AsyncFriendly,
{
    /// The [`distances::pq::Hybrid`] is an enum consisting of either a full-precision
    /// vector or a quantized vector.
    ///
    /// This accessor can return either.
    type Element<'a>
        = distances::pq::Hybrid<Vec<T>, Vec<u8>>
    where
        Self: 'a;

    /// The generalized reference form of `Element`.
    type ElementRef<'a> = distances::pq::Hybrid<&'a [T], &'a [u8]>;

    // Choose to panic on an out-of-bounds access rather than propagate an error.
    type GetError = Panics;

    /// The default behavior of `get_element` returns a full-precision vector. The
    /// implementation of [`Fill`] is how the `max_fp_vecs_per_fill` is used
    ///
    fn get_element(
        &mut self,
        id: Self::Id,
    ) -> impl Future<Output = Result<Self::Element<'_>, Self::GetError>> + Send {
        // SAFETY: We've decided to live with UB that can result from potentially mixing
        // unsynchronized reads and writes on the underlying memory.
        #[allow(clippy::expect_used)]
        std::future::ready(Ok(distances::pq::Hybrid::Full(
            self.provider
                .full_vectors
                .get_vector_sync(id.into_usize())
                .expect("Full vector provider failed to retrieve element"),
        )))
    }
}

impl<T, D> BuildDistanceComputer for HybridAccessor<'_, T, D>
where
    T: VectorRepr,
    D: AsyncFriendly,
{
    type DistanceComputerError = ANNError;
    type DistanceComputer = distances::pq::HybridComputer<T>;

    fn build_distance_computer(
        &self,
    ) -> Result<Self::DistanceComputer, Self::DistanceComputerError> {
        let metric = self.provider.quant_vectors.metric();
        Ok(distances::pq::HybridComputer::new(
            self.provider.quant_vectors.distance_computer(),
            T::distance(metric, Some(self.provider.full_vectors.dim())),
        ))
    }
}

impl<T, D> workingset::Fill<distances::pq::HybridMap<T, u8>> for HybridAccessor<'_, T, D>
where
    T: VectorRepr,
    D: AsyncFriendly,
{
    type Error = ANNError;
    type View<'a>
        = distances::pq::View<'a, T, u8>
    where
        Self: 'a;

    async fn fill<'a, Itr>(
        &'a mut self,
        state: &'a mut distances::pq::HybridMap<T, u8>,
        itr: Itr,
    ) -> Result<Self::View<'a>, Self::Error>
    where
        Itr: ExactSizeIterator<Item = Self::Id> + Clone + Send + Sync,
        Self: 'a,
    {
        let map = state.get_mut();
        map.prepare(itr.clone());
        let threshold = self.provider.max_fp_vecs_per_fill;
        itr.enumerate().try_for_each(|(i, id)| -> ANNResult<()> {
            match map.entry(id) {
                workingset::map::Entry::Seeded(_) => {}
                workingset::map::Entry::Occupied(occupied) => {
                    if i < threshold && !occupied.get().is_full() {
                        *occupied.into_mut() = distances::pq::Hybrid::Full(
                            self.provider
                                .full_vectors
                                .get_vector_sync(id.into_usize())?,
                        );
                    }
                }
                workingset::map::Entry::Vacant(vacant) => {
                    let element = if i < threshold {
                        let vec = self
                            .provider
                            .full_vectors
                            .get_vector_sync(id.into_usize())?;

                        distances::pq::Hybrid::Full(vec)
                    } else {
                        let vec = self
                            .provider
                            .quant_vectors
                            .get_vector_sync(id.into_usize())?;

                        distances::pq::Hybrid::Quant(vec)
                    };

                    vacant.insert(element);
                }
            }
            Ok(())
        })?;

        Ok(map.view())
    }
}

////////////////
// Strategies //
////////////////

/// Perform a search entirely in the full-precision space.
///
/// Starting points are not filtered out of the final results.
impl<T, Q, D> SearchStrategy<RocksdbProvider<T, Q, D>, &[T]> for FullPrecision
where
    T: VectorRepr,
    Q: AsyncFriendly,
    D: AsyncFriendly + DeletionCheck,
{
    type QueryComputer = T::QueryDistance;
    type SearchAccessor<'a> = FullAccessor<'a, T, Q, D>;
    type SearchAccessorError = Panics;

    fn search_accessor<'a>(
        &'a self,
        provider: &'a RocksdbProvider<T, Q, D>,
        _context: &'a DefaultContext,
    ) -> Result<Self::SearchAccessor<'a>, Self::SearchAccessorError> {
        Ok(FullAccessor::new(provider))
    }
}

impl<T, Q, D> DefaultPostProcessor<RocksdbProvider<T, Q, D>, &[T]> for FullPrecision
where
    T: VectorRepr,
    Q: AsyncFriendly,
    D: AsyncFriendly + DeletionCheck,
{
    default_post_processor!(glue::Pipeline<glue::FilterStartPoints, RemoveDeletedIdsAndCopy>);
}

/// An [`glue::SearchPostProcess`] implementation that reranks PQ vectors.
#[derive(Debug, Default, Clone, Copy)]
pub struct Rerank;

impl<'a, T, D> glue::SearchPostProcess<QuantAccessor<'a, T, D>, &[T]> for Rerank
where
    T: VectorRepr,
    D: AsyncFriendly + DeletionCheck,
{
    type Error = Panics;

    fn post_process<I, B>(
        &self,
        accessor: &mut QuantAccessor<'a, T, D>,
        query: &[T],
        _computer: &pq::distance::QueryComputer<Arc<FixedChunkPQTable>>,
        candidates: I,
        output: &mut B,
    ) -> impl Future<Output = Result<usize, Self::Error>> + Send
    where
        I: Iterator<Item = Neighbor<u32>>,
        B: SearchOutputBuffer<u32> + ?Sized,
    {
        let provider = &accessor.provider;
        let checker = accessor.as_deletion_check();
        let f = T::distance(provider.metric, Some(provider.full_vectors.dim()));

        // Filter before computing the full precision distances.
        let mut reranked: Vec<(u32, f32)> = candidates
            .filter_map(|n| {
                if checker.deletion_check(n.id) {
                    None
                } else {
                    #[allow(clippy::expect_used)]
                    let vec = provider
                        .full_vectors
                        .get_vector_sync(n.id.into_usize())
                        .expect("Full vector provider failed to retrieve element");
                    Some((n.id, f.evaluate_similarity(query, &vec)))
                }
            })
            .collect();

        // Sort the full precision distances.
        reranked
            .sort_unstable_by(|a, b| (a.1).partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        // Store the reranked results.
        std::future::ready(Ok(output.extend(reranked)))
    }
}

/// Perform a search entirely in the quantized space.
impl<T, D> SearchStrategy<RocksdbProvider<T, QuantVectorProvider, D>, &[T]> for Hybrid
where
    T: VectorRepr,
    D: AsyncFriendly + DeletionCheck,
{
    type QueryComputer = pq::distance::QueryComputer<Arc<FixedChunkPQTable>>;
    type SearchAccessor<'a> = QuantAccessor<'a, T, D>;
    type SearchAccessorError = Panics;

    fn search_accessor<'a>(
        &'a self,
        provider: &'a RocksdbProvider<T, QuantVectorProvider, D>,
        _context: &'a DefaultContext,
    ) -> Result<Self::SearchAccessor<'a>, Self::SearchAccessorError> {
        Ok(QuantAccessor::new(provider))
    }
}

/// Starting points are filtered out of the final results and results are reranked using
/// the full-precision data.
impl<T, D> DefaultPostProcessor<RocksdbProvider<T, QuantVectorProvider, D>, &[T]> for Hybrid
where
    T: VectorRepr,
    D: AsyncFriendly + DeletionCheck,
{
    default_post_processor!(glue::Pipeline<glue::FilterStartPoints, Rerank>);
}

// Pruning
impl<T, Q, D> PruneStrategy<RocksdbProvider<T, Q, D>> for FullPrecision
where
    T: VectorRepr,
    Q: AsyncFriendly,
    D: AsyncFriendly,
{
    type WorkingSet = map::Map<u32, Box<[T]>, map::Ref<[T]>>;
    type DistanceComputer<'a> = T::Distance;
    type PruneAccessor<'a> = FullAccessor<'a, T, Q, D>;
    type PruneAccessorError = diskann::error::Infallible;

    fn prune_accessor<'a>(
        &'a self,
        provider: &'a RocksdbProvider<T, Q, D>,
        _context: &'a DefaultContext,
    ) -> Result<Self::PruneAccessor<'a>, Self::PruneAccessorError> {
        Ok(FullAccessor::new(provider))
    }

    fn create_working_set(&self, capacity: usize) -> Self::WorkingSet {
        map::Builder::new(map::Capacity::Default).build(capacity)
    }
}

impl<T, D> PruneStrategy<RocksdbProvider<T, QuantVectorProvider, D>> for Hybrid
where
    T: VectorRepr,
    D: AsyncFriendly,
{
    type WorkingSet = distances::pq::HybridMap<T, u8>;
    type DistanceComputer<'a> = distances::pq::HybridComputer<T>;
    type PruneAccessor<'a> = HybridAccessor<'a, T, D>;
    type PruneAccessorError = diskann::error::Infallible;

    fn prune_accessor<'a>(
        &'a self,
        provider: &'a RocksdbProvider<T, QuantVectorProvider, D>,
        _context: &'a DefaultContext,
    ) -> Result<Self::PruneAccessor<'a>, Self::PruneAccessorError> {
        Ok(HybridAccessor::new(provider))
    }

    fn create_working_set(&self, capacity: usize) -> Self::WorkingSet {
        distances::pq::HybridMap::with_capacity(capacity)
    }
}

impl<T, Q, D> InsertStrategy<RocksdbProvider<T, Q, D>, &[T]> for FullPrecision
where
    T: VectorRepr,
    Q: AsyncFriendly,
    D: AsyncFriendly + DeletionCheck,
{
    type PruneStrategy = Self;
    fn prune_strategy(&self) -> Self::PruneStrategy {
        *self
    }
}

impl<T, D> InsertStrategy<RocksdbProvider<T, QuantVectorProvider, D>, &[T]> for Hybrid
where
    T: VectorRepr,
    D: AsyncFriendly + DeletionCheck,
{
    type PruneStrategy = Self;
    fn prune_strategy(&self) -> Self::PruneStrategy {
        *self
    }
}

impl<T, Q, D, B> MultiInsertStrategy<RocksdbProvider<T, Q, D>, B> for FullPrecision
where
    T: VectorRepr,
    Q: AsyncFriendly,
    D: AsyncFriendly + DeletionCheck,
    B: for<'a> Batch<Element<'a> = &'a [T]> + Debug,
{
    type Seed = map::Builder<u32, map::Ref<[T]>>;
    type WorkingSet = map::Map<u32, Box<[T]>, map::Ref<[T]>>;
    type FinishError = diskann::error::Infallible;
    type InsertStrategy = Self;

    fn insert_strategy(&self) -> Self::InsertStrategy {
        *self
    }

    fn finish<Itr>(
        &self,
        _provider: &RocksdbProvider<T, Q, D>,
        _ctx: &DefaultContext,
        batch: &std::sync::Arc<B>,
        ids: Itr,
    ) -> impl std::future::Future<Output = Result<Self::Seed, Self::FinishError>> + Send
    where
        Itr: ExactSizeIterator<Item = u32> + Send,
    {
        let overlay = map::Overlay::from_batch(batch.clone(), ids);
        let builder = map::Builder::new(map::Capacity::Default).with_overlay(overlay);
        std::future::ready(Ok(builder))
    }
}

impl<T, D, B> MultiInsertStrategy<RocksdbProvider<T, QuantVectorProvider, D>, B> for Hybrid
where
    T: VectorRepr,
    D: AsyncFriendly + DeletionCheck,
    B: for<'a> Batch<Element<'a> = &'a [T]> + Debug,
{
    type Seed = distances::pq::Overlay<T, u8>;
    type WorkingSet = distances::pq::HybridMap<T, u8>;
    type FinishError = diskann::error::Infallible;
    type InsertStrategy = Self;

    fn insert_strategy(&self) -> Self::InsertStrategy {
        *self
    }

    fn finish<Itr>(
        &self,
        _provider: &RocksdbProvider<T, QuantVectorProvider, D>,
        _ctx: &DefaultContext,
        batch: &std::sync::Arc<B>,
        ids: Itr,
    ) -> impl std::future::Future<Output = Result<Self::Seed, Self::FinishError>> + Send
    where
        Itr: ExactSizeIterator<Item = u32> + Send,
    {
        let overlay = Self::Seed::from_batch(batch.clone(), ids);
        std::future::ready(Ok(overlay))
    }
}

/// Inplace Delete
///
impl<T, Q, D> InplaceDeleteStrategy<RocksdbProvider<T, Q, D>> for FullPrecision
where
    T: VectorRepr,
    Q: AsyncFriendly,
    D: AsyncFriendly + DeletionCheck,
{
    type DeleteElementError = Panics;
    type DeleteElement<'a> = &'a [T];
    type DeleteElementGuard = Box<[T]>;
    type PruneStrategy = Self;
    type DeleteSearchAccessor<'a> = FullAccessor<'a, T, Q, D>;
    type SearchPostProcessor = RemoveDeletedIdsAndCopy;
    type SearchStrategy = Self;
    fn search_strategy(&self) -> Self::SearchStrategy {
        Self
    }

    fn prune_strategy(&self) -> Self::PruneStrategy {
        Self
    }

    fn search_post_processor(&self) -> Self::SearchPostProcessor {
        RemoveDeletedIdsAndCopy
    }

    async fn get_delete_element<'a>(
        &'a self,
        provider: &'a RocksdbProvider<T, Q, D>,
        _context: &'a DefaultContext,
        id: u32,
    ) -> Result<Self::DeleteElementGuard, Self::DeleteElementError> {
        #[allow(clippy::expect_used)]
        let elt = provider
            .full_vectors
            .get_vector_sync(id.into_usize())
            .expect("Failed to get delete element")
            .into();
        Ok(elt)
    }
}

impl<T, D> InplaceDeleteStrategy<RocksdbProvider<T, QuantVectorProvider, D>> for Hybrid
where
    T: VectorRepr,
    D: AsyncFriendly + DeletionCheck,
{
    type DeleteElementError = Panics;
    type DeleteElement<'a> = &'a [T];
    type DeleteElementGuard = Box<[T]>;
    type PruneStrategy = Self;
    type DeleteSearchAccessor<'a> = QuantAccessor<'a, T, D>;
    type SearchPostProcessor = Rerank;
    type SearchStrategy = Self;
    fn search_strategy(&self) -> Self::SearchStrategy {
        *self
    }

    fn prune_strategy(&self) -> Self::PruneStrategy {
        *self
    }

    fn search_post_processor(&self) -> Self::SearchPostProcessor {
        Rerank
    }

    async fn get_delete_element<'a>(
        &'a self,
        provider: &'a RocksdbProvider<T, QuantVectorProvider, D>,
        _context: &'a DefaultContext,
        id: u32,
    ) -> Result<Self::DeleteElementGuard, Self::DeleteElementError> {
        #[allow(clippy::expect_used)]
        let elt = provider
            .full_vectors
            .get_vector_sync(id.into_usize())
            .expect("Failed to get delete element")
            .into();
        Ok(elt)
    }
}

/// Stored parameters for reconstructing a rocksdb-backed provider.
///
/// The `bytes` / `max_record_size` / `leaf_page_size` fields are kept for
/// schema parity with `super::bf_tree::BfTreeParams`; rocksdb does not
/// expose 1:1 equivalents, so they are unused at config-build time.
#[derive(Serialize, Deserialize, Clone)]
pub struct RocksdbParams {
    pub bytes: usize,
    pub max_record_size: usize,
    pub leaf_page_size: usize,
}

impl RocksdbParams {
    /// Build a rocksdb `Config` from the saved parameters and a file path.
    /// When `is_memory` is true, the config uses an in-memory (tempdir) backend.
    #[allow(clippy::expect_used)]
    pub fn to_config(&self, path: &std::path::Path, is_memory: bool) -> Config {
        if is_memory {
            Config::in_memory().expect("failed to create in-memory rocksdb config")
        } else {
            Config::new(path)
        }
    }
}

#[derive(Serialize, Deserialize, Clone)]
pub struct QuantParams {
    pub num_pq_bytes: usize,
    pub max_fp_vecs_per_fill: usize,
    pub params_quant: RocksdbParams,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct SavedParams {
    pub max_points: usize,
    pub frozen_points: NonZeroUsize,
    pub dim: usize,
    pub metric: String,
    pub max_degree: u32,
    pub prefix: String,
    pub params_vector: RocksdbParams,
    pub params_neighbor: RocksdbParams,
    pub quant_params: Option<QuantParams>,
    pub graph_params: Option<GraphParams>,
    /// Whether the original model was in-memory (`true`) or on-disk (`false`).
    pub is_memory: bool,
}

/// The element type of the full-precision vectors stored in the index.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum VectorDtype {
    F32,
    F16,
    U8,
    I8,
}

/// A trait for mapping concrete vector element types to their [`VectorDtype`]
/// discriminant at compile time.
pub trait AsVectorDtype {
    const DATA_TYPE: VectorDtype;
}

impl AsVectorDtype for f32 {
    const DATA_TYPE: VectorDtype = VectorDtype::F32;
}

impl AsVectorDtype for half::f16 {
    const DATA_TYPE: VectorDtype = VectorDtype::F16;
}

impl AsVectorDtype for i8 {
    const DATA_TYPE: VectorDtype = VectorDtype::I8;
}

impl AsVectorDtype for u8 {
    const DATA_TYPE: VectorDtype = VectorDtype::U8;
}

/// Graph configuration parameters persisted alongside the index.
/// These are needed to reconstruct the `DiskANNIndex` config on load.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct GraphParams {
    /// l_build is the search list size used during index construction.
    /// When inserting a new vector into the DiskANN graph, the algorithm
    /// performs a greedy search to find the best neighbors to connect to.
    /// l_build controls how many candidate nodes are tracked during that search.
    pub l_build: usize,
    /// alpha is the pruning aggressiveness parameter used during graph
    /// construction. During pruning, when deciding whether to keep a candidate
    /// neighbor k for node i, the algorithm checks if there's already a
    /// closer neighbor j that "occludes" k. The occlusion test is is governed by alpha.
    pub alpha: f32,
    /// backedge_ratio controls how many reverse (back) edges are added after
    /// pruning during graph construction.
    pub backedge_ratio: f32,
    /// vector_dtype indicates the data type of the vectors stored in the index, which is necessary for correctly interpreting the raw bytes of the vectors when loading the index from disk.
    pub vector_dtype: VectorDtype,
}

/// Helper struct for generating consistent file paths for RocksdbProvider persistence.
/// Centralizes all path patterns to avoid hardcoded strings throughout the codebase.
pub struct RocksdbPaths;

impl RocksdbPaths {
    /// Returns the path for the parameters JSON file
    pub fn params_json(prefix: &str) -> String {
        format!("{}_params.json", prefix)
    }

    /// Returns the path for the vectors BfTree file
    pub fn vectors_bftree(prefix: &str) -> std::path::PathBuf {
        std::path::PathBuf::from(format!("{}_vectors.bftree", prefix))
    }

    /// Returns the path for the neighbors BfTree file
    pub fn neighbors_bftree(prefix: &str) -> std::path::PathBuf {
        std::path::PathBuf::from(format!("{}_neighbors.bftree", prefix))
    }

    /// Returns the path for the quantized vectors BfTree file
    pub fn quant_bftree(prefix: &str) -> std::path::PathBuf {
        std::path::PathBuf::from(format!("{}_quant.bftree", prefix))
    }

    /// Returns the path for the delete bitmap file
    pub fn delete_bin(prefix: &str) -> String {
        format!("{}_delete.bin", prefix)
    }

    /// Returns the path for the PQ pivots file
    pub fn pq_pivots_bin(prefix: &str) -> String {
        format!("{}_pq_pivots.bin", prefix)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::graph::provider::async_::common::TableBasedDeletes;

    #[tokio::test]
    async fn test_data_provider_and_delete_interface() {
        let ctx = &DefaultContext;
        let provider = RocksdbProvider::new_empty(
            RocksdbProviderParameters {
                max_points: 10,
                num_start_points: NonZeroUsize::new(2).unwrap(),
                dim: 5,
                metric: Metric::L2,
                max_fp_vecs_per_fill: None,
                max_degree: 64,
                vector_provider_config: Config::default(),
                quant_vector_provider_config: Config::default(),
                neighbor_list_provider_config: Config::default(),
                graph_params: None,
            },
            NoStore,
            TableBasedDeletes,
        )
        .unwrap();

        // Iterator
        //
        assert_eq!((&provider).into_iter(), 0..(10 + 2));

        let iter = provider.iter();
        for i in iter.clone() {
            assert_eq!(provider.to_external_id(ctx, i).unwrap(), i);
            assert_eq!(provider.to_internal_id(ctx, &i).unwrap(), i);
            assert_eq!(
                provider.status_by_internal_id(ctx, i).await.unwrap(),
                ElementStatus::Valid
            );
            assert_eq!(
                provider.status_by_external_id(ctx, &i).await.unwrap(),
                ElementStatus::Valid
            );

            // Delete by external ID.
            //
            provider.delete(ctx, &i).await.unwrap();
            assert_eq!(
                provider.status_by_internal_id(ctx, i).await.unwrap(),
                ElementStatus::Deleted
            );
            assert_eq!(
                provider.status_by_external_id(ctx, &i).await.unwrap(),
                ElementStatus::Deleted
            );
        }

        // Call `release` to "undelete" it ID.
        //
        for i in iter.clone() {
            // set adjacency list to non-empty before release
            provider
                .neighbor_provider
                .set_neighbors(i, &[1, 2])
                .unwrap();
            provider.release(ctx, i).await.unwrap();
            assert_eq!(
                provider.status_by_internal_id(ctx, i).await.unwrap(),
                ElementStatus::Valid
            );
            assert_eq!(
                provider.status_by_external_id(ctx, &i).await.unwrap(),
                ElementStatus::Valid
            );
            // check that adjacency list was reset after release
            let mut neighbors = AdjacencyList::new();
            provider
                .neighbor_provider
                .get_neighbors(i, &mut neighbors)
                .unwrap();
            assert!(neighbors.to_vec().is_empty());

            // Put it back to "deleted" to test `clear`.
            //
            provider.delete(ctx, &i).await.unwrap();
        }

        provider.clear_delete_set();
        for i in iter.clone() {
            assert_eq!(
                provider.status_by_internal_id(ctx, i).await.unwrap(),
                ElementStatus::Valid
            );
            assert_eq!(
                provider.status_by_external_id(ctx, &i).await.unwrap(),
                ElementStatus::Valid
            );
        }

        // out-of-bound set-element fails.
        //
        assert!(
            provider
                .set_element(ctx, &100, &[1.0, 2.0, 3.0, 4.0])
                .await
                .is_err()
        );
    }

    /// This functionality test targets scenarios of empty neighbor lists and ensures:
    /// 1. A new vector's neighbor list is empty
    /// 2. A vector's neighbor list could be set to empty
    /// 3. A non-existant vector's neighbor list is empty
    ///
    #[tokio::test]
    async fn test_empty_neighbor_list() {
        let num_points = 100u32;
        let ctx = &DefaultContext;
        let provider = RocksdbProvider::<f32, _, _>::new_empty(
            RocksdbProviderParameters {
                max_points: num_points as usize,
                num_start_points: NonZeroUsize::new(2).unwrap(),
                dim: 3,
                metric: Metric::L2,
                max_fp_vecs_per_fill: None,
                max_degree: 64,
                vector_provider_config: Config::default(),
                quant_vector_provider_config: Config::default(),
                neighbor_list_provider_config: Config::default(),
                graph_params: None,
            },
            NoStore,
            TableBasedDeletes,
        )
        .unwrap();

        let neighbor_accessor = &mut provider.neighbors();

        // Insert new vectors without neighbors and empty neighbor list is
        // expected for each newly inserted vector
        //
        for i in 0..num_points {
            let vector = vec![i as f32, (i + 1) as f32, (i + 2) as f32];
            provider.set_element(ctx, &i, &vector).await.unwrap();

            // First attempt should fail as NotFound
            let mut out = AdjacencyList::new();
            assert!(neighbor_accessor.get_neighbors(i, &mut out).await.is_err());

            // After we set the empty neighbor list, our attempt should succeed
            neighbor_accessor.set_neighbors(i, &[]).await.unwrap();
            neighbor_accessor.get_neighbors(i, &mut out).await.unwrap();

            assert!(out.is_empty());
        }

        // Add a non-empty neighbor list for a vector and then set it to empty
        // In the end, an empty neighbor list is expected for the vector
        //
        for i in 0..num_points {
            let mut out = AdjacencyList::new();
            let neighbors = vec![10, 20, 30, 40, 50, 60, 70, 80, 90, 100];
            neighbor_accessor
                .set_neighbors(i, &neighbors)
                .await
                .unwrap();

            neighbor_accessor.get_neighbors(i, &mut out).await.unwrap();

            assert_eq!(&*out, &[10, 20, 30, 40, 50, 60, 70, 80, 90, 100]); // len = 10

            neighbor_accessor.set_neighbors(i, &[]).await.unwrap();
            neighbor_accessor.get_neighbors(i, &mut out).await.unwrap();

            assert!(out.is_empty());
        }

        // Non-existant vectors have empty neighbor lists
        //
        let mut out = AdjacencyList::from_iter_untrusted([10, 20, 30, 40, 50, 60, 70, 80, 90, 100]); // len = 10

        // Attempt to access non-existant vector's neighbor list should fail as NotFound
        assert!(
            neighbor_accessor
                .get_neighbors(200, &mut out)
                .await
                .is_err()
        );
        assert!(out.is_empty());
    }
}
