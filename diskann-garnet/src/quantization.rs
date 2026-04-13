use std::sync::RwLock;

use diskann_quantization::{
    algorithms::TransformKind,
    alloc::{GlobalAllocator, ScopedAllocator},
    spherical::{
        self, Data, PreScale, SphericalQuantizer, SupportedMetric,
        iface::{self, Opaque, OpaqueMut, Quantizer},
    },
};
use diskann_utils::views::MatrixView;
use diskann_vector::{DistanceFunction, PreprocessedDistanceFunction, distance::Metric};
use thiserror::Error;

use crate::provider::{GarnetDistanceComputer, GarnetQueryComputer};

#[derive(Debug, Error)]
pub enum GarnetQuantizerError {
    #[error("Quantization training error: {0}")]
    Training(Box<dyn std::error::Error + Send + Sync + 'static>),
    #[error("Quantization alloc error: {0}")]
    Alloc(Box<dyn std::error::Error + Send + Sync + 'static>),
    #[error("Query computer error: {0}")]
    QueryComputer(Box<dyn std::error::Error + Send + Sync + 'static>),
    #[error("Binary quantization error: {0}")]
    Compression(Box<dyn std::error::Error + Send + Sync + 'static>),
    #[error("No quantizer found")]
    NoQuantizer,
}

pub trait GarnetQuantizer: Send + Sync {
    fn is_prepared(&self) -> bool;
    fn required_vectors(&self) -> usize;
    fn canonical_bytes(&self) -> usize;
    fn train(&self, metric: Metric, data: MatrixView<f32>) -> Result<(), GarnetQuantizerError>;
    fn compress(&self, v: &[f32], into: &mut [u8]) -> Result<(), GarnetQuantizerError>;
    fn distance_computer(&self) -> Result<GarnetDistanceComputer, GarnetQuantizerError>;
    fn query_computer(&self, query: &[f32]) -> Result<GarnetQueryComputer, GarnetQuantizerError>;
}

pub trait DynDistanceComputer: Send + Sync {
    fn evaluate_similarity(&self, a: &[u8], b: &[u8]) -> f32;
}

pub trait DynQueryComputer: Send + Sync {
    fn evaluate_similarity(&self, a: &[u8]) -> f32;
}

pub struct Spherical1Bit {
    dim: usize,
    inner: RwLock<Option<spherical::iface::Impl<1, GlobalAllocator>>>,
}

impl Spherical1Bit {
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            inner: RwLock::new(None),
        }
    }
}

impl GarnetQuantizer for Spherical1Bit {
    fn is_prepared(&self) -> bool {
        self.inner.read().unwrap().is_some()
    }

    fn required_vectors(&self) -> usize {
        1000
    }

    fn canonical_bytes(&self) -> usize {
        Data::<1, GlobalAllocator>::canonical_bytes(self.dim)
    }

    fn train(
        &self,
        metric_type: Metric,
        data: MatrixView<f32>,
    ) -> Result<(), GarnetQuantizerError> {
        let mut rng = rand::rng();
        let quantizer = SphericalQuantizer::train(
            data.as_view(),
            TransformKind::DoubleHadamard {
                target_dim: diskann_quantization::algorithms::transforms::TargetDim::Same,
            },
            SupportedMetric::try_from(metric_type).unwrap(),
            PreScale::ReciprocalMeanNorm,
            &mut rng,
            GlobalAllocator,
        )
        .map_err(|e| GarnetQuantizerError::Training(Box::new(e)))?;

        let mut inner = self.inner.write().unwrap();
        *inner = Some(
            spherical::iface::Impl::<1>::new(quantizer)
                .map_err(|e| GarnetQuantizerError::Alloc(Box::new(e)))?,
        );

        Ok(())
    }

    fn compress(&self, v: &[f32], into: &mut [u8]) -> Result<(), GarnetQuantizerError> {
        let guard = self.inner.read().unwrap();
        if let Some(quantizer) = &*guard {
            spherical::iface::Quantizer::<GlobalAllocator>::compress(
                quantizer,
                v,
                OpaqueMut::new(into),
                ScopedAllocator::global(),
            )
            .map_err(|e| GarnetQuantizerError::Compression(Box::new(e)))?;
            Ok(())
        } else {
            Err(GarnetQuantizerError::NoQuantizer)
        }
    }

    fn distance_computer(&self) -> Result<GarnetDistanceComputer, GarnetQuantizerError> {
        let guard = self.inner.read().unwrap();
        if let Some(quantizer) = &*guard {
            let computer = quantizer
                .distance_computer(GlobalAllocator)
                .map_err(|e| GarnetQuantizerError::Alloc(Box::new(e)))?;
            Ok(GarnetDistanceComputer::new(computer, true))
        } else {
            Err(GarnetQuantizerError::NoQuantizer)
        }
    }

    fn query_computer(&self, query: &[f32]) -> Result<GarnetQueryComputer, GarnetQuantizerError> {
        let guard = self.inner.read().unwrap();
        if let Some(quantizer) = &*guard {
            let computer = quantizer
                .fused_query_computer(
                    query,
                    iface::QueryLayout::FullPrecision,
                    true,
                    GlobalAllocator,
                    ScopedAllocator::global(),
                )
                .map_err(|e| GarnetQuantizerError::QueryComputer(Box::new(e)))?;
            Ok(GarnetQueryComputer::new(computer))
        } else {
            Err(GarnetQuantizerError::NoQuantizer)
        }
    }
}

impl DynDistanceComputer for iface::DistanceComputer {
    fn evaluate_similarity(&self, a: &[u8], b: &[u8]) -> f32 {
        <Self as DistanceFunction<Opaque<'_>, Opaque<'_>, _>>::evaluate_similarity(
            self,
            Opaque::new(a),
            Opaque::new(b),
        )
        .unwrap()
    }
}

impl DynQueryComputer for iface::QueryComputer {
    fn evaluate_similarity(&self, a: &[u8]) -> f32 {
        <Self as PreprocessedDistanceFunction<Opaque<'_>, _>>::evaluate_similarity(
            self,
            Opaque::new(a),
        )
        .unwrap()
    }
}

// TODO: pub struct MinMax8Bit {}
