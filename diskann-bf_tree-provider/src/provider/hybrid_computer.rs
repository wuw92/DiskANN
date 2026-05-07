/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::sync::Arc;

use diskann::utils::VectorRepr;
use diskann_providers::model::graph::provider::async_::distances::pq::Hybrid;
use diskann_quantization::{
    alloc::{GlobalAllocator, Poly, ScopedAllocator},
    spherical::iface::{DistanceComputer, Opaque, OpaqueMut, Quantizer, QueryComputer},
};
use diskann_vector::{DistanceFunction, PreprocessedDistanceFunction};

pub struct QuantQueryComputer(pub QueryComputer<GlobalAllocator>);

impl PreprocessedDistanceFunction<&[u8], f32> for QuantQueryComputer {
    fn evaluate_similarity(&self, x: &[u8]) -> f32 {
        self.0
            .evaluate_similarity(Opaque::new(x))
            .expect("spherical query distance failed")
    }
}

/// Distance computer that handles mixed full-precision and
/// spherical-quantized operands.
pub struct HybridComputer<T: VectorRepr> {
    quant: DistanceComputer<GlobalAllocator>,
    full: T::Distance,
    quantizer: Arc<Poly<dyn Quantizer>>,
}

impl<T: VectorRepr> HybridComputer<T> {
    pub fn new(
        quant: DistanceComputer<GlobalAllocator>,
        full: T::Distance,
        quantizer: Arc<Poly<dyn Quantizer>>,
    ) -> Self {
        Self {
            quant,
            full,
            quantizer,
        }
    }

    fn mixed_distance(&self, fp: &[T], q: &[u8]) -> f32 {
        let fp_f32 = T::as_f32(fp).expect("f32 conversion failure");
        let mut buf = vec![0u8; self.quantizer.bytes()];
        self.quantizer
            .compress(&fp_f32, OpaqueMut::new(&mut buf), ScopedAllocator::global())
            .expect("spherical compression failed");
        self.quant
            .evaluate_similarity(Opaque::new(&buf), Opaque::new(q))
            .expect("spherical distance failed")
    }
}

impl<T> DistanceFunction<Hybrid<&[T], &[u8]>, Hybrid<&[T], &[u8]>> for HybridComputer<T>
where
    T: VectorRepr,
{
    fn evaluate_similarity(&self, x: Hybrid<&[T], &[u8]>, y: Hybrid<&[T], &[u8]>) -> f32 {
        match (x, y) {
            (Hybrid::Full(x), Hybrid::Full(y)) => self.full.evaluate_similarity(x, y),
            (Hybrid::Quant(x), Hybrid::Quant(y)) => self
                .quant
                .evaluate_similarity(Opaque::new(x), Opaque::new(y))
                .expect("spherical distance failed"),
            (Hybrid::Full(x), Hybrid::Quant(y)) => self.mixed_distance(x, y),
            (Hybrid::Quant(x), Hybrid::Full(y)) => self.mixed_distance(y, x),
        }
    }
}
