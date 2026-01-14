// SPDX-FileCopyrightText: 2020 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
// SPDX-FileCopyrightText: 2020 Rafael de Santiago <r.santiago@ufsc.br>
//
// SPDX-License-Identifier: Apache-2.0

use std::{
    iter::Sum,
    ops::{AddAssign, DivAssign, MulAssign, RemAssign, SubAssign},
};

pub mod bitwise;
pub mod c_api;
pub mod convert;
pub mod dense;
pub mod dense_gpu;
pub mod dense_v2;
pub mod error;
pub mod quantum_execution;
pub mod sparse;

pub trait FloatOps:
    num_traits::Float
    + num_traits::FloatConst
    + AddAssign
    + SubAssign
    + MulAssign
    + DivAssign
    + RemAssign
    + Sum
    + Send
    + Sync
{
    fn small_epsilon() -> f32;
}

impl FloatOps for f32 {
    fn small_epsilon() -> f32 {
        1e-6
    }
}
impl FloatOps for f64 {
    fn small_epsilon() -> f32 {
        1e-15
    }
}

pub type DenseSimulator = quantum_execution::QubitManager<dense::Dense<f32>>;
pub type DenseGPUSimulator =
    quantum_execution::QubitManager<dense_gpu::DenseGPU<cubecl::wgpu::WgpuRuntime, f32>>;
pub type DenseV2Simulator = quantum_execution::QubitManager<dense_v2::DenseV2<f32>>;
pub type SparseSimulator = quantum_execution::QubitManager<sparse::Sparse<f32>>;
