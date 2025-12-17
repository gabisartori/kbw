// SPDX-FileCopyrightText: 2025 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
//
// SPDX-License-Identifier: Apache-2.0

use crate::{
    bitwise::get_ctrl_mask,
    quantum_execution::{ExecutionFeatures, QuantumExecution},
    FloatOps,
};
use cubecl::prelude::*;
use ket::execution::Capability;
use rand::distr::weighted::WeightedIndex;
use rand::prelude::*;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use std::marker::PhantomData;

pub struct DenseGPU<R: Runtime, F: Float + CubeElement + FloatOps> {
    state_real: cubecl::server::Handle,
    state_imag: cubecl::server::Handle,
    state_size: usize,
    num_qubits: usize,

    client: ComputeClient<R::Server>,
    _f: PhantomData<F>,
}

#[cube]
fn compute_ket_kernel(
    control_mask: u32,
    half_block_size: u32,
    full_block_size: u32,
) -> (u32, u32, bool) {
    let ket0 = ABSOLUTE_POS_X * full_block_size + ABSOLUTE_POS_Y;
    let ket1 = ket0 + half_block_size;

    (ket0, ket1, ket0 & control_mask == control_mask)
}

#[cube(launch_unchecked)]
fn gate_x_kernel<F: Float>(
    state_real: &mut Array<F>,
    state_imag: &mut Array<F>,
    control_mask: u32,
    half_block_size: u32,
    full_block_size: u32,
) {
    let ket = compute_ket_kernel(control_mask, half_block_size, full_block_size);

    if ket.2 {
        let old0 = state_real[ket.0];
        state_real[ket.0] = state_real[ket.1];
        state_real[ket.1] = old0;

        let old0 = state_imag[ket.0];
        state_imag[ket.0] = state_imag[ket.1];
        state_imag[ket.1] = old0;
    }
}

#[cube(launch_unchecked)]
fn gate_y_kernel<F: Float>(
    state_real: &mut Array<F>,
    state_imag: &mut Array<F>,
    control_mask: u32,
    half_block_size: u32,
    full_block_size: u32,
) {
    let ket = compute_ket_kernel(control_mask, half_block_size, full_block_size);

    if ket.2 {
        let old0_real = state_real[ket.0];
        let old0_imag = state_imag[ket.0];

        state_real[ket.0] = -state_imag[ket.1];
        state_imag[ket.0] = state_real[ket.1];

        state_real[ket.1] = old0_imag;
        state_imag[ket.1] = -old0_real;
    }
}

#[cube(launch_unchecked)]
fn gate_z_kernel<F: Float>(
    state_real: &mut Array<F>,
    state_imag: &mut Array<F>,
    control_mask: u32,
    half_block_size: u32,
    full_block_size: u32,
) {
    let ket = compute_ket_kernel(control_mask, half_block_size, full_block_size);

    if ket.2 {
        state_real[ket.1] = -state_real[ket.1];
        state_imag[ket.1] = -state_imag[ket.1];
    }
}

#[cube(launch_unchecked)]
fn gate_h_kernel<F: Float>(
    state_real: &mut Array<F>,
    state_imag: &mut Array<F>,
    control_mask: u32,
    half_block_size: u32,
    full_block_size: u32,

    frac_1_sqrt_2: F,
) {
    let ket = compute_ket_kernel(control_mask, half_block_size, full_block_size);

    if ket.2 {
        let old0_real = state_real[ket.0];
        let old0_imag = state_imag[ket.0];

        let old1_real = state_real[ket.1];
        let old1_imag = state_imag[ket.1];

        state_real[ket.0] = (old0_real + old1_real) * frac_1_sqrt_2;
        state_imag[ket.0] = (old0_imag + old1_imag) * frac_1_sqrt_2;

        state_real[ket.1] = (old0_real - old1_real) * frac_1_sqrt_2;
        state_imag[ket.1] = (old0_imag - old1_imag) * frac_1_sqrt_2;
    }
}

#[cube]
fn complex_mul_kernel<F: Float>(lhs_real: F, lhs_imag: F, rhs_real: F, rhs_imag: F) -> (F, F) {
    (
        lhs_real * rhs_real - lhs_imag * rhs_imag,
        lhs_real * rhs_imag + lhs_imag * rhs_real,
    )
}

#[cube]
fn complex_add_kernel<F: Float>(lhs_real: F, lhs_imag: F, rhs_real: F, rhs_imag: F) -> (F, F) {
    (lhs_real + rhs_real, lhs_imag + rhs_imag)
}

#[cube(launch_unchecked)]
fn gate_p_kernel<F: Float>(
    state_real: &mut Array<F>,
    state_imag: &mut Array<F>,
    control_mask: u32,
    half_block_size: u32,
    full_block_size: u32,

    angle_real: F,
    angle_imag: F,
) {
    let ket = compute_ket_kernel(control_mask, half_block_size, full_block_size);

    if ket.2 {
        let (r, i) =
            complex_mul_kernel::<F>(state_real[ket.1], state_imag[ket.1], angle_real, angle_imag);
        state_real[ket.1] = r;
        state_imag[ket.1] = i;
    }
}

#[cube(launch_unchecked)]
fn gate_rx_kernel<F: Float>(
    state_real: &mut Array<F>,
    state_imag: &mut Array<F>,
    control_mask: u32,
    half_block_size: u32,
    full_block_size: u32,

    cos_theta_2: F,
    sin_theta_2: F,
) {
    let ket = compute_ket_kernel(control_mask, half_block_size, full_block_size);

    if ket.2 {
        let (ket0_cos_real, ket0_cos_imag) = complex_mul_kernel::<F>(
            cos_theta_2,
            F::new(0.0),
            state_real[ket.0],
            state_imag[ket.0],
        );

        let (ket1_sin_real, ket1_sin_imag) = complex_mul_kernel::<F>(
            F::new(0.0),
            sin_theta_2,
            state_real[ket.1],
            state_imag[ket.1],
        );

        let (ket0_sin_real, ket0_sin_imag) = complex_mul_kernel::<F>(
            F::new(0.0),
            sin_theta_2,
            state_real[ket.0],
            state_imag[ket.0],
        );

        let (ket1_cos_real, ket1_cos_imag) = complex_mul_kernel::<F>(
            cos_theta_2,
            F::new(0.0),
            state_real[ket.1],
            state_imag[ket.1],
        );

        let (r, i) =
            complex_add_kernel::<F>(ket0_cos_real, ket0_cos_imag, ket1_sin_real, ket1_sin_imag);
        state_real[ket.0] = r;
        state_imag[ket.0] = i;

        let (r, i) =
            complex_add_kernel::<F>(ket0_sin_real, ket0_sin_imag, ket1_cos_real, ket1_cos_imag);
        state_real[ket.1] = r;
        state_imag[ket.1] = i;
    }
}

#[cube(launch_unchecked)]
fn gate_ry_kernel<F: Float>(
    state_real: &mut Array<F>,
    state_imag: &mut Array<F>,
    control_mask: u32,
    half_block_size: u32,
    full_block_size: u32,

    cos_theta_2: F,
    sin_theta_2: F,
) {
    let ket = compute_ket_kernel(control_mask, half_block_size, full_block_size);

    if ket.2 {
        let (ket0_cos_real, ket0_cos_imag) = complex_mul_kernel::<F>(
            cos_theta_2,
            F::new(0.0),
            state_real[ket.0],
            state_imag[ket.0],
        );

        let (ket1_sin_real, ket1_sin_imag) = complex_mul_kernel::<F>(
            -sin_theta_2,
            F::new(0.0),
            state_real[ket.1],
            state_imag[ket.1],
        );

        let (ket0_sin_real, ket0_sin_imag) = complex_mul_kernel::<F>(
            sin_theta_2,
            F::new(0.0),
            state_real[ket.0],
            state_imag[ket.0],
        );

        let (ket1_cos_real, ket1_cos_imag) = complex_mul_kernel::<F>(
            cos_theta_2,
            F::new(0.0),
            state_real[ket.1],
            state_imag[ket.1],
        );

        let (r, i) =
            complex_add_kernel::<F>(ket0_cos_real, ket0_cos_imag, ket1_sin_real, ket1_sin_imag);
        state_real[ket.0] = r;
        state_imag[ket.0] = i;

        let (r, i) =
            complex_add_kernel::<F>(ket0_sin_real, ket0_sin_imag, ket1_cos_real, ket1_cos_imag);
        state_real[ket.1] = r;
        state_imag[ket.1] = i;
    }
}

#[cube(launch_unchecked)]
fn gate_rz_kernel<F: Float>(
    state_real: &mut Array<F>,
    state_imag: &mut Array<F>,
    control_mask: u32,
    half_block_size: u32,
    full_block_size: u32,
    angle_real: F,
    angle_imag: F,
) {
    let ket = compute_ket_kernel(control_mask, half_block_size, full_block_size);

    if ket.2 {
        let (r, i) = complex_mul_kernel::<F>(
            state_real[ket.0],
            state_imag[ket.0],
            angle_real,
            -angle_imag,
        );
        state_real[ket.0] = r;
        state_imag[ket.0] = i;

        let (r, i) =
            complex_mul_kernel::<F>(state_real[ket.1], state_imag[ket.1], angle_real, angle_imag);
        state_real[ket.1] = r;
        state_imag[ket.1] = i;
    }
}

#[cube(launch_unchecked)]
fn measure_p1_kernel<F: Float>(
    state_real: &Array<F>,
    state_imag: &Array<F>,
    prob: &mut Array<F>,
    target_mask: u32,
) {
    let state = (CUBE_POS_X << 10) + UNIT_POS_X;
    prob[state] = if (state & target_mask) == target_mask {
        state_real[state] * state_real[state] + state_imag[state] * state_imag[state]
    } else {
        F::new(0.0)
    }
}

#[cube(launch_unchecked)]
fn measure_collapse_kernel<F: Float>(
    state_real: &mut Array<F>,
    state_imag: &mut Array<F>,
    target_mask: u32,
    result: u32,
    p: F,
) {
    let state = (CUBE_POS_X << 10) + UNIT_POS_X;
    state_real[state] = if (state & target_mask) == result {
        state_real[state] * p
    } else {
        F::new(0.0)
    };
    state_imag[state] = if (state & target_mask) == result {
        state_imag[state] * p
    } else {
        F::new(0.0)
    };
}

fn compute_cube_size(num_qubits: usize, target: usize) -> (u32, u32, CubeCount, CubeDim) {
    let half_block_size = 1u32 << target;
    let full_block_size = half_block_size << 1;
    let num_blocks = 1u32 << (num_qubits - target - 1);

    let mut cube_count_x = 1;
    let mut cube_count_y = 1;
    let mut cube_dim_x = num_blocks;
    let mut cube_dim_y = half_block_size;

    while cube_dim_x * cube_dim_y > 1024 {
        if cube_dim_x > cube_dim_y {
            cube_dim_x >>= 1;
            cube_count_x <<= 1;
        } else {
            cube_dim_y >>= 1;
            cube_count_y <<= 1;
        }
    }

    //println!("num_qubits={num_qubits}, target={target} => half_block_size={half_block_size}, full_block_size={full_block_size}, num_blocks={num_blocks}, cube_count=({cube_count_x}, {cube_count_y}), cube_dim=({cube_dim_x}, {cube_dim_y})");

    (
        half_block_size,
        full_block_size,
        CubeCount::new_2d(cube_count_x, cube_count_y),
        CubeDim::new_2d(cube_dim_x, cube_dim_y),
    )
}

impl<R: Runtime, F: Float + CubeElement + FloatOps> QuantumExecution for DenseGPU<R, F> {
    fn new(num_qubits: usize) -> crate::error::Result<Self>
    where
        Self: Sized,
    {
        let device = Default::default();
        let client = R::client(&device);

        let state_size = 1usize << num_qubits;
        let mut data = vec![F::new(0.0); state_size];

        let state_imag = client.create(F::as_bytes(&data));

        data[0] = F::new(1.0);
        let state_real = client.create(F::as_bytes(&data));

        Ok(DenseGPU {
            state_real,
            state_imag,
            state_size,
            num_qubits,
            client,
            _f: PhantomData,
        })
    }

    fn pauli_x(&mut self, target: usize, control: &[usize]) {
        //println!("Call X {target}");

        let (half_block_size, full_block_size, cube_count, cube_dim) =
            compute_cube_size(self.num_qubits, target);
        unsafe {
            gate_x_kernel::launch_unchecked::<F, R>(
                &self.client,
                cube_count,
                cube_dim,
                ArrayArg::from_raw_parts::<F>(&self.state_real, self.state_size, 1),
                ArrayArg::from_raw_parts::<F>(&self.state_imag, self.state_size, 1),
                ScalarArg::new(get_ctrl_mask(control)),
                ScalarArg::new(half_block_size),
                ScalarArg::new(full_block_size),
            );
        }

        //let state_real = self.client.read_one(self.state_real.clone());
        //let state_real = F::from_bytes(&state_real);
        //println!("state_real={state_real:?}")
    }

    fn pauli_y(&mut self, target: usize, control: &[usize]) {
        let (half_block_size, full_block_size, cube_count, cube_dim) =
            compute_cube_size(self.num_qubits, target);
        unsafe {
            gate_y_kernel::launch_unchecked::<F, R>(
                &self.client,
                cube_count,
                cube_dim,
                ArrayArg::from_raw_parts::<F>(&self.state_real, self.state_size, 1),
                ArrayArg::from_raw_parts::<F>(&self.state_imag, self.state_size, 1),
                ScalarArg::new(get_ctrl_mask(control)),
                ScalarArg::new(half_block_size),
                ScalarArg::new(full_block_size),
            );
        }
    }

    fn pauli_z(&mut self, target: usize, control: &[usize]) {
        let (half_block_size, full_block_size, cube_count, cube_dim) =
            compute_cube_size(self.num_qubits, target);
        unsafe {
            gate_z_kernel::launch_unchecked::<F, R>(
                &self.client,
                cube_count,
                cube_dim,
                ArrayArg::from_raw_parts::<F>(&self.state_real, self.state_size, 1),
                ArrayArg::from_raw_parts::<F>(&self.state_imag, self.state_size, 1),
                ScalarArg::new(get_ctrl_mask(control)),
                ScalarArg::new(half_block_size),
                ScalarArg::new(full_block_size),
            );
        }
    }

    fn hadamard(&mut self, target: usize, control: &[usize]) {
        let (half_block_size, full_block_size, cube_count, cube_dim) =
            compute_cube_size(self.num_qubits, target);
        unsafe {
            gate_h_kernel::launch_unchecked::<F, R>(
                &self.client,
                cube_count,
                cube_dim,
                ArrayArg::from_raw_parts::<F>(&self.state_real, self.state_size, 1),
                ArrayArg::from_raw_parts::<F>(&self.state_imag, self.state_size, 1),
                ScalarArg::new(get_ctrl_mask(control)),
                ScalarArg::new(half_block_size),
                ScalarArg::new(full_block_size),
                ScalarArg::new(F::FRAC_1_SQRT_2()),
            );
        }
    }

    fn phase(&mut self, lambda: f64, target: usize, control: &[usize]) {
        let (half_block_size, full_block_size, cube_count, cube_dim) =
            compute_cube_size(self.num_qubits, target);
        unsafe {
            gate_p_kernel::launch_unchecked::<F, R>(
                &self.client,
                cube_count,
                cube_dim,
                ArrayArg::from_raw_parts::<F>(&self.state_real, self.state_size, 1),
                ArrayArg::from_raw_parts::<F>(&self.state_imag, self.state_size, 1),
                ScalarArg::new(get_ctrl_mask(control)),
                ScalarArg::new(half_block_size),
                ScalarArg::new(full_block_size),
                ScalarArg::new(F::from(lambda.cos()).unwrap()),
                ScalarArg::new(F::from(lambda.sin()).unwrap()),
            );
        }
    }

    fn rx(&mut self, theta: f64, target: usize, control: &[usize]) {
        let (half_block_size, full_block_size, cube_count, cube_dim) =
            compute_cube_size(self.num_qubits, target);
        unsafe {
            gate_rx_kernel::launch_unchecked::<F, R>(
                &self.client,
                cube_count,
                cube_dim,
                ArrayArg::from_raw_parts::<F>(&self.state_real, self.state_size, 1),
                ArrayArg::from_raw_parts::<F>(&self.state_imag, self.state_size, 1),
                ScalarArg::new(get_ctrl_mask(control)),
                ScalarArg::new(half_block_size),
                ScalarArg::new(full_block_size),
                ScalarArg::new(F::from((theta / 2.0).cos()).unwrap()),
                ScalarArg::new(F::from(-(theta / 2.0).sin()).unwrap()),
            );
        }
    }

    fn ry(&mut self, theta: f64, target: usize, control: &[usize]) {
        let (half_block_size, full_block_size, cube_count, cube_dim) =
            compute_cube_size(self.num_qubits, target);
        unsafe {
            gate_ry_kernel::launch_unchecked::<F, R>(
                &self.client,
                cube_count,
                cube_dim,
                ArrayArg::from_raw_parts::<F>(&self.state_real, self.state_size, 1),
                ArrayArg::from_raw_parts::<F>(&self.state_imag, self.state_size, 1),
                ScalarArg::new(get_ctrl_mask(control)),
                ScalarArg::new(half_block_size),
                ScalarArg::new(full_block_size),
                ScalarArg::new(F::from((theta / 2.0).cos()).unwrap()),
                ScalarArg::new(F::from((theta / 2.0).sin()).unwrap()),
            );
        }
    }

    fn rz(&mut self, theta: f64, target: usize, control: &[usize]) {
        let (half_block_size, full_block_size, cube_count, cube_dim) =
            compute_cube_size(self.num_qubits, target);
        unsafe {
            gate_rz_kernel::launch_unchecked::<F, R>(
                &self.client,
                cube_count,
                cube_dim,
                ArrayArg::from_raw_parts::<F>(&self.state_real, self.state_size, 1),
                ArrayArg::from_raw_parts::<F>(&self.state_imag, self.state_size, 1),
                ScalarArg::new(get_ctrl_mask(control)),
                ScalarArg::new(half_block_size),
                ScalarArg::new(full_block_size),
                ScalarArg::new(F::from((theta / 2.0).cos()).unwrap()),
                ScalarArg::new(F::from((theta / 2.0).sin()).unwrap()),
            );
        }
    }

    fn measure<RNG: rand::Rng>(&mut self, target: usize, rng: &mut RNG) -> bool {
        let prob = self
            .client
            .empty(self.state_size * core::mem::size_of::<F>());

        let (cube_count, cube_dim) = if self.num_qubits <= 10 {
            (1, self.state_size)
        } else {
            (1 << (self.num_qubits - 10), 1024)
        };

        let target_mask = 1 << target;

        unsafe {
            measure_p1_kernel::launch_unchecked::<F, R>(
                &self.client,
                CubeCount::new_1d(cube_count as u32),
                CubeDim::new_1d(cube_dim as u32),
                ArrayArg::from_raw_parts::<F>(&self.state_real, self.state_size, 1),
                ArrayArg::from_raw_parts::<F>(&self.state_imag, self.state_size, 1),
                ArrayArg::from_raw_parts::<F>(&prob, self.state_size, 1),
                ScalarArg::new(target_mask),
            );
        }

        let prob = self.client.read_one(prob);
        let prob = F::from_bytes(&prob);

        //println!("prob={prob:?}");

        let p1: F = prob.par_iter().copied().sum();

        let p0 = match F::one() - p1 {
            p0 if p0 >= F::zero() => p0,
            _ => F::zero(),
        };

        //println!("P0={p0}; P1={p1}; target={target}");

        let result = WeightedIndex::new([p0.to_f64().unwrap(), p1.to_f64().unwrap()])
            .unwrap()
            .sample(rng)
            == 1;

        let p = F::one() / <F as num_traits::Float>::sqrt(if result { p1 } else { p0 });

        unsafe {
            measure_collapse_kernel::launch_unchecked::<F, R>(
                &self.client,
                CubeCount::new_1d(cube_count as u32),
                CubeDim::new_1d(cube_dim as u32),
                ArrayArg::from_raw_parts::<F>(&self.state_real, self.state_size, 1),
                ArrayArg::from_raw_parts::<F>(&self.state_imag, self.state_size, 1),
                ScalarArg::new(target_mask),
                ScalarArg::new(if result { target_mask } else { 0 }),
                ScalarArg::new(p),
            );
        }

        result
    }

    fn dump(&mut self, _qubits: &[usize]) -> ket::process::DumpData {
        todo!()
    }

    fn clear(&mut self) {
        let mut data = vec![F::new(0.0); self.state_size];

        self.state_imag = self.client.create(F::as_bytes(&data));

        data[0] = F::new(1.0);
        self.state_real = self.client.create(F::as_bytes(&data));
    }

    fn save(&self) -> Vec<u8> {
        unimplemented!("save quantum state is not available for KBW::DENSE::GPU")
    }

    fn load(&mut self, _data: &[u8]) {
        unimplemented!("load quantum state is not available for KBW::DENSE::GPU")
    }
}

impl<R: Runtime, F: Float + CubeElement + FloatOps> ExecutionFeatures for DenseGPU<R, F> {
    fn feature_measure() -> Capability {
        Capability::Advanced
    }

    fn feature_sample() -> Capability {
        Capability::Unsupported
    }

    fn feature_exp_value() -> Capability {
        Capability::Unsupported
    }

    fn feature_dump() -> Capability {
        Capability::Unsupported
    }

    fn feature_need_decomposition() -> bool {
        false
    }

    fn feature_allow_live() -> bool {
        true
    }

    fn supports_gradient() -> bool {
        false
    }
}
