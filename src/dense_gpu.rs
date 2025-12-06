use crate::quantum_execution::{ExecutionFeatures, QuantumExecution};
use cubecl::prelude::*;
use ket::execution::Capability;
use std::{f32::consts::FRAC_1_SQRT_2, marker::PhantomData};

pub struct DenseGPU<R: Runtime, F: Float + CubeElement> {
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
    state_size: u32,
    offset_x: u32,
    offset_y: u32,
) -> (u32, u32, bool) {
    let ket0 = (CUBE_POS_X << offset_x)
        + UNIT_POS_X * full_block_size
        + (CUBE_POS_Y << offset_y)
        + UNIT_POS_Y;
    let ket1 = ket0 + half_block_size;

    (
        ket0,
        ket1,
        ket1 < state_size && ket0 & control_mask == control_mask,
    )
}

#[cube(launch_unchecked)]
fn gate_x_kernel<F: Float>(
    state_real: &mut Array<F>,
    state_imag: &mut Array<F>,
    control_mask: u32,
    half_block_size: u32,
    full_block_size: u32,
    state_size: u32,
    offset_x: u32,
    offset_y: u32,
) {
    let ket = compute_ket_kernel(
        control_mask,
        half_block_size,
        full_block_size,
        state_size,
        offset_x,
        offset_y,
    );

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
    state_size: u32,
    offset_x: u32,
    offset_y: u32,
) {
    let ket = compute_ket_kernel(
        control_mask,
        half_block_size,
        full_block_size,
        state_size,
        offset_x,
        offset_y,
    );

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
    state_size: u32,
    offset_x: u32,
    offset_y: u32,
) {
    let ket = compute_ket_kernel(
        control_mask,
        half_block_size,
        full_block_size,
        state_size,
        offset_x,
        offset_y,
    );

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
    state_size: u32,
    offset_x: u32,
    offset_y: u32,
    frac_1_sqrt_2: F,
) {
    let ket = compute_ket_kernel(
        control_mask,
        half_block_size,
        full_block_size,
        state_size,
        offset_x,
        offset_y,
    );

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
fn complex_mul_kernel<F: Float>(
    lhs_real: F,
    lhs_imag: F,
    rhs_real: F,
    rhs_imag: F,
    result_real: &mut F,
    result_imag: &mut F,
) {
    *result_real = lhs_real * rhs_real - lhs_imag * rhs_imag;
    *result_imag = lhs_real * rhs_imag + lhs_imag * rhs_real;
}

#[cube]
fn complex_add_kernel<F: Float>(
    lhs_real: F,
    lhs_imag: F,
    rhs_real: F,
    rhs_imag: F,
    result_real: &mut F,
    result_imag: &mut F,
) {
    *result_real = lhs_real + rhs_real;
    *result_imag = lhs_imag + rhs_imag;
}

#[cube(launch_unchecked)]
fn gate_p_kernel<F: Float>(
    state_real: &mut Array<F>,
    state_imag: &mut Array<F>,
    control_mask: u32,
    half_block_size: u32,
    full_block_size: u32,
    state_size: u32,
    offset_x: u32,
    offset_y: u32,
    angle_real: F,
    angle_imag: F,
) {
    let ket = compute_ket_kernel(
        control_mask,
        half_block_size,
        full_block_size,
        state_size,
        offset_x,
        offset_y,
    );

    if ket.2 {
        complex_mul_kernel::<F>(
            state_real[ket.1],
            state_imag[ket.1],
            angle_real,
            angle_imag,
            &mut state_real[ket.1],
            &mut state_imag[ket.1],
        );
    }
}

#[cube(launch_unchecked)]
fn gate_rx_kernel<F: Float>(
    state_real: &mut Array<F>,
    state_imag: &mut Array<F>,
    control_mask: u32,
    half_block_size: u32,
    full_block_size: u32,
    state_size: u32,
    offset_x: u32,
    offset_y: u32,
    cos_theta_2: F,
    sin_theta_2: F,
) {
    let ket = compute_ket_kernel(
        control_mask,
        half_block_size,
        full_block_size,
        state_size,
        offset_x,
        offset_y,
    );

    if ket.2 {
        let mut ket0_cos_real = F::new(0.0);
        let mut ket0_cos_imag = F::new(0.0);
        complex_mul_kernel::<F>(
            cos_theta_2,
            F::new(0.0),
            state_real[ket.0],
            state_imag[ket.0],
            &mut ket0_cos_real,
            &mut ket0_cos_imag,
        );

        let mut ket1_sin_real = F::new(0.0);
        let mut ket1_sin_imag = F::new(0.0);
        complex_mul_kernel::<F>(
            F::new(0.0),
            sin_theta_2,
            state_real[ket.1],
            state_imag[ket.1],
            &mut ket1_sin_real,
            &mut ket1_sin_imag,
        );

        let mut ket0_sin_real = F::new(0.0);
        let mut ket0_sin_imag = F::new(0.0);
        complex_mul_kernel::<F>(
            F::new(0.0),
            sin_theta_2,
            state_real[ket.0],
            state_imag[ket.0],
            &mut ket0_sin_real,
            &mut ket0_sin_imag,
        );

        let mut ket1_cos_real = F::new(0.0);
        let mut ket1_cos_imag = F::new(0.0);
        complex_mul_kernel::<F>(
            cos_theta_2,
            F::new(0.0),
            state_real[ket.1],
            state_imag[ket.1],
            &mut ket1_cos_real,
            &mut ket1_cos_imag,
        );

        complex_add_kernel::<F>(
            ket0_cos_real,
            ket0_cos_imag,
            ket1_sin_real,
            ket1_sin_imag,
            &mut state_real[ket.0],
            &mut state_imag[ket.0],
        );

        complex_add_kernel::<F>(
            ket0_sin_real,
            ket0_sin_imag,
            ket1_cos_real,
            ket1_cos_imag,
            &mut state_real[ket.1],
            &mut state_imag[ket.1],
        );
    }
}

#[cube(launch_unchecked)]
fn gate_ry_kernel<F: Float>(
    state_real: &mut Array<F>,
    state_imag: &mut Array<F>,
    control_mask: u32,
    half_block_size: u32,
    full_block_size: u32,
    state_size: u32,
    offset_x: u32,
    offset_y: u32,
    cos_theta_2: F,
    sin_theta_2: F,
) {
    let ket = compute_ket_kernel(
        control_mask,
        half_block_size,
        full_block_size,
        state_size,
        offset_x,
        offset_y,
    );

    if ket.2 {
        let mut ket0_cos_real = F::new(0.0);
        let mut ket0_cos_imag = F::new(0.0);
        complex_mul_kernel::<F>(
            cos_theta_2,
            F::new(0.0),
            state_real[ket.0],
            state_imag[ket.0],
            &mut ket0_cos_real,
            &mut ket0_cos_imag,
        );

        let mut ket1_sin_real = F::new(0.0);
        let mut ket1_sin_imag = F::new(0.0);
        complex_mul_kernel::<F>(
            -sin_theta_2,
            F::new(0.0),
            state_real[ket.1],
            state_imag[ket.1],
            &mut ket1_sin_real,
            &mut ket1_sin_imag,
        );

        let mut ket0_sin_real = F::new(0.0);
        let mut ket0_sin_imag = F::new(0.0);
        complex_mul_kernel::<F>(
            sin_theta_2,
            F::new(0.0),
            state_real[ket.0],
            state_imag[ket.0],
            &mut ket0_sin_real,
            &mut ket0_sin_imag,
        );

        let mut ket1_cos_real = F::new(0.0);
        let mut ket1_cos_imag = F::new(0.0);
        complex_mul_kernel::<F>(
            cos_theta_2,
            F::new(0.0),
            state_real[ket.1],
            state_imag[ket.1],
            &mut ket1_cos_real,
            &mut ket1_cos_imag,
        );

        complex_add_kernel::<F>(
            ket0_cos_real,
            ket0_cos_imag,
            ket1_sin_real,
            ket1_sin_imag,
            &mut state_real[ket.0],
            &mut state_imag[ket.0],
        );

        complex_add_kernel::<F>(
            ket0_sin_real,
            ket0_sin_imag,
            ket1_cos_real,
            ket1_cos_imag,
            &mut state_real[ket.1],
            &mut state_imag[ket.1],
        );
    }
}

#[cube(launch_unchecked)]
fn gate_rz_kernel<F: Float>(
    state_real: &mut Array<F>,
    state_imag: &mut Array<F>,
    control_mask: u32,
    half_block_size: u32,
    full_block_size: u32,
    state_size: u32,
    offset_x: u32,
    offset_y: u32,
    angle_real: F,
    angle_imag: F,
) {
    let ket = compute_ket_kernel(
        control_mask,
        half_block_size,
        full_block_size,
        state_size,
        offset_x,
        offset_y,
    );

    if ket.2 {
        complex_mul_kernel::<F>(
            state_real[ket.0],
            state_imag[ket.0],
            angle_real,
            -angle_imag,
            &mut state_real[ket.0],
            &mut state_imag[ket.0],
        );

        complex_mul_kernel::<F>(
            state_real[ket.1],
            state_imag[ket.1],
            angle_real,
            angle_imag,
            &mut state_real[ket.1],
            &mut state_imag[ket.1],
        );
    }
}

#[cube(launch_unchecked)]
fn measure_p1_kernel<F: Float>(
    state_real: &Array<F>,
    state_imag: &Array<F>,
    state_size: u32,
    prob: &mut Array<F>,
    target_mask: u32,
) {
    let state = (CUBE_POS_X << 10) + UNIT_POS_X;
    if state < state_size {
        prob[state] = if state & target_mask == target_mask {
            state_real[state] * state_real[state] + state_imag[state] * state_imag[state]
        } else {
            F::new(0.0)
        }
    }
}

fn compute_cube_size(num_qubits: usize, target: usize) -> (u32, u32, CubeCount, CubeDim, u32, u32) {
    let half_block_size = 1u32 << target;
    let full_block_size = half_block_size << 1;
    let num_blocks = 1u32 << (num_qubits - target - 1);

    let cube_count_x = num_blocks.div_ceil(1024);
    let cube_count_y = half_block_size.div_ceil(1024);
    let mut cube_dim_x = std::cmp::min(1024, num_blocks);
    let mut cube_dim_y = std::cmp::min(1024, half_block_size);

    let mut offset_x = if cube_count_x == 1 { 0 } else { 10 };
    let mut offset_y = if cube_count_y == 1 { 0 } else { 10 };

    while cube_dim_x * cube_dim_y > 1024 {
        if cube_count_x > cube_count_y {
            offset_x += 1;
            cube_dim_x >>= 1;
        } else {
            offset_y += 1;
            cube_dim_y >>= 1;
        }
    }

    (
        half_block_size,
        full_block_size,
        CubeCount::new_2d(cube_count_x, cube_count_y),
        CubeDim::new_2d(cube_dim_x, cube_dim_y),
        offset_x,
        offset_y,
    )
}

fn compute_control_mask(control: &[usize]) -> u32 {
    let mut control_mask = 0u32;
    for c in control {
        control_mask |= 1 << c;
    }
    control_mask
}

impl<R: Runtime, F: Float + CubeElement> QuantumExecution for DenseGPU<R, F> {
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
        let (half_block_size, full_block_size, cube_count, cube_dim, offset_x, offset_y) =
            compute_cube_size(self.num_qubits, target);
        unsafe {
            gate_x_kernel::launch_unchecked::<F, R>(
                &self.client,
                cube_count,
                cube_dim,
                ArrayArg::from_raw_parts::<F>(&self.state_real, self.state_size, 1),
                ArrayArg::from_raw_parts::<F>(&self.state_imag, self.state_size, 1),
                ScalarArg::new(compute_control_mask(control)),
                ScalarArg::new(half_block_size),
                ScalarArg::new(full_block_size),
                ScalarArg::new(self.state_size as u32),
                ScalarArg::new(offset_x),
                ScalarArg::new(offset_y),
            );
        }
    }

    fn pauli_y(&mut self, target: usize, control: &[usize]) {
        let (half_block_size, full_block_size, cube_count, cube_dim, offset_x, offset_y) =
            compute_cube_size(self.num_qubits, target);
        unsafe {
            gate_y_kernel::launch_unchecked::<F, R>(
                &self.client,
                cube_count,
                cube_dim,
                ArrayArg::from_raw_parts::<F>(&self.state_real, self.state_size, 1),
                ArrayArg::from_raw_parts::<F>(&self.state_imag, self.state_size, 1),
                ScalarArg::new(compute_control_mask(control)),
                ScalarArg::new(half_block_size),
                ScalarArg::new(full_block_size),
                ScalarArg::new(self.state_size as u32),
                ScalarArg::new(offset_x),
                ScalarArg::new(offset_y),
            );
        }
    }

    fn pauli_z(&mut self, target: usize, control: &[usize]) {
        let (half_block_size, full_block_size, cube_count, cube_dim, offset_x, offset_y) =
            compute_cube_size(self.num_qubits, target);
        unsafe {
            gate_z_kernel::launch_unchecked::<F, R>(
                &self.client,
                cube_count,
                cube_dim,
                ArrayArg::from_raw_parts::<F>(&self.state_real, self.state_size, 1),
                ArrayArg::from_raw_parts::<F>(&self.state_imag, self.state_size, 1),
                ScalarArg::new(compute_control_mask(control)),
                ScalarArg::new(half_block_size),
                ScalarArg::new(full_block_size),
                ScalarArg::new(self.state_size as u32),
                ScalarArg::new(offset_x),
                ScalarArg::new(offset_y),
            );
        }
    }

    fn hadamard(&mut self, target: usize, control: &[usize]) {
        let (half_block_size, full_block_size, cube_count, cube_dim, offset_x, offset_y) =
            compute_cube_size(self.num_qubits, target);
        unsafe {
            gate_h_kernel::launch_unchecked::<F, R>(
                &self.client,
                cube_count,
                cube_dim,
                ArrayArg::from_raw_parts::<F>(&self.state_real, self.state_size, 1),
                ArrayArg::from_raw_parts::<F>(&self.state_imag, self.state_size, 1),
                ScalarArg::new(compute_control_mask(control)),
                ScalarArg::new(half_block_size),
                ScalarArg::new(full_block_size),
                ScalarArg::new(self.state_size as u32),
                ScalarArg::new(offset_x),
                ScalarArg::new(offset_y),
                ScalarArg::new(F::new(FRAC_1_SQRT_2)),
            );
        }
    }

    fn phase(&mut self, lambda: f64, target: usize, control: &[usize]) {
        let (half_block_size, full_block_size, cube_count, cube_dim, offset_x, offset_y) =
            compute_cube_size(self.num_qubits, target);
        unsafe {
            gate_p_kernel::launch_unchecked::<F, R>(
                &self.client,
                cube_count,
                cube_dim,
                ArrayArg::from_raw_parts::<F>(&self.state_real, self.state_size, 1),
                ArrayArg::from_raw_parts::<F>(&self.state_imag, self.state_size, 1),
                ScalarArg::new(compute_control_mask(control)),
                ScalarArg::new(half_block_size),
                ScalarArg::new(full_block_size),
                ScalarArg::new(self.state_size as u32),
                ScalarArg::new(offset_x),
                ScalarArg::new(offset_y),
                ScalarArg::new(F::new(lambda.cos() as f32)),
                ScalarArg::new(F::new(lambda.sin() as f32)),
            );
        }
    }

    fn rx(&mut self, theta: f64, target: usize, control: &[usize]) {
        let (half_block_size, full_block_size, cube_count, cube_dim, offset_x, offset_y) =
            compute_cube_size(self.num_qubits, target);
        unsafe {
            gate_rx_kernel::launch_unchecked::<F, R>(
                &self.client,
                cube_count,
                cube_dim,
                ArrayArg::from_raw_parts::<F>(&self.state_real, self.state_size, 1),
                ArrayArg::from_raw_parts::<F>(&self.state_imag, self.state_size, 1),
                ScalarArg::new(compute_control_mask(control)),
                ScalarArg::new(half_block_size),
                ScalarArg::new(full_block_size),
                ScalarArg::new(self.state_size as u32),
                ScalarArg::new(offset_x),
                ScalarArg::new(offset_y),
                ScalarArg::new(F::new((theta / 2.0).cos() as f32)),
                ScalarArg::new(F::new(-(theta / 2.0).sin() as f32)),
            );
        }
    }

    fn ry(&mut self, theta: f64, target: usize, control: &[usize]) {
        let (half_block_size, full_block_size, cube_count, cube_dim, offset_x, offset_y) =
            compute_cube_size(self.num_qubits, target);
        unsafe {
            gate_ry_kernel::launch_unchecked::<F, R>(
                &self.client,
                cube_count,
                cube_dim,
                ArrayArg::from_raw_parts::<F>(&self.state_real, self.state_size, 1),
                ArrayArg::from_raw_parts::<F>(&self.state_imag, self.state_size, 1),
                ScalarArg::new(compute_control_mask(control)),
                ScalarArg::new(half_block_size),
                ScalarArg::new(full_block_size),
                ScalarArg::new(self.state_size as u32),
                ScalarArg::new(offset_x),
                ScalarArg::new(offset_y),
                ScalarArg::new(F::new((theta / 2.0).cos() as f32)),
                ScalarArg::new(F::new((theta / 2.0).sin() as f32)),
            );
        }
    }

    fn rz(&mut self, theta: f64, target: usize, control: &[usize]) {
        let (half_block_size, full_block_size, cube_count, cube_dim, offset_x, offset_y) =
            compute_cube_size(self.num_qubits, target);
        unsafe {
            gate_rz_kernel::launch_unchecked::<F, R>(
                &self.client,
                cube_count,
                cube_dim,
                ArrayArg::from_raw_parts::<F>(&self.state_real, self.state_size, 1),
                ArrayArg::from_raw_parts::<F>(&self.state_imag, self.state_size, 1),
                ScalarArg::new(compute_control_mask(control)),
                ScalarArg::new(half_block_size),
                ScalarArg::new(full_block_size),
                ScalarArg::new(self.state_size as u32),
                ScalarArg::new(offset_x),
                ScalarArg::new(offset_y),
                ScalarArg::new(F::new((theta / 2.0).cos() as f32)),
                ScalarArg::new(F::new((theta / 2.0).sin() as f32)),
            );
        }
    }

    fn measure<RNG: rand::Rng>(&mut self, target: usize, rng: &mut RNG) -> bool {
        let prob = self.client.empty(self.state_size);

        unsafe {
            measure_p1_kernel::launch_unchecked::<F, R>(
                &self.client,
                CubeCount::new_1d(if self.num_qubits > 10 {
                    1 << (self.num_qubits - 10)
                } else {
                    1
                }),
                CubeDim::new_1d(if self.num_qubits > 10 {
                    1024
                } else {
                    self.state_size as u32
                }),
                ArrayArg::from_raw_parts::<F>(&self.state_real, self.state_size, 1),
                ArrayArg::from_raw_parts::<F>(&self.state_imag, self.state_size, 1),
                ScalarArg::new(self.state_size as u32),
                ArrayArg::from_raw_parts::<F>(&prob, self.state_size, 1),
                ScalarArg::new(1 << target),
            );
        }

        todo!()
    }

    fn dump(&mut self, qubits: &[usize]) -> ket::process::DumpData {
        todo!()
    }

    fn clear(&mut self) {
        todo!()
    }

    fn save(&self) -> Vec<u8> {
        todo!()
    }

    fn load(&mut self, data: &[u8]) {
        todo!()
    }
}

impl<R: Runtime, F: Float + CubeElement> ExecutionFeatures for DenseGPU<R, F> {
    fn feature_measure() -> Capability {
        Capability::Advanced
    }

    fn feature_sample() -> Capability {
        Capability::Advanced
    }

    fn feature_exp_value() -> Capability {
        Capability::Advanced
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
