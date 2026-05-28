// SPDX-FileCopyrightText: 2026 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
//
// SPDX-License-Identifier: Apache-2.0

use cubecl::prelude::*;

use crate::dense_gpu::ConstantFloat;

pub(super) fn compute_cube_size(
    num_qubits: usize,
    target: usize,
) -> (u32, u32, (u32, u32), (u32, u32), CubeCount, CubeDim) {
    let half_block_size = 1u32 << target;
    let full_block_size = target + 1;
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

    let mut cube_count_2_x = 1;
    let mut cube_count_2_y = 1;

    while cube_count_x * cube_count_y > 65535 {
        if cube_count_x > cube_count_y {
            cube_count_x >>= 1;
            cube_count_2_x <<= 1;
        } else {
            cube_count_y >>= 1;
            cube_count_2_y <<= 1;
        }
    }

    let stride_x = cube_count_x * cube_dim_x;
    let stride_y = cube_count_y * cube_dim_y;

    (
        half_block_size,
        full_block_size as u32,
        (cube_count_2_x, cube_count_2_y),
        (stride_x, stride_y),
        CubeCount::new_2d(cube_count_x, cube_count_y),
        CubeDim::new_2d(cube_dim_x, cube_dim_y),
    )
}

#[cube]
pub(super) const fn compute_ket(
    offset: usize,
    control_mask: u32,
    half_block_size: u32,
    full_block_size: u32,
    count_offset_x: u32,
    count_offset_y: u32,
) -> (usize, usize, bool) {
    let ket0 = offset
        + (((ABSOLUTE_POS_X + count_offset_x) << full_block_size)
            + (ABSOLUTE_POS_Y + count_offset_y)) as usize;
    let ket1 = ket0 + half_block_size as usize;

    (
        ket0,
        ket1,
        ket0 & control_mask as usize == control_mask as usize,
    )
}

#[cube(launch_unchecked)]
pub(super) fn gate_x<F: Float>(
    state_real: &mut Array<F>,
    state_imag: &mut Array<F>,
    offset: usize,
    control_mask: u32,
    half_block_size: u32,
    full_block_size: u32,
    count_offset_x: u32,
    count_offset_y: u32,
) {
    let ket = compute_ket(
        offset,
        control_mask,
        half_block_size,
        full_block_size,
        count_offset_x,
        count_offset_y,
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
pub(super) fn gate_y<F: Float>(
    state_real: &mut Array<F>,
    state_imag: &mut Array<F>,
    offset: usize,
    control_mask: u32,
    half_block_size: u32,
    full_block_size: u32,
    count_offset_x: u32,
    count_offset_y: u32,
) {
    let ket = compute_ket(
        offset,
        control_mask,
        half_block_size,
        full_block_size,
        count_offset_x,
        count_offset_y,
    );

    if ket.2 {
        let old0_real = state_real[ket.0];
        let old0_imag = state_imag[ket.0];

        let old1_real = state_real[ket.1];
        let old1_imag = state_imag[ket.1];

        state_real[ket.0] = old1_imag;
        state_imag[ket.0] = -old1_real;

        state_real[ket.1] = -old0_imag;
        state_imag[ket.1] = old0_real;
    }
}

#[cube(launch_unchecked)]
pub(super) fn gate_z<F: Float>(
    state_real: &mut Array<F>,
    state_imag: &mut Array<F>,
    offset: usize,
    control_mask: u32,
    half_block_size: u32,
    full_block_size: u32,
    count_offset_x: u32,
    count_offset_y: u32,
) {
    let ket = compute_ket(
        offset,
        control_mask,
        half_block_size,
        full_block_size,
        count_offset_x,
        count_offset_y,
    );

    if ket.2 {
        state_real[ket.1] = -state_real[ket.1];
        state_imag[ket.1] = -state_imag[ket.1];
    }
}

#[cube(launch_unchecked)]
pub(super) fn gate_h<F: Float + LaunchArg + ConstantFloat>(
    state_real: &mut Array<F>,
    state_imag: &mut Array<F>,
    offset: usize,
    control_mask: u32,
    half_block_size: u32,
    full_block_size: u32,
    count_offset_x: u32,
    count_offset_y: u32,
) {
    let ket = compute_ket(
        offset,
        control_mask,
        half_block_size,
        full_block_size,
        count_offset_x,
        count_offset_y,
    );

    if ket.2 {
        let r0 = state_real[ket.0];
        let i0 = state_imag[ket.0];

        let r1 = state_real[ket.1];
        let i1 = state_imag[ket.1];

        let k = F::FRAC_1_SQRT_2;

        let sum_r = r0 + r1;
        let diff_r = r0 - r1;

        let sum_i = i0 + i1;
        let diff_i = i0 - i1;

        state_real[ket.0] = sum_r * k;
        state_imag[ket.0] = sum_i * k;

        state_real[ket.1] = diff_r * k;
        state_imag[ket.1] = diff_i * k;
    }
}

#[cube]
fn complex_mul<F: Float>(a_r: F, a_i: F, b_r: F, b_i: F) -> (F, F) {
    let real = fma(a_r, b_r, -(a_i * b_i));
    let imag = fma(a_r, b_i, a_i * b_r);
    (real, imag)
}

#[cube(launch_unchecked)]
pub(super) fn gate_p<F: Float + LaunchArg>(
    state_real: &mut Array<F>,
    state_imag: &mut Array<F>,
    offset: usize,
    control_mask: u32,
    half_block_size: u32,
    full_block_size: u32,
    angle_real: F,
    angle_imag: F,
    count_offset_x: u32,
    count_offset_y: u32,
) {
    let ket = compute_ket(
        offset,
        control_mask,
        half_block_size,
        full_block_size,
        count_offset_x,
        count_offset_y,
    );

    if ket.2 {
        let (r, i) = complex_mul::<F>(state_real[ket.1], state_imag[ket.1], angle_real, angle_imag);
        state_real[ket.1] = r;
        state_imag[ket.1] = i;
    }
}

#[cube(launch_unchecked)]
pub(super) fn gate_rx<F: Float + LaunchArg + ConstantFloat>(
    state_real: &mut Array<F>,
    state_imag: &mut Array<F>,
    offset: usize,
    control_mask: u32,
    half_block_size: u32,
    full_block_size: u32,
    cos_theta_2: F,
    sin_theta_2: F,
    count_offset_x: u32,
    count_offset_y: u32,
) {
    let ket = compute_ket(
        offset,
        control_mask,
        half_block_size,
        full_block_size,
        count_offset_x,
        count_offset_y,
    );

    if ket.2 {
        let ket0_re = state_real[ket.0];
        let ket1_re = state_real[ket.1];
        let ket0_im = state_imag[ket.0];
        let ket1_im = state_imag[ket.1];

        state_real[ket.0] = fma(ket0_re, cos_theta_2, -ket1_im * sin_theta_2);
        state_imag[ket.0] = fma(ket0_im, cos_theta_2, ket1_re * sin_theta_2);

        state_real[ket.1] = fma(ket1_re, cos_theta_2, -ket0_im * sin_theta_2);
        state_imag[ket.1] = fma(ket1_im, cos_theta_2, ket0_re * sin_theta_2);
    }
}

#[cube(launch_unchecked)]
pub(super) fn gate_ry<F: Float + LaunchArg + ConstantFloat>(
    state_real: &mut Array<F>,
    state_imag: &mut Array<F>,
    offset: usize,
    control_mask: u32,
    half_block_size: u32,
    full_block_size: u32,
    cos_theta_2: F,
    sin_theta_2: F,
    count_offset_x: u32,
    count_offset_y: u32,
) {
    let ket = compute_ket(
        offset,
        control_mask,
        half_block_size,
        full_block_size,
        count_offset_x,
        count_offset_y,
    );

    if ket.2 {
        let ket0_re = state_real[ket.0];
        let ket1_re = state_real[ket.1];
        let ket0_im = state_imag[ket.0];
        let ket1_im = state_imag[ket.1];

        state_real[ket.0] = fma(ket0_re, cos_theta_2, ket1_re * -sin_theta_2);
        state_imag[ket.0] = fma(ket0_im, cos_theta_2, ket1_im * -sin_theta_2);

        state_real[ket.1] = fma(ket1_re, cos_theta_2, ket0_re * sin_theta_2);
        state_imag[ket.1] = fma(ket1_im, cos_theta_2, ket0_im * sin_theta_2);
    }
}

#[cube(launch_unchecked)]
pub(super) fn gate_rz<F: Float + LaunchArg>(
    state_real: &mut Array<F>,
    state_imag: &mut Array<F>,
    offset: usize,
    control_mask: u32,
    half_block_size: u32,
    full_block_size: u32,
    angle_real: F,
    angle_imag: F,
    count_offset_x: u32,
    count_offset_y: u32,
) {
    let ket = compute_ket(
        offset,
        control_mask,
        half_block_size,
        full_block_size,
        count_offset_x,
        count_offset_y,
    );

    if ket.2 {
        let ket0_re = state_real[ket.0];
        let ket1_re = state_real[ket.1];
        let ket0_im = state_imag[ket.0];
        let ket1_im = state_imag[ket.1];

        let (r, i) = complex_mul::<F>(ket0_re, ket0_im, angle_real, -angle_imag);
        state_real[ket.0] = r;
        state_imag[ket.0] = i;

        let (r, i) = complex_mul::<F>(ket1_re, ket1_im, angle_real, angle_imag);
        state_real[ket.1] = r;
        state_imag[ket.1] = i;
    }
}

#[cube(launch_unchecked)]
pub(super) fn measure_p1<F: Float>(
    state_real: &Array<F>,
    state_imag: &Array<F>,
    iter_offset: usize,
    block_offset: usize,
    prob: &mut Array<F>,
    target: u32,
    mask: u32,
) {
    let compressed_idx = ABSOLUTE_POS_X + iter_offset as u32;

    let expanded_local_idx =
        ((((compressed_idx >> target) << 1) | 1) << target) | (compressed_idx & mask);

    let read_idx = block_offset + expanded_local_idx as usize;
    let re = state_real[read_idx];
    let im = state_imag[read_idx];

    prob[compressed_idx as usize] = re * re + im * im;
}

#[allow(clippy::useless_conversion)]
#[cube(launch_unchecked)]
pub(super) fn measure_collapse<F: Float + LaunchArg + ConstantFloat>(
    state_real: &mut Array<F>,
    state_imag: &mut Array<F>,
    offset: usize,
    target_mask: u32,
    result: u32,
    p: F,
) {
    let state = ABSOLUTE_POS_X + offset as u32;
    let state_index = state as usize;

    state_real[state_index] = if (state & target_mask) == result {
        state_real[state_index] * p
    } else {
        F::ZERO.into()
    };
    state_imag[state_index] = if (state & target_mask) == result {
        state_imag[state_index] * p
    } else {
        F::ZERO.into()
    };
}

#[cube]
pub(super) const fn parity_u32(x: u32) -> u32 {
    let mut v = x;
    v ^= v >> 16;
    v ^= v >> 8;
    v ^= v >> 4;
    v ^= v >> 2;
    v ^= v >> 1;
    v & 1
}

#[allow(clippy::useless_conversion)]
#[cube(launch_unchecked)]
pub(super) fn exp_value<F: Float + ConstantFloat>(
    state_real: &Array<F>,
    state_imag: &Array<F>,
    offset: usize,
    prob: &mut Array<F>,
    target_mask: u32,
) {
    let state = ABSOLUTE_POS_X + offset as u32;
    let state_index = state as usize;
    let sign: F = if parity_u32(state & target_mask) == 1 {
        F::MINUS_ONE.into()
    } else {
        F::ONE.into()
    };

    prob[state_index] = sign
        * (state_real[state_index] * state_real[state_index]
            + state_imag[state_index] * state_imag[state_index]);
}

#[cube(launch_unchecked)]
pub(super) fn copy_swap<F: Float>(
    old_state_real: &Array<F>,
    old_state_imag: &Array<F>,
    new_state_real: &mut Array<F>,
    new_state_imag: &mut Array<F>,
    offset: u32,
    global_index: u32,
    local_index: u32,
) {
    let state = ABSOLUTE_POS_X + offset;
    let global_bit = (state >> global_index) & 1;
    let local_bit = (state >> local_index) & 1;

    let src_state = state & !((1 << global_index) | (1 << local_index));
    let src_state = src_state | (global_bit << local_index) | (local_bit << global_index);

    new_state_real[state as usize] = old_state_real[src_state as usize];
    new_state_imag[state as usize] = old_state_imag[src_state as usize];
}

#[allow(clippy::useless_conversion)]
#[cube(launch_unchecked)]
pub(super) fn init_state<F: Float + ConstantFloat>(
    state_real: &mut Array<F>,
    state_imag: &mut Array<F>,
    offset: u32,
) {
    let state = (offset + ABSOLUTE_POS_X) as usize;
    state_real[state] = if state == 0 { F::ONE } else { F::ZERO };
    state_imag[state] = F::ZERO;
}
