// SPDX-FileCopyrightText: 2026 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
//
// SPDX-License-Identifier: Apache-2.0

use cubecl::prelude::*;

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

        state_real[ket.0] = state_imag[ket.1];
        state_imag[ket.0] = -state_real[ket.1];

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
pub(super) fn gate_h<F: Float + LaunchArg>(
    state_real: &mut Array<F>,
    state_imag: &mut Array<F>,
    offset: usize,
    control_mask: u32,
    half_block_size: u32,
    full_block_size: u32,
    frac_1_sqrt_2: F,
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

        state_real[ket.0] = (old0_real + old1_real) * frac_1_sqrt_2;
        state_imag[ket.0] = (old0_imag + old1_imag) * frac_1_sqrt_2;

        state_real[ket.1] = (old0_real - old1_real) * frac_1_sqrt_2;
        state_imag[ket.1] = (old0_imag - old1_imag) * frac_1_sqrt_2;
    }
}

#[cube]
pub(super) fn complex_mul<F: Float>(lhs_real: F, lhs_imag: F, rhs_real: F, rhs_imag: F) -> (F, F) {
    (
        lhs_real * rhs_real - lhs_imag * rhs_imag,
        lhs_real * rhs_imag + lhs_imag * rhs_real,
    )
}

#[cube]
pub(super) fn complex_add<F: Float>(lhs_real: F, lhs_imag: F, rhs_real: F, rhs_imag: F) -> (F, F) {
    (lhs_real + rhs_real, lhs_imag + rhs_imag)
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
pub(super) fn gate_rx<F: Float + LaunchArg>(
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
        let (ket0_cos_real, ket0_cos_imag) = complex_mul::<F>(
            cos_theta_2,
            F::new(0.0),
            state_real[ket.0],
            state_imag[ket.0],
        );

        let (ket1_sin_real, ket1_sin_imag) = complex_mul::<F>(
            F::new(0.0),
            sin_theta_2,
            state_real[ket.1],
            state_imag[ket.1],
        );

        let (ket0_sin_real, ket0_sin_imag) = complex_mul::<F>(
            F::new(0.0),
            sin_theta_2,
            state_real[ket.0],
            state_imag[ket.0],
        );

        let (ket1_cos_real, ket1_cos_imag) = complex_mul::<F>(
            cos_theta_2,
            F::new(0.0),
            state_real[ket.1],
            state_imag[ket.1],
        );

        let (r, i) = complex_add::<F>(ket0_cos_real, ket0_cos_imag, ket1_sin_real, ket1_sin_imag);
        state_real[ket.0] = r;
        state_imag[ket.0] = i;

        let (r, i) = complex_add::<F>(ket0_sin_real, ket0_sin_imag, ket1_cos_real, ket1_cos_imag);
        state_real[ket.1] = r;
        state_imag[ket.1] = i;
    }
}

#[cube(launch_unchecked)]
pub(super) fn gate_ry<F: Float + LaunchArg>(
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
        let (ket0_cos_real, ket0_cos_imag) = complex_mul::<F>(
            cos_theta_2,
            F::new(0.0),
            state_real[ket.0],
            state_imag[ket.0],
        );

        let (ket1_sin_real, ket1_sin_imag) = complex_mul::<F>(
            -sin_theta_2,
            F::new(0.0),
            state_real[ket.1],
            state_imag[ket.1],
        );

        let (ket0_sin_real, ket0_sin_imag) = complex_mul::<F>(
            sin_theta_2,
            F::new(0.0),
            state_real[ket.0],
            state_imag[ket.0],
        );

        let (ket1_cos_real, ket1_cos_imag) = complex_mul::<F>(
            cos_theta_2,
            F::new(0.0),
            state_real[ket.1],
            state_imag[ket.1],
        );

        let (r, i) = complex_add::<F>(ket0_cos_real, ket0_cos_imag, ket1_sin_real, ket1_sin_imag);
        state_real[ket.0] = r;
        state_imag[ket.0] = i;

        let (r, i) = complex_add::<F>(ket0_sin_real, ket0_sin_imag, ket1_cos_real, ket1_cos_imag);
        state_real[ket.1] = r;
        state_imag[ket.1] = i;
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
        let (r, i) = complex_mul::<F>(
            state_real[ket.0],
            state_imag[ket.0],
            angle_real,
            -angle_imag,
        );
        state_real[ket.0] = r;
        state_imag[ket.0] = i;

        let (r, i) = complex_mul::<F>(state_real[ket.1], state_imag[ket.1], angle_real, angle_imag);
        state_real[ket.1] = r;
        state_imag[ket.1] = i;
    }
}

#[cube(launch_unchecked)]
pub(super) fn measure_p1<F: Float>(
    state_real: &Array<F>,
    state_imag: &Array<F>,
    offset: usize,
    prob: &mut Array<F>,
    target: u32,
    mask: u32,
) {
    let state = ABSOLUTE_POS_X + offset as u32;
    let state = ((((state >> target) << 1) | 1) << target) | (state & mask);
    let state = state as usize;

    prob[ABSOLUTE_POS_X as usize + offset] =
        state_real[state] * state_real[state] + state_imag[state] * state_imag[state]
}

#[cube(launch_unchecked)]
pub(super) fn measure_collapse<F: Float + LaunchArg>(
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
        F::new(0.0)
    };
    state_imag[state_index] = if (state & target_mask) == result {
        state_imag[state_index] * p
    } else {
        F::new(0.0)
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

#[cube(launch_unchecked)]
pub(super) fn exp_value<F: Float>(
    state_real: &Array<F>,
    state_imag: &Array<F>,
    offset: usize,
    prob: &mut Array<F>,
    target_mask: u32,
) {
    let state = ABSOLUTE_POS_X + offset as u32;
    let state_index = state as usize;
    prob[state_index] = if parity_u32(state & target_mask) == 1 {
        F::new(-1.0)
    } else {
        F::new(1.0)
    } * (state_real[state_index] * state_real[state_index]
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

#[cube(launch_unchecked)]
pub(super) fn init_state<F: Float>(
    state_real: &mut Array<F>,
    state_imag: &mut Array<F>,
    offset: u32,
) {
    let state = (offset + ABSOLUTE_POS_X) as usize;
    state_real[state] = if state == 0 { F::new(1.0) } else { F::new(0.0) };
    state_imag[state] = F::new(0.0);
}
