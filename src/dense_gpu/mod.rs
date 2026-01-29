// SPDX-FileCopyrightText: 2025 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
//
// SPDX-License-Identifier: Apache-2.0

mod kernel;

use crate::{
    bitwise::{get_ctrl_mask, is_one_at},
    block::BlockSimulator,
    quantum_execution::{ExecutionFeatures, QuantumExecution},
    FloatOps,
};
use cubecl::{prelude::*, server::Handle};
use itertools::Itertools;
use ket::{
    execution::Capability,
    prelude::{Hamiltonian, Pauli},
    process::DumpData,
};
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use std::marker::PhantomData;

pub struct DenseGPU<R: Runtime, F: Float + CubeElement + FloatOps> {
    state_real: cubecl::server::Handle,
    state_imag: cubecl::server::Handle,
    state_size: usize,
    num_qubits: usize,

    client: ComputeClient<R>,
    _f: PhantomData<F>,
}

impl<R: Runtime, F: Float + CubeElement + FloatOps> QuantumExecution for DenseGPU<R, F> {
    fn new(num_qubits: usize) -> crate::error::Result<Self>
    where
        Self: Sized,
    {
        let device = Default::default();
        let client = R::client(&device);

        let state_size = 1usize << num_qubits;
        let mut state = vec![F::zero(); state_size];
        let state_imag = client.create_from_slice(F::as_bytes(&state));
        state[0] = F::one();
        let state_real = client.create_from_slice(F::as_bytes(&state));

        Ok(Self {
            state_real,
            state_imag,
            state_size,
            num_qubits,
            client,
            _f: PhantomData,
        })
    }

    fn pauli_x(&mut self, target: usize, control: &[usize]) {
        let (half_block_size, full_block_size, cube_count, cube_dim) =
            kernel::compute_cube_size(self.num_qubits, target);
        unsafe {
            kernel::gate_x::launch_unchecked::<F, R>(
                &self.client,
                cube_count,
                cube_dim,
                ArrayArg::from_raw_parts::<F>(&self.state_real, self.state_size, 1),
                ArrayArg::from_raw_parts::<F>(&self.state_imag, self.state_size, 1),
                ScalarArg::new(get_ctrl_mask(control)),
                ScalarArg::new(half_block_size),
                ScalarArg::new(full_block_size),
            )
            .unwrap();
        }
    }

    fn pauli_y(&mut self, target: usize, control: &[usize]) {
        let (half_block_size, full_block_size, cube_count, cube_dim) =
            kernel::compute_cube_size(self.num_qubits, target);
        unsafe {
            kernel::gate_y::launch_unchecked::<F, R>(
                &self.client,
                cube_count,
                cube_dim,
                ArrayArg::from_raw_parts::<F>(&self.state_real, self.state_size, 1),
                ArrayArg::from_raw_parts::<F>(&self.state_imag, self.state_size, 1),
                ScalarArg::new(get_ctrl_mask(control)),
                ScalarArg::new(half_block_size),
                ScalarArg::new(full_block_size),
            )
            .unwrap();
        }
    }

    fn pauli_z(&mut self, target: usize, control: &[usize]) {
        let (half_block_size, full_block_size, cube_count, cube_dim) =
            kernel::compute_cube_size(self.num_qubits, target);
        unsafe {
            kernel::gate_z::launch_unchecked::<F, R>(
                &self.client,
                cube_count,
                cube_dim,
                ArrayArg::from_raw_parts::<F>(&self.state_real, self.state_size, 1),
                ArrayArg::from_raw_parts::<F>(&self.state_imag, self.state_size, 1),
                ScalarArg::new(get_ctrl_mask(control)),
                ScalarArg::new(half_block_size),
                ScalarArg::new(full_block_size),
            )
            .unwrap();
        }
    }

    fn hadamard(&mut self, target: usize, control: &[usize]) {
        let (half_block_size, full_block_size, cube_count, cube_dim) =
            kernel::compute_cube_size(self.num_qubits, target);
        unsafe {
            kernel::gate_h::launch_unchecked::<F, R>(
                &self.client,
                cube_count,
                cube_dim,
                ArrayArg::from_raw_parts::<F>(&self.state_real, self.state_size, 1),
                ArrayArg::from_raw_parts::<F>(&self.state_imag, self.state_size, 1),
                ScalarArg::new(get_ctrl_mask(control)),
                ScalarArg::new(half_block_size),
                ScalarArg::new(full_block_size),
                ScalarArg::new(F::FRAC_1_SQRT_2()),
            )
            .unwrap();
        }
    }

    fn phase(&mut self, lambda: f64, target: usize, control: &[usize]) {
        let (half_block_size, full_block_size, cube_count, cube_dim) =
            kernel::compute_cube_size(self.num_qubits, target);
        unsafe {
            kernel::gate_p::launch_unchecked::<F, R>(
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
            )
            .unwrap();
        }
    }

    fn rx(&mut self, theta: f64, target: usize, control: &[usize]) {
        let (half_block_size, full_block_size, cube_count, cube_dim) =
            kernel::compute_cube_size(self.num_qubits, target);
        unsafe {
            kernel::gate_rx::launch_unchecked::<F, R>(
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
            )
            .unwrap();
        }
    }

    fn ry(&mut self, theta: f64, target: usize, control: &[usize]) {
        let (half_block_size, full_block_size, cube_count, cube_dim) =
            kernel::compute_cube_size(self.num_qubits, target);
        unsafe {
            kernel::gate_ry::launch_unchecked::<F, R>(
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
            )
            .unwrap();
        }
    }

    fn rz(&mut self, theta: f64, target: usize, control: &[usize]) {
        let (half_block_size, full_block_size, cube_count, cube_dim) =
            kernel::compute_cube_size(self.num_qubits, target);
        unsafe {
            kernel::gate_rz::launch_unchecked::<F, R>(
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
            )
            .unwrap();
        }
    }

    fn measure_p1(&mut self, target: usize) -> f64 {
        let prob_size = 1 << (self.num_qubits - 1);
        let prob = self.client.empty(prob_size * core::mem::size_of::<F>());

        let (cube_count, cube_dim) = if self.num_qubits <= 11 {
            (1, prob_size)
        } else {
            (1 << (self.num_qubits - 11), 1024)
        };

        let mask = (1 << target) - 1;

        unsafe {
            kernel::measure_p1::launch_unchecked::<F, R>(
                &self.client,
                CubeCount::new_1d(cube_count as u32),
                CubeDim::new_1d(cube_dim as u32),
                ArrayArg::from_raw_parts::<F>(&self.state_real, self.state_size, 1),
                ArrayArg::from_raw_parts::<F>(&self.state_imag, self.state_size, 1),
                ArrayArg::from_raw_parts::<F>(&prob, prob_size, 1),
                ScalarArg::new(target as u32),
                ScalarArg::new(mask),
            )
            .unwrap();
        }

        let prob = self.client.read_one(prob);
        let prob = F::from_bytes(&prob);

        prob.par_iter().copied().sum::<F>().to_f64().unwrap()
    }

    fn measure_collapse(&mut self, target: usize, result: bool, p: f64) {
        let (cube_count, cube_dim) = if self.num_qubits <= 10 {
            (1, self.state_size)
        } else {
            (1 << (self.num_qubits - 10), 1024)
        };

        let mask = 1 << target;

        unsafe {
            kernel::measure_collapse::launch_unchecked::<F, R>(
                &self.client,
                CubeCount::new_1d(cube_count as u32),
                CubeDim::new_1d(cube_dim as u32),
                ArrayArg::from_raw_parts::<F>(&self.state_real, self.state_size, 1),
                ArrayArg::from_raw_parts::<F>(&self.state_imag, self.state_size, 1),
                ScalarArg::new(mask),
                ScalarArg::new(if result { mask } else { 0 }),
                ScalarArg::new(F::from(p).unwrap()),
            )
            .unwrap();
        }
    }

    fn dump(&mut self, qubits: &[usize]) -> DumpData {
        let state_real = self.client.read_one(self.state_real.clone());
        let state_real = F::from_bytes(&state_real);
        let state_imag = self.client.read_one(self.state_imag.clone());
        let state_imag = F::from_bytes(&state_imag);

        let (basis_states, amplitudes_real, amplitudes_imag): (Vec<_>, Vec<_>, Vec<_>) = state_real
            .iter()
            .zip(state_imag)
            .enumerate()
            .filter(|(_state, (r, i))| {
                num_traits::real::Real::sqrt(**r * **r + **i * **i)
                    > F::from(F::small_epsilon()).unwrap()
            })
            .map(|(state, (r, i))| {
                let state = qubits
                    .iter()
                    .rev()
                    .enumerate()
                    .map(|(index, qubit)| usize::from(is_one_at(state, *qubit)) << index)
                    .reduce(|a, b| a | b)
                    .unwrap_or(0);

                (
                    Vec::from([state as u64]),
                    r.to_f64().unwrap(),
                    i.to_f64().unwrap(),
                )
            })
            .multiunzip();

        DumpData {
            basis_states,
            amplitudes_real,
            amplitudes_imag,
        }
    }

    fn exp_value(&mut self, hamiltonian: &Hamiltonian<usize>) -> f64 {
        let (cube_count, cube_dim) = if self.num_qubits <= 10 {
            (1, self.state_size)
        } else {
            (1 << (self.num_qubits - 10), 1024)
        };

        hamiltonian
            .products
            .iter()
            .map(|pauli_terms| {
                pauli_terms.iter().for_each(|term| match term.pauli {
                    Pauli::PauliX => self.hadamard(term.qubit, &[]),
                    Pauli::PauliY => {
                        self.phase(-std::f64::consts::FRAC_PI_2, term.qubit, &[]);
                        self.hadamard(term.qubit, &[]);
                    }
                    Pauli::PauliZ => {}
                });

                let prob = self
                    .client
                    .empty(self.state_size * core::mem::size_of::<F>());

                let mut target_mask = 0;
                for q in pauli_terms.iter().map(|term| term.qubit) {
                    target_mask |= 1 << q;
                }

                unsafe {
                    kernel::exp_value::launch_unchecked::<F, R>(
                        &self.client,
                        CubeCount::new_1d(cube_count as u32),
                        CubeDim::new_1d(cube_dim as u32),
                        ArrayArg::from_raw_parts::<F>(&self.state_real, self.state_size, 1),
                        ArrayArg::from_raw_parts::<F>(&self.state_imag, self.state_size, 1),
                        ArrayArg::from_raw_parts::<F>(&prob, self.state_size, 1),
                        ScalarArg::new(target_mask),
                    )
                    .unwrap();
                }

                let prob = self.client.read_one(prob);
                let prob = F::from_bytes(&prob);

                let result: F = prob.par_iter().copied().sum();

                pauli_terms.iter().for_each(|term| match term.pauli {
                    Pauli::PauliX => self.hadamard(term.qubit, &[]),
                    Pauli::PauliY => {
                        self.hadamard(term.qubit, &[]);
                        self.phase(std::f64::consts::FRAC_PI_2, term.qubit, &[]);
                    }
                    Pauli::PauliZ => {}
                });

                result.to_f64().unwrap()
            })
            .zip(&hamiltonian.coefficients)
            .map(|(result, coefficient)| result * *coefficient)
            .sum()
    }

    fn clear(&mut self) {
        let state_size = 1usize << self.num_qubits;
        let mut state = vec![F::zero(); state_size];
        self.state_imag = self.client.create_from_slice(F::as_bytes(&state));
        state[0] = F::one();
        self.state_real = self.client.create_from_slice(F::as_bytes(&state));
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
        Capability::Advanced
    }

    fn feature_exp_value() -> Capability {
        Capability::Advanced
    }

    fn feature_dump() -> Capability {
        Capability::Advanced
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

impl<R: Runtime, F: Float + CubeElement + FloatOps>
    BlockSimulator<Self, (ComputeClient<R>, Handle, Handle)> for DenseGPU<R, F>
{
    fn new_blocks(
        num_local_qubits: usize,
        num_global_qubits: usize,
    ) -> crate::error::Result<(Vec<Self>, (ComputeClient<R>, Handle, Handle))> {
        let num_qubits = num_local_qubits + num_global_qubits;

        let device = Default::default();
        let client = R::client(&device);

        let state_size = 1usize << num_qubits;
        let mut state = vec![F::zero(); state_size];
        let state_imag = client.create_from_slice(F::as_bytes(&state));
        state[0] = F::one();
        let state_real = client.create_from_slice(F::as_bytes(&state));

        let mut simulators = Vec::new();

        let sub_state_size = 1 << num_local_qubits;
        for i in 0..(1 << num_global_qubits) {
            let offset = (i * sub_state_size * core::mem::size_of::<F>()) as u64;
            let sub_state_real = state_real.clone().offset_start(offset);
            let sub_state_imag = state_imag.clone().offset_start(offset);

            simulators.push(Self {
                state_real: sub_state_real,
                state_imag: sub_state_imag,
                state_size: sub_state_size,
                num_qubits: num_local_qubits,
                client: client.clone(),
                _f: PhantomData,
            });
        }

        Ok((simulators, (client, state_real, state_imag)))
    }

    fn swap(
        global_data: &mut (ComputeClient<R>, Handle, Handle),
        simulators: &mut [Self],
        num_global_qubits: usize,
        num_local_qubits: usize,
        global_qubit: usize,
        local_qubit: usize,
    ) {
        #[cfg(debug_assertions)]
        {
            println!("\x1b[1;95mBlock Data SWAP\x1b[0m global_qubit={global_qubit}, local_qubit={local_qubit}");
        }
        let num_qubits = num_local_qubits + num_global_qubits;
        let state_size = 1usize << num_qubits;

        let (client, state_real, state_imag) = global_data;

        let new_state_real = client.empty(state_size * core::mem::size_of::<F>());
        let new_state_imag = client.empty(state_size * core::mem::size_of::<F>());

        let (cube_count, cube_dim) = if num_qubits <= 10 {
            (1, state_size)
        } else {
            (1 << (num_qubits - 10), 1024)
        };

        let (cube_count_2, cube_count) = if cube_count <= (1 << 24) {
            (1, cube_count)
        } else {
            (cube_count >> 24, 1 << 24)
        };

        for i in 0..cube_count_2 {
            unsafe {
                kernel::copy_swap::launch_unchecked::<F, R>(
                    client,
                    CubeCount::new_1d(cube_count),
                    CubeDim::new_1d(cube_dim as u32),
                    ArrayArg::from_raw_parts::<F>(state_real, state_size, 1),
                    ArrayArg::from_raw_parts::<F>(state_imag, state_size, 1),
                    ArrayArg::from_raw_parts::<F>(&new_state_real, state_size, 1),
                    ArrayArg::from_raw_parts::<F>(&new_state_imag, state_size, 1),
                    ScalarArg::new(i * cube_count),
                    ScalarArg::new(1 << (global_qubit + num_local_qubits)),
                    ScalarArg::new(1 << (local_qubit)),
                )
                .unwrap();
            }
        }

        let sub_state_size = 1 << num_local_qubits;

        for i in 0..(1 << num_global_qubits) {
            let offset = (i * sub_state_size * core::mem::size_of::<F>()) as u64;
            let sub_state_real = new_state_real.clone().offset_start(offset);
            let sub_state_imag = new_state_imag.clone().offset_start(offset);

            simulators[i].state_real = sub_state_real;
            simulators[i].state_imag = sub_state_imag;
        }
    }
}
