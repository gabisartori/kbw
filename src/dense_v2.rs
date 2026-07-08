// SPDX-FileCopyrightText: 2024 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
//
// SPDX-License-Identifier: Apache-2.0

//! In-place dense state-vector simulator (V2).
//!
//! # Algorithm
//!
//! Unlike the double-buffered [`crate::dense`] backend, `DenseV2` stores only
//! a **single** state vector.  Gates operate on pairs of amplitudes
//! `(|…0…⟩, |…1…⟩)` in-place by splitting each block of `2^(target+1)`
//! elements at the midpoint.  This halves the memory footprint at the cost of
//! a slightly more complex iteration pattern.
//!
//! The inner loop is parallelised with [Rayon](https://docs.rs/rayon).
//!
//! # Limitations
//!
//! * Maximum **32 qubits**.
//! * All operations run on the CPU.

use std::fmt::Display;

use crate::block::BlockSimulator;
use crate::quantum_execution::QuantumExecution;
use crate::{
    bitwise::{get_ctrl_mask, is_one_at},
    FloatOps,
};
use itertools::Itertools;
use ket::error::KetError;
use ket::execution::DumpData;
use num::complex::ComplexFloat;
use num::Zero;
use num::{Complex, One};
use rayon::prelude::*;

/// In-place dense state-vector simulator generic over float type `F`.
///
/// Use the pre-defined type alias [`DenseV2`](crate::DenseV2) (`f32`) for
/// typical workloads.
///
/// The inner `Vec` stores amplitudes as `[a_0, a_1, …, a_{2^n - 1}]`.
pub struct DenseV2<F: FloatOps>(Vec<Complex<F>>);

impl<F: FloatOps> DenseV2<F> {
    fn gate<G>(&mut self, gate_impl: G, target: usize, control: &[usize])
    where
        G: Fn((&mut Complex<F>, &mut Complex<F>)) + std::marker::Sync,
    {
        let half_block_size = 1 << target;
        let full_block_size = half_block_size << 1;
        let ctrl_mask: usize = get_ctrl_mask(control);

        let inner_gate =
            |chunk_id: usize, (upper, lower): (&mut [Complex<F>], &mut [Complex<F>])| {
                upper
                    .par_iter_mut()
                    .zip(lower.par_iter_mut())
                    .enumerate()
                    .for_each(|(index, op)| {
                        if ((chunk_id * full_block_size + index) & ctrl_mask) == ctrl_mask {
                            gate_impl(op);
                        }
                    });
            };

        self.0
            .par_chunks_mut(full_block_size)
            .enumerate()
            .for_each(|(chunk_id, state)| {
                inner_gate(chunk_id, state.split_at_mut(half_block_size));
            });
    }
}

impl<F: FloatOps> QuantumExecution for DenseV2<F> {
    fn new(num_qubits: usize) -> Result<Self, KetError>
    where
        Self: Sized,
    {
        if num_qubits > 32 {
            return Err(KetError::ExecutionFailed);
        }

        let num_states = 1 << num_qubits;
        let mut state = vec![Complex::<F>::zero(); num_states];
        state[0] = Complex::<F>::one();

        Ok(Self(state))
    }

    fn pauli_x(&mut self, target: usize, control: &[usize]) {
        self.gate(
            |(ket0, ket1)| {
                std::mem::swap(ket0, ket1);
            },
            target,
            control,
        );
    }

    fn pauli_y(&mut self, target: usize, control: &[usize]) {
        self.gate(
            |(ket0, ket1)| {
                std::mem::swap(ket0, ket1);
                *ket0 *= -Complex::<F>::i();
                *ket1 *= Complex::<F>::i();
            },
            target,
            control,
        );
    }

    fn pauli_z(&mut self, target: usize, control: &[usize]) {
        self.gate(
            |(_ket0, ket1)| {
                *ket1 *= -Complex::<F>::one();
            },
            target,
            control,
        );
    }

    fn hadamard(&mut self, target: usize, control: &[usize]) {
        self.gate(
            |(ket0, ket1)| {
                let tmp_ket0 = *ket0;
                let tmp_ket1 = *ket1;
                *ket0 = (tmp_ket0 + tmp_ket1) * F::FRAC_1_SQRT_2();
                *ket1 = (tmp_ket0 - tmp_ket1) * F::FRAC_1_SQRT_2();
            },
            target,
            control,
        );
    }

    fn phase(&mut self, lambda: f64, target: usize, control: &[usize]) {
        let phase = Complex::<F>::exp(Complex::<F>::i() * F::from(lambda).unwrap());

        self.gate(
            |(_ket0, ket1): (&mut Complex<F>, &mut Complex<F>)| {
                *ket1 *= phase;
            },
            target,
            control,
        );
    }

    fn rx(&mut self, theta: f64, target: usize, control: &[usize]) {
        let cons_theta_2 = Complex::<F>::from(F::cos(F::from(theta / 2.0).unwrap()));
        let sin_theta_2 = -Complex::<F>::i() * F::sin(F::from(theta / 2.0).unwrap());

        self.gate(
            |(ket0, ket1)| {
                let tmp_ket0 = *ket0;
                let tmp_ket1 = *ket1;
                *ket0 = cons_theta_2 * tmp_ket0 + sin_theta_2 * tmp_ket1;
                *ket1 = sin_theta_2 * tmp_ket0 + cons_theta_2 * tmp_ket1;
            },
            target,
            control,
        );
    }

    fn ry(&mut self, theta: f64, target: usize, control: &[usize]) {
        let cons_theta_2 = Complex::<F>::from(F::cos(F::from(theta / 2.0).unwrap()));
        let p_sin_theta_2 = Complex::<F>::from(F::sin(F::from(theta / 2.0).unwrap()));
        let m_sin_theta_2 = -p_sin_theta_2;

        self.gate(
            |(ket0, ket1)| {
                let tmp_ket0 = *ket0;
                let tmp_ket1 = *ket1;
                *ket0 = cons_theta_2 * tmp_ket0 + m_sin_theta_2 * tmp_ket1;
                *ket1 = p_sin_theta_2 * tmp_ket0 + cons_theta_2 * tmp_ket1;
            },
            target,
            control,
        );
    }

    fn rz(&mut self, theta: f64, target: usize, control: &[usize]) {
        let phase_0 = Complex::<F>::exp(Complex::<F>::i() * F::from(-theta / 2.0).unwrap());
        let phase_1 = Complex::<F>::exp(Complex::<F>::i() * F::from(theta / 2.0).unwrap());

        self.gate(
            |(ket0, ket1)| {
                *ket0 *= phase_0;
                *ket1 *= phase_1;
            },
            target,
            control,
        );
    }

    fn measure_p1(&mut self, target: usize) -> f64 {
        let state_mask = 1 << target;
        self.0
            .par_iter()
            .enumerate()
            .map(|(state, amp)| {
                if state & state_mask == state_mask {
                    amp.norm().powi(2)
                } else {
                    F::zero()
                }
            })
            .sum::<F>()
            .to_f64()
            .unwrap()
    }

    fn measure_collapse(&mut self, target: usize, result: bool, p: f64) {
        let state_mask = 1 << target;

        self.0.par_iter_mut().enumerate().for_each(|(state, amp)| {
            *amp = if (state & state_mask == state_mask) == result {
                *amp * F::from(p).unwrap()
            } else {
                Complex::<F>::zero()
            };
        });
    }

    fn dump(&mut self, qubits: &[usize]) -> DumpData {
        let (basis_states, amplitudes_real, amplitudes_imag): (Vec<_>, Vec<_>, Vec<_>) = self
            .0
            .iter()
            .enumerate()
            .filter(|(_state, amp)| amp.norm() > F::from(F::small_epsilon()).unwrap())
            .map(|(state, amp)| {
                let state = qubits
                    .iter()
                    .rev()
                    .enumerate()
                    .map(|(index, qubit)| usize::from(is_one_at(state, *qubit)) << index)
                    .reduce(|a, b| a | b)
                    .unwrap_or(0);

                (
                    Vec::from([state as u64]),
                    amp.re.to_f64().unwrap(),
                    amp.im.to_f64().unwrap(),
                )
            })
            .multiunzip();

        DumpData {
            basis_states,
            amplitudes_real,
            amplitudes_imag,
        }
    }

    fn clear(&mut self) {
        self.0.fill(Complex::<F>::zero());
        self.0[0] = Complex::<F>::one();
    }
}

impl<F: FloatOps + Display> BlockSimulator<Self, ()> for DenseV2<F> {
    fn new_blocks(
        num_local_qubits: usize,
        num_global_qubits: usize,
    ) -> Result<(Vec<Self>, ()), KetError> {
        let mut simulators = vec![Self::new(num_local_qubits)?];
        let num_states = 1 << num_local_qubits;

        for _ in 1..(1 << num_global_qubits) {
            simulators.push(Self(vec![Complex::<F>::zero(); num_states]));
        }

        Ok((simulators, ()))
    }

    fn swap(
        (): &mut (),
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
        let mut new_states =
            vec![vec![Complex::<F>::zero(); 1 << num_local_qubits]; 1 << num_global_qubits];

        new_states
            .iter_mut()
            .enumerate()
            .for_each(|(global_index, global)| {
                global
                    .iter_mut()
                    .enumerate()
                    .for_each(|(local_index, state)| {
                        let g_bit_pos = global_qubit;
                        let l_bit_pos = local_qubit;

                        let g_bit = (global_index >> g_bit_pos) & 1;
                        let l_bit = (local_index >> l_bit_pos) & 1;

                        let src_global = (global_index & !(1 << g_bit_pos)) | (l_bit << g_bit_pos);

                        let src_local = (local_index & !(1 << l_bit_pos)) | (g_bit << l_bit_pos);

                        *state = simulators[src_global].0[src_local];
                    });
            });

        for (s, new_state) in simulators.iter_mut().zip(new_states.drain(..)) {
            s.0 = new_state;
        }
    }

    fn print_global_state((): &(), simulators: &[Self]) {
        println!("==============");
        for (global, s) in simulators.iter().enumerate() {
            for (local, amp) in s.0.iter().enumerate() {
                println!("{global:>2} {local:<2}: ({:.4} {:+.4})", amp.re(), amp.im());
            }
        }
        println!("==============");
    }
}
