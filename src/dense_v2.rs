// SPDX-FileCopyrightText: 2024 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
//
// SPDX-License-Identifier: Apache-2.0

use crate::bitwise::*;
use crate::error::{KBWError, Result};
use crate::quantum_execution::{ExecutionFeatures, QuantumExecution};
use itertools::Itertools;
use ket::execution::*;
use log::error;
use num::{complex::Complex64, Zero};
use rand::distr::weighted::WeightedIndex;
use rand::prelude::*;
use rayon::prelude::*;
use std::f64::consts::FRAC_1_SQRT_2;

pub struct DenseV2(Vec<Complex64>);

impl DenseV2 {
    fn gate<F>(&mut self, gate_impl: F, target: usize, control: &[usize])
    where
        F: Fn((&mut Complex64, &mut Complex64)) + std::marker::Sync,
    {
        let half_block_size = 1 << target;
        let full_block_size = half_block_size << 1;

        let inner_gate = |chunk_id: usize, (upper, lower): (&mut [Complex64], &mut [Complex64])| {
            upper
                .par_iter_mut()
                .zip(lower.par_iter_mut())
                .enumerate()
                .for_each(|(index, op)| {
                    if ctrl_check(chunk_id * full_block_size + index, control) {
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

impl QuantumExecution for DenseV2 {
    fn new(num_qubits: usize) -> Result<Self>
    where
        Self: Sized,
    {
        if num_qubits > 32 {
            error!("dense implementation supports up to 32 qubits");
            return Err(KBWError::UnsupportedNumberOfQubits);
        }

        let num_states = 1 << num_qubits;
        let mut state = Vec::new();
        state.resize(num_states, Complex64::zero());
        state[0] = Complex64::new(1.0, 0.0);

        Ok(DenseV2(state))
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
                *ket0 *= -Complex64::i();
                *ket1 *= Complex64::i();
            },
            target,
            control,
        );
    }

    fn pauli_z(&mut self, target: usize, control: &[usize]) {
        self.gate(
            |(_ket0, ket1)| {
                *ket1 *= -Complex64::from(1.0);
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
                *ket0 = (tmp_ket0 + tmp_ket1) * FRAC_1_SQRT_2;
                *ket1 = (tmp_ket0 - tmp_ket1) * FRAC_1_SQRT_2;
            },
            target,
            control,
        );
    }

    fn phase(&mut self, lambda: f64, target: usize, control: &[usize]) {
        let phase = Complex64::exp(lambda * Complex64::i());

        self.gate(
            |(_ket0, ket1): (&mut Complex64, &mut Complex64)| {
                *ket1 *= phase;
            },
            target,
            control,
        );
    }

    fn rx(&mut self, theta: f64, target: usize, control: &[usize]) {
        let cons_theta_2 = Complex64::from(f64::cos(theta / 2.0));
        let sin_theta_2 = -Complex64::i() * f64::sin(theta / 2.0);

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
        let cons_theta_2 = Complex64::from(f64::cos(theta / 2.0));
        let p_sin_theta_2 = Complex64::from(f64::sin(theta / 2.0));
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
        let phase_0 = Complex64::exp(-theta / 2.0 * Complex64::i());
        let phase_1 = Complex64::exp(theta / 2.0 * Complex64::i());

        self.gate(
            |(ket0, ket1)| {
                *ket0 *= phase_0;
                *ket1 *= phase_1;
            },
            target,
            control,
        );
    }

    fn measure<R: Rng>(&mut self, target: usize, rng: &mut R) -> bool {
        let p1: f64 = self
            .0
            .par_iter()
            .enumerate()
            .map(|(state, amp)| {
                if is_one_at(state, target) {
                    amp.norm().powi(2)
                } else {
                    0.0
                }
            })
            .sum();

        let p0 = match 1.0 - p1 {
            p0 if p0 >= 0.0 => p0,
            _ => 0.0,
        };

        let result = WeightedIndex::new([p0, p1]).unwrap().sample(rng) == 1;

        let p = 1.0 / f64::sqrt(if result { p1 } else { p0 });

        self.0.par_iter_mut().enumerate().for_each(|(state, amp)| {
            *amp = if is_one_at(state, target) == result {
                *amp * p
            } else {
                Complex64::zero()
            };
        });

        result
    }

    fn dump(&mut self, qubits: &[usize]) -> DumpData {
        let (basis_states, amplitudes_real, amplitudes_imag): (Vec<_>, Vec<_>, Vec<_>) = self
            .0
            .iter()
            .enumerate()
            .filter(|(_state, amp)| amp.norm() > 1e-15)
            .map(|(state, amp)| {
                let state = qubits
                    .iter()
                    .rev()
                    .enumerate()
                    .map(|(index, qubit)| (is_one_at(state, *qubit) as usize) << index)
                    .reduce(|a, b| a | b)
                    .unwrap_or(0);

                (Vec::from([state as u64]), amp.re, amp.im)
            })
            .multiunzip();

        DumpData {
            basis_states,
            amplitudes_real,
            amplitudes_imag,
        }
    }

    fn clear(&mut self) {
        self.0.fill(Complex64::zero());
        self.0[0] = Complex64::new(1.0, 0.0);
    }
}

impl ExecutionFeatures for DenseV2 {
    fn feature_measure() -> FeatureStatus {
        FeatureStatus::ValidAfter
    }

    fn feature_sample() -> FeatureStatus {
        FeatureStatus::ValidAfter
    }

    fn feature_exp_value() -> FeatureStatus {
        FeatureStatus::ValidAfter
    }

    fn feature_dump() -> FeatureStatus {
        FeatureStatus::ValidAfter
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
