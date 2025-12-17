// SPDX-FileCopyrightText: 2024 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
//
// SPDX-License-Identifier: Apache-2.0

use crate::error::{KBWError, Result};
use crate::quantum_execution::{ExecutionFeatures, QuantumExecution};
use crate::{bitwise::*, FloatOps};
use itertools::Itertools;
use ket::execution::*;
use ket::process::DumpData;
use log::error;
use num::Zero;
use num::{Complex, One};
use rand::distr::weighted::WeightedIndex;
use rand::prelude::*;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
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
        state.resize(num_states, Complex::<F>::zero());
        state[0] = Complex::<F>::one();

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

    fn measure<R: Rng>(&mut self, target: usize, rng: &mut R) -> bool {
        let state_mask = 1 << target;
        let p1 = self
            .0
            .par_iter()
            .enumerate()
            .map(|(state, amp)| {
                if state & state_mask == state_mask {
                    amp.norm().powi(2)
                } else {
                    F::zero()
                }
            })
            .sum();

        let p0 = match F::one() - p1 {
            p0 if p0 >= F::zero() => p0,
            _ => F::zero(),
        };

        let result = WeightedIndex::new([p0.to_f64().unwrap(), p1.to_f64().unwrap()])
            .unwrap()
            .sample(rng)
            == 1;

        let p = F::one() / F::sqrt(if result { p1 } else { p0 });

        self.0.par_iter_mut().enumerate().for_each(|(state, amp)| {
            *amp = if (state & state_mask == state_mask) == result {
                *amp * p
            } else {
                Complex::<F>::zero()
            };
        });

        result
    }

    fn dump(&mut self, qubits: &[usize]) -> DumpData {
        let (basis_states, amplitudes_real, amplitudes_imag): (Vec<_>, Vec<_>, Vec<_>) = self
            .0
            .iter()
            .enumerate()
            .filter(|(_state, amp)| amp.norm() > F::from(1e-15).unwrap())
            .map(|(state, amp)| {
                let state = qubits
                    .iter()
                    .rev()
                    .enumerate()
                    .map(|(index, qubit)| (is_one_at(state, *qubit) as usize) << index)
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

    fn save(&self) -> Vec<u8> {
        unimplemented!()
    }

    fn load(&mut self, _data: &[u8]) {
        unimplemented!()
    }
}

impl<F: FloatOps> ExecutionFeatures for DenseV2<F> {
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
