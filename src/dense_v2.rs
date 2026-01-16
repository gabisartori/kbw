// SPDX-FileCopyrightText: 2024 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
//
// SPDX-License-Identifier: Apache-2.0

use crate::blocking::BlockingSimulator;
use crate::error::{KBWError, Result};
use crate::quantum_execution::{ExecutionFeatures, QuantumExecution};
use crate::{
    bitwise::{get_ctrl_mask, is_one_at},
    FloatOps,
};
use itertools::Itertools;
use ket::execution::Capability;
use ket::process::DumpData;
use log::error;
use num::Zero;
use num::{Complex, One};
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

impl<F: FloatOps> BlockingSimulator<Self> for DenseV2<F> {
    fn new_zero(num_qubits: usize) -> Result<Self> {
        let num_states = 1 << num_qubits;
        let state = vec![Complex::<F>::zero(); num_states];
        Ok(Self(state))
    }

    fn swap(
        simulators: &mut [Self],
        num_global_qubits: usize,
        num_local_qubits: usize,
        global_qubit: usize,
        local_qubit: usize,
    ) {
        let mut new_states =
            vec![vec![Complex::<F>::zero(); 1 << num_local_qubits]; 1 << num_global_qubits];

        new_states
            .par_iter_mut()
            .enumerate()
            .for_each(|(global_index, global)| {
                global
                    .par_iter_mut()
                    .enumerate()
                    .for_each(|(local_index, state)| {
                        *state = if ((global_index & (1 << global_qubit)) != 0)
                            != ((local_index & (1 << local_qubit)) != 0)
                        {
                            simulators[global_index ^ (1 << global_qubit)].0
                                [local_index ^ (1 << local_qubit)]
                        } else {
                            simulators[global_index].0[local_index]
                        };
                    });
            });

        for (s, new_state) in simulators.iter_mut().zip(new_states.drain(..)) {
            s.0 = new_state;
        }
    }
}
