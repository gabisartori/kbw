// SPDX-FileCopyrightText: 2020 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
// SPDX-FileCopyrightText: 2020 Rafael de Santiago <r.santiago@ufsc.br>
//
// SPDX-License-Identifier: Apache-2.0

use crate::error::Result;
use crate::quantum_execution::QuantumExecution;
use crate::FloatOps;
use crate::{bitwise::*, quantum_execution::ExecutionFeatures};
use itertools::Itertools;
use ket::execution::*;
use ket::process::DumpData;
use num::complex::Complex;
use num::One;
use rand::distr::weighted::WeightedIndex;
use rand::prelude::*;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use twox_hash::xxhash64::RandomState;

type StateMap<F> = HashMap<Vec<u64>, Complex<F>, RandomState>;

#[derive(Serialize, Deserialize)]
pub struct Sparse<F: FloatOps> {
    state_0: StateMap<F>,
    state_1: StateMap<F>,
    state: bool,
    num_states: usize,
}

impl<F: FloatOps> Sparse<F> {
    fn get_states(&mut self) -> (&mut StateMap<F>, &mut StateMap<F>) {
        self.state = !self.state;
        if self.state {
            (&mut self.state_1, &mut self.state_0)
        } else {
            (&mut self.state_0, &mut self.state_1)
        }
    }

    fn get_current_state_mut(&mut self) -> &mut StateMap<F> {
        if self.state {
            &mut self.state_0
        } else {
            &mut self.state_1
        }
    }

    fn get_current_state(&self) -> &StateMap<F> {
        if self.state {
            &self.state_0
        } else {
            &self.state_1
        }
    }
}

impl<F: FloatOps> QuantumExecution for Sparse<F> {
    fn new(num_qubits: usize) -> Result<Self> {
        let num_states = (num_qubits + 64) / 64;

        let mut state_0 = StateMap::<F>::default();

        let zero = vec![0; num_states];

        state_0.insert(zero, Complex::<F>::one());

        Ok(Sparse {
            state_0,
            state_1: StateMap::<F>::default(),
            state: true,
            num_states,
        })
    }

    fn pauli_x(&mut self, target: usize, control: &[usize]) {
        let (current_state, next_state) = self.get_states();

        current_state.drain().for_each(|(state, amp)| {
            next_state.insert(
                if ctrl_check_vec(&state, control) {
                    bit_flip_vec(state, target)
                } else {
                    state
                },
                amp,
            );
        });
    }

    fn pauli_y(&mut self, target: usize, control: &[usize]) {
        let (current_state, next_state) = self.get_states();

        current_state.drain().for_each(|(state, mut amp)| {
            if ctrl_check_vec(&state, control) {
                amp *= if is_one_at_vec(&state, target) {
                    -Complex::<F>::i()
                } else {
                    Complex::<F>::i()
                };
                next_state.insert(bit_flip_vec(state, target), amp);
            } else {
                next_state.insert(state, amp);
            }
        });
    }

    fn pauli_z(&mut self, target: usize, control: &[usize]) {
        let current_state = self.get_current_state_mut();

        current_state.par_iter_mut().for_each(|(state, amp)| {
            if ctrl_check_vec(state, control) && is_one_at_vec(state, target) {
                *amp = -*amp;
            }
        });
    }

    fn hadamard(&mut self, target: usize, control: &[usize]) {
        let (current_state, next_state) = self.get_states();

        current_state.drain().for_each(|(state, mut amp)| {
            if ctrl_check_vec(&state, control) {
                amp *= F::FRAC_1_SQRT_2();
                let state_flipped = bit_flip_vec(Vec::clone(&state), target);

                match next_state.get_mut(&state_flipped) {
                    Some(c_amp) => {
                        *c_amp += amp;
                        if c_amp.norm() < F::from(1e-15).unwrap() {
                            next_state.remove(&state_flipped);
                        }
                    }
                    None => {
                        next_state.insert(state_flipped, amp);
                    }
                }

                amp = if is_one_at_vec(&state, target) {
                    -amp
                } else {
                    amp
                };

                match next_state.get_mut(&state) {
                    Some(c_amp) => {
                        *c_amp += amp;
                        if c_amp.norm() < F::from(1e-15).unwrap() {
                            next_state.remove(&state);
                        }
                    }
                    None => {
                        next_state.insert(state, amp);
                    }
                }
            } else {
                next_state.insert(state, amp);
            }
        });
    }

    fn phase(&mut self, lambda: f64, target: usize, control: &[usize]) {
        let current_state = self.get_current_state_mut();

        let phase = Complex::<F>::exp(Complex::<F>::i() * F::from(lambda).unwrap());

        current_state.par_iter_mut().for_each(|(state, amp)| {
            if ctrl_check_vec(state, control) && is_one_at_vec(state, target) {
                *amp *= phase;
            }
        });
    }

    fn rx(&mut self, theta: f64, target: usize, control: &[usize]) {
        let (current_state, next_state) = self.get_states();

        let cons_theta_2 = Complex::<F>::from(F::cos(F::from(theta / 2.0).unwrap()));
        let sin_theta_2 = -Complex::<F>::i() * F::sin(F::from(theta / 2.0).unwrap());

        current_state.drain().for_each(|(state, amp)| {
            if ctrl_check_vec(&state, control) {
                let state_flipped = bit_flip_vec(Vec::clone(&state), target);

                match next_state.get_mut(&state_flipped) {
                    Some(c_amp) => {
                        *c_amp += amp * sin_theta_2;
                        if c_amp.norm() < F::from(1e-15).unwrap() {
                            next_state.remove(&state_flipped);
                        }
                    }
                    None => {
                        next_state.insert(state_flipped, amp * sin_theta_2);
                    }
                }

                match next_state.get_mut(&state) {
                    Some(c_amp) => {
                        *c_amp += amp * cons_theta_2;
                        if c_amp.norm() < F::from(1e-15).unwrap() {
                            next_state.remove(&state);
                        }
                    }
                    None => {
                        next_state.insert(state, amp * cons_theta_2);
                    }
                }
            } else {
                next_state.insert(state, amp);
            }
        });
    }

    fn ry(&mut self, theta: f64, target: usize, control: &[usize]) {
        let (current_state, next_state) = self.get_states();

        let cons_theta_2 = Complex::<F>::from(F::cos(F::from(theta / 2.0).unwrap()));
        let p_sin_theta_2 = Complex::<F>::from(F::sin(F::from(theta / 2.0).unwrap()));
        let m_sin_theta_2 = -p_sin_theta_2;

        current_state.drain().for_each(|(state, amp)| {
            if ctrl_check_vec(&state, control) {
                let state_flipped = bit_flip_vec(Vec::clone(&state), target);
                let flipped_amp = amp
                    * if is_one_at_vec(&state, target) {
                        m_sin_theta_2
                    } else {
                        p_sin_theta_2
                    };

                match next_state.get_mut(&state_flipped) {
                    Some(c_amp) => {
                        *c_amp += flipped_amp;
                        if c_amp.norm() < F::from(1e-15).unwrap() {
                            next_state.remove(&state_flipped);
                        }
                    }
                    None => {
                        next_state.insert(state_flipped, flipped_amp);
                    }
                }

                match next_state.get_mut(&state) {
                    Some(c_amp) => {
                        *c_amp += amp * cons_theta_2;
                        if c_amp.norm() < F::from(1e-15).unwrap() {
                            next_state.remove(&state);
                        }
                    }
                    None => {
                        next_state.insert(state, amp * cons_theta_2);
                    }
                }
            } else {
                next_state.insert(state, amp);
            }
        });
    }

    fn rz(&mut self, theta: f64, target: usize, control: &[usize]) {
        let current_state = self.get_current_state_mut();

        let phase_0 = Complex::<F>::exp(Complex::<F>::i() * F::from(-theta / 2.0).unwrap());
        let phase_1 = Complex::<F>::exp(Complex::<F>::i() * F::from(theta / 2.0).unwrap());

        current_state.par_iter_mut().for_each(|(state, amp)| {
            if ctrl_check_vec(state, control) {
                if is_one_at_vec(state, target) {
                    *amp *= phase_1;
                } else {
                    *amp *= phase_0;
                }
            }
        });
    }

    fn measure<R: Rng>(&mut self, target: usize, rng: &mut R) -> bool {
        let (current_state, next_state) = self.get_states();

        let p1 = current_state
            .iter()
            .map(|(state, amp)| {
                if is_one_at_vec(state, target) {
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

        current_state.drain().for_each(|(state, amp)| {
            if is_one_at_vec(&state, target) == result {
                next_state.insert(state, amp * p);
            }
        });

        result
    }

    fn dump(&mut self, qubits: &[usize]) -> DumpData {
        let state = self.get_current_state();

        let (basis_states, amplitudes_real, amplitudes_imag): (Vec<_>, Vec<_>, Vec<_>) = state
            .iter()
            .sorted_by_key(|x| x.0)
            .map(|(state, amp)| {
                let mut state: Vec<u64> = qubits
                    .iter()
                    .rev()
                    .chunks(64)
                    .into_iter()
                    .map(|qubits| {
                        qubits
                            .into_iter()
                            .enumerate()
                            .map(|(index, qubit)| (is_one_at_vec(state, *qubit) as usize) << index)
                            .reduce(|a, b| a | b)
                            .unwrap_or(0) as u64
                    })
                    .collect();
                state.reverse();

                (state, amp.re.to_f64().unwrap(), amp.im.to_f64().unwrap())
            })
            .multiunzip();

        DumpData {
            basis_states,
            amplitudes_real,
            amplitudes_imag,
        }
    }

    fn clear(&mut self) {
        self.state_0 = StateMap::<F>::default();
        self.state_1 = StateMap::<F>::default();

        let zero = vec![0; self.num_states];

        self.state_0.insert(zero, Complex::<F>::one());
        self.state = true;
    }

    fn save(&self) -> Vec<u8> {
        unimplemented!()
    }

    fn load(&mut self, _data: &[u8]) {
        unimplemented!()
    }
}

impl<F: FloatOps> ExecutionFeatures for Sparse<F> {
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
