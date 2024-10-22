// SPDX-FileCopyrightText: 2020 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
// SPDX-FileCopyrightText: 2020 Rafael de Santiago <r.santiago@ufsc.br>
//
// SPDX-License-Identifier: Apache-2.0

use crate::bitwise::*;
use crate::error::Result;
use crate::quantum_execution::QuantumExecution;
use itertools::Itertools;
use ket::execution::*;
use num::complex::Complex64;
use rand::distributions::WeightedIndex;
use rand::prelude::*;
use rayon::prelude::*;
use std::{collections::HashMap, f64::consts::FRAC_1_SQRT_2};
use twox_hash::xxhash64::RandomState;

type StateMap = HashMap<Vec<u64>, Complex64, RandomState>;

pub struct Sparse {
    state_0: StateMap,
    state_1: StateMap,
    state: bool,
}

impl Sparse {
    fn get_states(&mut self) -> (&mut StateMap, &mut StateMap) {
        self.state = !self.state;
        if self.state {
            (&mut self.state_1, &mut self.state_0)
        } else {
            (&mut self.state_0, &mut self.state_1)
        }
    }

    fn get_current_state_mut(&mut self) -> &mut StateMap {
        if self.state {
            &mut self.state_0
        } else {
            &mut self.state_1
        }
    }

    fn get_current_state(&self) -> &StateMap {
        if self.state {
            &self.state_0
        } else {
            &self.state_1
        }
    }
}

impl QuantumExecution for Sparse {
    fn new(num_qubits: usize) -> Result<Self> {
        let num_states = (num_qubits + 64) / 64;

        let mut state_0 = StateMap::default();

        let zero = vec![0; num_states];

        state_0.insert(zero, Complex64::new(1.0, 0.0));

        Ok(Sparse {
            state_0,
            state_1: StateMap::default(),
            state: true,
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
                    -Complex64::i()
                } else {
                    Complex64::i()
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
                amp *= FRAC_1_SQRT_2;
                let state_flipped = bit_flip_vec(Vec::clone(&state), target);

                match next_state.get_mut(&state_flipped) {
                    Some(c_amp) => {
                        *c_amp += amp;
                        if c_amp.norm() < 1e-15 {
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
                        if c_amp.norm() < 1e-15 {
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

        let phase = Complex64::exp(lambda * Complex64::i());

        current_state.par_iter_mut().for_each(|(state, amp)| {
            if ctrl_check_vec(state, control) && is_one_at_vec(state, target) {
                *amp *= phase;
            }
        });
    }

    fn rx(&mut self, theta: f64, target: usize, control: &[usize]) {
        let (current_state, next_state) = self.get_states();

        let cons_theta_2 = Complex64::from(f64::cos(theta / 2.0));
        let sin_theta_2 = -Complex64::i() * f64::sin(theta / 2.0);

        current_state.drain().for_each(|(state, amp)| {
            if ctrl_check_vec(&state, control) {
                let state_flipped = bit_flip_vec(Vec::clone(&state), target);

                match next_state.get_mut(&state_flipped) {
                    Some(c_amp) => {
                        *c_amp += amp * sin_theta_2;
                        if c_amp.norm() < 1e-15 {
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
                        if c_amp.norm() < 1e-15 {
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

        let cons_theta_2 = Complex64::from(f64::cos(theta / 2.0));
        let p_sin_theta_2 = Complex64::from(f64::sin(theta / 2.0));
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
                        if c_amp.norm() < 1e-15 {
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
                        if c_amp.norm() < 1e-15 {
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

        let phase_0 = Complex64::exp(-theta / 2.0 * Complex64::i());
        let phase_1 = Complex64::exp(theta / 2.0 * Complex64::i());

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

        let p1: f64 = current_state
            .iter()
            .map(|(state, amp)| {
                if is_one_at_vec(state, target) {
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

                (state, amp.re, amp.im)
            })
            .multiunzip();

        DumpData {
            basis_states,
            amplitudes_real,
            amplitudes_imag,
        }
    }
}
