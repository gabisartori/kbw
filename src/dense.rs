// SPDX-FileCopyrightText: 2020 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
// SPDX-FileCopyrightText: 2020 Rafael de Santiago <r.santiago@ufsc.br>
//
// SPDX-License-Identifier: Apache-2.0

//! Double-buffered dense state-vector simulator.
//!
//! # Algorithm
//!
//! The quantum state is stored as a complex-valued vector of length `2^n`.
//! Gate application is parallelised with [Rayon](https://docs.rs/rayon) by
//! computing each output amplitude from the previous ("current") buffer and
//! writing into the alternate ("next") buffer.  After every gate the two
//! buffers swap roles via a boolean flag so that no allocation or copying is
//! needed.
//!
//! # Limitations
//!
//! * Maximum **32 qubits** (memory: `2 × 2^32 × 8 B ≈ 64 GiB` for `f32`).
//! * All operations run on the CPU; for GPU-accelerated dense simulation see
//!   [`crate::dense_gpu`].

use crate::quantum_execution::QuantumExecution;
use crate::{
    bitwise::{bit_flip, get_ctrl_mask, is_one_at},
    FloatOps,
};
use itertools::Itertools;
use ket::error::KetError;
use ket::execution::DumpData;
use num::complex::Complex;
use num::{One, Zero};
use rayon::prelude::*;

/// Double-buffered dense state-vector simulator generic over float type `F`.
///
/// Use the pre-defined type alias [`Dense`](crate::Dense) (`f32`) for
/// typical workloads, or instantiate `Dense<f64>` when higher precision is
/// required.
///
/// Gates are applied in parallel across all `2^n` amplitudes using Rayon.
pub struct Dense<F: FloatOps> {
    state_0: Vec<Complex<F>>,
    state_1: Vec<Complex<F>>,
    /// `true` → current state lives in `state_0`; `false` → in `state_1`.
    state: bool,
    num_states: usize,
}

impl<F: FloatOps> Dense<F> {
    fn get_states(&mut self) -> (&mut [Complex<F>], &mut [Complex<F>]) {
        self.state = !self.state;
        if self.state {
            (&mut self.state_1, &mut self.state_0)
        } else {
            (&mut self.state_0, &mut self.state_1)
        }
    }

    fn get_current_state(&self) -> &[Complex<F>] {
        if self.state {
            &self.state_0
        } else {
            &self.state_1
        }
    }
}

impl<F: FloatOps> QuantumExecution for Dense<F> {
    fn new(num_qubits: usize) -> Result<Self, KetError> {
        if num_qubits > 32 {
            return Err(KetError::ExecutionFailed);
        }

        let num_states = 1 << num_qubits;
        let mut state_0 = Vec::new();
        let mut state_1 = Vec::new();
        state_0.resize(num_states, Complex::<F>::zero());
        state_1.resize(num_states, Complex::<F>::zero());

        state_0[0] = Complex::<F>::one();

        Ok(Self {
            state: true,
            state_0,
            state_1,
            num_states,
        })
    }

    fn pauli_x(&mut self, target: usize, control: &[usize]) {
        let (current_state, next_state) = self.get_states();

        let ctrl_mask: usize = get_ctrl_mask(control);

        next_state
            .par_iter_mut()
            .enumerate()
            .for_each(|(state, amp)| {
                *amp = current_state[if (state & ctrl_mask) == ctrl_mask {
                    bit_flip(state, target)
                } else {
                    state
                }];
            });
    }

    fn pauli_y(&mut self, target: usize, control: &[usize]) {
        let (current_state, next_state) = self.get_states();

        let ctrl_mask: usize = get_ctrl_mask(control);

        next_state
            .par_iter_mut()
            .enumerate()
            .for_each(|(state, amp)| {
                if (state & ctrl_mask) == ctrl_mask {
                    *amp = current_state[bit_flip(state, target)]
                        * if is_one_at(state, target) {
                            Complex::<F>::i()
                        } else {
                            -Complex::<F>::i()
                        };
                } else {
                    *amp = current_state[state];
                }
            });
    }

    fn pauli_z(&mut self, target: usize, control: &[usize]) {
        let (current_state, next_state) = self.get_states();

        let ctrl_mask: usize = get_ctrl_mask(control);

        next_state
            .par_iter_mut()
            .enumerate()
            .for_each(|(state, amp)| {
                if (state & ctrl_mask) == ctrl_mask && is_one_at(state, target) {
                    *amp = -current_state[state];
                } else {
                    *amp = current_state[state];
                }
            });
    }

    fn hadamard(&mut self, target: usize, control: &[usize]) {
        let (current_state, next_state) = self.get_states();

        let ctrl_mask: usize = get_ctrl_mask(control);

        next_state
            .par_iter_mut()
            .enumerate()
            .for_each(|(state, amp)| {
                if (state & ctrl_mask) == ctrl_mask {
                    *amp = current_state[bit_flip(state, target)] * F::FRAC_1_SQRT_2();
                } else {
                    *amp = Complex::<F>::zero();
                }
            });

        current_state
            .par_iter_mut()
            .enumerate()
            .for_each(|(state, amp)| {
                if (state & ctrl_mask) == ctrl_mask {
                    *amp *= if is_one_at(state, target) {
                        -F::FRAC_1_SQRT_2()
                    } else {
                        F::FRAC_1_SQRT_2()
                    };
                }
            });

        next_state
            .par_iter_mut()
            .zip(current_state.par_iter())
            .for_each(|(next_amp, current_amp)| {
                *next_amp += *current_amp;
            });
    }

    fn phase(&mut self, lambda: f64, target: usize, control: &[usize]) {
        let (current_state, next_state) = self.get_states();

        let phase = Complex::<F>::exp(Complex::<F>::i() * F::from(lambda).unwrap());

        let ctrl_mask: usize = get_ctrl_mask(control);

        next_state
            .par_iter_mut()
            .enumerate()
            .for_each(|(state, amp)| {
                if (state & ctrl_mask) == ctrl_mask && is_one_at(state, target) {
                    *amp = current_state[state] * phase;
                } else {
                    *amp = current_state[state];
                }
            });
    }

    fn rx(&mut self, theta: f64, target: usize, control: &[usize]) {
        let (current_state, next_state) = self.get_states();

        let cons_theta_2 = Complex::<F>::from(F::cos(F::from(theta / 2.0).unwrap()));
        let sin_theta_2 = -Complex::<F>::i() * F::sin(F::from(theta / 2.0).unwrap());

        let ctrl_mask: usize = get_ctrl_mask(control);

        next_state
            .par_iter_mut()
            .enumerate()
            .for_each(|(state, amp)| {
                if (state & ctrl_mask) == ctrl_mask {
                    *amp = current_state[bit_flip(state, target)] * sin_theta_2;
                } else {
                    *amp = Complex::<F>::zero();
                }
            });

        current_state
            .par_iter_mut()
            .enumerate()
            .for_each(|(state, amp)| {
                if (state & ctrl_mask) == ctrl_mask {
                    *amp *= cons_theta_2;
                }
            });

        next_state
            .par_iter_mut()
            .zip(current_state.par_iter())
            .for_each(|(next_amp, current_amp)| {
                *next_amp += *current_amp;
            });
    }

    fn ry(&mut self, theta: f64, target: usize, control: &[usize]) {
        let (current_state, next_state) = self.get_states();

        let cons_theta_2 = Complex::<F>::from(F::cos(F::from(theta / 2.0).unwrap()));
        let p_sin_theta_2 = Complex::<F>::from(F::sin(F::from(theta / 2.0).unwrap()));
        let m_sin_theta_2 = -p_sin_theta_2;

        let ctrl_mask: usize = get_ctrl_mask(control);

        next_state
            .par_iter_mut()
            .enumerate()
            .for_each(|(state, amp)| {
                if (state & ctrl_mask) == ctrl_mask {
                    *amp = current_state[bit_flip(state, target)]
                        * if is_one_at(state, target) {
                            p_sin_theta_2
                        } else {
                            m_sin_theta_2
                        };
                } else {
                    *amp = Complex::<F>::zero();
                }
            });

        current_state
            .par_iter_mut()
            .enumerate()
            .for_each(|(state, amp)| {
                if (state & ctrl_mask) == ctrl_mask {
                    *amp *= cons_theta_2;
                }
            });

        next_state
            .par_iter_mut()
            .zip(current_state.par_iter())
            .for_each(|(next_amp, current_amp)| {
                *next_amp += *current_amp;
            });
    }

    fn rz(&mut self, theta: f64, target: usize, control: &[usize]) {
        let (current_state, next_state) = self.get_states();

        let phase_0 = Complex::<F>::exp(Complex::<F>::i() * F::from(-theta / 2.0).unwrap());
        let phase_1 = Complex::<F>::exp(Complex::<F>::i() * F::from(theta / 2.0).unwrap());

        let ctrl_mask: usize = get_ctrl_mask(control);

        next_state
            .par_iter_mut()
            .enumerate()
            .for_each(|(state, amp)| {
                if (state & ctrl_mask) == ctrl_mask {
                    *amp = current_state[state]
                        * if is_one_at(state, target) {
                            phase_1
                        } else {
                            phase_0
                        };
                } else {
                    *amp = current_state[state];
                }
            });
    }

    fn measure_p1(&mut self, target: usize) -> f64 {
        let current_state = self.get_current_state();

        current_state
            .par_iter()
            .enumerate()
            .map(|(state, amp)| {
                if is_one_at(state, target) {
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
        let (current_state, next_state) = self.get_states();

        next_state
            .par_iter_mut()
            .enumerate()
            .for_each(|(state, amp)| {
                *amp = if is_one_at(state, target) == result {
                    current_state[state] * F::from(p).unwrap()
                } else {
                    Complex::<F>::zero()
                };
            });
    }

    fn dump(&mut self, qubits: &[usize]) -> DumpData {
        let state = self.get_current_state();
        let (basis_states, amplitudes_real, amplitudes_imag): (Vec<_>, Vec<f64>, Vec<f64>) = state
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
        self.state_0.clear();
        self.state_1.clear();
        self.state_0.resize(self.num_states, Complex::<F>::zero());
        self.state_1.resize(self.num_states, Complex::<F>::zero());
        self.state_0[0] = Complex::<F>::one();
        self.state = true;
    }
}
