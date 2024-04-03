// SPDX-FileCopyrightText: 2020 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
// SPDX-FileCopyrightText: 2020 Rafael de Santiago <r.santiago@ufsc.br>
//
// SPDX-License-Identifier: Apache-2.0

use itertools::Itertools;
use ket::{Angle, LiveExecution};
use log::{debug, info, trace};
use num::Integer;
use rand::{rngs::StdRng, Rng, SeedableRng};

use crate::{
    convert::{from_dump_to_prob, from_prob_to_shots},
    error::Result,
};
pub trait QuantumExecution {
    fn new(num_qubits: usize) -> Result<Self>
    where
        Self: Sized;
    fn pauli_x(&mut self, target: usize, control: &[usize]);
    fn pauli_y(&mut self, target: usize, control: &[usize]);
    fn pauli_z(&mut self, target: usize, control: &[usize]);
    fn hadamard(&mut self, target: usize, control: &[usize]);
    fn phase(&mut self, lambda: f64, target: usize, control: &[usize]);
    fn rx(&mut self, theta: f64, target: usize, control: &[usize]);
    fn ry(&mut self, theta: f64, target: usize, control: &[usize]);
    fn rz(&mut self, theta: f64, target: usize, control: &[usize]);
    fn measure<R: Rng>(&mut self, target: usize, rng: &mut R) -> bool;
    fn dump(&mut self, qubits: &[usize]) -> ket::DumpData;
    fn debug_state(&self) -> Option<String> {
        None
    }
}

pub struct QubitManager<S: QuantumExecution> {
    simulator: S,
    qubit_stack: Vec<usize>,
    qubit_map: Vec<usize>,
    rng: StdRng,
    result: Option<ket::ir::ResultData>,
}

impl<S: QuantumExecution + 'static> QubitManager<S> {
    pub fn new(num_qubits: usize) -> Result<Self> {
        let seed = std::env::var("KBW_SEED")
            .unwrap_or_default()
            .parse::<u64>()
            .unwrap_or_else(|_| rand::random());

        info!("KBW seed={}", seed);

        Ok(QubitManager {
            simulator: S::new(num_qubits)?,
            qubit_stack: (0..num_qubits).collect_vec(),
            qubit_map: (0..num_qubits).collect_vec(),
            rng: StdRng::seed_from_u64(seed),
            result: None,
        })
    }

    pub fn configuration(num_qubits: usize, live: bool, decompose: bool) -> ket::Configuration {
        ket::Configuration {
            allow_measure: true,
            allow_sample: true,
            allow_exp_value: true,
            allow_dump: true,
            valid_after_measure: true,
            continue_after_sample: true,
            continue_after_exp_value: true,
            continue_after_dump: true,
            decompose,
            live_quantum_execution: if live {
                Some(Box::new(Self::new(num_qubits).unwrap()))
            } else {
                None
            },
            num_qubits,
            batch_execution: if live {
                None
            } else {
                Some(Box::new(Self::new(num_qubits).unwrap()))
            },
            execution_timeout: None,
        }
    }
}

impl<S: QuantumExecution> ket::LiveExecution for QubitManager<S> {
    fn alloc(&mut self, target: usize) {
        let qubit_index = self.qubit_stack.pop().unwrap();
        self.qubit_map[target] = qubit_index;

        debug!("alloc target={}->{}", target, qubit_index);
    }

    fn free(&mut self, target: usize) {
        self.qubit_stack.push(self.qubit_map[target]);
    }

    fn gate(&mut self, gate: &ket::QuantumGate, target: usize, control: &[usize]) {
        let target = self.qubit_map[target];
        let control = &control.iter().map(|x| self.qubit_map[*x]).collect_vec();

        debug!(
            "apply gate={:?}, target={}, control={:?}",
            gate, target, control
        );

        match gate {
            ket::QuantumGate::RotationX(angle) => match angle {
                Angle::Scalar(theta) => self.simulator.rx(*theta, target, control),
                Angle::PiFraction { top, bottom } => self.simulator.rx(
                    std::f64::consts::PI * *top as f64 / *bottom as f64,
                    target,
                    control,
                ),
            },
            ket::QuantumGate::RotationY(angle) => match angle {
                Angle::Scalar(theta) => self.simulator.ry(*theta, target, control),
                Angle::PiFraction { top, bottom } => self.simulator.ry(
                    std::f64::consts::PI * *top as f64 / *bottom as f64,
                    target,
                    control,
                ),
            },
            ket::QuantumGate::RotationZ(angle) => match angle {
                Angle::Scalar(theta) => self.simulator.rz(*theta, target, control),
                Angle::PiFraction { top, bottom } => self.simulator.rz(
                    std::f64::consts::PI * *top as f64 / *bottom as f64,
                    target,
                    control,
                ),
            },
            ket::QuantumGate::Phase(angle) => match angle {
                Angle::Scalar(theta) => self.simulator.phase(*theta, target, control),
                Angle::PiFraction { top, bottom } => self.simulator.phase(
                    std::f64::consts::PI * *top as f64 / *bottom as f64,
                    target,
                    control,
                ),
            },
            ket::QuantumGate::Hadamard => self.simulator.hadamard(target, control),
            ket::QuantumGate::PauliX => self.simulator.pauli_x(target, control),
            ket::QuantumGate::PauliY => self.simulator.pauli_y(target, control),
            ket::QuantumGate::PauliZ => self.simulator.pauli_z(target, control),
        }

        trace!(
            "after gate={:?}, target={}, control={:?}\n{}",
            gate,
            target,
            control,
            self.simulator.debug_state().unwrap_or_default()
        );
    }

    fn measure(&mut self, qubits: &[usize]) -> u64 {
        let qubits = qubits.iter().map(|x| self.qubit_map[*x]).collect_vec();

        debug!("measuring qubits={:?}", qubits);

        let result = qubits
            .iter()
            .rev()
            .enumerate()
            .map(|(index, qubit)| (self.simulator.measure(*qubit, &mut self.rng) as u64) << index)
            .reduce(|a, b| a | b)
            .unwrap_or(0);

        trace!(
            "after measurement qubits={:?}\n{}",
            qubits,
            self.simulator.debug_state().unwrap_or_default()
        );

        result
    }

    fn exp_value(&mut self, hamiltonian: &ket::PauliHamiltonian) -> f64 {
        hamiltonian
            .products
            .iter()
            .map(|pauli_terms| {
                pauli_terms.iter().for_each(|term| match term.pauli {
                    ket::Pauli::PauliX => self.simulator.hadamard(self.qubit_map[term.qubit], &[]),
                    ket::Pauli::PauliY => {
                        self.simulator.phase(
                            -std::f64::consts::FRAC_PI_2,
                            self.qubit_map[term.qubit],
                            &[],
                        );
                        self.simulator.hadamard(self.qubit_map[term.qubit], &[]);
                    }
                    ket::Pauli::PauliZ => {}
                });

                let dump_data = self.simulator.dump(
                    &pauli_terms
                        .iter()
                        .map(|term| self.qubit_map[term.qubit])
                        .collect_vec(),
                );
                let probabilities = from_dump_to_prob(dump_data);

                let result: f64 = probabilities
                    .basis_states
                    .iter()
                    .zip(probabilities.probabilities.iter())
                    .map(|(state, prob)| {
                        let parity = if state
                            .iter()
                            .fold(0, |acc, bit| acc + bit.count_ones())
                            .is_even()
                        {
                            1.0
                        } else {
                            -1.0
                        };
                        *prob * parity
                    })
                    .sum();

                pauli_terms.iter().for_each(|term| match term.pauli {
                    ket::Pauli::PauliX => self.simulator.hadamard(self.qubit_map[term.qubit], &[]),
                    ket::Pauli::PauliY => {
                        self.simulator.hadamard(self.qubit_map[term.qubit], &[]);
                        self.simulator.phase(
                            std::f64::consts::FRAC_PI_2,
                            self.qubit_map[term.qubit],
                            &[],
                        )
                    }
                    ket::Pauli::PauliZ => {}
                });

                result
            })
            .zip(&hamiltonian.coefficients)
            .map(|(result, coefficient)| result * *coefficient)
            .sum()
    }

    fn sample(&mut self, qubits: &[usize], shots: u64) -> (Vec<u64>, Vec<u64>) {
        let qubits = qubits.iter().map(|x| self.qubit_map[*x]).collect_vec();

        let data = self.simulator.dump(&qubits);
        from_prob_to_shots(from_dump_to_prob(data), shots, &mut self.rng)
    }

    fn dump(&mut self, qubits: &[usize]) -> ket::DumpData {
        let qubits = qubits.iter().map(|x| self.qubit_map[*x]).collect_vec();

        self.simulator.dump(&qubits)
    }
}

impl<S: QuantumExecution> ket::BatchExecution for QubitManager<S> {
    fn submit_execution(&mut self, instructions: &[ket::Instruction]) {
        self.result = Some(ket::ir::ResultData::default());
        for instruction in instructions {
            match instruction {
                ket::Instruction::Alloc { target } => self.alloc(*target),
                ket::Instruction::Free { target } => self.free(*target),
                ket::Instruction::Gate {
                    gate,
                    target,
                    control,
                } => self.gate(gate, *target, control),
                ket::Instruction::Measure { qubits, output } => {
                    assert!(self.result.as_ref().unwrap().measurements.len() == *output);
                    let result = self.measure(qubits);
                    self.result.as_mut().unwrap().measurements.push(result);
                }
                ket::Instruction::ExpValue {
                    hamiltonian,
                    output,
                } => {
                    assert!(self.result.as_ref().unwrap().exp_values.len() == *output);
                    let result = self.exp_value(hamiltonian);
                    self.result.as_mut().unwrap().exp_values.push(result);
                }
                ket::Instruction::Sample {
                    qubits,
                    shots,
                    output,
                } => {
                    assert!(self.result.as_ref().unwrap().samples.len() == *output);
                    let result = self.sample(qubits, *shots);
                    self.result.as_mut().unwrap().samples.push(result);
                }
                ket::Instruction::Dump { qubits, output } => {
                    assert!(self.result.as_ref().unwrap().dumps.len() == *output);
                    let result = self.dump(qubits);
                    self.result.as_mut().unwrap().dumps.push(result);
                }
            }
        }
    }

    fn get_result(&mut self) -> ket::ResultData {
        let result_data = self.result.take();
        info!("result={:?}", result_data);
        result_data.unwrap()
    }

    fn get_status(&self) -> ket::ExecutionStatus {
        ket::ExecutionStatus::Completed
    }
}
