use itertools::Itertools;
use ket::{Angle, LiveExecution};
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
}

pub struct QubitManager<S: QuantumExecution> {
    simulator: S,
    qubit_stack: Vec<usize>,
    qubit_map: Vec<usize>,
    rng: StdRng,
}

impl<S: QuantumExecution + 'static> QubitManager<S> {
    pub fn new(num_qubits: usize) -> Result<Self> {
        let seed = std::env::var("KBW_SEED")
            .unwrap_or_default()
            .parse::<u64>()
            .unwrap_or_else(|_| rand::random());
        Ok(QubitManager {
            simulator: S::new(num_qubits)?,
            qubit_stack: (0..num_qubits).collect_vec(),
            qubit_map: (0..num_qubits).collect_vec(),
            rng: StdRng::seed_from_u64(seed),
        })
    }

    pub fn features(num_qubits: usize) -> ket::Features {
        ket::Features {
            allow_measure: true,
            allow_sample: true,
            allow_exp_value: true,
            allow_dump: true,
            valid_after_measure: true,
            continue_after_sample: true,
            continue_after_exp_value: true,
            continue_after_dump: true,
            decompose: false,
            live_quantum_execution: Some(Box::new(Self::new(num_qubits).unwrap())),
            num_qubits,
        }
    }
}

impl<S: QuantumExecution> ket::LiveExecution for QubitManager<S> {
    fn alloc(&mut self, target: usize) {
        let qubit_index = self.qubit_stack.pop().unwrap();
        self.qubit_map[target] = qubit_index;
    }

    fn free(&mut self, target: usize) {
        self.qubit_stack.push(self.qubit_map[target]);
    }

    fn gate(&mut self, gate: &ket::QuantumGate, target: usize, control: &[usize]) {
        match gate {
            ket::QuantumGate::PauliX(angle) => match angle {
                Angle::Scalar(theta) => self.simulator.rx(*theta, target, control),
                Angle::PiFraction { top, bottom } => {
                    if *top == 1 && *bottom == 1 {
                        self.simulator.pauli_x(target, control)
                    } else {
                        self.simulator.rx(
                            std::f64::consts::PI * *top as f64 / *bottom as f64,
                            target,
                            control,
                        )
                    }
                }
            },
            ket::QuantumGate::PauliY(angle) => match angle {
                Angle::Scalar(theta) => self.simulator.ry(*theta, target, control),
                Angle::PiFraction { top, bottom } => {
                    if *top == 1 && *bottom == 1 {
                        self.simulator.pauli_y(target, control)
                    } else {
                        self.simulator.ry(
                            std::f64::consts::PI * *top as f64 / *bottom as f64,
                            target,
                            control,
                        )
                    }
                }
            },
            ket::QuantumGate::PauliZ(angle) => match angle {
                Angle::Scalar(theta) => self.simulator.rz(*theta, target, control),
                Angle::PiFraction { top, bottom } => {
                    if *top == 1 && *bottom == 1 {
                        self.simulator.pauli_z(target, control)
                    } else {
                        self.simulator.rz(
                            std::f64::consts::PI * *top as f64 / *bottom as f64,
                            target,
                            control,
                        )
                    }
                }
            },
            ket::QuantumGate::Hadamard => self.simulator.hadamard(target, control),
        }
    }

    fn measure(&mut self, qubits: &[usize]) -> i64 {
        qubits
            .iter()
            .rev()
            .map(|qubit| self.qubit_map[*qubit])
            .enumerate()
            .map(|(index, qubit)| (self.simulator.measure(qubit, &mut self.rng) as i64) << index)
            .reduce(|a, b| a | b)
            .unwrap_or(0)
    }

    fn exp_value(&mut self, hamiltonian: &ket::PauliHamiltonian) -> f64 {
        hamiltonian
            .products
            .iter()
            .map(|pauli_terms| {
                pauli_terms.iter().for_each(|term| match term.pauli {
                    ket::Pauli::PauliX => self.simulator.hadamard(term.qubit, &[]),
                    ket::Pauli::PauliY => {
                        self.simulator.hadamard(term.qubit, &[]);
                        self.simulator
                            .phase(-std::f64::consts::FRAC_PI_2, term.qubit, &[])
                    }
                    ket::Pauli::PauliZ => {}
                });

                let dump_data = self
                    .simulator
                    .dump(&pauli_terms.iter().map(|term| term.qubit).collect_vec());
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
                    ket::Pauli::PauliX => self.simulator.hadamard(term.qubit, &[]),
                    ket::Pauli::PauliY => {
                        self.simulator.hadamard(term.qubit, &[]);
                        self.simulator
                            .phase(-std::f64::consts::FRAC_PI_2, term.qubit, &[])
                    }
                    ket::Pauli::PauliZ => {}
                });

                result
            })
            .zip(&hamiltonian.coefficients)
            .map(|(result, coefficient)| result * *coefficient)
            .sum()
    }

    fn sample(&mut self, qubits: &[usize], shots: u64) -> Vec<(Vec<u64>, u64)> {
        let data = self.simulator.dump(qubits);
        from_prob_to_shots(from_dump_to_prob(data), shots, &mut self.rng)
    }

    fn dump(&mut self, qubits: &[usize]) -> ket::DumpData {
        self.simulator.dump(qubits)
    }
}

pub fn run<S: QuantumExecution + 'static>(
    quantum_code: &[ket::Instruction],
    metadata: &ket::Metadata,
) -> Result<ket::ir::ResultData> {
    let mut sim = QubitManager::<S>::new(metadata.qubit_simultaneous)?;

    let mut result = ket::ir::ResultData::default();

    for instruction in quantum_code {
        match instruction {
            ket::Instruction::Alloc { target } => sim.alloc(*target),
            ket::Instruction::Free { target } => sim.free(*target),
            ket::Instruction::Gate {
                gate,
                target,
                control,
            } => sim.gate(gate, *target, control),
            ket::Instruction::Measure { qubits, output } => {
                assert!(result.measurements.len() == *output);
                result.measurements.push(sim.measure(qubits));
            }
            ket::Instruction::ExpValue {
                hamiltonian,
                output,
            } => {
                assert!(result.exp_values.len() == *output);
                result.exp_values.push(sim.exp_value(hamiltonian));
            }
            ket::Instruction::Sample {
                qubits,
                shots,
                output,
            } => {
                assert!(result.samples.len() == *output);
                result.samples.push(sim.sample(qubits, *shots));
            }
            ket::Instruction::Dump { qubits, output } => {
                assert!(result.dumps.len() == *output);
                result.dumps.push(sim.dump(qubits));
            }
        }
    }

    Ok(result)
}
