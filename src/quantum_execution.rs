// SPDX-FileCopyrightText: 2020 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
// SPDX-FileCopyrightText: 2020 Rafael de Santiago <r.santiago@ufsc.br>
//
// SPDX-License-Identifier: Apache-2.0

use itertools::Itertools;
use ket::execution::*;
use ket::ir::gate::Param;
use ket::prelude::*;
use num::Integer;
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::f64::consts::FRAC_PI_2;

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
    fn dump(&mut self, qubits: &[usize]) -> DumpData;

    fn gradients(&mut self) -> Option<Vec<f64>> {
        None
    }

    fn sample<R: Rng>(&mut self, qubits: &[usize], shots: usize, rng: &mut R) -> Sample {
        let data = self.dump(qubits);
        from_prob_to_shots(from_dump_to_prob(data), shots, rng)
    }

    fn exp_value<Q: Qubit>(&mut self, hamiltonian: &Hamiltonian<Q>) -> f64 {
        hamiltonian
            .products
            .iter()
            .map(|pauli_terms| {
                pauli_terms.iter().for_each(|term| match term.pauli {
                    Pauli::PauliX => self.hadamard(term.qubit.index(), &[]),
                    Pauli::PauliY => {
                        self.phase(-FRAC_PI_2, term.qubit.index(), &[]);
                        self.hadamard(term.qubit.index(), &[]);
                    }
                    Pauli::PauliZ => {}
                });

                let dump_data = self.dump(
                    &pauli_terms
                        .iter()
                        .map(|term| term.qubit.index())
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
                    Pauli::PauliX => self.hadamard(term.qubit.index(), &[]),
                    Pauli::PauliY => {
                        self.hadamard(term.qubit.index(), &[]);
                        self.phase(FRAC_PI_2, term.qubit.index(), &[])
                    }
                    Pauli::PauliZ => {}
                });

                result
            })
            .zip(&hamiltonian.coefficients)
            .map(|(result, coefficient)| result * *coefficient)
            .sum()
    }

    fn clear(&mut self);
}

pub struct QubitManager<S: QuantumExecution> {
    simulator: S,
    rng: StdRng,
    results: ResultData,
    parameters: Vec<f64>,
}

pub trait ExecutionFeatures {
    fn feature_measure() -> FeatureStatus;
    fn feature_sample() -> FeatureStatus;
    fn feature_exp_value() -> FeatureStatus;
    fn feature_dump() -> FeatureStatus;
    fn feature_need_decomposition() -> bool;
    fn feature_allow_live() -> bool;
    fn supports_gradient() -> bool;
}

impl<S: QuantumExecution + ExecutionFeatures + 'static> QubitManager<S> {
    pub fn new(num_qubits: usize) -> Result<Self> {
        let seed = std::env::var("KBW_SEED")
            .unwrap_or_default()
            .parse::<u64>()
            .unwrap_or_else(|_| rand::random());

        Ok(QubitManager {
            simulator: S::new(num_qubits)?,
            rng: StdRng::seed_from_u64(seed),
            results: Default::default(),
            parameters: Vec::new(),
        })
    }

    pub fn configuration(
        num_qubits: usize,
        use_live: bool,
        coupling_graph: Option<Vec<(usize, usize)>>,
        gradient: GradientStatus,
    ) -> Configuration {
        let execution = if coupling_graph.is_none() && use_live && S::feature_allow_live() {
            ket::execution::QuantumExecution::Live(Box::new(Self::new(num_qubits).unwrap()))
        } else {
            ket::execution::QuantumExecution::Batch(Box::new(Self::new(num_qubits).unwrap()))
        };

        Configuration {
            measure: S::feature_measure(),
            sample: S::feature_sample(),
            exp_value: S::feature_exp_value(),
            dump: S::feature_dump(),
            execution: Some(execution),
            num_qubits,
            qpu: if coupling_graph.is_some() {
                Some(QPU::new(
                    coupling_graph,
                    num_qubits,
                    U2Gates::All,
                    U4Gate::CX,
                ))
            } else if S::feature_need_decomposition() {
                Some(QPU::new(None, num_qubits, U2Gates::All, U4Gate::CX))
            } else {
                None
            },
            gradient,
        }
    }
}

impl<S: QuantumExecution> QubitManager<S> {
    fn get_param(&self, param: Param) -> f64 {
        match param {
            Param::Value(value) => value,
            Param::Ref {
                index, multiplier, ..
            } => self.parameters[index] * multiplier,
        }
    }

    fn gate<Q: Qubit>(&mut self, gate: QuantumGate, target: Q, control: &[Q]) {
        let target = target.index();
        let control = &control.iter().map(|x| x.index()).collect_vec();

        match gate {
            QuantumGate::RotationX(theta) => {
                self.simulator.rx(self.get_param(theta), target, control)
            }
            QuantumGate::RotationY(theta) => {
                self.simulator.ry(self.get_param(theta), target, control)
            }
            QuantumGate::RotationZ(theta) => {
                self.simulator.rz(self.get_param(theta), target, control)
            }
            QuantumGate::Phase(lambda) => {
                self.simulator
                    .phase(self.get_param(lambda), target, control)
            }
            QuantumGate::Hadamard => self.simulator.hadamard(target, control),
            QuantumGate::PauliX => self.simulator.pauli_x(target, control),
            QuantumGate::PauliY => self.simulator.pauli_y(target, control),
            QuantumGate::PauliZ => self.simulator.pauli_z(target, control),
        }
    }

    fn measure<Q: Qubit>(&mut self, qubits: &[Q]) -> u64 {
        let qubits = qubits.iter().map(|x| x.index()).collect_vec();

        let result = qubits
            .iter()
            .rev()
            .enumerate()
            .map(|(index, qubit)| (self.simulator.measure(*qubit, &mut self.rng) as u64) << index)
            .reduce(|a, b| a | b)
            .unwrap_or(0);

        result
    }

    fn exp_value<Q: Qubit>(&mut self, hamiltonian: &Hamiltonian<Q>) -> f64 {
        self.simulator.exp_value(hamiltonian)
    }

    fn sample<Q: Qubit>(&mut self, qubits: &[Q], shots: usize) -> Sample {
        let qubits = qubits.iter().map(|x| x.index()).collect_vec();
        self.simulator.sample(&qubits, shots, &mut self.rng)
    }

    fn dump<Q: Qubit>(&mut self, qubits: &[Q]) -> DumpData {
        let qubits = qubits.iter().map(|x| x.index()).collect_vec();
        self.simulator.dump(&qubits)
    }

    fn free_aux(&mut self, _aux_group: usize, _num_qubits: usize) {
        todo!()
    }
}

impl<S: QuantumExecution> LiveExecution for QubitManager<S> {
    fn gate(&mut self, gate: QuantumGate, target: LogicalQubit, control: &[LogicalQubit]) {
        self.gate(gate, target, control)
    }

    fn measure(&mut self, qubits: &[LogicalQubit]) -> u64 {
        self.measure(qubits)
    }

    fn exp_value(&mut self, hamiltonian: &Hamiltonian<LogicalQubit>) -> f64 {
        self.exp_value(hamiltonian)
    }

    fn sample(&mut self, qubits: &[LogicalQubit], shots: usize) -> Sample {
        self.sample(qubits, shots)
    }

    fn dump(&mut self, qubits: &[LogicalQubit]) -> DumpData {
        self.dump(qubits)
    }

    fn free_aux(&mut self, aux_group: usize, num_qubits: usize) {
        self.free_aux(aux_group, num_qubits)
    }
}

impl<S: QuantumExecution + ExecutionFeatures> QubitManager<S> {
    fn submit_execution<Q: Qubit + Clone>(&mut self, instructions: &[Instruction<Q>]) {
        let pb = indicatif::ProgressBar::new(instructions.len() as u64);
        pb.set_style(
            indicatif::ProgressStyle::with_template(
                "KBW: {percent_precise}% {wide_bar} Time: {elapsed}/{duration} (ETA: {eta})",
            )
            .unwrap(),
        );
        for instruction in instructions.iter().cloned() {
            match instruction {
                Instruction::Gate {
                    gate,
                    target,
                    control,
                } => self.gate(gate, target, &control),
                Instruction::Measure { qubits, index } => {
                    let result = self.measure(&qubits);
                    let measurements = &mut self.results.measurements;
                    if measurements.len() <= index {
                        measurements.resize(index + 1, 0);
                    }
                    measurements[index] = result;
                }
                Instruction::Sample {
                    qubits,
                    index,
                    shots,
                } => {
                    let result = self.sample(&qubits, shots);
                    let samples = &mut self.results.samples;
                    if samples.len() <= index {
                        samples.resize(index + 1, Default::default());
                    }
                    samples[index] = result;
                }
                Instruction::Dump { qubits, index } => {
                    let result = self.dump(&qubits);
                    let dumps = &mut self.results.dumps;
                    if dumps.len() <= index {
                        dumps.resize(index + 1, Default::default());
                    }
                    dumps[index] = result;
                }
                Instruction::ExpValue { hamiltonian, index } => {
                    let result = self.exp_value(&hamiltonian);
                    let exp_values = &mut self.results.exp_values;
                    if exp_values.len() <= index {
                        exp_values.resize(index + 1, 0.0);
                    }
                    exp_values[index] = result;
                }
                _ => {}
            }
            pb.inc(1);
        }
        pb.finish_and_clear();
        if S::supports_gradient() {
            self.results.gradients = self.simulator.gradients();
        }
    }
}

impl<S: QuantumExecution + ExecutionFeatures> BatchExecution for QubitManager<S> {
    fn submit_execution(
        &mut self,
        logical_circuit: &[Instruction<LogicalQubit>],
        physical_circuit: Option<&[Instruction<PhysicalQubit>]>,
        parameters: &[f64],
    ) {
        if !parameters.is_empty() {
            self.parameters = parameters.to_owned();
        }
        if let Some(physical_circuit) = physical_circuit {
            self.submit_execution(physical_circuit);
        } else {
            self.submit_execution(logical_circuit);
        }
    }

    fn get_results(&mut self) -> ResultData {
        self.results.clone()
    }

    fn clear(&mut self) {
        self.results = Default::default();
        self.parameters.clear();
        self.simulator.clear();
    }
}
