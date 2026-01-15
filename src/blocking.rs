// SPDX-FileCopyrightText: 2026 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
//
// SPDX-License-Identifier: Apache-2.0

use crate::quantum_execution::{ExecutionFeatures, QuantumExecution};

pub trait BlockingSimulator<S: QuantumExecution> {
    fn new_zero(num_qubits: usize) -> crate::error::Result<S>;

    fn new_one(num_qubits: usize) -> crate::error::Result<S> {
        S::new(num_qubits)
    }

    fn swap(
        simulators: &mut [S],
        num_global_qubits: usize,
        num_local_qubits: usize,
        global_qubit: usize,
        local_qubit: usize,
    );
}

enum Gate {
    X,
    Y,
    Z,
    H,
    P(f64),
    RX(f64),
    RY(f64),
    RZ(f64),
}

pub struct Blocking<S: QuantumExecution + BlockingSimulator<S>> {
    num_global_qubits: usize,
    num_local_qubits: usize,
    simulators: Vec<S>,
    logical_qubit_map: Vec<usize>,
    next_local_qubit: usize,
    gate_queue: Vec<(Gate, usize, Vec<usize>, Vec<bool>)>,
}

impl<S: QuantumExecution + BlockingSimulator<S>> Blocking<S> {
    const fn get_next_local_qubit(&mut self) -> usize {
        let qubit = self.next_local_qubit;
        self.next_local_qubit = (qubit + 1) % self.num_local_qubits;
        qubit
    }

    const fn is_local(&self, qubit: usize) -> bool {
        qubit < (1 << self.num_local_qubits)
    }

    fn get_local_qubit(&mut self, qubit: usize) -> usize {
        let mapped_qubit = self.logical_qubit_map[qubit];
        if self.is_local(mapped_qubit) {
            mapped_qubit
        } else {
            self.execute_gates();
            let global_qubit = mapped_qubit >> self.num_local_qubits;
            let local_qubit = self.get_next_local_qubit();

            S::swap(
                &mut self.simulators,
                self.num_global_qubits,
                self.num_local_qubits,
                global_qubit,
                local_qubit,
            );

            let local_qubit_index = self
                .logical_qubit_map
                .iter()
                .position(|val| *val == local_qubit)
                .unwrap();
            self.logical_qubit_map.swap(qubit, local_qubit_index);

            local_qubit
        }
    }

    fn get_control_local_qubit(&self, control: &[usize]) -> (Vec<usize>, Vec<bool>) {
        let mut new_control = Vec::new();
        let mut execute = vec![true; self.simulators.len()];
        for c in control {
            let c = self.logical_qubit_map[*c];
            if self.is_local(c) {
                new_control.push(c);
            } else {
                let c = 1usize << (c >> self.num_local_qubits);
                for (i, s) in execute.iter_mut().enumerate() {
                    *s &= (i & c) == c;
                }
            }
        }

        (new_control, execute)
    }

    fn execute_gates(&mut self) {
        let queue = std::mem::take(&mut self.gate_queue);

        for (i, s) in self.simulators.iter_mut().enumerate() {
            for (gate, target, control, execute) in &queue {
                if execute[i] {
                    match gate {
                        Gate::X => {
                            s.pauli_x(*target, control);
                        }
                        Gate::Y => {
                            s.pauli_y(*target, control);
                        }
                        Gate::Z => {
                            s.pauli_z(*target, control);
                        }
                        Gate::H => {
                            s.hadamard(*target, control);
                        }
                        Gate::P(lambda) => {
                            s.phase(*lambda, *target, control);
                        }
                        Gate::RX(theta) => s.rx(*theta, *target, control),
                        Gate::RY(theta) => s.ry(*theta, *target, control),
                        Gate::RZ(theta) => s.rz(*theta, *target, control),
                    }
                }
            }
        }
    }
}

impl<S: QuantumExecution + BlockingSimulator<S>> QuantumExecution for Blocking<S> {
    fn new(num_qubits: usize) -> crate::error::Result<Self>
    where
        Self: Sized,
    {
        if num_qubits == 1 {
            return Err(crate::error::KBWError::UnsupportedNumberOfQubits);
        }

        let block_size = std::env::var("KBW_BLOCK_SIZE")
            .unwrap_or_default()
            .parse::<usize>()
            .unwrap_or(std::cmp::min(num_qubits - 1, 16));

        let num_local_qubits = block_size;
        let num_global_qubits = num_qubits - block_size;

        let mut simulators = vec![S::new_one(num_local_qubits)?];
        for _ in 1..(1 << num_global_qubits) {
            simulators.push(S::new_zero(num_local_qubits)?)
        }

        let logical_qubit_map: Vec<_> = (0..num_qubits).collect();

        Ok(Self {
            num_global_qubits,
            num_local_qubits,
            simulators,
            logical_qubit_map,
            next_local_qubit: 0,
            gate_queue: Vec::new(),
        })
    }

    fn pauli_x(&mut self, target: usize, control: &[usize]) {
        let target = self.get_local_qubit(target);
        let (control, execute) = self.get_control_local_qubit(control);
        self.gate_queue.push((Gate::X, target, control, execute));
    }

    fn pauli_y(&mut self, target: usize, control: &[usize]) {
        let target = self.get_local_qubit(target);
        let (control, execute) = self.get_control_local_qubit(control);
        self.gate_queue.push((Gate::Y, target, control, execute));
    }

    fn pauli_z(&mut self, target: usize, control: &[usize]) {
        let target = self.get_local_qubit(target);
        let (control, execute) = self.get_control_local_qubit(control);
        self.gate_queue.push((Gate::Z, target, control, execute));
    }

    fn hadamard(&mut self, target: usize, control: &[usize]) {
        let target = self.get_local_qubit(target);
        let (control, execute) = self.get_control_local_qubit(control);
        self.gate_queue.push((Gate::H, target, control, execute));
    }

    fn phase(&mut self, lambda: f64, target: usize, control: &[usize]) {
        let target = self.get_local_qubit(target);
        let (control, execute) = self.get_control_local_qubit(control);
        self.gate_queue
            .push((Gate::P(lambda), target, control, execute));
    }

    fn rx(&mut self, theta: f64, target: usize, control: &[usize]) {
        let target = self.get_local_qubit(target);
        let (control, execute) = self.get_control_local_qubit(control);
        self.gate_queue
            .push((Gate::RX(theta), target, control, execute));
    }

    fn ry(&mut self, theta: f64, target: usize, control: &[usize]) {
        let target = self.get_local_qubit(target);
        let (control, execute) = self.get_control_local_qubit(control);
        self.gate_queue
            .push((Gate::RY(theta), target, control, execute));
    }

    fn rz(&mut self, theta: f64, target: usize, control: &[usize]) {
        let target = self.get_local_qubit(target);
        let (control, execute) = self.get_control_local_qubit(control);
        self.gate_queue
            .push((Gate::RZ(theta), target, control, execute));
    }

    fn measure_p1(&mut self, target: usize) -> f64 {
        let target = self.get_local_qubit(target);
        self.simulators
            .iter_mut()
            .map(|s| s.measure_p1(target))
            .sum()
    }

    fn measure_collapse(&mut self, target: usize, result: bool, p: f64) {
        let target = self.get_local_qubit(target);
        for s in &mut self.simulators {
            s.measure_collapse(target, result, p)
        }
    }

    fn dump(&mut self, _qubits: &[usize]) -> ket::process::DumpData {
        todo!()
    }

    fn clear(&mut self) {
        self.gate_queue.clear();
        for s in &mut self.simulators {
            s.clear();
        }
    }

    fn save(&self) -> Vec<u8> {
        unimplemented!()
    }

    fn load(&mut self, _data: &[u8]) {
        unimplemented!()
    }
}

impl<S: QuantumExecution + BlockingSimulator<S> + ExecutionFeatures> ExecutionFeatures
    for Blocking<S>
{
    fn feature_measure() -> ket::execution::Capability {
        S::feature_measure()
    }

    fn feature_sample() -> ket::execution::Capability {
        S::feature_sample()
    }

    fn feature_exp_value() -> ket::execution::Capability {
        S::feature_exp_value()
    }

    fn feature_dump() -> ket::execution::Capability {
        S::feature_dump()
    }

    fn feature_need_decomposition() -> bool {
        S::feature_need_decomposition()
    }

    fn feature_allow_live() -> bool {
        S::feature_allow_live()
    }

    fn supports_gradient() -> bool {
        false
    }
}
