// SPDX-FileCopyrightText: 2026 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
//
// SPDX-License-Identifier: Apache-2.0

use bimap::BiMap;
use ket::process::DumpData;

use crate::quantum_execution::{ExecutionFeatures, QuantumExecution};

pub trait BlockSimulator<S: QuantumExecution, G> {
    fn new_blocks(
        num_local_qubits: usize,
        num_global_qubits: usize,
    ) -> crate::error::Result<(Vec<S>, G)>;

    fn swap(
        global_data: &mut G,
        simulators: &mut [S],
        num_global_qubits: usize,
        num_local_qubits: usize,
        global_qubit: usize,
        local_qubit: usize,
    );

    fn print_global_state(_global_data: &G, _simulators: &[S]) {
        unimplemented!()
    }
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

pub struct Block<G, S: QuantumExecution + BlockSimulator<S, G>> {
    num_global_qubits: usize,
    num_local_qubits: usize,
    simulators: Vec<S>,
    global_data: G,
    logical_to_physical: BiMap<usize, usize>,
    qubit_last_access: Vec<u64>,
    access_counter: u64,
    gate_queue: Vec<(Gate, usize, Vec<usize>, Vec<bool>)>,
    #[cfg(feature = "swap_count")]
    swap_count: usize,
}

impl<G, S: QuantumExecution + BlockSimulator<S, G>> Block<G, S> {
    fn touch(&mut self, physical_qubit: usize) {
        if physical_qubit < self.num_local_qubits {
            self.access_counter += 1;
            self.qubit_last_access[physical_qubit] = self.access_counter;
        }
    }

    fn get_next_local_qubit(&self) -> usize {
        self.qubit_last_access
            .iter()
            .enumerate()
            .min_by_key(|(_, &timestamp)| timestamp)
            .map(|(index, _)| index)
            .unwrap_or(0) // Fallback for 0 qubits, though usually checked in new()
    }

    const fn is_local(&self, qubit: usize) -> bool {
        qubit < self.num_local_qubits
    }

    fn get_local_qubit(&mut self, logical_qubit: usize) -> usize {
        let physical_qubit = *self
            .logical_to_physical
            .get_by_left(&logical_qubit)
            .unwrap();

        if self.is_local(physical_qubit) {
            self.touch(physical_qubit);
            physical_qubit
        } else {
            self.execute_gates();
            let global_qubit = physical_qubit - self.num_local_qubits;
            let local_qubit = self.get_next_local_qubit();

            S::swap(
                &mut self.global_data,
                &mut self.simulators,
                self.num_global_qubits,
                self.num_local_qubits,
                global_qubit,
                local_qubit,
            );
            #[cfg(feature = "swap_count")]
            {
                self.swap_count += 1;
                unsafe {
                    std::env::set_var("KBW_BLOCK_SWAP_COUNT", format!("{}", self.swap_count));
                }
            }

            let old_logical_qubit = *self.logical_to_physical.get_by_right(&local_qubit).unwrap();

            self.logical_to_physical.insert(logical_qubit, local_qubit);
            self.logical_to_physical
                .insert(old_logical_qubit, physical_qubit);

            // #[cfg(debug_assertions)]
            // {
            //     println!(
            //         "\x1b[1;95mlogical <> physical {:?}\x1b[0m",
            //         self.logical_to_physical
            //     );
            // }
            self.touch(local_qubit);

            local_qubit
        }
    }

    fn get_control_local_qubit(&self, control: &[usize]) -> (Vec<usize>, Vec<bool>) {
        let mut new_control = Vec::new();
        let mut execute = vec![true; self.simulators.len()];
        for c in control {
            let c = *self.logical_to_physical.get_by_left(c).unwrap();
            if self.is_local(c) {
                new_control.push(c);
            } else {
                let c = 1usize << (c - self.num_local_qubits);
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

impl<G, S: QuantumExecution + BlockSimulator<S, G>> QuantumExecution for Block<G, S> {
    fn new(num_qubits: usize) -> crate::error::Result<Self>
    where
        Self: Sized,
    {
        if num_qubits == 1 {
            return Err(crate::error::KBWError::UnsupportedNumberOfQubits);
        }

        let block_size = std::cmp::min(
            num_qubits - 1,
            std::env::var("KBW_BLOCK_SIZE")
                .unwrap_or_default()
                .parse::<usize>()
                .unwrap_or(16),
        );

        // #[cfg(debug_assertions)]
        // {
        //     println!("\x1b[1;95mBlock Size {block_size}\x1b[0m");
        // }

        let num_local_qubits = block_size;
        let num_global_qubits = num_qubits - block_size;

        let (simulators, global_data) = S::new_blocks(num_local_qubits, num_global_qubits)?;

        let logical_to_physical = (0..num_qubits).map(|i| (i, i)).collect();
        let qubit_last_access = vec![0; num_local_qubits];

        Ok(Self {
            num_global_qubits,
            num_local_qubits,
            simulators,
            global_data,
            logical_to_physical,
            qubit_last_access,
            access_counter: 0,
            gate_queue: Vec::new(),
            #[cfg(feature = "swap_count")]
            swap_count: 0,
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
        self.execute_gates();
        let target = self.get_local_qubit(target);
        self.simulators
            .iter_mut()
            .map(|s| s.measure_p1(target))
            .sum()
    }

    fn measure_collapse(&mut self, target: usize, result: bool, p: f64) {
        let target = self.get_local_qubit(target);
        for s in &mut self.simulators {
            s.measure_collapse(target, result, p);
        }
    }

    fn dump(&mut self, logical_qubits: &[usize]) -> ket::process::DumpData {
        self.execute_gates();

        let physical_qubits: Vec<_> = logical_qubits
            .iter()
            .map(|q| *self.logical_to_physical.get_by_left(q).unwrap())
            .collect();

        let mut global_dump = DumpData::default();

        for (global_index, simulator) in self.simulators.iter_mut().enumerate() {
            let local_dump = simulator.dump(&physical_qubits);
            for (s, (r, i)) in local_dump.basis_states.iter().zip(
                local_dump
                    .amplitudes_real
                    .iter()
                    .zip(local_dump.amplitudes_imag.iter()),
            ) {
                let global_state = physical_qubits
                    .iter()
                    .rev()
                    .enumerate()
                    .filter(|(_, q)| **q >= self.num_local_qubits)
                    .map(|(index, qubit)| (index, *qubit - self.num_local_qubits))
                    .map(|(index, qubit)| {
                        let bit = (global_index >> qubit) & 1;
                        bit << index
                    })
                    .reduce(|a, b| a | b)
                    .unwrap_or(0);

                let state = s[0] as usize | global_state;
                global_dump.basis_states.push(vec![state as u64]);
                global_dump.amplitudes_real.push(*r);
                global_dump.amplitudes_imag.push(*i);
            }
        }

        #[cfg(debug_assertions)]
        {
            use itertools::Itertools;

            let (basis_states, amplitudes_real, amplitudes_imag): (Vec<_>, Vec<_>, Vec<_>) =
                global_dump
                    .basis_states
                    .drain(..)
                    .zip(global_dump.amplitudes_real.drain(..))
                    .zip(global_dump.amplitudes_imag.drain(..))
                    .map(|((basis, real), imag)| (basis, real, imag))
                    .sorted_by(|a, b| a.0.cmp(&b.0))
                    .multiunzip();

            global_dump.basis_states = basis_states;
            global_dump.amplitudes_real = amplitudes_real;
            global_dump.amplitudes_imag = amplitudes_imag;
        }

        global_dump
    }

    fn clear(&mut self) {
        *self = Self::new(self.num_global_qubits + self.num_local_qubits).unwrap();
    }

    fn save(&self) -> Vec<u8> {
        unimplemented!()
    }

    fn load(&mut self, _data: &[u8]) {
        unimplemented!()
    }
}

impl<G, S: QuantumExecution + BlockSimulator<S, G> + ExecutionFeatures> ExecutionFeatures
    for Block<G, S>
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
