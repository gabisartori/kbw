// SPDX-FileCopyrightText: 2024 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
//
// SPDX-License-Identifier: Apache-2.0

use kbw::{
    quantum_execution::{ExecutionFeatures, QuantumExecution, QubitManager},
    DenseGPUBlock,
};
use ket::prelude::*;

fn play<S: QuantumExecution + ExecutionFeatures + 'static>() -> Result<(), KetError> {
    let size = 28;
    unsafe {
        std::env::set_var("KBW_BLOCK_SIZE", "20");
    }

    let (target, execution) = QubitManager::<S>::configuration(size, true, None, None, None, false);
    let mut process = Process::new(target, execution);
    let qubits: Vec<_> = (0..size).map(|_| process.alloc().unwrap()).collect();

    for (i, qubit) in qubits[..].iter().enumerate() {
        process.gate(QuantumGate::PauliX, *qubit)?;

        // let m = process.measure(&qubits)?;
        // let m = process.get_measure(m).unwrap();
        // println!("{i:>2}: {m:0size$b}");
    }

    let m = process.measure(&qubits)?;
    let m = process.get_measure(m).unwrap();
    let expected = (1 << size) - 1;
    println!("{:0size$b} == {}? {}", m, expected, m == expected);
    Ok(())
}

fn main() -> Result<(), KetError> {
    play::<DenseGPUBlock>()
}
