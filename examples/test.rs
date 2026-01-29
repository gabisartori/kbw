// SPDX-FileCopyrightText: 2024 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
//
// SPDX-License-Identifier: Apache-2.0

use kbw::{
    quantum_execution::{ExecutionFeatures, QuantumExecution, QubitManager},
    DenseGPUBlock, DenseV2, DenseV2Block,
};
use ket::prelude::*;

fn play<S: QuantumExecution + ExecutionFeatures + 'static>() -> Result<(), KetError> {
    let size = 4;
    unsafe {
        std::env::set_var("KBW_BLOCK_SIZE", "3");
    }
    let (target, execution) = QubitManager::<S>::configuration(size, true, None, None, None, false);

    let mut process = Process::new(target, execution);

    let qubits: Vec<_> = (0..size).map(|_| process.alloc().unwrap()).collect();

    let d = process.dump(&qubits)?;
    print!("{:#?}", process.get_dump(d).unwrap());
    println!("------------");
    for qubit in qubits.iter().rev() {
        process.gate(QuantumGate::Hadamard, *qubit)?;
        let d = process.dump(&qubits)?;
        print!("{:#?}", process.get_dump(d).unwrap());
        println!("------------");
    }

    Ok(())
}

fn main() -> Result<(), KetError> {
    play::<DenseV2>()?;
    println!("==========");
    play::<DenseV2Block>()?;
    println!("==========");
    play::<DenseGPUBlock>()
}
