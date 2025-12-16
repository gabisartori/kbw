// SPDX-FileCopyrightText: 2025 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
//
// SPDX-License-Identifier: Apache-2.0

use std::f64::consts::{FRAC_PI_2, FRAC_PI_4};

use kbw::DenseGPUSimulator;
use ket::prelude::*;

fn main() -> Result<(), KetError> {
    let size = 16;
    let (target, execution) =
        DenseGPUSimulator::configuration(size, false, None, None, None, false);

    let mut process = Process::new(target, execution);

    let qubits: Vec<_> = (0..size).map(|_| process.alloc().unwrap()).collect();

    for qubit in &qubits {
        process.gate(QuantumGate::RotationY(FRAC_PI_2.into()), *qubit)?;
    }

    let steps = ((FRAC_PI_4) * f64::sqrt((1 << size) as f64)) as i64;

    for _ in 0..steps {
        ctrl(&mut process, &qubits[1..], |process| {
            process.gate(QuantumGate::PauliZ, qubits[0])
        })?;

        around(
            &mut process,
            |process| {
                for qubit in &qubits {
                    process.gate(QuantumGate::Hadamard, *qubit)?;
                }

                for qubit in &qubits {
                    process.gate(QuantumGate::PauliX, *qubit)?;
                }
                Ok(())
            },
            |process| {
                ctrl(process, &qubits[1..], |process| {
                    process.gate(QuantumGate::PauliZ, qubits[0])
                })
            },
        )?;
    }

    let result = process.measure(&qubits)?;

    process.execute()?;

    let result = process.get_measure(result).unwrap();

    println!("Result {:#064b}", result);

    Ok(())
}
