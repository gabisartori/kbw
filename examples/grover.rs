// SPDX-FileCopyrightText: 2024 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
//
// SPDX-License-Identifier: Apache-2.0

use std::f64::consts::FRAC_PI_4;

use kbw::DenseSimulator;
use ket::prelude::*;

fn main() -> Result<(), KetError> {
    let config = DenseSimulator::configuration(
        12,
        // 0--1--2--3
        // |  |  |  |
        // 4--5--6--7
        // |  |  |  |
        // 8--9--A--B
        Some(vec![
            (0, 4),
            (0, 1),
            (1, 2),
            (1, 5),
            (2, 3),
            (2, 6),
            (3, 7),
            (4, 8),
            (4, 5),
            (5, 9),
            (5, 6),
            (6, 10),
            (6, 7),
            (7, 11),
            (8, 9),
            (9, 10),
            (10, 11),
        ]),
    );

    let mut process = Process::new(config);

    let size = 6;

    let qubits: Vec<_> = (0..size).map(|_| process.alloc().unwrap()).collect();

    for qubit in &qubits {
        process.gate(QuantumGate::Hadamard, *qubit)?;
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

    let sample = process.measure(&qubits)?;
    process.transpile();
    process.execute()?;

    println!("Result: {:#012b}", process.get_measure(sample).unwrap());

    Ok(())
}
