// SPDX-FileCopyrightText: 2024 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
//
// SPDX-License-Identifier: Apache-2.0

use std::f64::consts::FRAC_PI_4;

use kbw::DenseSimulator;
use ket::prelude::*;

fn main() -> Result<(), KetError> {
    //set_log_level(4);

    let config = DenseSimulator::configuration(12, false, Some(ket::ex_arch::GRID12.into()));

    let mut process = Process::new(config);

    let size = 7;

    let qubits: Vec<_> = (0..size).map(|_| process.alloc().unwrap()).collect();

    for qubit in &qubits {
        process.gate(QuantumGate::Hadamard, *qubit)?;
    }

    let mut dumps = vec![];

    let steps = ((FRAC_PI_4) * f64::sqrt((1 << size) as f64)) as i64;

    for _ in 0..steps {
        ctrl(&mut process, &qubits[1..], |process| {
            process.gate(QuantumGate::PauliZ, qubits[0])
        })?;

        //        dumps.push(process.dump(&qubits)?);

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

        //      dumps.push(process.dump(&qubits)?);
    }
    dumps.push(process.dump(&qubits)?);

    process.transpile();
    process.execute()?;

    for dump in dumps {
        if let Some(plot) = process.get_dump(dump).map(plot_dump) {
            plot.show();
        }
    }
    Ok(())
}
