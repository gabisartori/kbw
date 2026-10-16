// TODO: License and Document this

// Allow dead code
#![allow(dead_code, unused_variables, unused_mut, unused_imports)]

use std::collections::HashMap;

use crate::quantum_execution::QuantumExecution;
use crate::{
    bitwise::{bit_flip, get_ctrl_mask, is_one_at},
    FloatOps,
};

use ket::error::KetError;
use ket::execution::DumpData;
use num::complex::Complex;
use num::{One, Zero};

pub struct Qmdd<F: FloatOps> { value: F }

impl<F: FloatOps> QuantumExecution for Qmdd<F> {
    fn new(num_qubits: usize) -> Result<Self, KetError> {
        Ok(Qmdd { value: F::zero() })
    }
    // Gates
    fn pauli_x(&mut self, target: usize, control: &[usize]) { }
    fn pauli_y(&mut self, target: usize, control: &[usize]) { }
    fn pauli_z(&mut self, target: usize, control: &[usize]) { }
    fn hadamard(&mut self, target: usize, control: &[usize]) { }
    fn phase(&mut self, lambda: f64, target: usize, control: &[usize]) { }
    fn rx(&mut self, theta: f64, target: usize, control: &[usize]) { }
    fn ry(&mut self, theta: f64, target: usize, control: &[usize]) { }
    fn rz(&mut self, theta: f64, target: usize, control: &[usize]) { }

    // Measurements
    fn measure_p1(&mut self, target: usize) -> f64 { 1.0 }
    fn measure_collapse(&mut self, target: usize, result: bool, p: f64) { }
    fn dump(&mut self, qubits: &[usize]) -> DumpData {
        DumpData {
            basis_states: vec![vec![0, 1], vec![1, 0]],
            amplitudes_real: vec![0.7, 0.7,],
            amplitudes_imag: vec![0.0, 0.0,],
        }
    }
    fn clear(&mut self) { }
}