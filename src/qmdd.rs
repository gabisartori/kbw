// TODO: License and Document this
use crate::quantum_execution::QuantumExecution;

use ket::error::KetError;
use ket::execution::DumpData;

const COMPLEX_ZERO: Complex = Complex { re: Quad::zero(), im: Quad::zero() };
const COMPLEX_ONE: Complex = Complex { re: Quad::one(), im: Quad::zero() };
const COMPLEX_INVERSE_SQRT_2: Complex = Complex { re: std::f64::consts::FRAC_1_SQRT_2, im: Quad::zero() };
const COMPLEX_NEG_INVERSE_SQRT_2: Complex = Complex { re: -std::f64::consts::FRAC_1_SQRT_2, im: Quad::zero() };

const I_MATRIX: [Complex; 4] = [
    COMPLEX_ONE,
    COMPLEX_ZERO,
    COMPLEX_ZERO,
    COMPLEX_ONE
];

const IF_FALSE_MATRIX: [Complex; 4] = [
    COMPLEX_ONE,
    COMPLEX_ZERO,
    COMPLEX_ZERO,
    COMPLEX_ZERO
];

const IF_TRUE_MATRIX: [Complex; 4] = [
    COMPLEX_ZERO,
    COMPLEX_ZERO,
    COMPLEX_ZERO,
    COMPLEX_ONE
];

const PAULI_X_MATRIX: [Complex; 4] = [
    Complex { re: Quad::zero(), im: Quad::zero() },
    Complex { re: Quad::one(), im: Quad::zero() },
    Complex { re: Quad::one(), im: Quad::zero() },
    Complex { re: Quad::zero(), im: Quad::zero() },
];

const PAULI_Y_MATRIX: [Complex; 4] = [
    Complex { re: Quad::zero(), im: Quad::zero() },
    Complex { re: Quad::zero(), im: -Quad::one() },
    Complex { re: Quad::zero(), im: Quad::one() },
    Complex { re: Quad::zero(), im: Quad::zero() },
];

const PAULI_Z_MATRIX: [Complex; 4] = [
    Complex { re: Quad::one(), im: Quad::zero() },
    Complex { re: Quad::zero(), im: Quad::zero() },
    Complex { re: Quad::zero(), im: Quad::zero() },
    Complex { re: -Quad::one(), im: Quad::zero() },
];

const HADAMARD_MATRIX: [Complex; 4] = [
    COMPLEX_INVERSE_SQRT_2,
    COMPLEX_INVERSE_SQRT_2,
    COMPLEX_INVERSE_SQRT_2,
    COMPLEX_NEG_INVERSE_SQRT_2
];

const M0_MATRIX: [Complex; 4] = [
    Complex { re: Quad::one(), im: Quad::zero() },
    Complex { re: Quad::zero(), im: Quad::zero() },
    Complex { re: Quad::zero(), im: Quad::zero() },
    Complex { re: Quad::zero(), im: Quad::zero() }
];

const M1_MATRIX: [Complex; 4] = [
    Complex { re: Quad::zero(), im: Quad::zero() },
    Complex { re: Quad::zero(), im: Quad::zero() },
    Complex { re: Quad::zero(), im: Quad::zero() },
    Complex { re: Quad::one(), im: Quad::zero() }
];

fn phase_matrix(lambda: f64) -> [Complex; 4] {
    [
        Complex { re: Quad::one(), im: Quad::zero() },
        Complex { re: Quad::zero(), im: Quad::zero() },
        Complex { re: Quad::zero(), im: Quad::zero() },
        Complex { re: lambda.cos(), im: lambda.sin() }
    ]
}

fn rx_matrix(theta: f64) -> [Complex; 4] {
    let half_theta = theta / 2.0;
    [
        Complex { re: half_theta.cos(), im: Quad::zero(), },
        Complex { re: Quad::zero(), im: -half_theta.sin(), },
        Complex { re: Quad::zero(), im: -half_theta.sin(), },
        Complex { re: half_theta.cos(), im: Quad::zero(), }
    ]
}

fn ry_matrix(theta: f64) -> [Complex; 4] {
    let half_theta = theta / 2.0;
    [
        Complex { re: half_theta.cos(), im: Quad::zero(), },
        Complex { re: -half_theta.sin(), im: Quad::zero(), },
        Complex { re: half_theta.sin(), im: Quad::zero(), },
        Complex { re: half_theta.cos(), im: Quad::zero(), }
    ]
}

fn rz_matrix(theta: f64) -> [Complex; 4] {
    let half_theta = theta / 2.0;
    [
        Complex { re: half_theta.cos(), im: -half_theta.sin() },
        Complex { re: Quad::zero(), im: Quad::zero() },
        Complex { re: Quad::zero(), im: Quad::zero() },
        Complex { re: half_theta.cos(), im: half_theta.sin() }
    ]
}

#[derive(Copy, Clone, Debug)]
struct Rational {
    numerator: i64,
    denominator: i64,
}

impl Rational {
    fn new(numerator: i64, denominator: i64) -> Self {
        let mut rational = Self { numerator, denominator };
        rational = rational.normalize();
        if rational.denominator < 0 {
            rational.numerator = -rational.numerator;
            rational.denominator = -rational.denominator;
        }
        rational
    }

    fn gdc(a: i64, b: i64) -> i64 {
        if b == 0 {
            a.abs()
        } else {
            Self::gdc(b, a % b)
        }
    }

    fn normalize(&self) -> Self {
        let gdc = Rational::gdc(self.numerator, self.denominator);
        Rational {
            numerator: self.numerator / gdc,
            denominator: self.denominator / gdc,
        }
    }
}

impl std::ops::Add for Rational {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        let numerator = self.numerator * other.denominator + other.numerator * self.denominator;
        let denominator = self.denominator * other.denominator;
        Rational::new(numerator, denominator)
    }
}

impl std::ops::Sub for Rational {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        let numerator = self.numerator * other.denominator - other.numerator * self.denominator;
        let denominator = self.denominator * other.denominator;
        Rational::new(numerator, denominator)
    }
}

impl std::ops::Neg for Rational {
    type Output = Self;

    fn neg(self) -> Self {
        Rational::new(-self.numerator, self.denominator)
    }
}

impl std::ops::Mul for Rational {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        let numerator = self.numerator * other.numerator;
        let denominator = self.denominator * other.denominator;
        Rational::new(numerator, denominator)
    }
}

impl std::ops::Mul<i64> for Rational {
    type Output = Self;

    fn mul(self, other: i64) -> Self {
        Rational::new(self.numerator * other, self.denominator)
    }
}

impl std::ops::Mul<Rational> for i64 {
    type Output = Rational;

    fn mul(self, other: Rational) -> Rational {
        Rational::new(self * other.numerator, other.denominator)
    }
}

impl std::ops::Div for Rational {
    type Output = Self;

    fn div(self, other: Self) -> Self {
        let numerator = self.numerator * other.denominator;
        let denominator = self.denominator * other.numerator;
        Rational::new(numerator, denominator)
    }
}

#[derive(Copy, Clone, Debug)]
struct Quad {
    a: Rational,
    b: Rational,
}

impl Quad {
    fn zero() -> Self {
        Self {
            a: Rational { numerator: 0, denominator: 1 },
            b: Rational { numerator: 0, denominator: 1 },
        }
    }

    fn one() -> Self {
        Self {
            a: Rational { numerator: 1, denominator: 1 },
            b: Rational { numerator: 0, denominator: 1 },
        }
    }
}

impl std::ops::Add for Quad {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self {
            a: self.a + other.a,
            b: self.b + other.b,
        }
    }
}

impl std::ops::Mul<i64> for Quad {
    type Output = Self;

    fn mul(self, other: i64) -> Self {
        Self {
            a: self.a * Rational::new(other, 1),
            b: self.b * Rational::new(other, 1),
        }
    }
}

impl std::ops::Sub for Quad {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        Self {
            a: self.a - other.a,
            b: self.b - other.b,
        }
    }
}

impl std::ops::Neg for Quad {
    type Output = Self;

    fn neg(self) -> Self {
        Self {
            a: -self.a,
            b: -self.b,
        }
    }
}

impl std::ops::Mul for Quad {
    type Output = Self;

    // a+b2 * c+d2 = ac + ad2 + bc2 + bd4 = (ac-bd) + (ad+bc)2
    fn mul(self, other: Self) -> Self {
        let a = self.a * other.a + 2 * self.b * other.b;
        let b = self.a * other.b + self.b * other.a;
        Self { a, b }
    }
}

impl std::ops::Div for Quad {
    type Output = Self;

    // (a+b2)/(c+d2) = (a+b2)(c-d2)/(c+d2)(c-d2) = (ac-2bd + (bc-ad)2)/(c^2-2d^2)
    fn div(self, other: Self) -> Self {
        let denominator = other.a * other.a - 2 * other.b * other.b;
        let a = self.a * other.a - 2 * self.b * other.b;
        let b = self.b * other.a - self.a * other.b;
        Self { a: a / denominator, b: b / denominator }
    }
}

#[derive(Copy, Clone, Debug)]
struct Complex {
    re: Quad,
    im: Quad,
}

impl Complex {
    // fn abs(&self) -> Quad {
    //     let re_sq = self.re * self.re;
    //     let im_sq = self.im * self.im;
    //     (re_sq + im_sq).sqrt()
    // }

    fn conjugate(&self) -> Self {
        Self {
            re: self.re,
            im: -self.im,
        }
    }

    fn norm(&self) -> Quad {
        let re_sq = self.re * self.re;
        let im_sq = self.im * self.im;
        re_sq + im_sq
    }
}

impl std::ops::Add for Complex {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self {
            re: self.re + other.re,
            im: self.im + other.im,
        }
    }
}

impl std::ops::Sub for Complex {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        Self {
            re: self.re - other.re,
            im: self.im - other.im,
        }
    }
}

impl std::ops::Neg for Complex {
    type Output = Self;

    fn neg(self) -> Self {
        Self {
            re: -self.re,
            im: -self.im,
        }
    }
}

impl std::ops::Mul for Complex {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        let re = self.re * other.re - self.im * other.im;
        let im = self.re * other.im + self.im * other.re;
        Self { re, im }
    }
}

impl std::ops::Div for Complex {
    type Output = Self;

    // (a+bi)/(c+di) = (a+bi)(c-di)/(c+di)(c-di) = (ac+bd + (bc-ad)i)/(c^2+d^2)
    fn div(self, other: Self) -> Self {
        let denominator = other.re * other.re + other.im * other.im;
        let re = (self.re * other.re + self.im * other.im) / denominator;
        let im = (self.im * other.re - self.re * other.im) / denominator;
        Self { re, im }
    }
}

type NodeId = usize;
type EdgeId = usize;
type Weight = Complex;

struct Node {
    id: NodeId,
    var: usize,
    edges: [EdgeId; 4]
}

struct Edge {
    weight: Weight,
    target: NodeId,
}

#[derive(Clone, Debug)]
pub struct Qmdd {
    num_vars: usize,
    terminal_node: NodeId,
    nodes: Vec<Node>,
    root: EdgeId,
}

impl Qmdd {
    /// Initializes the structure, creating the true node and setting the root edge to point to it with a weight of 1.
    fn new(num_vars: usize) -> Self {}

    /// Rebuild the full computational-basis amplitudes for the current QMDD.
    ///
    /// Each basis state is traversed through the graph by selecting the edge
    /// corresponding to the bit value of each qubit. The chosen path multiplies
    /// the root weight by every traversed edge weight until the terminal node is
    /// reached.
    fn reconstruct_state(&self) -> Vec<Complex> {
        let dim = 1usize << self.num_vars;
        let mut state = vec![COMPLEX_ZERO; dim];

        if self.root.node == self.terminal_node {
            state[0] = self.root.weight;
            return state;
        }
        for basis in 0..dim {
            let mut value = self.root.weight;
            let mut current_edge = self.root;
            let mut current_node = &self.nodes[current_edge.node];
            loop {
                let bit = current_node.var;
                let bit_index = (basis >> bit) & 1;
                let child_index = if bit_index == 0 { 0 } else { 2 };
                let next_edge = self.get_ei(current_edge, child_index);
                // TODO: Impl MulAssign
                value = value * next_edge.weight;

                if current_edge.node == self.terminal_node {
                    break;
                }

                current_edge = next_edge;
                current_node = &self.nodes[current_edge.node];
            }

            state[basis] = value;
        }
        state
    }

    #[cfg(test)]
    fn traverse(&self) -> usize {
        let mut visited = std::collections::HashSet::new();
        let mut stack = vec![self.root.node];
        let mut count = 0;
        loop {
            if let Some(node_id) = stack.pop() {
                if !visited.insert(node_id) {
                    continue;
                }
                count += 1;
                let node = &self.nodes[node_id];
                for &edge in &node.edges {
                    if edge.node != self.terminal_node {
                        stack.push(edge.node);
                    }
                }
            } else {
                break;
            }
        }
        count
    }
}

/// A struct representing a quantum gate, including its control qubits, target qubits, and the corresponding matrix representation.
#[derive(Clone, Debug)]
struct Gate {
    controls: Vec<usize>,
    target: usize,
    matrix: [Complex; 4],
}

impl Gate {
    fn new(controls: Vec<usize>, target: usize, matrix: [Complex; 4]) -> Self {
        Self {
            controls,
            target,
            matrix,
        }
    }
}

impl QuantumExecution for Qmdd {
    fn new(num_qubits: usize) -> Result<Self, KetError> {
        Ok(Self::new(num_qubits))
    }
    // Gates
    fn pauli_x(&mut self, target: usize, control: &[usize]) {
        self.apply_gate(Gate::new(control.to_vec(), target, PAULI_X_MATRIX));
    }

    fn pauli_y(&mut self, target: usize, control: &[usize]) {
        self.apply_gate(Gate::new(control.to_vec(), target, PAULI_Y_MATRIX));
    }

    fn pauli_z(&mut self, target: usize, control: &[usize]) {
        self.apply_gate(Gate::new(control.to_vec(), target, PAULI_Z_MATRIX));
    }

    fn hadamard(&mut self, target: usize, control: &[usize]) {
        self.apply_gate(Gate::new(control.to_vec(), target, HADAMARD_MATRIX));
    }

    fn phase(&mut self, lambda: f64, target: usize, control: &[usize]) {
        self.apply_gate(Gate::new(control.to_vec(), target, phase_matrix(lambda)));
    }

    fn rx(&mut self, theta: f64, target: usize, control: &[usize]) {
        self.apply_gate(Gate::new(control.to_vec(), target, rx_matrix(theta)));
    }

    fn ry(&mut self, theta: f64, target: usize, control: &[usize]) {
        self.apply_gate(Gate::new(control.to_vec(), target, ry_matrix(theta)));
    }

    fn rz(&mut self, theta: f64, target: usize, control: &[usize]) {
        self.apply_gate(Gate::new(control.to_vec(), target, rz_matrix(theta)));
    }

    // Measurements
    fn measure_p1(&mut self, target: usize) -> f64 {}

    fn measure_collapse(&mut self, target: usize, result: bool, p: f64) {
        let matrix = if result { M1_MATRIX } else { M0_MATRIX };
        let gate = Gate::new(vec![], target, matrix);
        let e = self.build_controlled_gate_edge(&gate);
        self.root = self.mul_edge(e, self.root);
        self.root.weight = self.root.weight * Complex::new(p, 0.0);
    }

    fn dump(&mut self, qubits: &[usize]) -> DumpData {}

    fn clear(&mut self) {
        *self = Self::new(self.num_vars);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_qubit_h_test() {
        let mut sim = Qmdd::new(1);
        sim.hadamard(0, &[]);
        let state = sim.reconstruct_state();
        let expected = vec![
            COMPLEX_INVERSE_SQRT_2,
            COMPLEX_INVERSE_SQRT_2,
        ];

        for (actual, expected) in state.iter().zip(expected.iter()) {
            assert!(
                (actual.re - expected.re).abs() < 1e-12,
                "unexpected real component: {actual:?} != {expected:?}"
            );
            assert!(
                (actual.im - expected.im).abs() < 1e-12,
                "unexpected imaginary component: {actual:?} != {expected:?}"
            );
        }
    }

    #[test]
    fn double_qubit_h_test() {
        let mut sim = Qmdd::new(2);
        sim.hadamard(0, &[]);
        sim.hadamard(1, &[]);
        let state = sim.reconstruct_state();
        let expected = vec![
            Complex::new(0.5, 0.0),
            Complex::new(0.5, 0.0),
            Complex::new(0.5, 0.0),
            Complex::new(0.5, 0.0),
        ];

        for (actual, expected) in state.iter().zip(expected.iter()) {
            assert!(
                (actual.re - expected.re).abs() < 1e-12,
                "unexpected real component: {actual:?} != {expected:?}"
            );
            assert!(
                (actual.im - expected.im).abs() < 1e-12,
                "unexpected imaginary component: {actual:?} != {expected:?}"
            );
        }
    }

    #[test]
    fn bell_test() {
        let mut sim = Qmdd::new(2);
        sim.hadamard(0, &[]);
        sim.pauli_x(1, &[0]);
        let state = sim.reconstruct_state();
        let expected = vec![
            COMPLEX_INVERSE_SQRT_2,
            COMPLEX_ZERO,
            COMPLEX_ZERO,
            COMPLEX_INVERSE_SQRT_2,
        ];

        for (actual, expected) in state.iter().zip(expected.iter()) {
            assert!(
                (actual.re - expected.re).abs() < 1e-12,
                "unexpected real component: {actual:?} != {expected:?}"
            );
            assert!(
                (actual.im - expected.im).abs() < 1e-12,
                "unexpected imaginary component: {actual:?} != {expected:?}"
            );
        }
    }

    #[test]
    fn node_counter_test() {
        let size = 5;
        let mut sim = Qmdd::new(size);
        for i in 0..size {
            sim.hadamard(i, &[]);
        }
        println!("Number of total nodes: {}", sim.nodes.len());
        println!("Number of currently used nodes: {}", sim.traverse());
        // let state = sim.reconstruct_state();
    }

    #[test]
    fn measure_collapse_hadamard() {
        let mut sim = Qmdd::new(1);
        sim.hadamard(0, &[]);
        let p1 = sim.measure_p1(0);
        let p = Quad::one()/p1.sqrt();
        assert!((p1 - 0.5).abs() < 1e-12, "unexpected probability: {p1:?} != 0.5");
        sim.measure_collapse(0, true, p);
        let p1 = sim.measure_p1(0);
        assert!((p1 - Quad::one()).abs() < 1e-12, "unexpected probability: {p1:?} != Quad::one()");
    }

    #[test]
    fn measure_collapse_bell() {
        let mut sim = Qmdd::new(2);
        sim.hadamard(0, &[]);
        sim.pauli_x(1, &[0]);
        let p1 = sim.measure_p1(0);
        let p = Quad::one()/p1.sqrt();
        assert!((p1 - 0.5).abs() < 1e-12, "unexpected probability: {p1:?} != 0.5");
        sim.measure_collapse(0, true, p);
        let p1 = sim.measure_p1(1);
        assert!((p1 - Quad::one()).abs() < 1e-12, "unexpected probability: {p1:?} != Quad::one()");
    }
}
