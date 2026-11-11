// TODO: License and Document this
use crate::quantum_execution::QuantumExecution;

use ket::error::KetError;
use ket::execution::DumpData;

use num::complex::Complex;

const COMPLEX_ZERO: Complex<f64> = Complex { re: 0.0, im: 0.0 };
const COMPLEX_ONE: Complex<f64> = Complex { re: 1.0, im: 0.0 };
const COMPLEX_INVERSE_SQRT_2: Complex<f64> = Complex { re: std::f64::consts::FRAC_1_SQRT_2, im: 0.0 };
const COMPLEX_NEG_INVERSE_SQRT_2: Complex<f64> = Complex { re: -std::f64::consts::FRAC_1_SQRT_2, im: 0.0 };

const I_MATRIX: [Complex<f64>; 4] = [
    COMPLEX_ONE,
    COMPLEX_ZERO,
    COMPLEX_ZERO,
    COMPLEX_ONE
];

const IF_FALSE_MATRIX: [Complex<f64>; 4] = [
    COMPLEX_ONE,
    COMPLEX_ZERO,
    COMPLEX_ZERO,
    COMPLEX_ZERO
];

const IF_TRUE_MATRIX: [Complex<f64>; 4] = [
    COMPLEX_ZERO,
    COMPLEX_ZERO,
    COMPLEX_ZERO,
    COMPLEX_ONE
];

const PAULI_X_MATRIX: [Complex<f64>; 4] = [
    Complex::<f64> { re: 0.0, im: 0.0 },
    Complex::<f64> { re: 1.0, im: 0.0 },
    Complex::<f64> { re: 1.0, im: 0.0 },
    Complex::<f64> { re: 0.0, im: 0.0 },
];

const PAULI_Y_MATRIX: [Complex<f64>; 4] = [
    Complex::<f64> { re: 0.0, im: 0.0 },
    Complex::<f64> { re: 0.0, im: -1.0 },
    Complex::<f64> { re: 0.0, im: 1.0 },
    Complex::<f64> { re: 0.0, im: 0.0 },
];

const PAULI_Z_MATRIX: [Complex<f64>; 4] = [
    Complex::<f64> { re: 1.0, im: 0.0 },
    Complex::<f64> { re: 0.0, im: 0.0 },
    Complex::<f64> { re: 0.0, im: 0.0 },
    Complex::<f64> { re: -1.0, im: 0.0 },
];

const HADAMARD_MATRIX: [Complex<f64>; 4] = [
    COMPLEX_INVERSE_SQRT_2,
    COMPLEX_INVERSE_SQRT_2,
    COMPLEX_INVERSE_SQRT_2,
    COMPLEX_NEG_INVERSE_SQRT_2
];

const M0_MATRIX: [Complex<f64>; 4] = [
    Complex::<f64> { re: 1.0, im: 0.0 },
    Complex::<f64> { re: 0.0, im: 0.0 },
    Complex::<f64> { re: 0.0, im: 0.0 },
    Complex::<f64> { re: 0.0, im: 0.0 }
];

const M1_MATRIX: [Complex<f64>; 4] = [
    Complex::<f64> { re: 0.0, im: 0.0 },
    Complex::<f64> { re: 0.0, im: 0.0 },
    Complex::<f64> { re: 0.0, im: 0.0 },
    Complex::<f64> { re: 1.0, im: 0.0 }
];

fn phase_matrix(lambda: f64) -> [Complex<f64>; 4] {
    [
        Complex::<f64> { re: 1.0, im: 0.0 },
        Complex::<f64> { re: 0.0, im: 0.0 },
        Complex::<f64> { re: 0.0, im: 0.0 },
        Complex::<f64> { re: lambda.cos(), im: lambda.sin() }
    ]
}

fn rx_matrix(theta: f64) -> [Complex<f64>; 4] {
    let half_theta = theta / 2.0;
    [
        Complex::<f64> { re: half_theta.cos(), im: 0.0, },
        Complex::<f64> { re: 0.0, im: -half_theta.sin(), },
        Complex::<f64> { re: 0.0, im: -half_theta.sin(), },
        Complex::<f64> { re: half_theta.cos(), im: 0.0, }
    ]
}

fn ry_matrix(theta: f64) -> [Complex<f64>; 4] {
    let half_theta = theta / 2.0;
    [
        Complex::<f64> { re: half_theta.cos(), im: 0.0, },
        Complex::<f64> { re: -half_theta.sin(), im: 0.0, },
        Complex::<f64> { re: half_theta.sin(), im: 0.0, },
        Complex::<f64> { re: half_theta.cos(), im: 0.0, }
    ]
}

fn rz_matrix(theta: f64) -> [Complex<f64>; 4] {
    let half_theta = theta / 2.0;
    [
        Complex::<f64> { re: half_theta.cos(), im: -half_theta.sin() },
        Complex::<f64> { re: 0.0, im: 0.0 },
        Complex::<f64> { re: 0.0, im: 0.0 },
        Complex::<f64> { re: half_theta.cos(), im: half_theta.sin() }
    ]
}

type NodeId = usize;

/// An edge in the QMDD, holding a weight and its target node
#[derive(Clone, Copy, Debug, PartialEq)]
struct Edge {
    weight: Complex<f64>,
    node: NodeId,
}

/// A node in the QMDD, holding a variable and its four outgoing edges
#[derive(Clone, Debug, PartialEq)]
struct Node {
    var: usize,
    children: [NodeId; 4],
    weights: [Complex<f64>; 4],
}

#[derive(Clone, Debug)]
pub struct Qmdd {
    num_vars: usize,
    true_node: NodeId,
    nodes: Vec<Node>,
    root: Edge,
}

impl Qmdd {
    /// Initializes the structure, creating the true node and setting the root edge to point to it with a weight of 1.
    fn new(num_vars: usize) -> Self {
        let mut q = Self {
            num_vars,
            true_node: 0,
            // Terminal node points to itself since it'll never actually be followed
            nodes: vec![Node {
                var: num_vars,
                children: [0; 4],
                weights: [COMPLEX_ONE; 4]
            }],
            root: Edge { weight: COMPLEX_ONE, node: 0 },
        };

        let mut root = Edge {
            weight: COMPLEX_ONE,
            node: q.true_node,
        };

        // Build the n-qubit identity operator as a Kronecker chain.
        for var in (0..num_vars).rev() {
            let i_edge = q.local_operator_edge(var, I_MATRIX);
            root = q.kro_edge(i_edge, root);
        }
        q.root = root;

        q
    }

    /// Creates a new node in the QMDD and inserts it to the node index, or returns an existing node if it already exists.
    /// It also checks for redundancy: if all children and weights are the same, it returns the child node instead of creating a new one.
    fn make_node(&mut self, var: usize, children: [NodeId; 4], weights: [Complex<f64>; 4]) -> NodeId {
        let redundant = children
            .iter()
            .zip(weights.iter())
            .all(|(child, w)| *child == children[0] && *w == weights[0]);

            if redundant {
            return children[0];
        }

        for (node_id, node) in self.nodes.iter().enumerate() {
            if node.var == var && node.children == children && node.weights == weights {
                return node_id;
            }
        }

        self.nodes.push(Node {
            var,
            children,
            weights,
        });
        self.nodes.len() - 1
    }

    fn node_var(&self, node: NodeId) -> usize {
        self.nodes[node].var
    }

    fn child(&self, node: NodeId, idx: usize) -> NodeId {
        self.nodes[node].children[idx]
    }

    fn weight_at(&self, node: NodeId, idx: usize) -> Complex<f64> {
        self.nodes[node].weights[idx]
    }

    fn edge_is_terminal(&self, e: Edge) -> bool {
        e.node == self.true_node
    }

    fn get_ei(&self, e: Edge, idx: usize) -> Edge {
        Edge {
            weight: self.weight_at(e.node, idx),
            node: self.child(e.node, idx),
        }
    }

    /// Algorithm for normalizing weights, described in section 3.3.2
    /// With the slight difference that node redundancy is checked in make_node, so this function doesn't need to check for it.
    fn normalize_weights(&mut self, weights: &mut [Complex<f64>; 4]) -> Complex<f64> {
        let zero = COMPLEX_ZERO;
        let pivot_idx = weights.iter().position(|w| *w != zero);
        let Some(pivot_idx) = pivot_idx else {
            return zero;
        };

        let pivot = weights[pivot_idx];
        for w in weights.iter_mut() {
            if *w != zero {
                *w = *w / pivot;
            }
        }
        weights[pivot_idx] = COMPLEX_ONE;
        pivot
    }

    /// Algorithm for applying edges, described in section 3.3.3
    fn add_edge(&mut self, e0: Edge, e1: Edge) -> Edge {
        if self.edge_is_terminal(e0) {
            if e0.weight == COMPLEX_ZERO {
                return e1;
            }
            if self.edge_is_terminal(e1) {
                return Edge {
                    weight: e0.weight + e1.weight,
                    node: self.true_node,
                };
            }
        }

        if self.node_var(e0.node) > self.node_var(e1.node) {
            return self.add_edge(e1, e0);
        }

        let mut z_children = [self.true_node; 4];
        let mut z_weights = [COMPLEX_ZERO; 4];

        for i in 0..4 {
            let ei0 = self.get_ei(e0, i);
            let q0 = Edge {
                weight: e0.weight * ei0.weight,
                node: ei0.node,
            };

            let q1 = if self.node_var(e0.node) == self.node_var(e1.node) {
                let ei1 = self.get_ei(e1, i);
                Edge {
                    weight: e1.weight * ei1.weight,
                    node: ei1.node,
                }
            } else {
                e1
            };

            let zi = self.add_edge(q0, q1);
            z_children[i] = zi.node;
            z_weights[i] = zi.weight;
        }

        let pivot = self.normalize_weights(&mut z_weights);
        let new_node = self.make_node(self.node_var(e0.node), z_children, z_weights);
        Edge {
            weight: pivot,
            node: new_node,
        }
    }

    /// Algorithm for multiplying edges, described in section 3.3.4
    fn mul_edge(&mut self, e0: Edge, e1: Edge) -> Edge {
        if self.edge_is_terminal(e0) {
            if e0.weight == COMPLEX_ZERO {
                return e0;
            }
            if e0.weight == COMPLEX_ONE {
                return e1;
            }
            return Edge {
                weight: e0.weight * e1.weight,
                node: e1.node,
            };
        }

        if self.node_var(e0.node) > self.node_var(e1.node) {
            return self.mul_edge(e1, e0);
        }

        let mut z_children = [self.true_node; 4];
        let mut z_weights = [COMPLEX_ZERO; 4];

        // 2x2 block-matrix multiplication across quadrants.
        for i in (0..4).step_by(2) {
            for j in 0..2 {
                let mut zij = Edge {
                    weight: COMPLEX_ZERO,
                    node: self.true_node,
                };

                for k in 0..2 {
                    let ei0 = self.get_ei(e0, i + k);
                    let q0 = Edge {
                        weight: e0.weight * ei0.weight,
                        node: ei0.node,
                    };

                    let q1 = if self.node_var(e0.node) == self.node_var(e1.node) {
                        let ei1 = self.get_ei(e1, j + 2 * k);
                        Edge {
                            weight: e1.weight * ei1.weight,
                            node: ei1.node,
                        }
                    } else {
                        e1
                    };

                    let q0q1 = self.mul_edge(q0, q1);
                    zij = self.add_edge(zij, q0q1);
                }

                z_children[i + j] = zij.node;
                z_weights[i + j] = zij.weight;
            }
        }

        let pivot = self.normalize_weights(&mut z_weights);
        let new_node = self.make_node(self.node_var(e0.node), z_children, z_weights);
        Edge {
            weight: pivot,
            node: new_node,
        }
    }

    /// Algorithm for Kronecker product of edges, described in section 3.3.5.1
    fn kro_edge(&mut self, e0: Edge, e1: Edge) -> Edge {
        if self.edge_is_terminal(e0) {
            if e0.weight == COMPLEX_ZERO {
                return e0;
            }
            if e0.weight == COMPLEX_ONE {
                return e1;
            }
            return Edge {
                weight: e0.weight * e1.weight,
                node: e1.node,
            };
        }

        let mut z_children = [self.true_node; 4];
        let mut z_weights = [COMPLEX_ZERO; 4];

        for i in 0..4 {
            let zi = self.kro_edge(self.get_ei(e0, i), e1);
            z_children[i] = zi.node;
            z_weights[i] = zi.weight;
        }

        let mut pivot = self.normalize_weights(&mut z_weights);
        pivot = pivot * e0.weight;
        let new_node = self.make_node(self.node_var(e0.node), z_children, z_weights);
        Edge {
            weight: pivot,
            node: new_node,
        }
    }

    /// Interface for applying an arbitrary quantum gate to the current QMDD, updating the root edge accordingly
    fn apply_gate(&mut self, gate: Gate) {
        let gate_edge = self.build_controlled_gate_edge(&gate);
        self.root = self.mul_edge(gate_edge, self.root);
    }

    // TODO: Not sure what this does
    fn local_operator_edge(&mut self, var: usize, weights: [Complex<f64>; 4]) -> Edge {
        let children = [self.true_node; 4];
        Edge {
            weight: COMPLEX_ONE,
            node: self.make_node(var, children, weights),
        }
    }

    /// Resuses logic from the paper to build a controlled gate edge from a given gate, which can then be applied to the QMDD.
    fn build_controlled_gate_edge(&mut self, gate: &Gate) -> Edge {
        debug_assert!(gate.target < self.num_vars);

        let mut controls = gate.controls.clone();
        controls.sort_unstable();
        controls.dedup();
        debug_assert!(controls
            .iter()
            .all(|&c| c < self.num_vars && c != gate.target));

        let mut identity_subtree = vec![
            Edge {
                weight: COMPLEX_ONE,
                node: self.true_node,
            };
            self.num_vars + 1
        ];

        for var in (0..self.num_vars).rev() {
            let i_edge = self.local_operator_edge(var, I_MATRIX);
            identity_subtree[var] = self.kro_edge(i_edge, identity_subtree[var + 1]);
        }

        let mut active_gate = Edge {
            weight: COMPLEX_ONE,
            node: self.true_node,
        };
        let mut inactive_gate = Edge {
            weight: COMPLEX_ZERO,
            node: self.true_node,
        };

        let mut next_control = controls.len();

        for var in (0..self.num_vars).rev() {
            let is_control = if next_control > 0 && controls[next_control - 1] == var {
                next_control -= 1;
                true
            } else {
                false
            };

            if var > gate.target {
                if is_control {
                    let p1_edge = self.local_operator_edge(var, IF_TRUE_MATRIX);
                    active_gate = self.kro_edge(p1_edge, active_gate);

                    let p0_edge = self.local_operator_edge(var, IF_FALSE_MATRIX);
                    let p0_part = self.kro_edge(p0_edge, identity_subtree[var + 1]);
                    let p1_part = self.kro_edge(p1_edge, inactive_gate);
                    inactive_gate = self.add_edge(p0_part, p1_part);
                } else {
                    let i_edge = self.local_operator_edge(var, I_MATRIX);
                    active_gate = self.kro_edge(i_edge, active_gate);
                    inactive_gate = self.kro_edge(i_edge, inactive_gate);
                }
            } else if var == gate.target {
                let i_edge = self.local_operator_edge(var, I_MATRIX);
                let u_edge = self.local_operator_edge(var, gate.matrix);

                let inactive_part = self.kro_edge(i_edge, inactive_gate);
                let active_part = self.kro_edge(u_edge, active_gate);
                active_gate = self.add_edge(inactive_part, active_part);
            } else if is_control {
                let p0_edge = self.local_operator_edge(var, IF_FALSE_MATRIX);
                let p1_edge = self.local_operator_edge(var, IF_TRUE_MATRIX);

                let p0_part = self.kro_edge(p0_edge, identity_subtree[var + 1]);
                let p1_part = self.kro_edge(p1_edge, active_gate);
                active_gate = self.add_edge(p0_part, p1_part);
            } else {
                let i_edge = self.local_operator_edge(var, I_MATRIX);
                active_gate = self.kro_edge(i_edge, active_gate);
            }
        }

        active_gate
    }

    /// Rebuild the full computational-basis amplitudes for the current QMDD.
    ///
    /// Each basis state is traversed through the graph by selecting the edge
    /// corresponding to the bit value of each qubit. The chosen path multiplies
    /// the root weight by every traversed edge weight until the terminal node is
    /// reached.
    fn reconstruct_state(&self) -> Vec<Complex<f64>> {
        let dim = 1usize << self.num_vars;
        let mut state = vec![COMPLEX_ZERO; dim];

        if self.root.node == self.true_node {
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

                if current_edge.node == self.true_node {
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
                for &child in &node.children {
                    if child != self.true_node {
                        stack.push(child);
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
    matrix: [Complex<f64>; 4],
}

impl Gate {
    fn new(controls: Vec<usize>, target: usize, matrix: [Complex<f64>; 4]) -> Self {
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
    fn measure_p1(&mut self, target: usize) -> f64 {
        // Fallback measurement strategy: derive probabilities from the same
        // projection used by dump so measurement semantics match ket output.
        // TODO: Fix this to use the QMDD directly instead of dumping to a state vector.
        let projected = self.dump(&[target]);

        projected
            .basis_states
            .iter()
            .zip(
                projected
                    .amplitudes_real
                    .iter()
                    .zip(projected.amplitudes_imag.iter()),
            )
            .filter(|(basis, _)| basis.first().copied().unwrap_or(0) == 1)
            .map(|(_, (re, im))| re * re + im * im)
            .sum()
    }

    fn measure_collapse(&mut self, target: usize, result: bool, p: f64) {
        let matrix = if result { M1_MATRIX } else { M0_MATRIX };
        let gate = Gate::new(vec![], target, matrix);
        let e = self.build_controlled_gate_edge(&gate);
        self.root = self.mul_edge(e, self.root);
        self.root.weight = self.root.weight / Complex::<f64>::new(p.sqrt(), 0.0);
    }

    fn dump(&mut self, qubits: &[usize]) -> DumpData {
        let output = self.reconstruct_state();
        let mut basis_states = Vec::new();
        let mut amplitudes_real = Vec::new();
        let mut amplitudes_imag = Vec::new();

        for (state, amp) in output.iter().enumerate() {
            let prob = amp.re * amp.re + amp.im * amp.im;
            if prob <= f64::EPSILON {
                continue;
            }

            let projected_state = qubits
                .iter()
                .rev()
                .enumerate()
                .map(|(index, qubit)| (((state >> qubit) & 1) as u64) << index)
                .reduce(|a, b| a | b)
                .unwrap_or(0);

            basis_states.push(vec![projected_state]);
            amplitudes_real.push(amp.re);
            amplitudes_imag.push(amp.im);
        }

        DumpData {
            basis_states,
            amplitudes_real,
            amplitudes_imag,
        }
    }

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
            Complex::<f64>::new(0.5, 0.0),
            Complex::<f64>::new(0.5, 0.0),
            Complex::<f64>::new(0.5, 0.0),
            Complex::<f64>::new(0.5, 0.0),
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
    fn measure_collapse_test() {
        let mut sim = Qmdd::new(1);
        sim.hadamard(0, &[]);
        let p1 = sim.measure_p1(0);
        assert!((p1 - 0.5).abs() < 1e-12, "unexpected probability: {p1:?} != 0.5");
        sim.measure_collapse(0, true, p1);
        let p = sim.measure_p1(0);
        assert!((p - 1.0).abs() < 1e-12, "unexpected probability: {p:?} != 1.0");
    }
}
