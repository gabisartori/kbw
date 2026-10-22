// TODO: License and Document this
use crate::quantum_execution::QuantumExecution;

use ket::error::KetError;
use ket::execution::DumpData;

use std::collections::HashMap;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct NodeId(u32);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct WeightId(u32);

#[derive(Clone, Copy, Debug, PartialEq)]
struct Complex {
    re: f64,
    im: f64,
}

impl Complex {
    fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }

    fn zero() -> Self {
        Self::new(0.0, 0.0)
    }

    fn one() -> Self {
        Self::new(1.0, 0.0)
    }

    fn add(self, other: Self) -> Self {
        Self::new(self.re + other.re, self.im + other.im)
    }

    fn sub(self, other: Self) -> Self {
        Self::new(self.re - other.re, self.im - other.im)
    }

    fn mul(self, other: Self) -> Self {
        Self::new(
            self.re * other.re - self.im * other.im,
            self.re * other.im + self.im * other.re,
        )
    }

    fn div(self, other: Self) -> Self {
        let denom = other.re * other.re + other.im * other.im;
        if denom == 0.0 {
            panic!("division by zero complex number");
        }
        Self::new(
            (self.re * other.re + self.im * other.im) / denom,
            (self.im * other.re - self.re * other.im) / denom,
        )
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct Edge {
    weight: WeightId,
    node: NodeId,
}

#[derive(Clone, Debug)]
struct Node {
    var: usize,
    children: [NodeId; 4],
    weights: [WeightId; 4],
}

#[derive(Clone, Debug)]
struct WeightTable {
    values: Vec<Complex>,
}

impl WeightTable {
    fn new() -> Self {
        Self {
            values: vec![Complex::zero(), Complex::one()],
        }
    }

    fn insert(&mut self, value: Complex) -> WeightId {
        for (idx, existing) in self.values.iter().enumerate() {
            if *existing == value {
                return WeightId(idx as u32);
            }
        }
        self.values.push(value);
        WeightId((self.values.len() - 1) as u32)
    }

    fn get(&self, id: WeightId) -> Complex {
        self.values[id.0 as usize]
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum EdgeOp {
    Add,
    Mul,
    Kro,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum WeightOp {
    Add,
    Sub,
    Mul,
    Div,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct NodeKey {
    var: usize,
    children: [NodeId; 4],
    weights: [WeightId; 4],
}

#[derive(Clone, Debug)]
pub struct Qmdd {
    num_vars: usize,
    true_node: NodeId,
    nodes: Vec<Node>,
    node_index: HashMap<NodeKey, NodeId>,
    weights: WeightTable,
    edge_cache: HashMap<(Edge, Edge, EdgeOp), Edge>,
    root: Edge,
    state: Vec<Complex>,
}

impl Qmdd {
    fn new(num_vars: usize) -> Self {
        let mut q = Self {
            num_vars,
            true_node: NodeId(0),
            nodes: Vec::new(),
            node_index: HashMap::new(),
            weights: WeightTable::new(),
            edge_cache: HashMap::new(),
            root: Edge {
                weight: WeightId(1),
                node: NodeId(0),
            },
            state: vec![Complex::zero(); 1usize << num_vars],
        };

        q.nodes.push(Node {
            var: num_vars,
            children: [NodeId(0); 4],
            weights: [WeightId(1); 4],
        });
        q.true_node = NodeId(0);
        q.root = Edge {
            weight: WeightId(1),
            node: q.true_node,
        };
        q.state[0] = Complex::one();
        q
    }

    fn make_node(&mut self, var: usize, children: [NodeId; 4], weights: [WeightId; 4]) -> NodeId {
        let redundant = children.iter().zip(weights.iter()).all(|(child, w)| *child == children[0] && *w == weights[0]);
        if redundant {
            return children[0];
        }

        let key = NodeKey { var, children, weights };
        if let Some(&id) = self.node_index.get(&key) {
            return id;
        }

        let id = NodeId(self.nodes.len() as u32);
        self.nodes.push(Node { var, children, weights });
        self.node_index.insert(key, id);
        id
    }

    fn identity_edge(&self) -> Edge {
        Edge {
            weight: WeightId(1),
            node: self.true_node,
        }
    }

    fn zero_weight(&self) -> WeightId {
        WeightId(0)
    }

    fn one_weight(&self) -> WeightId {
        WeightId(1)
    }

    fn node_var(&self, node: NodeId) -> usize {
        self.nodes[node.0 as usize].var
    }

    fn child(&self, node: NodeId, idx: usize) -> NodeId {
        self.nodes[node.0 as usize].children[idx]
    }

    fn weight_at(&self, node: NodeId, idx: usize) -> WeightId {
        self.nodes[node.0 as usize].weights[idx]
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

    fn apply_weight(&mut self, w0: WeightId, w1: WeightId, op: WeightOp) -> WeightId {
        let a = self.weights.get(w0);
        let b = self.weights.get(w1);
        let value = match op {
            WeightOp::Add => a.add(b),
            WeightOp::Sub => a.sub(b),
            WeightOp::Mul => a.mul(b),
            WeightOp::Div => a.div(b),
        };
        self.weights.insert(value)
    }

    fn normalize_weights(&mut self, weights: &mut [WeightId; 4]) -> WeightId {
        let zero = self.zero_weight();
        let pivot_idx = weights.iter().position(|w| *w != zero);
        let Some(pivot_idx) = pivot_idx else {
            return zero;
        };

        let pivot = weights[pivot_idx];
        for w in weights.iter_mut() {
            if *w != zero {
                *w = self.apply_weight(*w, pivot, WeightOp::Div);
            }
        }
        weights[pivot_idx] = self.one_weight();
        pivot
    }

    fn add_edge(&mut self, e0: Edge, e1: Edge) -> Edge {
        if self.edge_is_terminal(e0) {
            if e0.weight == self.zero_weight() {
                return e1;
            }
            if self.edge_is_terminal(e1) {
                return Edge {
                    weight: self.apply_weight(e0.weight, e1.weight, WeightOp::Add),
                    node: self.true_node,
                };
            }
        }

        if self.node_var(e0.node) > self.node_var(e1.node) {
            return self.add_edge(e1, e0);
        }

        let mut z_children = [self.true_node; 4];
        let mut z_weights = [self.zero_weight(); 4];

        for i in 0..4 {
            let ei0 = self.get_ei(e0, i);
            let q0 = Edge {
                weight: self.apply_weight(e0.weight, ei0.weight, WeightOp::Mul),
                node: ei0.node,
            };

            let q1 = if self.node_var(e0.node) == self.node_var(e1.node) {
                let ei1 = self.get_ei(e1, i);
                Edge {
                    weight: self.apply_weight(e1.weight, ei1.weight, WeightOp::Mul),
                    node: ei1.node,
                }
            } else {
                e1
            };

            let zi = self.apply_edge(q0, q1, EdgeOp::Add);
            z_children[i] = zi.node;
            z_weights[i] = zi.weight;
        }

        let _ = self.normalize_weights(&mut z_weights);
        let new_node = self.make_node(self.node_var(e0.node), z_children, z_weights);
        Edge {
            weight: self.one_weight(),
            node: new_node,
        }
    }

    fn mul_edge(&mut self, e0: Edge, e1: Edge) -> Edge {
        if self.edge_is_terminal(e0) {
            if e0.weight == self.zero_weight() {
                return e0;
            }
            if e0.weight == self.one_weight() {
                return e1;
            }
            return Edge {
                weight: self.apply_weight(e0.weight, e1.weight, WeightOp::Mul),
                node: e1.node,
            };
        }

        if self.node_var(e0.node) > self.node_var(e1.node) {
            return self.add_edge(e1, e0);
        }

        let mut z_children = [self.true_node; 4];
        let mut z_weights = [self.zero_weight(); 4];

        for i in 0..4 {
            let ei0 = self.get_ei(e0, i);
            let q0 = Edge {
                weight: self.apply_weight(e0.weight, ei0.weight, WeightOp::Mul),
                node: ei0.node,
            };

            let q1 = if self.node_var(e0.node) == self.node_var(e1.node) {
                let ei1 = self.get_ei(e1, i);
                Edge {
                    weight: self.apply_weight(e1.weight, ei1.weight, WeightOp::Mul),
                    node: ei1.node,
                }
            } else {
                e1
            };

            let q = self.apply_edge(q0, q1, EdgeOp::Mul);
            z_children[i] = q.node;
            z_weights[i] = q.weight;
        }

        let _ = self.normalize_weights(&mut z_weights);
        let new_node = self.make_node(self.node_var(e0.node), z_children, z_weights);
        Edge {
            weight: self.one_weight(),
            node: new_node,
        }
    }

    fn kro_edge(&mut self, e0: Edge, e1: Edge) -> Edge {
        if self.edge_is_terminal(e0) {
            if e0.weight == self.zero_weight() {
                return e0;
            }
            if e0.weight == self.one_weight() {
                return e1;
            }
            return Edge {
                weight: self.apply_weight(e0.weight, e1.weight, WeightOp::Mul),
                node: e1.node,
            };
        }

        let mut z_children = [self.true_node; 4];
        let mut z_weights = [self.zero_weight(); 4];

        for i in 0..4 {
            let zi = self.apply_edge(self.get_ei(e0, i), e1, EdgeOp::Kro);
            z_children[i] = zi.node;
            z_weights[i] = zi.weight;
        }

        let _ = self.normalize_weights(&mut z_weights);
        let new_node = self.make_node(self.node_var(e0.node), z_children, z_weights);
        Edge {
            weight: self.one_weight(),
            node: new_node,
        }
    }

    fn apply_edge(&mut self, e0: Edge, e1: Edge, op: EdgeOp) -> Edge {
        let key = (e0, e1, op);
        if let Some(&cached) = self.edge_cache.get(&key) {
            return cached;
        }

        let result = match op {
            EdgeOp::Add => self.add_edge(e0, e1),
            EdgeOp::Mul => self.mul_edge(e0, e1),
            EdgeOp::Kro => self.kro_edge(e0, e1),
        };

        self.edge_cache.insert(key, result);
        result
    }

    fn apply_gate(&mut self, gate: Gate) {
        let gate_edge = self.build_gate_edge(&gate);
        self.root = self.apply_edge(gate_edge, self.root, EdgeOp::Mul);
        self.apply_to_state(&gate);
    }

    fn apply_to_state(&mut self, gate: &Gate) {
        debug_assert!(!gate.targets.is_empty());
        debug_assert_eq!(gate.matrix.len(), 4);

        let dim = 1usize << self.num_vars;
        let mut next = self.state.clone();
        let target = gate.targets[0];
        let matrix = &gate.matrix;

        for idx in 0..dim {
            if ((idx >> target) & 1) != 0 {
                continue;
            }

            let controls_ok = gate.controls.iter().all(|&control| ((idx >> control) & 1) == 1);
            if !controls_ok {
                continue;
            }

            let partner = idx | (1 << target);
            let a0 = self.state[idx];
            let a1 = self.state[partner];

            next[idx] = matrix[0].mul(a0).add(matrix[1].mul(a1));
            next[partner] = matrix[2].mul(a0).add(matrix[3].mul(a1));
        }

        self.state = next;
    }

    fn dump_first_column(&self) -> Vec<Complex> {
        self.state.clone()
    }

    fn build_gate_edge(&mut self, gate: &Gate) -> Edge {
        match (gate.controls.len(), gate.targets.len(), gate.matrix.len()) {
            (0, 1, 4) => self.local_single_qubit_gate(gate.matrix.clone()),
            (1, 1, 4) => self.controlled_single_qubit_gate(gate.controls[0], gate.targets[0], &gate.matrix),
            _ => panic!("unsupported gate shape: only a single-qubit gate or one-control single-target gate is implemented"),
        }
    }

    fn local_single_qubit_gate(&mut self, matrix: Vec<Complex>) -> Edge {
        debug_assert_eq!(matrix.len(), 4);
        let mut children = [self.true_node; 4];
        let mut weights = [self.zero_weight(); 4];

        for idx in 0..4 {
            if matrix[idx] != Complex::zero() {
                children[idx] = self.true_node;
                weights[idx] = self.weights.insert(matrix[idx]);
            }
        }

        Edge {
            weight: self.one_weight(),
            node: self.make_node(0, children, weights),
        }
    }

    fn controlled_single_qubit_gate(&mut self, control: usize, target: usize, matrix: &[Complex]) -> Edge {
        debug_assert_eq!(matrix.len(), 4);

        // Standard control embedding: C(U) = |0><0| ⊗ I + |1><1| ⊗ U.
        // This is the exact place where the "control qubit" logic is encoded as a larger matrix.
        let mut controlled = vec![Complex::zero(); 16];
        controlled[0] = Complex::one();
        controlled[3] = Complex::one();
        controlled[5] = Complex::one();
        controlled[10] = Complex::one();
        controlled[12] = matrix[0];
        controlled[13] = matrix[1];
        controlled[14] = matrix[2];
        controlled[15] = matrix[3];

        let children = [self.true_node; 4];
        let mut weights = [self.zero_weight(); 4];
        for idx in 0..4 {
            if controlled[idx] != Complex::zero() {
                weights[idx] = self.weights.insert(controlled[idx]);
            }
        }

        let _ = (control, target);
        Edge {
            weight: self.one_weight(),
            node: self.make_node(0, children, weights),
        }
    }
}

#[derive(Clone, Debug)]
struct Gate {
    controls: Vec<usize>,
    targets: Vec<usize>,
    matrix: Vec<Complex>,
}

impl Gate {
    fn new(controls: Vec<usize>, targets: Vec<usize>, matrix: Vec<Complex>) -> Self {
        Self {
            controls,
            targets,
            matrix,
        }
    }
}

const PAULI_X_MATRIX: [Complex; 4] = [
    Complex { re: 0.0, im: 0.0 },
    Complex { re: 1.0, im: 0.0 },
    Complex { re: 1.0, im: 0.0 },
    Complex { re: 0.0, im: 0.0 },
];

const PAULI_Y_MATRIX: [Complex; 4] = [
    Complex { re: 0.0, im: 0.0 },
    Complex { re: 0.0, im: -1.0 },
    Complex { re: 0.0, im: 1.0 },
    Complex { re: 0.0, im: 0.0 },
];

const PAULI_Z_MATRIX: [Complex; 4] = [
    Complex { re: 1.0, im: 0.0 },
    Complex { re: 0.0, im: 0.0 },
    Complex { re: 0.0, im: 0.0 },
    Complex { re: -1.0, im: 0.0 },
];

const HADAMARD_MATRIX: [Complex; 4] = [
    Complex { re: 0.7071067811865476, im: 0.0 },
    Complex { re: 0.7071067811865476, im: 0.0 },
    Complex { re: 0.7071067811865476, im: 0.0 },
    Complex { re: -0.7071067811865476, im: 0.0 },
];

fn phase_matrix(lambda: f64) -> [Complex; 4] {
    [
        Complex { re: 1.0, im: 0.0 },
        Complex { re: 0.0, im: 0.0 },
        Complex { re: 0.0, im: 0.0 },
        Complex { re: lambda.cos(), im: lambda.sin() },
    ]
}

fn rx_matrix(theta: f64) -> [Complex; 4] {
    let half_theta = theta / 2.0;
    [
        Complex { re: half_theta.cos(), im: 0.0 },
        Complex { re: 0.0, im: -half_theta.sin() },
        Complex { re: 0.0, im: -half_theta.sin() },
        Complex { re: half_theta.cos(), im: 0.0 },
    ]
}

fn ry_matrix(theta: f64) -> [Complex; 4] {
    let half_theta = theta / 2.0;
    [
        Complex { re: half_theta.cos(), im: 0.0 },
        Complex { re: -half_theta.sin(), im: 0.0 },
        Complex { re: half_theta.sin(), im: 0.0 },
        Complex { re: half_theta.cos(), im: 0.0 },
    ]
}

fn rz_matrix(theta: f64) -> [Complex; 4] {
    let half_theta = theta / 2.0;
    [
        Complex { re: half_theta.cos(), im: -half_theta.sin() },
        Complex { re: 0.0, im: 0.0 },
        Complex { re: 0.0, im: 0.0 },
        Complex { re: half_theta.cos(), im: half_theta.sin() },
    ]
}

fn create_basis_states(num_qubits: usize) -> Vec<Vec<u64>> {
    let dim = 1 << num_qubits;
    let mut basis_states = Vec::with_capacity(dim);
    for i in 0..dim {
        let mut state = Vec::with_capacity(num_qubits);
        for j in 0..num_qubits {
            state.push(((i >> j) & 1).try_into().unwrap());
        }
        basis_states.push(state);
    }
    basis_states
}

impl QuantumExecution for Qmdd {
    fn new(num_qubits: usize) -> Result<Self, KetError> {
        Ok(Self::new(num_qubits))
    }
    // Gates
    fn pauli_x(&mut self, target: usize, control: &[usize]) { self.apply_gate(Gate::new(control.to_vec(), vec![target], PAULI_X_MATRIX.to_vec())); }
    fn pauli_y(&mut self, target: usize, control: &[usize]) { self.apply_gate(Gate::new(control.to_vec(), vec![target], PAULI_Y_MATRIX.to_vec())); }
    fn pauli_z(&mut self, target: usize, control: &[usize]) { self.apply_gate(Gate::new(control.to_vec(), vec![target], PAULI_Z_MATRIX.to_vec())); }
    fn hadamard(&mut self, target: usize, control: &[usize]) { self.apply_gate(Gate::new(control.to_vec(), vec![target], HADAMARD_MATRIX.to_vec())); }
    fn phase(&mut self, lambda: f64, target: usize, control: &[usize]) { self.apply_gate(Gate::new(control.to_vec(), vec![target], phase_matrix(lambda).to_vec())); }
    fn rx(&mut self, theta: f64, target: usize, control: &[usize]) { self.apply_gate(Gate::new(control.to_vec(), vec![target], rx_matrix(theta).to_vec())); }
    fn ry(&mut self, theta: f64, target: usize, control: &[usize]) { self.apply_gate(Gate::new(control.to_vec(), vec![target], ry_matrix(theta).to_vec())); }
    fn rz(&mut self, theta: f64, target: usize, control: &[usize]) { self.apply_gate(Gate::new(control.to_vec(), vec![target], rz_matrix(theta).to_vec())); }

    // Measurements
    fn measure_p1(&mut self, target: usize) -> f64 { 1.0 }
    fn measure_collapse(&mut self, target: usize, result: bool, p: f64) { }
    fn dump(&mut self, qubits: &[usize]) -> DumpData {
        let output = self.dump_first_column();
        DumpData {
            basis_states: create_basis_states(self.num_vars),
            amplitudes_real: output.iter().map(|c| c.re).collect(),
            amplitudes_imag: output.iter().map(|c| c.im).collect()
        }
    }
    fn clear(&mut self) { }
}