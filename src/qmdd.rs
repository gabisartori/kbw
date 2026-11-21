// TODO: License and Document this
use crate::quantum_execution::QuantumExecution;

use ket::error::KetError;
use ket::execution::DumpData;

use std::collections::HashMap;
use std::f64::consts::FRAC_1_SQRT_2;
use std::hash::{Hash, Hasher};

const PAULI_X_MATRIX: [ComplexId; 4] = [
    ComplexTable::ZERO,
    ComplexTable::ONE,
    ComplexTable::ONE,
    ComplexTable::ZERO
];

const PAULI_Y_MATRIX: [ComplexId; 4] = [
    ComplexTable::ZERO,
    ComplexTable::NEG_I,
    ComplexTable::I,
    ComplexTable::ZERO
];

const PAULI_Z_MATRIX: [ComplexId; 4] = [
    ComplexTable::ONE,
    ComplexTable::ZERO,
    ComplexTable::ZERO,
    ComplexTable::NEG_ONE
];

const HADAMARD_MATRIX: [ComplexId; 4] = [
    ComplexTable::A,
    ComplexTable::A,
    ComplexTable::A,
    ComplexTable::NEG_A
];

const M0_MATRIX: [ComplexId; 4] = [
    ComplexTable::ONE,
    ComplexTable::ZERO,
    ComplexTable::ZERO,
    ComplexTable::ZERO
];

const M1_MATRIX: [ComplexId; 4] = [
    ComplexTable::ZERO,
    ComplexTable::ZERO,
    ComplexTable::ZERO,
    ComplexTable::ONE
];

#[derive(Debug, Clone, Copy)]
pub struct ComplexVal {
    pub real: f64,
    pub imag: f64,
}

impl ComplexVal {
    pub fn new(real: f64, imag: f64) -> Self {
        Self {
            real: if real == 0.0 { 0.0 } else { real },
            imag: if imag == 0.0 { 0.0 } else { imag },
        }
    }

    pub fn re(&self) -> f64 {
        self.real
    }

    pub fn im(&self) -> f64 {
        self.imag
    }

    pub fn mag2(&self) -> f64 {
        let r = self.re();
        let i = self.im();
        r * r + i * i
    }

    pub fn from_float(x: f64, real: bool) -> Self {
        if real {
            Self::new(x, 0.0)
        } else {
            Self::new(0.0, x)
        }
    }
}

impl std::ops::Add for ComplexVal {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self::new(self.real + other.real, self.imag + other.imag)
    }
}

impl std::ops::Sub for ComplexVal {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        Self::new(self.real - other.real, self.imag - other.imag)
    }
}

impl std::ops::Mul for ComplexVal {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        Self::new(
            self.real * other.real - self.imag * other.imag,
            self.real * other.imag + self.imag * other.real,
        )
    }
}

impl std::ops::Div for ComplexVal {
    type Output = Self;

    fn div(self, other: Self) -> Self {
        assert!(
            other.real != 0.0 || other.imag != 0.0,
            "Error in ComplexVal::div(): Division by zero."
        );

        if other.real.abs() >= other.imag.abs() {
            let ratio = other.imag / other.real;
            let denominator = other.real + other.imag * ratio;
            Self::new(
                (self.real + self.imag * ratio) / denominator,
                (self.imag - self.real * ratio) / denominator,
            )
        } else {
            let ratio = other.real / other.imag;
            let denominator = other.imag + other.real * ratio;
            Self::new(
                (self.real * ratio + self.imag) / denominator,
                (self.imag * ratio - self.real) / denominator,
            )
        }
    }
}

impl PartialEq for ComplexVal {
    fn eq(&self, other: &Self) -> bool {
        self.real.to_bits() == other.real.to_bits()
            && self.imag.to_bits() == other.imag.to_bits()
    }
}

impl Eq for ComplexVal {}

impl Hash for ComplexVal {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.real.to_bits().hash(state);
        self.imag.to_bits().hash(state);
    }
}

pub type ComplexId = usize;

#[derive(Debug, Clone)]
pub struct ComplexTable {
    i2complex: Vec<ComplexVal>,
    complex2i: HashMap<ComplexVal, ComplexId>,
    sums: HashMap<(ComplexId, ComplexId), ComplexId>,
    prods: HashMap<(ComplexId, ComplexId), ComplexId>,
    quots: HashMap<(ComplexId, ComplexId), ComplexId>,
}

impl ComplexTable {
    pub const ZERO: ComplexId = 0;    // 0 + 0j
    pub const ONE: ComplexId = 1;     // 1 + 0j
    pub const A: ComplexId = 2;       // 1/sqrt(2) + 0j
    pub const NEG_ONE: ComplexId = 3; // -1 + 0j
    pub const I: ComplexId = 4;       // 0 + 1j
    pub const NEG_I: ComplexId = 5;   // 0 - 1j
    pub const NEG_A: ComplexId = 6;   // -1/sqrt(2) + 0j
    pub const B: ComplexId = 7;       // 1/sqrt(2) + 1/sqrt(2)j
    pub const C: ComplexId = 8;       // 1/sqrt(2) - 1/sqrt(2)j

    pub fn new() -> Self {
        let mut table = Self {
            i2complex: Vec::new(),
            complex2i: HashMap::new(),
            sums: HashMap::new(),
            prods: HashMap::new(),
            quots: HashMap::new(),
        };
        table.reset();
        table
    }

    pub fn reset(&mut self) {
        self.i2complex.clear();
        self.complex2i.clear();
        self.sums.clear();
        self.prods.clear();
        self.quots.clear();

        self.intern(0.0, 0.0); // 0: ZERO
        self.intern(1.0, 0.0); // 1: ONE
        self.intern(FRAC_1_SQRT_2, 0.0); // 2: A
        self.intern(-1.0, 0.0); // 3: NEG_ONE
        self.intern(0.0, 1.0); // 4: I
        self.intern(0.0, -1.0); // 5: NEG_I
        self.intern(-FRAC_1_SQRT_2, 0.0); // 6: NEG_A
        self.intern(FRAC_1_SQRT_2, FRAC_1_SQRT_2); // 7: B
        self.intern(FRAC_1_SQRT_2, -FRAC_1_SQRT_2); // 8: C
    }

    pub fn from_float(&mut self, x: f64, real: bool) -> ComplexId {
        let x = ComplexVal::from_float(x, real);
        self.intern(x.real, x.imag)
    }

    pub fn intern(&mut self, real: f64, imag: f64) -> ComplexId {
        let val = ComplexVal::new(real, imag);
        if let Some(&id) = self.complex2i.get(&val) {
            return id;
        }
        let id = self.i2complex.len();
        self.i2complex.push(val);
        self.complex2i.insert(val, id);
        id
    }

    #[inline]
    pub fn get(&self, id: ComplexId) -> ComplexVal {
        self.i2complex[id]
    }

    pub fn add(&mut self, first: ComplexId, second: ComplexId) -> ComplexId {
        if first > second {
            return self.add(second, first);
        }
        if first == Self::ZERO {
            return second;
        }
        if let Some(&cached) = self.sums.get(&(first, second)) {
            return cached;
        }

        let lhs = self.i2complex[first];
        let rhs = self.i2complex[second];

        let val = lhs + rhs;
        let res = self.intern(val.real, val.imag);

        self.sums.insert((first, second), res);
        res
    }

    pub fn mul(&mut self, first: ComplexId, second: ComplexId) -> ComplexId {
        if first > second {
            return self.mul(second, first);
        }
        if first == Self::ZERO || second == Self::ONE {
            return first;
        }
        if second == Self::ZERO || first == Self::ONE {
            return second;
        }
        if let Some(&cached) = self.prods.get(&(first, second)) {
            return cached;
        }

        let lhs = self.i2complex[first];
        let rhs = self.i2complex[second];

        let val = lhs * rhs;
        let res = self.intern(val.real, val.imag);

        self.prods.insert((first, second), res);
        res
    }

    pub fn mul3(&mut self, a: ComplexId, b: ComplexId, c: ComplexId) -> ComplexId {
        let ab = self.mul(a, b);
        self.mul(ab, c)
    }

    pub fn div(&mut self, num: ComplexId, den: ComplexId) -> ComplexId {
        assert!(den != Self::ZERO, "Error in ComplexTable::div(): Division by zero.");
        if num == Self::ZERO || den == Self::ONE {
            return num;
        }
        if num == den {
            return Self::ONE;
        }
        if let Some(&cached) = self.quots.get(&(num, den)) {
            return cached;
        }

        let lhs = self.i2complex[num];
        let rhs = self.i2complex[den];
        let res_val = lhs / rhs;
        let res = self.intern(res_val.real, res_val.imag);

        self.quots.insert((num, den), res);
        res
    }

    pub fn argmax(&self, indices: &[ComplexId], tolerance: f64) -> ComplexId {
        assert!(!indices.is_empty(), "Error in ComplexTable::argmax(): No values passed.");
        let mut greatest = indices[0];
        let mut max = if indices.len() > 1 {
            self.i2complex[greatest].mag2()
        } else {
            0.0
        };

        for &idx in &indices[1..] {
            let mag = self.i2complex[idx].mag2();
            if mag - max > tolerance {
                greatest = idx;
                max = mag;
            }
        }
        greatest
    }
}

// ============================================================================
// 2. QMDD Vertices, Edges, and Simulator
// ============================================================================

pub type VertexId = usize;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Edge {
    pub dest: VertexId,
    pub weight: ComplexId,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Edges {
    Terminal,
    Vector([Edge; 2]),
    Matrix([Edge; 4]),
}

impl Edges {
    pub fn as_slice(&self) -> &[Edge] {
        match self {
            Edges::Terminal => &[],
            Edges::Vector(e) => e,
            Edges::Matrix(e) => e,
        }
    }

    pub fn is_vector(&self) -> bool {
        matches!(self, Edges::Vector(_))
    }

    pub fn is_terminal(&self) -> bool {
        matches!(self, Edges::Terminal)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct VertexKey {
    variable: usize,
    edges: Edges,
}

#[derive(Debug, Clone)]
pub struct Vertex {
    pub variable: usize,
    pub prob: f64,
    pub edges: Edges,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NormRule {
    FirstNonZero,
    MaxMagnitude,
}

pub struct Qmdd {
    num_vars: usize,
    complex: ComplexTable,
    vertices: Vec<Vertex>,
    unique_table: HashMap<VertexKey, VertexId>,
    sums: HashMap<(Edge, Edge), Edge>,
    prods: HashMap<(Edge, Edge), Edge>,
    terminal: VertexId,
    state: Edge,
}

impl Qmdd {
    /// Creates a simulator with capability for `num_vars`, initialized to the |0...0> ground state.
    pub fn new(num_vars: usize) -> Self {
        let mut sim = Self {
            num_vars,
            complex: ComplexTable::new(),
            vertices: Vec::new(),
            unique_table: HashMap::new(),
            sums: HashMap::new(),
            prods: HashMap::new(),
            terminal: 0,
            state: Edge {
                dest: 0,
                weight: ComplexTable::ONE,
            },
        };

        sim.terminal = sim.create_terminal(num_vars);
        sim.state = sim.ground_state();
        sim
    }

    /// Exposes the internal complex number table so callers can construct custom gates
    /// using standard constants (`ComplexTable::ONE`, `ComplexTable::A`, etc.) or `intern(...)`.
    pub fn complex_table(&mut self) -> &mut ComplexTable {
        &mut self.complex
    }

    /// Applies a single-qubit 2x2 gate (in row-major order `[e00, e01, e10, e11]`)
    /// at `target` qubit index, conditioned on all `controls` being in state |1>.
    pub fn apply_gate(&mut self, gate: [ComplexId; 4], target: usize, controls: &[usize]) {
        assert!(target < self.num_vars, "Target qubit index out of bounds.");
        for &c in controls {
            assert!(c < self.num_vars, "Control qubit index out of bounds.");
            assert!(c != target, "Control and target qubits must be distinct.");
        }

        let gate_edge = self.construct_gate(gate, target, controls);
        let level = self.vertices[gate_edge.dest]
            .variable
            .min(self.vertices[self.state.dest].variable);
        self.state = self.multiply(gate_edge, self.state, level);
    }

    /// Returns all non-zero basis states and their `(re, im)` amplitudes via preorder DFS.
    // TODO: Replace dump by this
    // TODO: Replace string with Vec<u64> as is the ket interface
    pub fn strong_simulate(&mut self) -> Vec<(Vec<u64>, f64, f64)> {
        assert!(self.state.weight != ComplexTable::ZERO, "State entry is a zero edge.");
        let mut results = Vec::new();
        let mut stack = vec![(self.state.dest, self.state.weight, vec![0])];

        while let Some((v_id, w, s)) = stack.pop() {
            let v = &self.vertices[v_id];
            if v.edges.is_terminal() {
                let c = self.complex.get(w);
                results.push((s, c.re(), c.im()));
                continue;
            }

            let edges = v.edges.as_slice();
            for i in [1usize, 0usize] {
                if edges[i].weight != ComplexTable::ZERO {
                    let s: u64 = s[0] | (i as u64) << v.variable;
                    let next_w = self.complex.mul(w, edges[i].weight);
                    stack.push((edges[i].dest, next_w, vec![s]));
                }
            }
        }
        results
    }

    fn reconstruct_state(&mut self) -> Vec<ComplexVal> {
        let dim = 1 << self.num_vars;
        let mut output = vec![ComplexVal::new(0.0, 0.0); dim];
                
        for state in 0..dim {
            let mut value = self.state.weight;
            let mut current_node = self.state.dest;
            loop {
                if self.vertices[current_node].edges.is_terminal() {
                    output[state] = self.complex.get(value);
                    break;
                }
                let bit = self.vertices[current_node].variable;
                let bit_index = (state >> bit) & 1;
                let next_edge = self.vertices[current_node].edges.as_slice()[bit_index];
                value = self.complex.mul(value, next_edge.weight);
                current_node = next_edge.dest;
            }
        }

        output
    }

    fn create_terminal(&mut self, width: usize) -> VertexId {
        let key = VertexKey {
            variable: width,
            edges: Edges::Terminal,
        };
        if let Some(&id) = self.unique_table.get(&key) {
            return id;
        }
        let id = self.vertices.len();
        self.vertices.push(Vertex {
            variable: width,
            prob: 1.0,
            edges: Edges::Terminal,
        });
        self.unique_table.insert(key, id);
        id
    }

    /// Normalizes a non-redundant set of edges
    fn normalize(&mut self, edges: &mut Edges, rule: NormRule) -> (ComplexId, f64) {
        let slice = edges.as_slice();
        if slice.is_empty() {
            return (ComplexTable::ONE, 1.0);
        }

        // Chooses a factor to normalize the edges based on the normalization rule
        // Not sure why the rule isn't always FirstNonZero. But the implementation is using MaxMagnitude so idk
        let factor = match rule {
            NormRule::FirstNonZero => slice
                .iter()
                .find(|e| e.weight != ComplexTable::ZERO)
                .map(|e| e.weight)
                .unwrap_or(ComplexTable::ZERO),
            NormRule::MaxMagnitude => {
                let weights: Vec<ComplexId> = slice.iter().map(|e| e.weight).collect();
                self.complex.argmax(&weights, 0.000001)
            }
        };

        // Divide all edges weights by the chosen factor
        if factor != ComplexTable::ZERO {
            match edges {
                Edges::Vector(arr) => {
                    for edge in arr.iter_mut() {
                        edge.weight = self.complex.div(edge.weight, factor);
                    }
                }
                Edges::Matrix(arr) => {
                    for edge in arr.iter_mut() {
                        edge.weight = self.complex.div(edge.weight, factor);
                    }
                }
                Edges::Terminal => {}
            }
        }

        // TODO: What the fuck
        let prob = match edges {
            Edges::Vector(arr) => {
                let p0 = self.vertices[arr[0].dest].prob * self.complex.get(arr[0].weight).mag2();
                let p1 = self.vertices[arr[1].dest].prob * self.complex.get(arr[1].weight).mag2();
                p0 + p1
            }
            _ => 1.0,
        };

        (factor, prob)
    }

    // TODO: Maybe I needed to explain this in the paper
    fn is_trivial(edges: &Edges) -> bool {
        if let Edges::Matrix(e) = edges {
            e[0].weight == e[3].weight
                && e[0].dest == e[3].dest
                && e[0].weight != ComplexTable::ZERO
                && e[1].weight == ComplexTable::ZERO
                && e[2].weight == ComplexTable::ZERO
        } else {
            false
        }
    }

    // Returns true if all edges are identical (same destination and weight).
    // Terminal edges are never redundant and Vectors do not apply
    fn is_redundant(edges: &Edges) -> bool {
        if let Edges::Matrix(e) = edges {
            e.iter()
                .all(|edge| edge.dest == e[0].dest && edge.weight == e[0].weight)
        } else {
            false
        }
    }

    fn create_vertex(&mut self, variable: usize, mut edges: Edges, rule: NormRule) -> Edge {
        let (weight, prob) = self.normalize(&mut edges, rule);

        if weight == ComplexTable::ZERO {
            return Edge {
                dest: self.terminal,
                weight: ComplexTable::ZERO,
            };
        }

        if Self::is_trivial(&edges) || Self::is_redundant(&edges) {
            return Edge {
                dest: edges.as_slice()[0].dest,
                weight,
            };
        }

        let key = VertexKey {
            variable,
            edges: edges.clone(),
        };

        let dest = if let Some(&existing_id) = self.unique_table.get(&key) {
            existing_id
        } else {
            let id = self.vertices.len();
            self.vertices.push(Vertex {
                variable,
                prob,
                edges,
            });
            self.unique_table.insert(key, id);
            id
        };

        Edge { dest, weight }
    }

    /// Creates a vector edge representing |0>^num_vars, which is the ground state of the system.
    fn ground_state(&mut self) -> Edge {
        let mut e = Edge {
            dest: self.terminal,
            weight: ComplexTable::ONE,
        };
        let zero_edge = Edge {
            dest: self.terminal,
            weight: ComplexTable::ZERO,
        };

        for variable in (0..self.num_vars).rev() {
            e = self.create_vertex(variable, Edges::Vector([e, zero_edge]), NormRule::MaxMagnitude);
        }
        e
    }

    fn add(&mut self, tensor1: Edge, tensor2: Edge, level: usize) -> Edge {
        if tensor1.weight == ComplexTable::ZERO {
            return Edge {
                dest: if tensor2.weight == ComplexTable::ZERO {
                    self.terminal
                } else {
                    tensor2.dest
                },
                weight: tensor2.weight,
            };
        }

        if tensor2.weight == ComplexTable::ZERO {
            return tensor1;
        }

        if tensor1.dest == tensor2.dest {
            return Edge {
                dest: tensor1.dest,
                weight: self.complex.add(tensor1.weight, tensor2.weight),
            };
        }

        let cache_key = if tensor1.dest <= tensor2.dest {
            (tensor1, tensor2)
        } else {
            (tensor2, tensor1)
        };

        if let Some(&cached) = self.sums.get(&cache_key) {
            return cached;
        }

        let is_vec = self.vertices[tensor1.dest].edges.is_vector()
            || self.vertices[tensor2.dest].edges.is_vector();
        let num_edges = if is_vec { 2 } else { 4 };
        let mut new_edges = Vec::with_capacity(num_edges);

        for i in 0..num_edges {
            let e0 = self.extract_sub_edge(tensor1, i, level);
            let e1 = self.extract_sub_edge(tensor2, i, level);
            new_edges.push(self.add(e0, e1, level + 1));
        }

        let edges = if is_vec {
            Edges::Vector([new_edges[0], new_edges[1]])
        } else {
            Edges::Matrix([new_edges[0], new_edges[1], new_edges[2], new_edges[3]])
        };

        let res = self.create_vertex(level, edges, NormRule::MaxMagnitude);
        self.sums.insert(cache_key, res);
        res
    }

    fn extract_sub_edge(&mut self, tensor: Edge, i: usize, level: usize) -> Edge {
        let v = &self.vertices[tensor.dest];
        if v.edges.is_terminal() || v.variable > level {
            if i == 0 || i == 3 {
                tensor
            } else {
                Edge {
                    dest: self.terminal,
                    weight: ComplexTable::ZERO,
                }
            }
        } else {
            let child = v.edges.as_slice()[i];
            Edge {
                dest: child.dest,
                weight: self.complex.mul(tensor.weight, child.weight),
            }
        }
    }

    // For this implementation, multiplication is always applied between a matrix and a vector
    // There is no need for matrix-matrix multiplication
    fn multiply(&mut self, matrix: Edge, vector: Edge, level: usize) -> Edge {
        if matrix.weight == ComplexTable::ZERO || vector.weight == ComplexTable::ZERO {
            return Edge {
                dest: self.terminal,
                weight: ComplexTable::ZERO,
            };
        }

        if self.vertices[matrix.dest].edges.is_terminal() {
            return Edge {
                dest: vector.dest,
                weight: self.complex.mul(vector.weight, matrix.weight),
            };
        }

        let cache_key = (matrix, vector);
        if let Some(&cached) = self.prods.get(&cache_key) {
            return cached;
        }

        let zero_edge = Edge {
            dest: self.terminal,
            weight: ComplexTable::ZERO,
        };
        let mut edges = [zero_edge, zero_edge];

        for i in 0..2 {
            for j in 0..2 {
                let v = self.vertices[vector.dest].edges.as_slice()[j];
                let m_var = self.vertices[matrix.dest].variable;
                let m = if m_var == level {
                    self.vertices[matrix.dest].edges.as_slice()[2 * i + j]
                } else if i == j {
                    Edge {
                        dest: matrix.dest,
                        weight: ComplexTable::ONE,
                    }
                } else {
                    zero_edge
                };

                let prod = self.multiply(m, v, level + 1);
                edges[i] = self.add(edges[i], prod, level + 1);
            }
        }

        let e = self.create_vertex(level, Edges::Vector(edges), NormRule::MaxMagnitude);
        let res = Edge {
            dest: e.dest,
            weight: self.complex.mul3(e.weight, matrix.weight, vector.weight),
        };
        self.prods.insert(cache_key, res);
        res
    }

    fn construct_gate(&mut self, gate: [ComplexId; 4], target: usize, controls: &[usize]) -> Edge {
        let mut ctrls = controls.to_vec();
        // Sort descending so deeper qubit indices are handled first
        ctrls.sort_unstable_by(|a, b| b.cmp(a));
        ctrls.dedup();

        let zero_edge = Edge {
            dest: self.terminal,
            weight: ComplexTable::ZERO,
        };
        let one_edge = Edge {
            dest: self.terminal,
            weight: ComplexTable::ONE,
        };

        let mut root: [Edge; 4] = [
            Edge { dest: self.terminal, weight: gate[0] },
            Edge { dest: self.terminal, weight: gate[1] },
            Edge { dest: self.terminal, weight: gate[2] },
            Edge { dest: self.terminal, weight: gate[3] },
        ];

        let mut q = 0;

        // Controls below target (index > target), all-if-true (activator = 3, antiactivator = 0)
        while q < ctrls.len() && ctrls[q] > target {
            let ctrl_idx = ctrls[q];
            for i in 0..2 {
                for j in 0..2 {
                    let edge_idx = 2 * i + j;
                    let mut reusable = [zero_edge; 4];
                    reusable[3] = root[edge_idx];
                    reusable[0] = if i == j { one_edge } else { zero_edge };
                    root[edge_idx] =
                        self.create_vertex(ctrl_idx, Edges::Matrix(reusable), NormRule::MaxMagnitude);
                }
            }
            q += 1;
        }

        // Target vertex
        let mut entry = self.create_vertex(target, Edges::Matrix(root), NormRule::MaxMagnitude);

        // Controls above target (index < target), all-if-true (activator = 3, antiactivator = 0)
        while q < ctrls.len() {
            let ctrl_idx = ctrls[q];
            let mut reusable = [zero_edge; 4];
            reusable[3] = entry;
            reusable[0] = one_edge;
            entry = self.create_vertex(ctrl_idx, Edges::Matrix(reusable), NormRule::MaxMagnitude);
            q += 1;
        }

        entry
    }

    fn renormalize(&mut self, p: f64) {
        let p = self.complex.from_float(p, true);
        self.state.weight = self.complex.mul(self.state.weight, p);
    }

    fn prob_one_dfs(
        &mut self,
        v_id: VertexId,
        acc_weight: ComplexId,
        target: usize,
        memo: &mut HashMap<(VertexId, ComplexId), f64>,
    ) -> f64 {
        if acc_weight == ComplexTable::ZERO {
            return 0.0;
        }

        let v = &self.vertices[v_id].clone();
        if v.edges.is_terminal() {
            return 0.0;
        }

        // If we jumped past `target` (target level was skipped in the DAG),
        // half of the assignments to the skipped `target` bit yield |1>.
        if v.variable < target {
            let w = self.complex.get(acc_weight).mag2();
            return 0.5 * w * self.subtree_norm_sqr(v_id);
        }

        let key = (v_id, acc_weight);
        if let Some(&cached) = memo.get(&key) {
            return cached;
        }

        let edges = v.edges.as_slice();
        let total = if v.variable == target {
            // Only the |1> branch (edges[1]) contributes to measuring |1> on `target`
            let e1 = edges[1];
            if e1.weight == ComplexTable::ZERO {
                0.0
            } else {
                let w1 = self.complex.mul(acc_weight, e1.weight);
                let mag2 = self.complex.get(w1).mag2();
                mag2 * self.subtree_norm_sqr(e1.dest)
            }
        } else {
            // Above target: explore both |0> (edges[0]) and |1> (edges[1])
            let mut sum = 0.0;
            for &edge in edges {
                if edge.weight != ComplexTable::ZERO {
                    let next_w = self.complex.mul(acc_weight, edge.weight);
                    sum += self.prob_one_dfs(edge.dest, next_w, target, memo);
                }
            }
            sum
        };

        memo.insert(key, total);
        total
    }

    /// Sum of |path_weight|^2 from `v_id` to terminal (assuming unit-norm state).
    fn subtree_norm_sqr(&self, v_id: VertexId) -> f64 {
        let v = &self.vertices[v_id];
        if v.edges.is_terminal() {
            return 1.0;
        }
        let edges = v.edges.as_slice();
        let mut sum = 0.0;
        for &edge in edges {
            if edge.weight != ComplexTable::ZERO {
                let w2 = self.complex.get(edge.weight).mag2();
                sum += w2 * self.subtree_norm_sqr(edge.dest);
            }
        }
        sum
    }
}

impl QuantumExecution for Qmdd {
    fn new(num_vars: usize) -> Result<Self, KetError> {
        Ok(Self::new(num_vars))
    }
    // Gates
    fn pauli_x(&mut self, target: usize, control: &[usize]) {
        self.apply_gate(PAULI_X_MATRIX, target, control);
    }

    fn pauli_y(&mut self, target: usize, control: &[usize]) {
        self.apply_gate(PAULI_Y_MATRIX, target, control);
    }

    fn pauli_z(&mut self, target: usize, control: &[usize]) {
        self.apply_gate(PAULI_Z_MATRIX, target, control);
    }

    fn hadamard(&mut self, target: usize, control: &[usize]) {
        self.apply_gate(HADAMARD_MATRIX, target, control);
    }

    fn phase(&mut self, lambda: f64, target: usize, control: &[usize]) {
        let c = self.complex.from_float(lambda.cos(), true);
        let s = self.complex.from_float(lambda.sin(), false);
        let p = self.complex.add(c, s);
        let phase_matrix = [
            ComplexTable::ONE,
            ComplexTable::ZERO,
            ComplexTable::ZERO,
            p
        ];
        self.apply_gate(phase_matrix, target, control);
    }

    fn rx(&mut self, theta: f64, target: usize, control: &[usize]) {
        let t = theta/2.0;
        let c = self.complex.from_float(t.cos(), true);
        let s = self.complex.from_float(-t.sin(), false);
        let rx = [
            c, s,
            s, c
        ];
        self.apply_gate(rx, target, control);
    }

    fn ry(&mut self, theta: f64, target: usize, control: &[usize]) {
        let t = theta/2.0;
        let c = self.complex.from_float(t.cos(), true);
        let s = self.complex.from_float(t.sin(), true);
        let sm = self.complex.from_float(-t.sin(), true);
        let ry = [
            c, sm,
            s, c
        ];
        self.apply_gate(ry, target, control);
    }

    fn rz(&mut self, theta: f64, target: usize, control: &[usize]) {
        let t = theta/2.0;
        let c = self.complex.from_float(t.cos(), true);
        let s = self.complex.from_float(t.sin(), false);
        let sm = self.complex.from_float(-t.sin(), false);
        let rz = [
            c, sm,
            s, c
        ];
        self.apply_gate(rz, target, control);
    }

    // Measurements
    fn measure_p1(&mut self, target: usize) -> f64 {
        assert!(target < self.num_vars, "Target qubit out of bounds");
        
        // Memoization cache to avoid redundant sub-graph traversals: (vertex_id, accumulated_weight_id) -> probability contribution
        let mut memo: HashMap<(VertexId, ComplexId), f64> = HashMap::new();
        
        self.prob_one_dfs(
            self.state.dest,
            self.state.weight,
            target,
            &mut memo
        )
    }

    // Apply collapse matrix appropriate to observed result then renormalize the state vector edge
    fn measure_collapse(&mut self, target: usize, result: bool, p: f64) {
        let matrix = if result { M1_MATRIX } else { M0_MATRIX };
        self.apply_gate(matrix, target, &[]);
        self.renormalize(p);
    }

    // TODO: Replace this by a function that doesn't need 2^n vector for any ammount of qubits
    // Strong simulate does it but I still need to project the states into the states created by *qubits*
    fn dump(&mut self, qubits: &[usize]) -> DumpData {
        let output = self.reconstruct_state();
        let mut basis_states = Vec::new();
        let mut amplitudes_real = Vec::new();
        let mut amplitudes_imag = Vec::new();

        for (state, amp) in output.iter().enumerate() {
            let prob = amp.mag2();
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
            amplitudes_real.push(amp.re());
            amplitudes_imag.push(amp.im());
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

    fn assert_states_close(actual: &[ComplexVal], expected: &[ComplexVal]) {
        assert_eq!(actual.len(), expected.len());
        for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (actual.re() - expected.re()).abs() < 1e-10
                    && (actual.im() - expected.im()).abs() < 1e-10,
                "State {index} differs: {actual:?} != {expected:?}"
            );
        }
    }

    fn measure_all_qubits(sim: &mut Qmdd) -> u64 {
        let mut result = 0u64;
        for qubit in 0..sim.num_vars {
            let p1 = sim.measure_p1(qubit);
            println!("Measured qubit {qubit}: p1 = {p1}");
            let outcome = p1 > 0.9;
            if outcome {
                result |= 1 << qubit;
            }
        }
        result
    }

    #[test]
    fn single_qubit_h() {
        let mut sim = Qmdd::new(1);
        sim.hadamard(0, &[]);
        let state = sim.reconstruct_state();
        let expected = vec![
            ComplexVal::new(FRAC_1_SQRT_2, 0.0),
            ComplexVal::new(FRAC_1_SQRT_2, 0.0)
        ];
        assert_states_close(&state, &expected);
    }

    #[test]
    fn double_qubit_h() {
        let mut sim = Qmdd::new(2);
        sim.hadamard(0, &[]);
        sim.hadamard(1, &[]);
        let state = sim.reconstruct_state();
        let expected = vec![
            ComplexVal::new(0.5, 0.0), // |00> amplitude (1/2)
            ComplexVal::new(0.5, 0.0), // |01> amplitude (1/2)
            ComplexVal::new(0.5, 0.0), // |10> amplitude (1/2)
            ComplexVal::new(0.5, 0.0), // |11> amplitude (1/2)
        ];
        assert_states_close(&state, &expected);
    }

    #[test]
    fn bell() {
        let mut sim = Qmdd::new(2);
        sim.hadamard(0, &[]);
        sim.pauli_x(1, &[0]);
        let state = sim.reconstruct_state();
        let expected = vec![
            ComplexVal::new(FRAC_1_SQRT_2, 0.0), // |00> amplitude (sqrt(2)/2)
            ComplexVal::new(0.0, 0.0), // |01> amplitude (0)
            ComplexVal::new(0.0, 0.0), // |10> amplitude (0)
            ComplexVal::new(FRAC_1_SQRT_2, 0.0), // |11> amplitude (sqrt(2)/2)
        ];
        assert_states_close(&state, &expected);
    }

    #[test]
    fn measure_collapse_hadamard() {
        let mut sim = Qmdd::new(1);
        sim.hadamard(0, &[]);
        let state = sim.reconstruct_state();
        let expected = vec![
            ComplexVal::new(FRAC_1_SQRT_2, 0.0), // |0> amplitude (sqrt(2)/2)
            ComplexVal::new(FRAC_1_SQRT_2, 0.0), // |1> amplitude (sqrt(2)/2)
        ];
        assert_states_close(&state, &expected);

        let p1 = sim.measure_p1(0);
        assert!((p1 - 0.5).abs() < 1e-12, "Wrong probability before measure: {p1:?} != 0.5");
        let p = 1.0/p1.sqrt();
        sim.measure_collapse(0, true, p);
        let p1 = sim.measure_p1(0);
        let state = sim.reconstruct_state();
        let expected = vec![
            ComplexVal::new(0.0, 0.0), // |0> amplitude (0)
            ComplexVal::new(1.0, 0.0), // |1> amplitude (1)
        ];
        assert_states_close(&state, &expected);
        assert!((p1 - 1.0).abs() < 1e-12, "Wrong probability after measure: {p1:?} != 1.0");
    }

    #[test]
    fn measure_collapse_bell() {
        let mut sim = Qmdd::new(2);
        sim.hadamard(0, &[]);
        sim.pauli_x(1, &[0]);

        let state = sim.reconstruct_state();
        let expected = vec![
            ComplexVal::new(FRAC_1_SQRT_2, 0.0), // |00> amplitude (sqrt(2)/2)
            ComplexVal::new(0.0, 0.0), // |01> amplitude (0)
            ComplexVal::new(0.0, 0.0), // |10> amplitude (0)
            ComplexVal::new(FRAC_1_SQRT_2, 0.0), // |11> amplitude (sqrt(2)/2)
        ];
        let p1 = sim.measure_p1(0);
        assert_states_close(&state, &expected);
        assert!((p1 - 0.5).abs() < 1e-12, "Wrong probability before measure: {p1:?} != 0.5");

        // Collapse the superposition to |11>
        let p = 1.0/p1.sqrt();
        sim.measure_collapse(0, true, p);
        let state = sim.reconstruct_state();
        let expected = vec![
            ComplexVal::new(0.0, 0.0), // |00> amplitude (0)
            ComplexVal::new(0.0, 0.0), // |01> amplitude (0)
            ComplexVal::new(0.0, 0.0), // |10> amplitude (0)
            ComplexVal::new(1.0, 0.0), // |11> amplitude (1)
        ];
        let p1 = sim.measure_p1(0);
        assert_states_close(&state, &expected);
        assert!((p1 - 1.0).abs() < 1e-12, "Wrong probability after measure: {p1:?} != 1.0");
    }

    #[test]
    fn measure_prob() {
        let size = 5;
        let mut sim = Qmdd::new(size);
        for i in 0..size {
            let p1 = sim.measure_p1(i);
            assert!((p1 - 0.0).abs() < 1e-12, "Wrong probability for qubit {i}: {p1:?} != 0.0");
        }
    }

    #[test]
    fn dump() {
        let mut sim = Qmdd::new(2);
        sim.hadamard(0, &[]);
        sim.pauli_x(1, &[0]);
        let dump = sim.dump(&[0, 1]);
        assert_eq!(dump.basis_states.len(), 2);
        assert_eq!(dump.amplitudes_real.len(), 2);
        assert_eq!(dump.amplitudes_imag.len(), 2);
        assert_eq!(dump.basis_states[0], vec![0]);
        assert_eq!(dump.basis_states[1], vec![3]);
        assert!((dump.amplitudes_real[0] - 0.7071067811865475).abs() < 1e-12);
        assert!((dump.amplitudes_real[1] - 0.7071067811865475).abs() < 1e-12);
        assert!((dump.amplitudes_imag[0] - 0.0).abs() < 1e-12);
        assert!((dump.amplitudes_imag[1] - 0.0).abs() < 1e-12);
    }

    #[test]
    fn big_hadamard() {
        let start = std::time::Instant::now();
        let size = 2500;
        let mut sim = Qmdd::new(size);
        for i in 0..size {
            sim.hadamard(i, &[]);
        }
        let end = std::time::Instant::now();
        println!("Time taken to simulate {} H gates: {:?}", size, end-start);
        // let state = sim.reconstruct_state();
        // let expected_amplitude = 1.0 / (2.0_f64).powi(size as i32 / 2);
        // for amp in state {
        //     assert!((amp.re() - expected_amplitude).abs() < 1e-12);
        //     assert!((amp.im()).abs() < 1e-12);
        // }
    }

    fn grover_algorithm(sim: &mut Qmdd, size: usize, marked_state: usize) {
        let qubits: Vec<usize> = (0..size).collect();
        let controls_0_to_n_minus_1: Vec<usize> = (0..size - 1).collect();

        // 1. Initial superposition across all qubits
        for &q in &qubits {
            sim.hadamard(q, &[]);
        }

        // 2. Grover iterations: floor((pi / 4) * sqrt(2^size))
        let iterations = ((std::f64::consts::PI / 4.0) * (2.0_f64.powi(size as i32)).sqrt()) as usize;

        for _ in 0..iterations {
            // --- Oracle: flip phase of `marked_state` ---
            for (i, &q) in qubits.iter().enumerate() {
                if ((marked_state >> (size - 1 - i)) & 1) == 0 {
                    sim.pauli_x(q, &[]);
                }
            }
            // Multi-controlled Z on the last qubit, controlled by all preceding qubits.
            sim.hadamard(size-1, &[]);
            sim.pauli_x(size-1, &controls_0_to_n_minus_1);
            sim.hadamard(size-1, &[]);
            // Uncompute X gates
            for (i, &q) in qubits.iter().enumerate() {
                if ((marked_state >> (size-1 - i)) & 1) == 0 {
                    sim.pauli_x(q, &[]);
                }
            }

            // --- Diffusion operator: invert about the mean ---
            for &q in &qubits {
                sim.hadamard(q, &[]);
                sim.pauli_x(q, &[]);
            }
            // Multi-controlled Z on the last qubit, controlled by all preceding qubits.
            sim.hadamard(size-1, &[]);
            sim.pauli_x(size-1, &controls_0_to_n_minus_1);
            sim.hadamard(size-1, &[]);
            // Uncompute X and H gates
            for &q in &qubits {
                sim.pauli_x(q, &[]);
                sim.hadamard(q, &[]);
            }
        }
    }

    #[test]
    fn grover() {
        let size = 7;
        let mut sim = Qmdd::new(size);
        let marked_state: usize = 17;

        grover_algorithm(&mut sim, size, marked_state);
        let result = measure_all_qubits(&mut sim);
        let state = sim.reconstruct_state();
        println!("{:?}", state);
        assert_eq!(result, marked_state as u64, "Manual measurement failed. Expected: {marked_state}, got: {result}");
        let mut highest_state = 0;
        let mut highest_amplitude = state[0].mag2();
        for (i, s) in state.iter().enumerate() {
            if s.mag2() > highest_amplitude {
                highest_amplitude = s.mag2();
                highest_state = i;
            }
        }

        assert_eq!(highest_state, marked_state, "State reconstruction failed. Expected: {marked_state}, got: {highest_state}");
    }
}
