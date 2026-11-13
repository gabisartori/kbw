// TODO: License and Document this
use crate::quantum_execution::QuantumExecution;

use ket::error::KetError;
use ket::execution::DumpData;

use std::collections::HashMap;
use std::f64::consts::FRAC_1_SQRT_2;

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

fn phase_matrix(lambda: f64, table: &mut ComplexTable) -> [ComplexId; 4] {
    let c = Complex::new(lambda.cos(), lambda.sin());
    [
        table.intern(Complex::ONE),
        table.intern(Complex::ZERO),
        table.intern(Complex::ZERO),
        table.intern(c),
    ]
}

fn rx_matrix(theta: f64, table: &mut ComplexTable) -> [ComplexId; 4] {
    let c = Complex::new((theta / 2.0).cos(), 0.0);
    let s = Complex::new(0.0, -(theta / 2.0).sin());
    [
        table.intern(c),
        table.intern(s),
        table.intern(s),
        table.intern(c),
    ]
}

fn ry_matrix(theta: f64, table: &mut ComplexTable) -> [ComplexId; 4] {
    let c = Complex::new((theta / 2.0).cos(), 0.0);
    let s = Complex::new(-(theta / 2.0).sin(), 0.0);
    [
        table.intern(c),
        table.intern(s),
        table.intern(-s),
        table.intern(c),
    ]
}

fn rz_matrix(theta: f64, table: &mut ComplexTable) -> [ComplexId; 4] {
    let c = Complex::new((theta / 2.0).cos(), (theta / 2.0).sin());
    let c_conj = Complex::new((theta / 2.0).cos(), -(theta / 2.0).sin());
    [
        table.intern(c_conj),
        table.intern(Complex::ZERO),
        table.intern(Complex::ZERO),
        table.intern(c),
    ]
}

// ============================================================================
// 1. Exact Algebraic Complex Numbers: (A + B/sqrt(2) + C*j + D/sqrt(2)*j) / E
// ============================================================================

use std::hash::{Hash, Hasher};

// ============================================================================
// 1. Hashable f64 Complex Numbers with Tolerance Support
// ============================================================================

#[derive(Debug, Clone, Copy)]
pub struct Complex {
    pub re: f64,
    pub im: f64,
}

impl Complex {
    pub const ZERO: Complex = Complex { re: 0.0, im: 0.0 };
    pub const ONE: Complex = Complex { re: 1.0, im: 0.0 };
    pub const I: Complex = Complex { re: 0.0, im: 1.0 };
    pub const NEG_I: Complex = Complex { re: 0.0, im: -1.0 };
    pub const A: Complex = Complex { re: FRAC_1_SQRT_2, im: 0.0 };
    pub const NEG_A: Complex = Complex { re: -FRAC_1_SQRT_2, im: 0.0 };
    pub const NEG_ONE: Complex = Complex { re: -1.0, im: 0.0 };

    pub fn new(re: f64, im: f64) -> Self {
        let re = if re == -0.0 { 0.0 } else { re };
        let im = if im == -0.0 { 0.0 } else { im };
        Self { re, im }
    }

    pub fn conj(&self) -> Self {
        Self::new(self.re, -self.im)
    }

    pub fn norm_sqr(&self) -> f64 {
        self.re * self.re + self.im * self.im
    }

    pub fn abs(&self) -> f64 {
        self.norm_sqr().sqrt()
    }
}

impl std::ops::Add for Complex {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Complex::new(self.re + rhs.re, self.im + rhs.im)
    }
}

impl std::ops::Sub for Complex {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Complex::new(self.re - rhs.re, self.im - rhs.im)
    }
}

impl std::ops::Neg for Complex {
    type Output = Self;
    fn neg(self) -> Self {
        Complex::new(-self.re, -self.im)
    }
}

impl std::ops::Mul for Complex {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        Complex::new(
            self.re * rhs.re - self.im * rhs.im,
            self.re * rhs.im + self.im * rhs.re,
        )
    }
}

impl std::ops::Div for Complex {
    type Output = Self;
    fn div(self, rhs: Self) -> Self {
        let denom = rhs.norm_sqr();
        Complex::new(
            (self.re * rhs.re + self.im * rhs.im) / denom,
            (self.im * rhs.re - self.re * rhs.im) / denom,
        )
    }
}

// Wrapper for hashability using bit-casting
#[derive(Debug, Clone, Copy)]
pub struct HashableComplex(pub Complex);

impl PartialEq for HashableComplex {
    fn eq(&self, other: &Self) -> bool {
        self.0.re.to_bits() == other.0.re.to_bits() &&
        self.0.im.to_bits() == other.0.im.to_bits()
    }
}

impl Eq for HashableComplex {}

impl Hash for HashableComplex {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.re.to_bits().hash(state);
        self.0.im.to_bits().hash(state);
    }
}

pub type ComplexId = usize;

pub struct ComplexTable {
    i2complex: Vec<Complex>,
    complex2i: HashMap<HashableComplex, ComplexId>,
    tolerance: f64,
    sums: HashMap<(ComplexId, ComplexId), ComplexId>,
    prods: HashMap<(ComplexId, ComplexId), ComplexId>,
    quots: HashMap<(ComplexId, ComplexId), ComplexId>,
}

impl ComplexTable {
    pub const ZERO: ComplexId = 0;
    pub const ONE: ComplexId = 1;
    pub const I: ComplexId = 2;
    pub const NEG_I: ComplexId = 3;
    pub const A: ComplexId = 4;
    pub const NEG_A: ComplexId = 5;
    pub const NEG_ONE: ComplexId = 6;

    pub fn new(tolerance: f64) -> Self {
        let mut table = Self {
            i2complex: Vec::new(),
            complex2i: HashMap::new(),
            tolerance,
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

        self.intern(Complex::ZERO);
        self.intern(Complex::ONE);
        self.intern(Complex::I);
        self.intern(Complex::NEG_I);
        self.intern(Complex::A);
        self.intern(Complex::NEG_A);
        self.intern(Complex::NEG_ONE);
    }

    /// Snaps a float value to the tolerance grid for approximate merging
    fn snap(&self, val: f64) -> f64 {
        if val == 0.0 {
            return 0.0;
        }
        (val / self.tolerance).round() * self.tolerance
    }

    pub fn intern(&mut self, c: Complex) -> ComplexId {
        let snapped = Complex::new(self.snap(c.re), self.snap(c.im));
        let wrapped = HashableComplex(snapped);

        if let Some(&id) = self.complex2i.get(&wrapped) {
            return id;
        }

        let id = self.i2complex.len();
        self.i2complex.push(snapped);
        self.complex2i.insert(wrapped, id);
        id
    }

    #[inline]
    pub fn get(&self, id: ComplexId) -> Complex {
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

        let res_val = self.i2complex[first] + self.i2complex[second];
        let res_id = self.intern(res_val);
        self.sums.insert((first, second), res_id);
        res_id
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

        let res_val = self.i2complex[first] * self.i2complex[second];
        let res_id = self.intern(res_val);
        self.prods.insert((first, second), res_id);
        res_id
    }

    pub fn div(&mut self, num: ComplexId, den: ComplexId) -> ComplexId {
        let den_val = self.i2complex[den];
        assert!(den_val.norm_sqr() > 1e-15, "Division by zero in ComplexTable::div()");
        if num == Self::ZERO || den == Self::ONE {
            return num;
        }
        if num == den {
            return Self::ONE;
        }
        if let Some(&cached) = self.quots.get(&(num, den)) {
            return cached;
        }

        let res_val = self.i2complex[num] / den_val;
        let res_id = self.intern(res_val);
        self.quots.insert((num, den), res_id);
        res_id
    }

    pub fn argmax(&self, indices: &[ComplexId]) -> ComplexId {
        assert!(!indices.is_empty(), "No values passed to argmax()");
        let mut greatest = indices[0];
        let mut max = self.i2complex[greatest].norm_sqr();

        for &idx in &indices[1..] {
            let norm = self.i2complex[idx].norm_sqr();
            if norm - max > self.tolerance {
                greatest = idx;
                max = norm;
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
            complex: ComplexTable::new(1e-12),
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
    pub fn strong_simulate(&self, decimals: u32) -> Vec<(String, f64, f64)> {
        assert!(self.state.weight != ComplexTable::ZERO, "State entry is a zero edge.");
        let mut results = Vec::new();
        let mut stack = vec![(self.state.dest, self.state.weight, String::new())];
        let pad = 10f64.powi(decimals as i32);

        while let Some((v_id, w, s)) = stack.pop() {
            let v = &self.vertices[v_id];
            if v.edges.is_terminal() {
                let c = self.complex.get(w);
                let re = (c.re * pad).round() / pad;
                let im = (c.im * pad).round() / pad;
                results.push((s, re, im));
                continue;
            }

            let edges = v.edges.as_slice();
            for i in [1usize, 0usize] {
                if edges[i].weight != ComplexTable::ZERO {
                    let bit = if i == 0 { '0' } else { '1' };
                    let mut next_s = String::with_capacity(s.len() + 1);
                    next_s.push(bit);
                    next_s.push_str(&s);
                    let next_w = self.complex_mul_readonly(w, edges[i].weight);
                    stack.push((edges[i].dest, next_w, next_s));
                }
            }
        }
        results
    }

    fn reconstruct_state(&mut self) -> Vec<Complex> {
        let dim = 1 << self.num_vars;
        let mut output = vec![Complex::new(0.0, 0.0); dim];
                
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

    // Helper for readonly traversal when `strong_simulate` doesn't hold `&mut self`
    fn complex_mul_readonly(&self, a: ComplexId, b: ComplexId) -> ComplexId {
        if a == ComplexTable::ZERO || b == ComplexTable::ONE {
            return a;
        }
        if b == ComplexTable::ZERO || a == ComplexTable::ONE {
            return b;
        }
        let key = if a <= b { (a, b) } else { (b, a) };
        if let Some(&id) = self.complex.prods.get(&key) {
            return id;
        }
        let lhs = self.complex.get(a);
        let rhs = self.complex.get(b);
        let val = HashableComplex(Complex::new(
            lhs.re * rhs.re - lhs.im * rhs.im,
            lhs.re * rhs.im + lhs.im * rhs.re,
        ));
        self.complex.complex2i[&val]
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

    fn normalize(&mut self, edges: &mut Edges) -> (ComplexId, f64) {
        let slice = edges.as_slice();
        if slice.is_empty() {
            return (ComplexTable::ONE, 1.0);
        }

        let weights: Vec<ComplexId> = slice.iter().map(|e| e.weight).collect();
        let factor = self.complex.argmax(&weights);

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

        let prob = match edges {
            Edges::Vector(arr) => {
                let p0 = self.vertices[arr[0].dest].prob * self.complex.get(arr[0].weight).norm_sqr();
                let p1 = self.vertices[arr[1].dest].prob * self.complex.get(arr[1].weight).norm_sqr();
                p0 + p1
            }
            _ => 1.0,
        };

        (factor, prob)
    }
    
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

    fn is_redundant(edges: &Edges) -> bool {
        if let Edges::Matrix(e) = edges {
            e.iter()
                .all(|edge| edge.dest == e[0].dest && edge.weight == e[0].weight)
        } else {
            false
        }
    }

    fn create_vertex(&mut self, variable: usize, mut edges: Edges) -> Edge {
        let (weight, prob) = self.normalize(&mut edges);

        let factor_val = self.complex.get(weight);
        if factor_val.norm_sqr() < 1e-15 {
            return Edge {
                dest: self.terminal,
                weight: ComplexTable::ZERO,
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
            e = self.create_vertex(variable, Edges::Vector([e, zero_edge]));
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

        let res = self.create_vertex(level, edges);
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

        let e = self.create_vertex(level, Edges::Vector(edges));
        let tmp = self.complex.mul(e.weight, matrix.weight);
        let res = Edge {
            dest: e.dest,
            weight: self.complex.mul(tmp, vector.weight),
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
                        self.create_vertex(ctrl_idx, Edges::Matrix(reusable));
                }
            }
            q += 1;
        }

        // Target vertex
        let mut entry = self.create_vertex(target, Edges::Matrix(root));

        // Controls above target (index < target), all-if-true (activator = 3, antiactivator = 0)
        while q < ctrls.len() {
            let ctrl_idx = ctrls[q];
            let mut reusable = [zero_edge; 4];
            reusable[3] = entry;
            reusable[0] = one_edge;
            entry = self.create_vertex(ctrl_idx, Edges::Matrix(reusable));
            q += 1;
        }

        entry
    }

    fn renormalize(&mut self, p: f64) {
        // self.state.weight = self.complex.mul_f64(self.state.weight, p);
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
        let m = phase_matrix(lambda, &mut self.complex);
        self.apply_gate(m, target, control);
    }

    fn rx(&mut self, theta: f64, target: usize, control: &[usize]) {
        let m = rx_matrix(theta, &mut self.complex);
        self.apply_gate(m, target, control);
    }

    fn ry(&mut self, theta: f64, target: usize, control: &[usize]) {
        let m = ry_matrix(theta, &mut self.complex);
        self.apply_gate(m, target, control);
    }

    fn rz(&mut self, theta: f64, target: usize, control: &[usize]) {
        let m = rz_matrix(theta, &mut self.complex);
        self.apply_gate(m, target, control);
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
        self.apply_gate(matrix, target, &[]);
        // TODO: Normalize weights after measurement collapse.
        self.renormalize(p);
        // self.root.weight = self.root.weight * Complex::<f64>::new(p, 0.0);
    }

    fn dump(&mut self, qubits: &[usize]) -> DumpData {
        let output = self.reconstruct_state();
        let mut basis_states = Vec::new();
        let mut amplitudes_real = Vec::new();
        let mut amplitudes_imag = Vec::new();

        for (state, amp) in output.iter().enumerate() {
            // let prob = amp.re * amp.re + amp.im * amp.im;
            let prob = 0.0;
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
        println!("H1 state: {:?}", state);
    }

    #[test]
    fn double_qubit_h_test() {
        let mut sim = Qmdd::new(2);
        sim.hadamard(0, &[]);
        sim.hadamard(1, &[]);
        let state = sim.reconstruct_state();
        println!("H2 state: {:?}", state);
    }

    #[test]
    fn bell_test() {
        let mut sim = Qmdd::new(2);
        sim.hadamard(0, &[]);
        sim.pauli_x(1, &[0]);
        let state = sim.reconstruct_state();
        println!("Bell state: {:?}", state);
    }

    #[test]
    fn measure_collapse_hadamard() {
        let mut sim = Qmdd::new(1);
        sim.hadamard(0, &[]);
        println!("Before measurement: {:?}", sim.reconstruct_state());
        let p1 = sim.measure_p1(0);
        let p = 1.0/p1.sqrt();
        sim.measure_collapse(0, true, p);
        let p1 = sim.measure_p1(0);
        println!("After measurement: {:?}", sim.reconstruct_state());
    }

    #[test]
    fn measure_collapse_bell() {
        let mut sim = Qmdd::new(2);
        sim.hadamard(0, &[]);
        sim.pauli_x(1, &[0]);
        let p1 = sim.measure_p1(0);
        let p = 1.0/p1.sqrt();
        // assert!((p1 - 0.5).abs() < 1e-12, "unexpected probability: {p1:?} != 0.5");
        println!("Before measurement: {:?}", sim.reconstruct_state());
        sim.measure_collapse(0, true, p);
        let p1 = sim.measure_p1(1);
        // assert!((p1 - 1.0).abs() < 1e-12, "unexpected probability: {p1:?} != 1.0");
        println!("After measurement: {:?}", sim.reconstruct_state());
    }
}
