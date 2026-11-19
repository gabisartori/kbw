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

// ============================================================================
// 1. Exact Algebraic Complex Numbers: (A + B/sqrt(2) + C*j + D/sqrt(2)*j) / E
// ============================================================================

fn gcd_two(mut a: i128, mut b: i128) -> i128 {
    a = a.abs();
    b = b.abs();
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a
}

fn gcd_five(a: i128, b: i128, c: i128, d: i128, e: i128) -> i128 {
    gcd_two(gcd_two(gcd_two(gcd_two(a, b), c), d), e)
}

#[derive(Debug, Clone, Copy, Hash)]
pub struct ComplexVal {
    pub a: i128,
    pub b: i128,
    pub c: i128,
    pub d: i128,
    pub e: i128,
}

impl ComplexVal {
    /// Creates a canonicalized complex value where gcd(a,b,c,d,e) == 1 and e > 0.
    pub fn new(a: i128, b: i128, c: i128, d: i128, e: i128) -> Self {
        assert!(e != 0, "Error in ComplexVal::new(): Division by zero.");
        let g = gcd_five(a, b, c, d, e);
        let sign = if e < 0 { -1 } else { 1 };
        let f = g * sign;
        Self {
            a: a / f,
            b: b / f,
            c: c / f,
            d: d / f,
            e: e / f,
        }
    }

    pub fn re(&self) -> f64 {
        (self.a as f64 + (self.b as f64) * FRAC_1_SQRT_2) / (self.e as f64)
    }

    pub fn im(&self) -> f64 {
        (self.c as f64 + (self.d as f64) * FRAC_1_SQRT_2) / (self.e as f64)
    }

    pub fn mag2(&self) -> f64 {
        let r = self.re();
        let i = self.im();
        r * r + i * i
    }

    pub fn from_float(x: f64, real: bool) -> Self {
        let p = (x * 1e12).round() as i128;
        let q = 1e12 as i128;
        if real {
            Self { a: p, b: 0, c: 0, d: 0, e: q }
        } else {
            Self { a: 0, b: 0, c: p, d: 0, e: q }
        }
    }
}

impl std::ops::Add for ComplexVal {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        let a = self.e * other.a + other.e * self.a;
        let b = self.e * other.b + other.e * self.b;
        let c = self.e * other.c + other.e * self.c;
        let d = self.e * other.d + other.e * self.d;
        let e = self.e * other.e;
        Self::new(a, b, c, d, e)
    }
}

impl std::ops::Sub for ComplexVal {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        let a = self.e * other.a - other.e * self.a;
        let b = self.e * other.b - other.e * self.b;
        let c = self.e * other.c - other.e * self.c;
        let d = self.e * other.d - other.e * self.d;
        let e = self.e * other.e;
        Self::new(a, b, c, d, e)
    }
}

impl std::ops::Mul for ComplexVal {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        let a = 2 * self.a * other.a + other.b * self.b - 2 * self.c * other.c - self.d * other.d;
        let b = 2 * (self.a * other.b + other.a * self.b - self.c * other.d - other.c * self.d);
        let c = 2 * self.a * other.c + 2 * other.a * self.c + self.b * other.d + other.b * self.d;
        let d = 2 * (self.a * other.d + other.a * self.d + self.b * other.c + other.b * self.c);
        let e = 2 * self.e * other.e;
        Self::new(a, b, c, d, e)
    }
}

impl std::cmp::PartialEq for ComplexVal {
    fn eq(&self, other: &Self) -> bool {
        (*self - *other).mag2() <= 1e-10
    }
}

impl std::cmp::Eq for ComplexVal {}

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

        self.intern(0, 0, 0, 0, 1);  // 0: ZERO
        self.intern(1, 0, 0, 0, 1);  // 1: ONE
        self.intern(0, 1, 0, 0, 1);  // 2: A
        self.intern(-1, 0, 0, 0, 1); // 3: NEG_ONE
        self.intern(0, 0, 1, 0, 1);  // 4: I
        self.intern(0, 0, -1, 0, 1); // 5: NEG_I
        self.intern(0, -1, 0, 0, 1); // 6: NEG_A
        self.intern(0, 1, 0, 1, 1);  // 7: B
        self.intern(0, 1, 0, -1, 1); // 8: C
    }

    pub fn from_float(&mut self, x: f64, real: bool) -> ComplexId {
        let x = ComplexVal::from_float(x, real);
        self.intern(x.a, x.b, x.c, x.d, x.e)
    }

    pub fn intern(&mut self, a: i128, b: i128, c: i128, d: i128, e: i128) -> ComplexId {
        let val = ComplexVal::new(a, b, c, d, e);
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

        let res = self.intern(
            lhs.e * rhs.a + rhs.e * lhs.a,
            lhs.e * rhs.b + rhs.e * lhs.b,
            lhs.e * rhs.c + rhs.e * lhs.c,
            lhs.e * rhs.d + rhs.e * lhs.d,
            lhs.e * rhs.e,
        );

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

        let res = self.intern(
            2 * lhs.a * rhs.a + rhs.b * lhs.b - 2 * lhs.c * rhs.c - lhs.d * rhs.d,
            2 * (lhs.a * rhs.b + rhs.a * lhs.b - lhs.c * rhs.d - rhs.c * lhs.d),
            2 * lhs.a * rhs.c + 2 * rhs.a * lhs.c + lhs.b * rhs.d + rhs.b * lhs.d,
            2 * (lhs.a * rhs.d + rhs.a * lhs.d + lhs.b * rhs.c + rhs.b * lhs.c),
            2 * lhs.e * rhs.e,
        );

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

        let temp1 = 2 * (rhs.a * rhs.a + rhs.c * rhs.c) + rhs.b * rhs.b + rhs.d * rhs.d;
        let temp2 = rhs.a * rhs.b + rhs.c * rhs.d;
        let temp3 = 2 * (lhs.a * rhs.a + lhs.c * rhs.c) + lhs.b * rhs.b + lhs.d * rhs.d;
        let temp4 = 2 * (lhs.a * rhs.b + lhs.b * rhs.a + lhs.c * rhs.d + lhs.d * rhs.d);
        let temp5 = 2 * (lhs.c * rhs.a - lhs.a * rhs.c) + lhs.d * rhs.b - lhs.b * rhs.d;
        let temp6 = 2 * (lhs.c * rhs.b + lhs.d * rhs.a - lhs.a * rhs.d - lhs.b * rhs.c);

        let res = self.intern(
            rhs.e * (temp1 * temp3 - 2 * temp2 * temp4),
            rhs.e * (temp1 * temp4 - 4 * temp2 * temp3),
            rhs.e * (temp1 * temp5 - 2 * temp2 * temp6),
            rhs.e * (temp1 * temp6 - 4 * temp2 * temp5),
            lhs.e * (temp1 * temp1 - 8 * temp2 * temp2),
        );

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
    pub fn strong_simulate(&self, decimals: u32) -> Vec<(String, f64, f64)> {
        assert!(self.state.weight != ComplexTable::ZERO, "State entry is a zero edge.");
        let mut results = Vec::new();
        let mut stack = vec![(self.state.dest, self.state.weight, String::new())];
        let pad = 10f64.powi(decimals as i32);

        while let Some((v_id, w, s)) = stack.pop() {
            let v = &self.vertices[v_id];
            if v.edges.is_terminal() {
                let c = self.complex.get(w);
                let re = (c.re() * pad).round() / pad;
                let im = (c.im() * pad).round() / pad;
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

    fn reconstruct_state(&mut self) -> Vec<ComplexVal> {
        let dim = 1 << self.num_vars;
        let mut output = vec![ComplexVal::new(0, 0, 0, 0, 1); dim];
                
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
        let val = ComplexVal::new(
            2 * lhs.a * rhs.a + rhs.b * lhs.b - 2 * lhs.c * rhs.c - lhs.d * rhs.d,
            2 * (lhs.a * rhs.b + rhs.a * lhs.b - lhs.c * rhs.d - rhs.c * lhs.d),
            2 * lhs.a * rhs.c + 2 * rhs.a * lhs.c + lhs.b * rhs.d + rhs.b * lhs.d,
            2 * (lhs.a * rhs.d + rhs.a * lhs.d + lhs.b * rhs.c + rhs.b * lhs.c),
            2 * lhs.e * rhs.e,
        );
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

    fn normalize(&mut self, edges: &mut Edges, rule: NormRule) -> (ComplexId, f64) {
        let slice = edges.as_slice();
        if slice.is_empty() {
            return (ComplexTable::ONE, 1.0);
        }

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
                let p0 = self.vertices[arr[0].dest].prob * self.complex.get(arr[0].weight).mag2();
                let p1 = self.vertices[arr[1].dest].prob * self.complex.get(arr[1].weight).mag2();
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

    fn calc_prob_recursive(
        &self,
        v_id: VertexId,
        weight_id: ComplexId,
        target: usize,
        memo: &mut HashMap<(VertexId, ComplexId), f64>,
    ) -> f64 {
        let v = &self.vertices[v_id];

        // If we hit the terminal or have gone past the target level down the tree, no contribution from this branch
        if v.edges.is_terminal() || v.variable < target {
            return 0.0;
        }

        let cache_key = (v_id, weight_id);
        if let Some(&prob) = memo.get(&cache_key) {
            return prob;
        }

        let mut total_prob = 0.0;
        let edges = v.edges.as_slice();

        // If this vertex corresponds exactly to our target qubit
        if v.variable == target {
            // Edge 1 corresponds to measuring |1> for this qubit
            let edge1 = &edges[1];
            if edge1.weight != ComplexTable::ZERO {
                // Combine the incoming weight with the edge weight
                let combined_weight = self.complex_mul_readonly(weight_id, edge1.weight);
                let c = self.complex.get(combined_weight);
                let path_prob = c.mag2() * self.vertices[edge1.dest].prob; // Multiplied by sub-diagram probability
                
                // Alternatively, if sub-diagrams are fully normalized, you can accumulate:
                // total_prob += c.norm_sqr();
                // Let's use the direct path probability calculation:
                total_prob += path_prob;
            }
        } else {
            // If we are above the target qubit level, we must branch down both paths (0 and 1)
            for i in 0..2 {
                let edge = &edges[i];
                if edge.weight != ComplexTable::ZERO {
                    let next_weight = self.complex_mul_readonly(weight_id, edge.weight);
                    total_prob += self.calc_prob_recursive(edge.dest, next_weight, target, memo);
                }
            }
        }

        memo.insert(cache_key, total_prob);
        total_prob
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

    // TODO: Implement Rz as a phase gate, and Rx/Ry as a combination of H and Rz.
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
        
        self.calc_prob_recursive(self.state.dest, self.state.weight, target, &mut memo)
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

    #[test]
    fn single_qubit_h_test() {
        let mut sim = Qmdd::new(1);
        sim.hadamard(0, &[]);
        let state = sim.reconstruct_state();
        let expected = vec![
            ComplexVal::new(0, 1, 0, 0, 1),
            ComplexVal::new(0, 1, 0, 0, 1)
        ];
        assert_eq!(state, expected);
    }

    #[test]
    fn double_qubit_h_test() {
        let mut sim = Qmdd::new(2);
        sim.hadamard(0, &[]);
        sim.hadamard(1, &[]);
        let state = sim.reconstruct_state();
        let expected = vec![
            ComplexVal::new(1, 0, 0, 0, 2), // |00> amplitude (1/2)
            ComplexVal::new(1, 0, 0, 0, 2), // |01> amplitude (1/2)
            ComplexVal::new(1, 0, 0, 0, 2), // |10> amplitude (1/2)
            ComplexVal::new(1, 0, 0, 0, 2), // |11> amplitude (1/2)
        ];
        assert_eq!(state, expected);
    }

    #[test]
    fn bell_test() {
        let mut sim = Qmdd::new(2);
        sim.hadamard(0, &[]);
        sim.pauli_x(1, &[0]);
        let state = sim.reconstruct_state();
        let expected = vec![
            ComplexVal::new(0, 1, 0, 0, 1), // |00> amplitude (sqrt(2)/2)
            ComplexVal::new(0, 0, 0, 0, 1), // |01> amplitude (0)
            ComplexVal::new(0, 0, 0, 0, 1), // |10> amplitude (0)
            ComplexVal::new(0, 1, 0, 0, 1), // |11> amplitude (sqrt(2)/2)
        ];
        assert_eq!(state, expected);
    }

    #[test]
    fn measure_collapse_hadamard() {
        let mut sim = Qmdd::new(1);
        sim.hadamard(0, &[]);
        let state = sim.reconstruct_state();
        let expected = vec![
            ComplexVal::new(0, 1, 0, 0, 1), // |0> amplitude (sqrt(2)/2)
            ComplexVal::new(0, 1, 0, 0, 1), // |1> amplitude (sqrt(2)/2)
        ];
        assert_eq!(state, expected);

        let p1 = sim.measure_p1(0);
        assert!((p1 - 0.5).abs() < 1e-12, "Wrong probability before measure: {p1:?} != 0.5");
        let p = 1.0/p1.sqrt();
        sim.measure_collapse(0, true, p);
        let p1 = sim.measure_p1(0);
        let state = sim.reconstruct_state();
        let expected = vec![
            ComplexVal::new(0, 0, 0, 0, 1), // |0> amplitude (1)
            ComplexVal::new(1, 0, 0, 0, 1), // |1> amplitude (0)
        ];
        assert_eq!(state, expected);
        assert!((p1 - 1.0).abs() < 1e-12, "Wrong probability after measure: {p1:?} != 1.0");
    }

    #[test]
    fn measure_collapse_bell() {
        let mut sim = Qmdd::new(2);
        sim.hadamard(0, &[]);
        sim.pauli_x(1, &[0]);

        let state = sim.reconstruct_state();
        let expected = vec![
            ComplexVal::new(0, 1, 0, 0, 1), // |00> amplitude (sqrt(2)/2)
            ComplexVal::new(0, 0, 0, 0, 1), // |01> amplitude (0)
            ComplexVal::new(0, 0, 0, 0, 1), // |10> amplitude (0)
            ComplexVal::new(0, 1, 0, 0, 1), // |11> amplitude (sqrt(2)/2)
        ];
        let p1 = sim.measure_p1(0);
        assert_eq!(state, expected);
        assert!((p1 - 0.5).abs() < 1e-12, "Wrong probability before measure: {p1:?} != 0.5");

        // Collapse the superposition to |11>
        let p = 1.0/p1.sqrt();
        sim.measure_collapse(0, true, p);
        let state = sim.reconstruct_state();
        let expected = vec![
            ComplexVal::new(0, 0, 0, 0, 1), // |00> amplitude (0)
            ComplexVal::new(0, 0, 0, 0, 1), // |01> amplitude (0)
            ComplexVal::new(0, 0, 0, 0, 1), // |10> amplitude (0)
            ComplexVal::new(1, 0, 0, 0, 1), // |11> amplitude (1)
        ];
        let p1 = sim.measure_p1(0);
        assert_eq!(state, expected);
        assert!((p1 - 1.0).abs() < 1e-12, "Wrong probability after measure: {p1:?} != 1.0");
    }
}
