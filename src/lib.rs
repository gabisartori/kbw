// SPDX-FileCopyrightText: 2020 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
// SPDX-FileCopyrightText: 2020 Rafael de Santiago <r.santiago@ufsc.br>
//
// SPDX-License-Identifier: Apache-2.0

pub mod bitwise;
pub mod c_api;
pub mod convert;
pub mod dense;
pub mod dense_v2;
pub mod error;
pub mod quantum_execution;
pub mod sparse;

pub type DenseSimulator = quantum_execution::QubitManager<dense::Dense>;
pub type DenseV2Simulator = quantum_execution::QubitManager<dense::Dense>;
pub type SparseSimulator = quantum_execution::QubitManager<sparse::Sparse>;
