// SPDX-FileCopyrightText: 2020 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
// SPDX-FileCopyrightText: 2020 Rafael de Santiago <r.santiago@ufsc.br>
//
// SPDX-License-Identifier: Apache-2.0

//! C-compatible ABI for embedding KBW in foreign-language runtimes.
//!
//! This module exports two `extern "C"` symbols:
//!
//! * [`kbw_make_configuration`]: create a [`QPUConfig`] for a named simulator
//!   and write a heap-allocated pointer to the caller-supplied out-parameter.
//! * [`kbw_build_info`]: return a static string describing the build
//!   (crate name, version, compiler, target triple).
//!
//! Callers are responsible for freeing the `QPUConfig` pointer returned by
//! `kbw_make_configuration` via the Ket runtime's own cleanup path.

use std::ffi::{c_char, CStr};

use crate::{Dense, DenseGPUBlock, DenseV2, Sparse, SparseV2};
use ket::{error::KetError, process::QPUConfig};

use crate::quantum_execution::QubitManager;

/// Create a [`QPUConfig`] for the requested simulator and write the heap
/// pointer into `*qpu_config`.
///
/// # Parameters
///
/// * `num_qubits`: number of qubits to allocate.
/// * `simulator`: NUL-terminated C string naming the backend.  Recognised
///   values (case-insensitive, whitespace ignored): `"dense"`, `"densev2"`,
///   `"densegpu"`, `"sparse"`, `"sparsev2"`.
/// * `use_live`: `true` for live (gate-by-gate) mode; `false` for batch mode.
/// * `gradient`: enable the parameter-shift-rule gradient strategy (batch
///   mode only).
/// * `coupling_graph_json`: JSON-encoded `Vec<[usize; 2]>` describing allowed
///   two-qubit pairs, or a JSON `null` for all-to-all connectivity.
/// * `qpu_config`: out-parameter; set to the newly allocated config on
///   success.
///
/// # Returns
///
/// A `KetError` integer code: `0` on success, non-zero on failure.
///
/// # Safety
///
/// * `simulator` must be a valid, NUL-terminated C string.
/// * `coupling_graph_json` must be a valid, NUL-terminated C string containing
///   either a JSON array of two-element integer arrays or `null`.
/// * `qpu_config` must be a non-null, aligned, writable pointer.
#[no_mangle]
pub unsafe extern "C" fn kbw_make_configuration(
    num_qubits: usize,
    simulator: *const c_char,
    use_live: bool,
    gradient: bool,
    coupling_graph_json: *const c_char,
    qpu_config: &mut *mut QPUConfig,
) -> i32 {
    let Ok(simulator) = (unsafe { CStr::from_ptr(simulator).to_str() }) else {
        return KetError::ExecutionFailed.error_code();
    };

    let mut simulator = simulator.to_lowercase();
    simulator.retain(|c| !c.is_whitespace());

    let Ok(Ok(coupling_graph)) = unsafe { CStr::from_ptr(coupling_graph_json) }
        .to_str()
        .map(serde_json::from_str)
    else {
        return KetError::SerdeError.error_code();
    };

    let config = match simulator.as_str() {
        "dense" => make_config::<Dense>(num_qubits, use_live, coupling_graph, gradient),
        "densev2" => make_config::<DenseV2>(num_qubits, use_live, coupling_graph, gradient),
        "densegpu" => make_config::<DenseGPUBlock>(num_qubits, use_live, coupling_graph, gradient),
        "sparse" => make_config::<Sparse>(num_qubits, use_live, coupling_graph, gradient),
        "sparsev2" => make_config::<SparseV2>(num_qubits, use_live, coupling_graph, gradient),
        _ => Err(KetError::ExecutionFailed),
    };

    match config {
        Ok(config) => {
            *qpu_config = Box::into_raw(Box::new(config));
            KetError::Success.error_code()
        }
        Err(err) => err.error_code(),
    }
}

fn make_config<S>(
    num_qubits: usize,
    use_live: bool,
    coupling_graph: Option<Vec<(usize, usize)>>,
    gradient: bool,
) -> Result<QPUConfig, KetError>
where
    S: crate::quantum_execution::QuantumExecution + 'static,
    QubitManager<S>: ket::execution::LiveExecution + ket::execution::BatchExecution,
{
    if coupling_graph.is_none() && use_live {
        QubitManager::<S>::make_live_config(num_qubits)
    } else {
        QubitManager::<S>::make_batch_config(num_qubits, coupling_graph, gradient)
    }
}

const BUILD_INFO: &str = build_info::format!("{} v{} [{} {}]", $.crate_info.name, $.crate_info.version, $.compiler, $.target);

/// Return a static string describing the KBW build.
///
/// Sets `*msg` to the UTF-8 bytes of the build-info string and `*size` to
/// its byte length.  The pointer is valid for the lifetime of the process.
///
/// Returns a `KetError` integer code (always `0` = success).
#[no_mangle]
pub const extern "C" fn kbw_build_info(msg: &mut *const u8, size: &mut usize) -> i32 {
    let bytes = BUILD_INFO.as_bytes();
    *size = bytes.len();
    *msg = bytes.as_ptr();
    KetError::Success.error_code()
}
