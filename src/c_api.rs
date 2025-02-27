// SPDX-FileCopyrightText: 2020 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
// SPDX-FileCopyrightText: 2020 Rafael de Santiago <r.santiago@ufsc.br>
//
// SPDX-License-Identifier: Apache-2.0

use crate::{error::KBWError, DenseSimulator, DenseV2Simulator, SparseSimulator};
use env_logger::Builder;
use ket::execution::*;
use log::LevelFilter;

#[no_mangle]
pub extern "C" fn kbw_set_log_level(level: u32) -> i32 {
    let level = match level {
        0 => LevelFilter::Off,
        1 => LevelFilter::Error,
        2 => LevelFilter::Warn,
        3 => LevelFilter::Info,
        4 => LevelFilter::Debug,
        5 => LevelFilter::Trace,
        _ => LevelFilter::max(),
    };

    Builder::new().filter_level(level).init();

    KBWError::Success.error_code()
}

pub mod error {
    use crate::error::{KBWError, Result};

    /// Returns the error message for the given error code.
    ///
    /// # Safety
    ///
    /// This functions is unsafe because it assumes that the error code is valid.
    #[no_mangle]
    pub unsafe extern "C" fn kbw_error_message(
        error_code: i32,
        buffer: *mut u8,
        buffer_size: usize,
        write_size: &mut usize,
    ) -> i32 {
        let msg = unsafe { KBWError::from_error_code(error_code) }.to_string();
        let msg = msg.as_bytes();
        *write_size = msg.len();
        if buffer_size >= *write_size {
            let buffer = unsafe { std::slice::from_raw_parts_mut(buffer, buffer_size) };
            buffer[..*write_size].copy_from_slice(msg);
            0
        } else {
            1
        }
    }

    pub fn wrapper(error: Result<()>) -> i32 {
        match error {
            Ok(_) => KBWError::Success.error_code(),
            Err(error) => error.error_code(),
        }
    }
}

/// # Safety
#[no_mangle]
pub unsafe extern "C" fn kbw_make_configuration(
    num_qubits: usize,
    simulator: i32,
    use_live: bool,
    coupling_graph: *const (usize, usize),
    coupling_graph_size: usize,
    gradient: bool,
    result: &mut *mut Configuration,
) -> i32 {
    let coupling_graph = if coupling_graph.is_null() {
        None
    } else {
        let coupling_graph =
            unsafe { std::slice::from_raw_parts(coupling_graph, coupling_graph_size) };
        Some(coupling_graph.to_vec())
    };

    *result = Box::into_raw(Box::new(match simulator {
        0 => DenseSimulator::configuration(
            num_qubits,
            use_live,
            coupling_graph,
            if gradient {
                GradientStatus::ParameterShift
            } else {
                GradientStatus::Disable
            },
        ),
        1 => SparseSimulator::configuration(
            num_qubits,
            use_live,
            coupling_graph,
            if gradient {
                GradientStatus::ParameterShift
            } else {
                GradientStatus::Disable
            },
        ),
        2 => DenseV2Simulator::configuration(
            num_qubits,
            use_live,
            coupling_graph,
            if gradient {
                GradientStatus::ParameterShift
            } else {
                GradientStatus::Disable
            },
        ),
        _ => return KBWError::InvalidSimulator.error_code(),
    }));

    KBWError::Success.error_code()
}

const BUILD_INFO: &str = build_info::format!("{} v{} [{} {}]", $.crate_info.name, $.crate_info.version, $.compiler, $.target);

#[no_mangle]
pub extern "C" fn kbw_build_info(msg: &mut *const u8, size: &mut usize) -> i32 {
    let bytes = BUILD_INFO.as_bytes();
    *size = bytes.len();
    *msg = bytes.as_ptr();
    KBWError::Success.error_code()
}
