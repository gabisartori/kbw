// SPDX-FileCopyrightText: 2020 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
// SPDX-FileCopyrightText: 2020 Rafael de Santiago <r.santiago@ufsc.br>
//
// SPDX-License-Identifier: Apache-2.0

use env_logger::Builder;
use log::LevelFilter;

use crate::{dense::Dense, error::KBWError, quantum_execution::QubitManager, sparse::Sparse};

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

    #[no_mangle]
    pub extern "C" fn kbw_error_message(error_code: i32, size: &mut usize) -> *const u8 {
        let msg = KBWError::from_error_code(error_code).to_str();
        *size = msg.len();
        msg.as_ptr()
    }

    pub fn wrapper(error: Result<()>) -> i32 {
        match error {
            Ok(_) => KBWError::Success.error_code(),
            Err(error) => error.error_code(),
        }
    }
}

#[no_mangle]
pub extern "C" fn kbw_make_configuration(
    num_qubits: usize,
    live: bool,
    use_sparse: bool,
    result: &mut *mut ket::Configuration,
) -> i32 {
    if use_sparse {
        *result = Box::into_raw(Box::new(QubitManager::<Sparse>::configuration(
            num_qubits, live,
        )));
    } else {
        *result = Box::into_raw(Box::new(QubitManager::<Dense>::configuration(
            num_qubits, live,
        )));
    }

    KBWError::Success.error_code()
}
