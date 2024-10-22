// SPDX-FileCopyrightText: 2020 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
// SPDX-FileCopyrightText: 2020 Rafael de Santiago <r.santiago@ufsc.br>
//
// SPDX-License-Identifier: Apache-2.0

use std::result;

/// Enumeration of possible errors in the KBW simulator.
#[derive(thiserror::Error, Debug, Clone, Copy)]
#[repr(i32)]
pub enum KBWError {
    #[error("The function call completed successfully.")]
    Success,

    #[error("An undefined error occurred.")]
    UndefinedError,

    #[error("The quantum execution has timed out.")]
    Timeout,

    #[error(
        "Cannot allocate more qubits. Ensure you are not deallocating too many qubits as dirty."
    )]
    OutOfQubits,

    #[error("The number of requested qubits is not supported.")]
    UnsupportedNumberOfQubits,

    #[error("The process is not yet ready for execution.")]
    NotReadyForExecution,

    #[error("The simulation mode is undefined.")]
    UndefinedSimMode,

    #[error("The data type is undefined.")]
    UndefinedDataType,

    #[error("The simulator is undefined.")]
    InvalidSimulator,
}

/// Result type for KBW library functions.
pub type Result<T> = result::Result<T, KBWError>;

impl KBWError {
    /// Returns the error code as an integer.
    pub fn error_code(&self) -> i32 {
        *self as i32
    }

    /// Converts an error code into a KBWError.
    ///
    /// # Safety
    ///
    /// This function is unsafe because it assumes that the error code is valid.
    pub unsafe fn from_error_code(error_code: i32) -> KBWError {
        unsafe { std::mem::transmute(error_code) }
    }
}

#[cfg(test)]
mod tests {
    use super::KBWError;

    #[test]
    fn success_is_zero() {
        assert!(KBWError::Success.error_code() == 0)
    }

    #[test]
    fn print_error_code() {
        let mut error_code = 0;
        loop {
            let error = unsafe { KBWError::from_error_code(error_code) };
            println!("#define KBW_{:#?} {}", error, error_code);

            if let KBWError::UndefinedError = error {
                break;
            } else {
                error_code += 1;
            }
        }
    }
}
