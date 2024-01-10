// SPDX-FileCopyrightText: 2020 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
// SPDX-FileCopyrightText: 2020 Rafael de Santiago <r.santiago@ufsc.br>
//
// SPDX-License-Identifier: Apache-2.0

use std::{error::Error, fmt::Display, result};

#[derive(Debug, Clone, Copy)]
#[repr(i32)]
pub enum KBWError {
    Success,
    Timeout,
    OutOfQubits,
    UnsupportedNumberOfQubits,
    NotReadyForExecution,
    UndefinedDataType,
    UndefinedSimMode,
    UndefinedError,
}

pub type Result<T> = result::Result<T, KBWError>;

impl KBWError {
    pub fn to_str(&self) -> &'static str {
        match self {
            KBWError::Success => "The function call completed successfully.",
            KBWError::UndefinedError => "An undefined error occurred.",
            KBWError::Timeout => "The quantum execution has timed out.",
            KBWError::OutOfQubits => "Cannot allocate more qubits. Ensure you are not deallocating too many qubits as dirty.",
            KBWError::UnsupportedNumberOfQubits => "The number of requested qubits is not supported.",
            KBWError::NotReadyForExecution => "The process is not yet ready for execution.",
            KBWError::UndefinedSimMode => "The simulation mode is undefined.",
            KBWError::UndefinedDataType => "The data type is undefined.",
        }
    }

    pub fn error_code(&self) -> i32 {
        *self as i32
    }

    pub fn from_error_code(error_code: i32) -> KBWError {
        unsafe { std::mem::transmute(error_code) }
    }
}

impl Display for KBWError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_str())
    }
}

impl Error for KBWError {}

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
            let error = KBWError::from_error_code(error_code);
            println!("#define KBW_{:#?} {}", error, error_code);

            if let KBWError::UndefinedError = error {
                break;
            } else {
                error_code += 1;
            }
        }
    }
}
