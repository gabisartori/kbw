use crate::{error::KBWError, run, run_and_set_result, Dense, QuantumExecution, Sparse};
use ket::Process;
use num::FromPrimitive;
use num_derive::{FromPrimitive, ToPrimitive};

use self::error::wrapper;

mod error {
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

#[derive(Debug, FromPrimitive, ToPrimitive)]
pub enum SimMode {
    Dense,
    Sparse,
}

#[no_mangle]
pub extern "C" fn kbw_run_and_set_result(process: &mut Process, sim_mode: i32) -> i32 {
    match SimMode::from_i32(sim_mode) {
        Some(sim_mode) => match sim_mode {
            SimMode::Dense => wrapper(run_and_set_result::<Dense>(process)),
            SimMode::Sparse => wrapper(run_and_set_result::<Sparse>(process)),
        },
        None => KBWError::UndefinedSimMode.error_code(),
    }
}

#[no_mangle]
pub unsafe extern "C" fn kbw_run_serialized(
    quantum_code: *const u8,
    quantum_code_size: usize,
    metrics: *const u8,
    metrics_size: usize,
    data_type: i32,
    sim_mode: i32,
    result: &mut *mut Vec<u8>,
) -> i32 {
    let quantum_code = unsafe { std::slice::from_raw_parts(quantum_code, quantum_code_size) };
    let metrics = unsafe { std::slice::from_raw_parts(metrics, metrics_size) };

    let data_type: ket::serialize::DataType = match ket::serialize::DataType::from_i32(data_type) {
        Some(data_type) => data_type,
        None => return KBWError::UndefinedDataType.error_code(),
    };

    let quantum_code: Vec<ket::code_block::CodeBlock> = match data_type {
        ket::serialize::DataType::JSON => serde_json::from_slice(quantum_code).unwrap(),
        ket::serialize::DataType::BIN => bincode::deserialize(quantum_code).unwrap(),
    };

    let quantum_code: Vec<&ket::code_block::CodeBlock> = quantum_code.iter().collect();

    let metrics: ket::ir::Metrics = match data_type {
        ket::serialize::DataType::JSON => serde_json::from_slice(metrics).unwrap(),
        ket::serialize::DataType::BIN => bincode::deserialize(metrics).unwrap(),
    };

    let result_data = match SimMode::from_i32(sim_mode) {
        Some(sim_mode) => match sim_mode {
            SimMode::Dense => {
                let mut sim = match Dense::new(&metrics) {
                    Ok(sim) => sim,
                    Err(error) => return error.error_code(),
                };
                match run(&mut sim, &quantum_code, &metrics) {
                    Ok(result_data) => result_data,
                    Err(error) => return error.error_code(),
                }
            }
            SimMode::Sparse => {
                let mut sim = match Sparse::new(&metrics) {
                    Ok(sim) => sim,
                    Err(error) => return error.error_code(),
                };
                match run(&mut sim, &quantum_code, &metrics) {
                    Ok(result_data) => result_data,
                    Err(error) => return error.error_code(),
                }
            }
        },
        None => return KBWError::UndefinedSimMode.error_code(),
    };

    let result_data: Vec<u8> = match data_type {
        ket::serialize::DataType::JSON => serde_json::to_vec(&result_data).unwrap(),
        ket::serialize::DataType::BIN => bincode::serialize(&result_data).unwrap(),
    };

    *result = Box::into_raw(Box::new(result_data));

    KBWError::Success.error_code()
}

#[no_mangle]
pub extern "C" fn kbw_result_get(result: &Vec<u8>, data: &mut *const u8, size: &mut usize) -> i32 {
    *data = result.as_ptr();
    *size = result.len();
    KBWError::Success.error_code()
}

#[no_mangle]
pub unsafe extern "C" fn kbw_result_delete(result: *mut Vec<u8>) -> i32 {
    unsafe { Box::from_raw(result) };
    KBWError::Success.error_code()
}

#[cfg(test)]
mod tests {
    use super::SimMode;
    use num::FromPrimitive;

    #[test]
    fn print_sim_mode() {
        let mut sim_mode = 0;
        while let Some(mode) = SimMode::from_i32(sim_mode) {
            println!("#define KBW_{:#?} {}", mode, sim_mode);
            sim_mode += 1;
        }
    }
}
