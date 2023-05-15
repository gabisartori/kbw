use std::collections::HashMap;

use num::complex::{Complex64, ComplexFloat};
use rand::{distributions::WeightedIndex, prelude::Distribution, Rng};
use rayon::prelude::*;

pub fn from_dump_vec_to_dump_prob(data: ket::DumpData) -> ket::DumpData {
    match data {
        ket::DumpData::Vector {
            basis_states,
            amplitudes_real,
            amplitudes_imag,
        } => {
            let probabilities = amplitudes_real
                .par_iter()
                .zip(amplitudes_imag)
                .map(|(real, imag)| Complex64::new(*real, imag).abs().powf(2.0))
                .collect();

            ket::DumpData::Probability {
                basis_states,
                probabilities,
            }
        }
        ket::DumpData::Probability { .. } => panic!(),
        ket::DumpData::Shots { .. } => panic!(),
    }
}

pub fn from_dump_prob_to_dump_shots<R: Rng>(
    data: ket::DumpData,
    shots: u64,
    rng: &mut R,
) -> ket::DumpData {
    match data {
        ket::DumpData::Vector { .. } => panic!(),
        ket::DumpData::Probability {
            basis_states,
            probabilities,
        } => {
            let mut count_map = HashMap::new();
            let dist = WeightedIndex::new(probabilities).unwrap();

            (0..shots).for_each(|_| {
                count_map
                    .entry(&basis_states[dist.sample(rng)])
                    .and_modify(|c| *c += 1)
                    .or_insert(1u32);
            });

            let mut basis_states = Vec::new();
            let mut count = Vec::new();

            count_map.drain().for_each(|(key, val)| {
                basis_states.push(key.clone());
                count.push(val)
            });

            ket::DumpData::Shots {
                basis_states,
                count,
                total: shots,
            }
        }
        ket::DumpData::Shots { .. } => panic!(),
    }
}
