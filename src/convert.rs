// SPDX-FileCopyrightText: 2020 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
// SPDX-FileCopyrightText: 2020 Rafael de Santiago <r.santiago@ufsc.br>
//
// SPDX-License-Identifier: Apache-2.0

use ket::execution::DumpData;
use num::complex::{Complex64, ComplexFloat};
use rand::{distributions::WeightedIndex, prelude::Distribution, Rng};
use rayon::prelude::*;
use std::collections::HashMap;

pub(crate) struct DumpProbability {
    pub(crate) basis_states: Vec<Vec<u64>>,
    pub(crate) probabilities: Vec<f64>,
}

pub(crate) fn from_dump_to_prob(data: DumpData) -> DumpProbability {
    let probabilities = data
        .amplitudes_real
        .par_iter()
        .zip(&data.amplitudes_imag)
        .map(|(real, imag)| Complex64::new(*real, *imag).abs().powf(2.0))
        .collect();

    DumpProbability {
        basis_states: data.basis_states,
        probabilities,
    }
}

pub(crate) fn from_prob_to_shots<R: Rng>(
    data: DumpProbability,
    shots: usize,
    rng: &mut R,
) -> (Vec<u64>, Vec<u64>) {
    let mut count_map = HashMap::new();
    let dist = WeightedIndex::new(data.probabilities).unwrap();

    (0..shots).for_each(|_| {
        count_map
            .entry(&data.basis_states[dist.sample(rng)])
            .and_modify(|c| *c += 1)
            .or_insert(1u64);
    });

    count_map
        .drain()
        .map(|(state, count)| (state[0], count))
        .unzip()
}
