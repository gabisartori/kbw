// SPDX-FileCopyrightText: 2020 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
// SPDX-FileCopyrightText: 2020 Rafael de Santiago <r.santiago@ufsc.br>
//
// SPDX-License-Identifier: Apache-2.0

use num::Integer;

pub(crate) fn bit_flip(state: usize, index: usize) -> usize {
    state ^ (1 << index)
}

pub(crate) fn bit_flip_vec(mut state: Vec<u64>, index: usize) -> Vec<u64> {
    let (outer_index, inner_index) = index.div_mod_floor(&64);
    state[outer_index] = bit_flip(state[outer_index] as usize, inner_index) as u64;
    state
}

pub(crate) fn is_one_at(state: usize, target: usize) -> bool {
    state & (1 << target) != 0
}

pub(crate) fn is_one_at_vec(state: &[u64], target: usize) -> bool {
    let (outer_index, inner_index) = target.div_mod_floor(&64);
    state[outer_index] & (1 << inner_index) != 0
}

pub(crate) fn ctrl_check(state: usize, control: &[usize]) -> bool {
    control.iter().all(|control| is_one_at(state, *control))
}

pub(crate) fn ctrl_check_vec(state: &[u64], control: &[usize]) -> bool {
    control.iter().all(|control| is_one_at_vec(state, *control))
}
