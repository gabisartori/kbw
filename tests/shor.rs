use num_traits::Pow;

#[test]
fn shor_21() {
    std::env::set_var("KBW_DUMP_TYPE", "probability");
    let n = 5;

    let p = ket::Process::new_ptr();
    let mut features = ket::Features::all();
    features.register_plugin("pown");

    p.borrow_mut().set_features(features);

    let reg1 = ket::Quant::new(&p, n).unwrap();
    ket::h(&reg1).unwrap();

    let reg2 = ket::Quant::new(&p, n).unwrap();
    ket::x(&reg2.at(n - 1)).unwrap();

    ket::plugin("pown", "5 21", &ket::Quant::cat(&[&reg1, &reg2]).unwrap()).unwrap();

    let d = ket::dump(&ket::Quant::cat(&[&reg1, &reg2]).unwrap()).unwrap();

    p.borrow_mut().prepare_for_execution().unwrap();

    kbw::run_and_set_result::<kbw::Dense>(&mut p.borrow_mut()).unwrap();
    for state in d.value().as_ref().unwrap().basis_states() {
        println!("{} {}", state[0] >> n, state[0] & (2.pow(n) - 1) as u64)
    }
    println!(
        "{}",
        d.value()
            .as_ref()
            .unwrap()
            .probabilities()
            .unwrap()
            .iter()
            .sum::<f64>()
    );
}
