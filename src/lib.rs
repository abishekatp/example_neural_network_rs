mod network;

use std::f64::consts::PI;

use ndarray::{array, Array2};
use network::{
    Activation::{Sigmoid, Tanh},
    DNN,
};

// to test our implementation of the model we can use the sin() function
// input_sample will have only one feature which we will give to the sin() function as input.
// sample output will be what actual value we will get from sin() function.
pub fn sin_function_prediction() {
    let mut inputs = Array2::zeros((1, 700));
    let mut outputs = Array2::zeros((1, 700));
    let mut i = 1;
    while i < 700 {
        let val = i as f64 * 0.01;
        inputs[[0, i]] = val;
        outputs[[0, i]] = val.sin();
        i += 1;
    }
    let mut dnn = DNN::new(1, vec![3, 1], vec![Tanh, Sigmoid], 0.1);
    dnn.train(inputs, outputs);

    let inputs = array![[PI / 2.0, PI]];
    dnn.evaluate(inputs);
}
