use std::cell::RefCell;
use std::rc::Rc;

use crate::network::{activations::Tanh, dense::Dense, Layer, LossType};
use crate::network::{predict, train};
use ndarray::Array2;

pub fn xor() {
    let x_train = Array2::from_shape_vec((2, 4), vec![0., 0., 1., 1., 0., 1., 0., 1.]).unwrap();
    let y_train = Array2::from_shape_vec((1, 4), vec![0., 1., 1., 0.]).unwrap();
    let network: Vec<Layer> = vec![
        Layer::Dense(Dense::new(2, 3)),
        Layer::Tanh(Tanh::new()),
        Layer::Dense(Dense::new(3, 1)),
        Layer::Tanh(Tanh::new()),
    ];
    let network = Rc::new(RefCell::new(network));
    let network_clone = Rc::clone(&network);
    train(
        network_clone,
        LossType::MSE,
        x_train.clone(),
        y_train,
        10000,
        0.01,
    );
    let network_clone = Rc::clone(&network);
    let output = predict(network_clone, x_train);
    println!("Output: {}", output);
}
