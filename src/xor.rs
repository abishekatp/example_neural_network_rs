use std::cell::RefCell;
use std::rc::Rc;

use crate::network::activations::Sigmoid;
use crate::network::dnn::{predict, train};
use crate::network::{
    activations::Tanh,
    dense::Dense,
    dnn::{LossType, DNN},
};
use ndarray::Array2;

// note: in case if you want to use Tanh activation at the ouput layer you have to use Tanh error method
pub fn _xor() {
    let x_train = Array2::from_shape_vec((2, 4), vec![0., 0., 1., 1., 0., 1., 0., 1.]).unwrap();
    let y_train = Array2::from_shape_vec((1, 4), vec![0., 1., 1., 0.]).unwrap();
    let network: Vec<DNN> = vec![
        DNN::Dense(Dense::new(2, 3)),
        DNN::Tanh(Tanh::new()),
        DNN::Dense(Dense::new(3, 1)),
        DNN::Tanh(Tanh::new()),
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

// note: in case if you want to use Sigmoid activation at the output layer you have to use BinaryCrossEntrophy error method
pub fn _xor_sigmoid() {
    let x_train = Array2::from_shape_vec((2, 4), vec![0., 0., 1., 1., 0., 1., 0., 1.]).unwrap();
    let y_train = Array2::from_shape_vec((1, 4), vec![0., 1., 1., 0.]).unwrap();
    let network: Vec<DNN> = vec![
        DNN::Dense(Dense::new(2, 3)),
        DNN::Tanh(Tanh::new()),
        DNN::Dense(Dense::new(3, 1)),
        DNN::Sigmoid(Sigmoid::new()),
    ];
    let network = Rc::new(RefCell::new(network));
    let network_clone = Rc::clone(&network);
    train(
        network_clone,
        LossType::BinaryCrossEntrophy,
        x_train.clone(),
        y_train,
        10000,
        0.02,
    );
    let network_clone = Rc::clone(&network);
    let output = predict(network_clone, x_train);
    println!("Output: {}", output);
}
