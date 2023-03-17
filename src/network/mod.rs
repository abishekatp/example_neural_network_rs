pub mod activations;
pub mod convolution;
pub mod dense;
pub mod losses;

use std::{cell::RefCell, rc::Rc};

use crate::network::activations::{Sigmoid, Tanh};
use crate::network::dense::Dense;
use crate::network::losses::{binary_cross_entropy, binary_cross_entropy_grad, mse, mse_grad};
use ndarray::Array2;

#[derive(Clone)]
pub enum Layer {
    Dense(Dense),
    Tanh(Tanh),
    Sigmoid(Sigmoid),
}

pub fn predict(network: Rc<RefCell<Vec<Layer>>>, input: Array2<f64>) -> Array2<f64> {
    let mut network = network.borrow_mut();
    let mut output = input;
    for layer in network.iter_mut() {
        match layer {
            Layer::Dense(dense) => {
                output = dense.forward(output);
            }
            Layer::Tanh(tanh) => {
                output = tanh.forward(output);
            }
            Layer::Sigmoid(sig) => {
                output = sig.forward(output);
            }
        }
    }
    return output;
}

fn grad(network: Rc<RefCell<Vec<Layer>>>, output_grad: Array2<f64>, learning_rate: f64) {
    let mut network = network.borrow_mut();
    let mut grad = output_grad;
    let mut current_layer = network.len() - 1;
    loop {
        let layer = network.get_mut(current_layer).unwrap();
        match layer {
            Layer::Dense(dense) => {
                grad = dense.backward(grad, learning_rate);
            }
            Layer::Tanh(tanh) => {
                grad = tanh.backward(grad);
            }
            Layer::Sigmoid(sig) => {
                grad = sig.backward(grad);
            }
        }
        if current_layer == 0 {
            break;
        }
        current_layer -= 1;
    }
}

pub enum LossType {
    MSE,
    BinaryCrossEntrophy,
}

pub fn train(
    network: Rc<RefCell<Vec<Layer>>>,
    loss_type: LossType,
    x_train: Array2<f64>,
    y_train: Array2<f64>,
    epochs: usize,
    learning_rate: f64,
) {
    for e in 0..epochs {
        let network_clone = Rc::clone(&network);
        let mut y_pred = predict(network_clone, x_train.clone());
        match loss_type {
            LossType::MSE => {
                let err = mse(y_pred.clone(), y_train.clone());
                y_pred = mse_grad(y_pred, y_train.clone());
                println!("epoch: {} MSE: {}", e, err);
            }
            LossType::BinaryCrossEntrophy => {
                let err = binary_cross_entropy(y_pred.clone(), y_train.clone());
                y_pred = binary_cross_entropy_grad(y_pred, y_train.clone());
                println!("epoch: {} BinaryCrossEntrophy: {}", e, err);
            }
        }

        let network_clone = Rc::clone(&network);
        grad(network_clone, y_pred, learning_rate);
    }
}
