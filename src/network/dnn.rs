use std::{cell::RefCell, rc::Rc};

use crate::network::activations::{ReLu, Sigmoid, Softmax, Tanh};
use crate::network::dense::Dense;
use crate::network::losses::{binary_cross_entropy, binary_cross_entropy_grad, mse, mse_grad};
use ndarray::Array2;

#[derive(Clone)]
pub enum DNN {
    Dense(Dense),
    Tanh(Tanh),
    Sigmoid(Sigmoid),
    Softmax(Softmax),
    ReLu(ReLu),
}

pub fn predict(network: Rc<RefCell<Vec<DNN>>>, input: Array2<f64>) -> Array2<f64> {
    let mut network = network.borrow_mut();
    let mut output = input;
    for layer in network.iter_mut() {
        match layer {
            DNN::Dense(dense) => {
                output = dense.forward(output);
            }
            DNN::Tanh(tanh) => {
                output = tanh.forward(output);
            }
            DNN::Sigmoid(sig) => {
                output = sig.forward(output);
            }
            DNN::Softmax(s) => {
                output = s.forward(output);
            }
            DNN::ReLu(r) => {
                output = r.forward(output);
            }
        }
    }
    return output;
}

fn grad(
    network: Rc<RefCell<Vec<DNN>>>,
    output_grad: Array2<f64>,
    learning_rate: f64,
) -> Array2<f64> {
    let mut network = network.borrow_mut();
    let mut grad = output_grad;
    let mut current_layer = network.len() - 1;
    loop {
        let layer = network.get_mut(current_layer).unwrap();
        match layer {
            DNN::Dense(dense) => {
                grad = dense.backward(grad, learning_rate);
            }
            DNN::Tanh(tanh) => {
                grad = tanh.backward(grad);
            }
            DNN::Sigmoid(sig) => {
                grad = sig.backward(grad);
            }
            DNN::Softmax(s) => {
                grad = s.backward(grad);
            }
            DNN::ReLu(r) => {
                grad = r.backward(grad);
            }
        }
        if current_layer == 0 {
            break;
        }
        current_layer -= 1;
    }
    grad
}

pub enum LossType {
    MSE,
    BinaryCrossEntrophy,
}

pub fn train(
    network: Rc<RefCell<Vec<DNN>>>,
    loss_type: LossType,
    x_train: Array2<f64>,
    y_train: Array2<f64>,
    epochs: usize,
    learning_rate: f64,
) -> Array2<f64> {
    let mut gradient: Array2<f64> = Array2::zeros(x_train.dim());
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
        gradient = grad(network_clone, y_pred, learning_rate);
    }
    gradient
}
