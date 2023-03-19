use std::{cell::RefCell, rc::Rc};

use crate::network::convolution::activations::{Sigmoid, Tanh};
use crate::network::convolution::Convolution;
use crate::network::dnn;
use crate::network::reshape::Reshape;
use ndarray::{Array2, Array3};

#[derive(Clone)]
pub enum CNN {
    Convolution(Convolution),
    Tanh(Tanh),
    Sigmoid(Sigmoid),
}

pub fn predict(network: Rc<RefCell<Vec<CNN>>>, input: Array3<f64>) -> Array3<f64> {
    let mut network = network.borrow_mut();
    let mut output = input;

    for layer in network.iter_mut() {
        match layer {
            CNN::Convolution(c) => {
                output = c.forward(output);
            }
            CNN::Tanh(tanh) => {
                output = tanh.forward(output);
            }
            CNN::Sigmoid(sig) => {
                output = sig.forward(output);
            }
        }
    }
    return output;
}

fn grad(network: Rc<RefCell<Vec<CNN>>>, output_grad: Array3<f64>, learning_rate: f64) {
    let mut network = network.borrow_mut();
    let mut grad = output_grad;
    let mut current_layer = network.len() - 1;
    loop {
        let layer = network.get_mut(current_layer).unwrap();
        match layer {
            CNN::Convolution(c) => {
                grad = c.backward(grad, learning_rate);
            }
            CNN::Tanh(tanh) => {
                grad = tanh.backward(grad);
            }
            CNN::Sigmoid(sig) => {
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
    network_cnn: Rc<RefCell<Vec<CNN>>>,
    network_dnn: Rc<RefCell<Vec<dnn::DNN>>>,
    loss_type: LossType,
    x_train: Array3<f64>,
    y_train: Array2<f64>,
    epochs: usize,
    learning_rate: f64,
) {
    for e in 0..epochs {
        let network_clone = Rc::clone(&network_cnn);
        let y_pred = predict(network_clone, x_train.clone());

        //dnn layers
        let reshape_layer = Reshape::new(y_pred.dim(), (y_pred.len(), 1));
        let reshaped_out = reshape_layer.forward(y_pred.clone());
        let network_clone = Rc::clone(&network_dnn);
        let output_grad = dnn::train(
            network_clone,
            dnn::LossType::MSE,
            reshaped_out,
            y_train.clone(),
            1,
            learning_rate,
        );
        let reshaped_grad = reshape_layer.backward(output_grad);

        let network_clone = Rc::clone(&network_cnn);
        grad(network_clone, reshaped_grad, learning_rate);
    }
}
