use std::{cell::RefCell, rc::Rc};

use crate::{
    network::{
        activations::Sigmoid,
        cnn::{self, CNN},
        convolution::{activations, Convolution},
        dense::Dense,
        dnn::DNN,
    },
    utils::read_csv,
};

pub fn mnist_conv() {
    let (x_train, y_train) = read_csv("./archive/mnist_train.csv", 100, vec![1.0], true);
    let network_dnn: Vec<DNN> = vec![
        DNN::Dense(Dense::new(2, 3)),
        DNN::Sigmoid(Sigmoid::new()),
        DNN::Dense(Dense::new(3, 1)),
        DNN::Sigmoid(Sigmoid::new()),
    ];
    let network_cnn: Vec<CNN> = vec![
        CNN::Convolution(Convolution::new((1, 28, 28), 3, 5)),
        CNN::Sigmoid(activations::Sigmoid::new()),
    ];
    let network_dnn = Rc::new(RefCell::new(network_dnn));
    let network_cnn = Rc::new(RefCell::new(network_cnn));
    for i in 0..x_train.ncols() {
        let x_col = x_train.column(i);
        let x_col = x_col
            .into_shape((1, 28, 28))
            .expect("Expecting (1,28,28) array")
            .to_owned();
        let y_col = y_train.column(i);
        let y_col = y_col
            .into_shape((1, 1))
            .expect("Expecting (1,1) label")
            .to_owned();

        let network_dnn_clone = Rc::clone(&network_dnn);
        let network_cnn_clone = Rc::clone(&network_cnn);
        cnn::train(
            network_cnn_clone,
            network_dnn_clone,
            cnn::LossType::MSE,
            x_col,
            y_col,
            1000,
            0.01,
        )
    }
}
