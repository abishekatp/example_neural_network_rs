use ndarray::{Array2, Axis};
use ndarray_rand::{rand_distr::Uniform, RandomExt};

#[derive(Debug)]
pub enum Activation {
    Tanh,
    ReLu,
    //Softmax and Sigmoid are used only in output layer.
    Softmax,
    Sigmoid,
}
//todo: Adam optimisation for dw and db.
//todo: Batch Normalization.
//todo: Softmax regression.
#[derive(Debug)]
pub struct CNN {
    // number of layers in the network(hidden and output).
    layers: usize,
    // number of units in each layer.
    units: Vec<usize>,
    // weights contains learning weight parameters for each layer of the network.
    // dimension of each weight is (no of units in a current layer l,no of units of previous layer(l-1) that is going to be input to this layer).
    weights: Vec<Array2<f64>>,
    // biases contains bias for each layer of the network.
    // dimension of each bias is (no of units in a current layer).
    biases: Vec<Array2<f64>>,
    // this vector contains activation function type for each layer.
    activations: Vec<Activation>,
    // learning_rate field stores the learning rate hyper parameter of the model.
    learning_rate: f64,
    // this variable stores no of input feature
    no_of_input_features: usize,
}

impl CNN {
    pub fn new(
        no_of_input_features: usize,
        units: Vec<usize>,
        activations: Vec<Activation>,
        learning_rate: f64,
    ) -> CNN {
        todo!()
    }

    pub fn train(&mut self, input_sample: Array2<f64>, output: Array2<f64>) -> Option<f64> {
        todo!()
    }

    //evaluate returns output value for given input
    pub fn evaluate(&mut self, input_sample: Array2<f64>) -> Array2<f64> {
        todo!()
    }

    fn forward_propagate(&mut self, input_sample: Array2<f64>) -> Vec<Array2<f64>> {
        todo!()
    }

    fn log_loss(&self, pred: Array2<f64>, output: Array2<f64>) -> Option<f64> {
        todo!()
    }

    fn backward_propagate(
        &mut self,
        input_sample: Array2<f64>,
        output: Array2<f64>,
        cache: Vec<Array2<f64>>,
    ) {
        todo!()
    }
}
