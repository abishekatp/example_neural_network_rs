use crate::dnn::{Sigmoid, Tanh, DNN};
use convolutions_rs::convolutions::*;
use convolutions_rs::Padding;
use ndarray::Array2;
use ndarray::{Array1, Array3, Array4};
use ndarray_rand::{rand_distr::Uniform, RandomExt};

#[derive(Debug, Clone)]
pub enum Activation {
    Tanh,
    ReLu,
    //Softmax and Sigmoid are used only in output layer.
    Softmax,
    Sigmoid,
}

#[derive(Debug, Clone)]
pub struct HyperPar {
    // filter dimension of a layer. usualy odd number dimension(f,f).
    pub filter_dim: usize,
    // no of such filters you want for the current layer
    pub channel_out: usize,
    // number outer of layers we want to add to the input.
    pub padding: usize,
    // when we do convolution number of steps we want to jump is strides.
    pub stride: usize,
    // activation function type for current layer.
    pub activation: Activation,
}

#[derive(Debug)]
pub struct Layer {
    // this struct contains all the hyper parameters for each layer
    hyper_parameters: HyperPar,
    // weights contains learning weight parameters for the current layer.
    weights: Array4<f64>,
    // bias of the current layer.
    bias: Array1<f64>,
    // no of channels in the input examples
    channel_in: usize,
}
#[derive(Debug)]
pub struct CNN {
    // number of layers in the network(hidden and output).
    no_of_layers: usize,
    // this vector contains the convoulution network layer information
    cnn_layers: Vec<Layer>,
    // this conatians the normal neural network layer information that will be appended at the end.
    fully_connected_layers: DNN,
    // learning_rate field stores the learning rate hyper parameter of the model.
    learning_rate: f64,
}

impl CNN {
    //input_shape is (no of channels in input, input no of featurs, input no of examples)
    pub fn new(
        hyper_parameters: Vec<HyperPar>,
        fully_connected_layers: DNN,
        learning_rate: f64,
        input_shape: (usize, usize, usize),
    ) -> CNN {
        let mut layers = vec![];
        let mut channel_in = input_shape.0;
        let input_rows = input_shape.1;
        let input_cols = input_shape.2;
        for h in hyper_parameters {
            // Kernel has shape (channels out, channels in, filter height, filter width)
            // channels in is the no of channels in the output of previous layer
            // channels out is the no of channels ouputted by current layer. it is same as no of filters in the current layer.
            let weights = Array4::random(
                (h.channel_out, channel_in, h.filter_dim, h.filter_dim),
                Uniform::new(0.01, 0.09),
            );
            // let ouput_shape = (input_rows - h.filter_dim + 1, input_cols - h.filter_dim + 1,h.channels_out);
            // but convolution_rs accepts one dimensional array.
            let bias = Array1::zeros(h.channel_out);
            layers.push(Layer {
                hyper_parameters: h.clone(),
                channel_in,
                weights,
                bias,
            });
            channel_in = h.channel_out;
        }
        CNN {
            no_of_layers: layers.len(),
            cnn_layers: layers,
            learning_rate,
            fully_connected_layers,
        }
    }

    // pub fn train(&mut self, input_sample: Array4<f64>, output: Array4<f64>) -> Option<f64> {
    //     todo!()
    // }

    // //evaluate returns output value for given input
    // pub fn evaluate(&mut self, input_sample: Array4<f64>) -> Array4<f64> {
    //     todo!()
    // }

    // input_sample will be vector of arrays. where each array is of dimension 3 (channels,filter height, filter width)
    // ouput is of dimension (no of input examples, no of units on the ouput layer)
    fn forward_propagate(&mut self, input_sample: Vec<Array3<f64>>, output: Array2<f64>) {
        let no_examples = input_sample.len();
        let mut cnn_input = input_sample;
        let mut new_input = vec![];
        for l in &self.cnn_layers {
            for i in cnn_input {
                // this crate take care of convolution multiplication.
                let conv_layer = ConvolutionLayer::new(
                    l.weights.clone(),
                    Some(l.bias.clone()),
                    l.hyper_parameters.stride,
                    Padding::Valid,
                );
                let output = conv_layer.convolve(&i);
                new_input.push(output);
            }
            cnn_input = new_input;
            new_input = vec![];
        }
        let no_features = cnn_input[0].len();
        // now we flatten the final ouput of the convolutional neural network
        // and then pass it to the fully connected neural network layers.
        let mut flatten = Array2::zeros((no_features, no_examples));
        let mut j = 0;
        for inp in cnn_input {
            let mut i = 0;
            for elem in inp.iter() {
                flatten[[i, j]] = elem.clone();
                i += 1;
            }
            j += 1;
        }
        let mut dnn = DNN::new(
            no_features,
            vec![50, 60, 1],
            vec![Tanh, Tanh, Sigmoid],
            self.learning_rate,
        );
        let cache = dnn.forward_propagate(flatten);
    }

    fn log_loss(&self, pred: Array4<f64>, output: Array4<f64>) -> Option<f64> {
        todo!()
    }

    fn backward_propagate(
        &mut self,
        input_sample: Array4<f64>,
        output: Array4<f64>,
        cache: Vec<Array4<f64>>,
    ) {
        todo!()
    }
}
