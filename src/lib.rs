use ndarray::Array2;
use ndarray_rand::{rand_distr::Uniform, RandomExt};

#[derive(Debug)]
pub enum Activation {
    Sigmoid,
    Tanh,
    ReLu,
}

#[derive(Debug)]
pub struct DNN {
    //no of layers in the network(hidden and output).
    layers: usize,
    // no of units in each layer.
    units: Vec<usize>,
    //weights contains weights for each layer of the network.
    //dimension of each weight is (no of units in a current layer,no of units of previous layer that is going to be input to this layer).
    weights: Vec<Array2<f64>>,
    //biases contains bias for each layer of the network.
    //dimension of each bias is (no of units in a current layer).
    biases: Vec<Array2<f64>>,
    // this vector contains activation function type for each layer.
    activations: Vec<Activation>,
}

impl DNN {
    pub fn new(
        no_of_input_features: usize,
        units: Vec<usize>,
        activations: Vec<Activation>,
    ) -> DNN {
        let layers = units.len();
        let mut weights: Vec<Array2<f64>> = vec![];
        let mut biases: Vec<Array2<f64>> = vec![];
        let mut pre_units = no_of_input_features;
        let mut units_iter = units.iter();
        while let Some(cur_units) = units_iter.next() {
            // we have to assign random values to weight of the each unit of the each layer.
            // if we assign 0 to all. then the result will be like we used single layer with single unit.
            // Also all the units in all the layers will learn the same thing instead of learning diffent features of the input sample.
            // we need to assign small random values so that when we use sigmoid or tanh activation function slope will not be nearly zero.
            // for large values when we calculate derivative slope will be zero for sigmoid and tanh functions.
            let rand_arr = Array2::random((*cur_units, pre_units), Uniform::new(0.001, 0.010));
            weights.push(rand_arr);
            // we can assign all 0's to the biases because it will not create the above problem.
            // we create the column vector to follow the convention.
            let zeros = Array2::zeros((*cur_units, 1));
            biases.push(zeros);

            pre_units = *cur_units;
        }
        DNN {
            layers,
            units,
            weights,
            biases,
            activations,
        }
    }
    // input_sample should have no of rows equal to the no of features passed to the new() function.
    // no of columns can be any number
    // input_sample dimension (no of features in input, no of examples in the input)
    // no of input features should match the no of units of previous layer in first hidden layer.
    pub fn forward_propagate(&self, input_sample: Array2<f64>) {
        let mut input_sample = input_sample;
        for i in 0..self.layers {
            let w = self
                .weights
                .get(i)
                .expect("Expecting weights for netowrk layer");
            //dimension(no of units in current layer,no of units in the previous layer) x dimension(no of input features,no of examples in the input)
            let z = w.dot(&input_sample);
        }
    }
}
