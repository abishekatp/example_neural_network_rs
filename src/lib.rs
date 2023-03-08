use std::ops::Add;

use ndarray::{Array2, Axis};
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
    // learning_rate field stores the learning rate hyper parameter of the model.
    learning_rate: f64,
}

impl DNN {
    pub fn new(
        no_of_input_features: usize,
        units: Vec<usize>,
        activations: Vec<Activation>,
        learning_rate: f64,
    ) -> DNN {
        let layers = units.len();
        if layers < 2 {
            panic!("Neural network should contain at least two layers")
        }
        let mut weights: Vec<Array2<f64>> = vec![];
        let mut biases: Vec<Array2<f64>> = vec![];
        let mut pre_units = no_of_input_features;
        let mut units_iter = units.iter();
        while let Some(cur_units) = units_iter.next() {
            // we have to assign random values to weights of the each unit of each layer of the network.
            // if we assign 0 to all. then the result will be equivalent to using single layer with single unit.
            // This is happening because all the units in all the layers will learn the same thing instead of learning diffent features of the input sample when we assign 0 to all.
            // we need to assign small random values so that when we use sigmoid or tanh activation function slope will not be nearly zero.
            // for large values of weights when we calculate derivative slope will be zero for sigmoid and tanh functions.
            // becasue these functions are flat line for large x axis values.
            let rand_arr = Array2::random((*cur_units, pre_units), Uniform::new(0.001, 0.010));
            weights.push(rand_arr);
            // we can assign all 0's to the biases because it will not create any of the above problems.
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
            learning_rate,
        }
    }

    // input_sample should have nuber of rows equal to the no_of_input_features passed to the new() function.
    // no of columns can be any number
    // output should contain the actual output of the final layer
    // output matrix should have number of rows = no of units in output layer.
    // output matrix should have number of columns = no columns in the input_sample.
    // Each column of output matrix corresponds to ouput of each column in the input_sample.
    pub fn train(&mut self, input_sample: Array2<f64>, output: Array2<f64>) {
        let cache = self.forward_propagate(input_sample.clone());
        self.backward_propagate(input_sample, output, cache);
    }

    // forward_propagate function will propagate through each layer of the network one by one
    // and store the predicted output of each layer in the the cache.
    // input_sample dimension (no of features in the input, no of examples in the input)
    fn forward_propagate(&mut self, input_sample: Array2<f64>) -> Vec<Array2<f64>> {
        //getting no of columns in the input sample
        // this ncols return the lenght of Axis(1)
        let no_of_examples = input_sample.ncols();
        let mut pre_out = input_sample;
        // cache will store the predicted outputs of each layere
        let mut cache: Vec<Array2<f64>> = vec![];
        dbg!(&no_of_examples);
        for i in 0..self.layers {
            let w = self
                .weights
                .get(i)
                .expect("Expecting weights for the current netowrk layer");
            let bias = self
                .biases
                .get(i)
                .expect("Expecting biases for the current network layer");
            let units = self
                .units
                .get(i)
                .expect("Expecting units of the current layer");
            // Z^[l] = W^[l] A^[l-1] + B^[l]
            // dimension of W^[l] is (no of units of layer l, no of units in layer l-1)
            // dimension of A^[l-1] is (no of units of layer l-1, no of input examples)
            // dimension of Z^[l] is (no of units of layer l,no of input examples)
            let z = w.dot(&pre_out);
            let bias = bias
                .broadcast((*units, no_of_examples))
                .expect("Expecting the broadcasted array");
            let mut z = z.add(bias);

            //apply the activation function to the output of the current layer
            let acti = self
                .activations
                .get(i)
                .expect("Expecting activation function type for current layer");
            match acti {
                Activation::Tanh => {
                    z.mapv_inplace(|val| val.tanh());
                }
                Activation::Sigmoid => {
                    z.mapv_inplace(|val| sigmoid(val));
                }
                Activation::ReLu => {
                    panic!("ReLu not supported yet");
                }
            }

            cache.push(z.clone());
            pre_out = z;
        }
        println!("{:#?}", &cache);
        cache
    }

    // Backward Propagate
    // backward_propagate will propagate throught each layer of the network from the last to the first.
    // In the process it will comput he dw[i],db[i] for each layer i and update their weights weights[i] and bias biases[i] based on the learning rate alpha.
    fn backward_propagate(
        &mut self,
        input_sample: Array2<f64>,
        output: Array2<f64>,
        cache: Vec<Array2<f64>>,
    ) {
        // getting no of columns in the input sample
        // this ncols return the lenght of Axis(1)
        let no_of_examples = input_sample.ncols();
        let mut updated_weights: Vec<Array2<f64>> = vec![];
        let mut update_biases: Vec<Array2<f64>> = vec![];
        //for output layer: as of now assuming output layer uses sigmoid activation function
        let cur_layer_output = cache
            .get(self.layers - 1)
            .expect("Expecting predicted output of the last layer");
        let prev_layer_out = cache
            .get(self.layers - 2)
            .expect("Expecting predicted output of the layer before layer")
            .clone();
        // if g(z) = (1/1+e^-z). for sigmoid function derivative with respect to z will be g'(z) = (1/1+e^-z)(1- (1/1+e^-z))=g(z)(1-g(z)).
        // note the g(Z^[l]) is the output of the layer l.
        // dL/dz = (dL/dg)(dg/dz), if g(z)=a. then dL/dz = (dL/da)(da/dz).
        // For logistic regression L(a,Y) = -y*log(a) - (1-y)log(1-a).
        // if you find these derivatives and substitute in dL/dz you will get dz = dL/dz = (dL/dg)(dg/dz) = a - y where a is predicted output and y is actual output.
        let mut dz = cur_layer_output.clone() - output;
        // we got dw = dL/dw = (dL/dz)(dz/dw) = (dz)(A^[l-1])^T
        // here * is broadcasted to all the elements of the matrix. for getting average we multiply by (1/m)
        // here (dz x A) dimenstions are (no of units in output layer,no of input examples)x(no of examples, no units in the previous layer)
        // the matrix multiplication will result in (no of units in ouput layer,no of units in previous layer)
        let dw = (dz.dot(&prev_layer_out.reversed_axes())) * (1.0 / no_of_examples as f64);
        // db is of dimension (no of units in current layer,1)
        let db = dz.sum_axis(Axis(1));

        // updating the learning parameters for output layer.
        let w = self
            .weights
            .get(self.layers - 1)
            .expect("Expecting mutable weights");
        let b = self
            .biases
            .get(self.layers - 1)
            .expect("Expecting mutable weights");
        let w = w - (dw * self.learning_rate);
        let b = b - (db * self.learning_rate);
        updated_weights.push(w);
        update_biases.push(b);

        let mut cur_layer = self.layers - 2;
        while cur_layer >= 0 {
            // this is A^[l]
            let mut cur_layer_output = cache
                .get(cur_layer)
                .expect("Expecting predicted output of the last layer")
                .clone();
            // this is A^[l-1]
            let pre_layer_output;
            if cur_layer != 0 {
                pre_layer_output = cache
                    .get(cur_layer - 1)
                    .expect("Expecting predicted output of the last layer")
                    .clone();
            } else {
                // in case of first hidden layer previous output is what user give as input
                pre_layer_output = input_sample.clone();
            }
            // this is W^[l+1]
            let next_weights = self
                .weights
                .get(cur_layer + 1)
                .expect("Expecting weights of next layer")
                .clone();

            // computing (tanh(Z^[l]))^2
            // next_weights for layer l we will get from layer l+1.
            // dimensions of weight of layer l is (no of units of layer l,no of units of layer l-1)
            // assuming all the hidden layers will have tanh activation function.
            // derivation of tanh is (1 - (tan(z))^2)
            cur_layer_output.mapv_inplace(|val| {
                let v = val.tanh();
                1.0 - (v * v)
            });
            // dz^[l] = W^[l+1]^T dz^[l+1] * g'^[l](Z^[l]).
            // dimension of W^[l+1] is (no of units of layer l+1, no of units in layer l)
            // dimension of dz^[l+1] is (no of units of layer l+1, no of examples of the input)
            // dimension of dz^[l] is (no of units of layer l, no of examples of the input)
            let cur_dz = next_weights.reversed_axes().dot(&dz) * cur_layer_output;
            // dw^[l] = (1/m)dz^[l] A^[l-1]^T
            // pre_layer_output: dimension of A^[l-1] is (no of units of l-1, no of examples of the input)
            // dimension of dw^[l] is (no of units of layer l, no of units of layer l-1)
            let dw = cur_dz.dot(&pre_layer_output.reversed_axes()) * (1.0 / no_of_examples as f64);
            // db^[l] = (1/m)np.sum(dz^[l], axis=1)
            // dimension of db^[l] is (no of units of layer l, 1)
            let db = dz.sum_axis(Axis(1)) * (1.0 / no_of_examples as f64);

            // updating the learning parameters for layer l.
            let w = self
                .weights
                .get(cur_layer)
                .expect("Expecting mutable weights");
            let b = self
                .biases
                .get(cur_layer)
                .expect("Expecting mutable weights");
            // W := W - (dW * alpha), B := B - (dB * alpha)
            let w = w - (dw * self.learning_rate);
            let b = b - (db * self.learning_rate);
            updated_weights.push(w);
            update_biases.push(b);
            dz = cur_dz;

            cur_layer -= 1;
        }
        updated_weights.reverse();
        update_biases.reverse();
        self.weights = updated_weights;
        self.biases = update_biases;
    }
}

fn sigmoid(val: f64) -> f64 {
    let exp_val = val.exp();
    // note that 1/(1+e^-z)= e^z/((e^z)+1)
    exp_val / (exp_val + 1.0)
}
