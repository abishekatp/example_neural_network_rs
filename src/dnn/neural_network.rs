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
pub struct DNN {
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
            // This is happening because all the units in all the layers will learn the same thing instead of learning diffent features of the input sample.
            // we need to assign small random values so that when we use sigmoid or tanh activation function slope(derivative) will not be nearly zero.
            // for large values of weights when we calculate derivative or slope, it will be zero for sigmoid and tanh functions.
            // becasue these functions are flat line for large x axis values.
            let rand_arr = Array2::random((*cur_units, pre_units), Uniform::new(0.01, 0.09));
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
            no_of_input_features,
        }
    }

    // input_sample should have nuber of rows equal to the no_of_input_features passed to the new() function.
    // no of columns can be any number based on how user want to train.
    // output should contain the actual output for the given.
    // output matrix should have number of rows = no of units in output layer.
    // output matrix should have number of columns = no columns in the input_sample.
    // Each column of output matrix corresponds to ouput of each column in the input_sample.
    pub fn train(&mut self, input_sample: Array2<f64>, output: Array2<f64>) -> Option<f64> {
        // dbg!(&self.weights, &self.biases);
        let cache = self.forward_propagate(input_sample.clone());
        self.backward_propagate(input_sample, output.clone(), cache.clone());
        let predicted_output = cache
            .last()
            .expect("Expecting predicted output of last layer")
            .clone();
        let loss = self.log_loss(predicted_output, output);
        loss
    }

    //evaluate returns output value for given input
    pub fn evaluate(&mut self, input_sample: Array2<f64>) -> Array2<f64> {
        if input_sample.nrows() != self.no_of_input_features {
            panic!("No of rows should match no of input features");
        }
        let cache = self.forward_propagate(input_sample.clone());
        cache
            .last()
            .expect("Expecting predicted output of last layer")
            .clone()
    }

    // forward_propagate function will propagate through each layer of the network one by one
    // and store the predicted output of each layer in the the cache.
    // Each neural network unit will have learning parameters W and B assoiciated with them.
    // In forward propagate we will use these parameters W(self.weights) and B(self.biases) to predict the output.
    // second layer will predict the output based on first layers output.
    // at the end we will have single predicted output from the output layer.
    // returns the cache which will contain the output of the each layer.
    pub fn forward_propagate(&mut self, input_sample: Array2<f64>) -> Vec<Array2<f64>> {
        //getting no of columns in the input sample
        // this ncols return the lenght of Axis(1)
        let no_of_examples = input_sample.ncols();
        let mut pre_out = input_sample;
        // cache will store the predicted outputs of each layere
        let mut cache: Vec<Array2<f64>> = vec![];
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
            // matrix adddition
            let mut z = z + bias;

            //apply the activation function to the output of the current layer
            let acti = self
                .activations
                .get(i)
                .expect("Expecting activation function type for current layer");
            match acti {
                Activation::Tanh => {
                    z.mapv_inplace(|val| val.tanh());
                }
                // SoftMax is for bi-class classification models.
                // It should be only used on ouput layer.
                Activation::Sigmoid => {
                    z.mapv_inplace(|val| sigmoid(val));
                }
                Activation::ReLu => {
                    z.mapv_inplace(|val| relu(0.0, val));
                }
                // SoftMax is for multi-class classification models.
                // It should be only used on ouput layer.
                // t(i)=e^(z(i)) and a^[l](i)=t(i)/sum of all t(i).
                // for this softmax activation also output will be same.
                Activation::Softmax => {
                    for j in 0..z.ncols() {
                        let mut sum = 0.0;
                        for i in 0..z.nrows() {
                            sum += z[[i, j]].exp();
                        }
                        for i in 0..z.nrows() {
                            z[[i, j]] = z[[i, j]].exp() / sum;
                        }
                    }
                }
            }
            cache.push(z.clone());
            pre_out = z;
        }
        cache
    }

    fn log_loss(&self, pred: Array2<f64>, output: Array2<f64>) -> Option<f64> {
        match self
            .activations
            .last()
            .expect("Expecting last activation function")
        {
            Activation::Softmax => {
                let pred = pred.mapv(|val| -1.0 * val.ln());
                // this is elementwise multiplication.
                // this multiplication is corresponding to -y * log(a). y will be 1 at only one row at each column
                let loss_pred = &output * &pred;
                let loss = loss_pred.sum() * (1.0 / output.ncols() as f64);
                Some(loss)
            }
            Activation::Sigmoid => {
                // In logistic regression we have seen log loss formula is L(a,Y) = -y*log(a) - (1-y)log(1-a).
                let log_pred = pred.mapv(|val| val.ln());
                let log_one_pred = pred.mapv(|val| (1.0 - val).ln());
                let pred = (&output * log_pred + (1.0 - &output) * log_one_pred) * -1.0;
                let loss = pred.sum() * (1.0 / output.ncols() as f64);
                Some(loss)
            }
            _ => None,
        }
    }

    // todos: when I get time I can add gradient check which will compare this gradient result with numerically computed gradient and show the correctness of this implementation.
    // here we are using gradient descent optimization. We can use Adam optimization for faster training.
    // I am not going to implement all these things here but we can test it out in python frameworks like tensorflow or pytorch.
    // some optimization techniques are Exponentially Weighted average,Root mean square(RMS), Adam optimisation, learning rate decay,
    // Batch Normalization(same as normalization to input but applied to all the inputs of the hidden layers).

    // backward_propagate will propagate through each layer of the network from the last layer to the first layer.
    // In the backward propagation we will calculate derivative of L with respect each learning parameter of the each neural network unit.
    // Intutively these derivative values will help us to minimize this loss value.
    // In the process it will comput he dw[i],db[i] for each layer i and update their corresponding weights weights[i] and bias biases[i].
    // here we will calculate derivative of L with respect to each learning parameter associated with each neural network unit.
    // these derivatives are dw = dL/dw and db = dL/dB. to calculate this derivative we used chain rules and computed the formulas.
    // For example to calculate dw we will use da=dL/dA and dA/dz and dz/dw.
    // But for simplification we have calculated these derivatives already as formulas which we are using here.
    // why do we use derivatives? Because derivatives will indicate us for the increase or decrease in the Loss value how much each parameter contributed.
    // For example the value dw = dL/dw will say how much weight w contributed for the loss value created by current iteration.
    pub fn backward_propagate(
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
        let last_layer_output = cache
            .get(self.layers - 1)
            .expect("Expecting predicted output of the last layer");
        let prev_layer_out = cache
            .get(self.layers - 2)
            .expect("Expecting predicted output of the layer before layer")
            .clone();
        let output_units = self
            .units
            .last()
            .expect("Expecting no of units in the last layer");

        // for Sigmoid activation:
        // if g(z) = (1/1+e^-z). for sigmoid function derivative with respect to z will be g'(z) = (1/1+e^-z)(1- (1/1+e^-z))=g(z)(1-g(z)).
        // note the g(Z^[l]) is the output of the layer l.
        // dL/dz = (dL/dg)(dg/dz), if g(z)=a. then dL/dz = (dL/da)(da/dz).
        // For logistic regression L(a,Y) = -y*log(a) - (1-y)log(1-a).
        // if you find these derivatives and substitute in dL/dz you will get dz = dL/dz = (dL/da)(da/dz) = a - y where a is predicted output and y is actual output.

        // note: the other way could be we can calculate dL/dA and dA/dz then multiply numerically.
        // Here as of now we assumed A will be sigmoid function. so found the equation A-Y.
        // for your reference dL/da = (-y/a) +(1-y / 1-a) and da/dz = a(1-a).

        // Note: we get this same derivative equation for softmax activation at the ouput layer.
        // dimension = (no of units in output layer, no of input examples)
        // in case of sigmoid it will have single row, in case of Softmax it will have multiple rows.
        let mut dz = last_layer_output - output;

        // note: dw and db are dL/dw and dL/db for current layer.
        // we got dw = dL/dw = (dL/dz)(dz/dw) = (dz)(A^[l-1])^T
        // here * is broadcasted to all the elements of the matrix. for getting average we multiply by (1/m)
        // here (dz x A) dimenstions are (no of units in output layer,no of input examples)x(no of examples, no units in the previous layer)
        // the matrix multiplication will result in (no of units in ouput layer,no of units in previous layer)
        let dw = (dz.dot(&prev_layer_out.reversed_axes())) * (1.0 / no_of_examples as f64);
        // db is of dimension (no of units in ouput layer,1)
        let db = dz.sum_axis(Axis(1)) * (1.0 / no_of_examples as f64);
        // sum_axis returns the 1 dimensional array. but we actually want column vector.
        let db = Array2::from_shape_vec((output_units.clone(), 1), db.to_vec())
            .expect("Expecting reshaped bias matrix");

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
        loop {
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
            let activation_fun = self
                .activations
                .get(cur_layer)
                .expect("Expecting activation function");
            match activation_fun {
                Activation::Tanh => {
                    // next_weights for layer l we will get from layer l+1.
                    // dimensions of weight of layer l is (no of units of layer l,no of units of layer l-1)
                    // derivative for tanh activation function.
                    // derivation of tanh is (1 - (tan(z))^2)
                    cur_layer_output.mapv_inplace(|val| {
                        let v = val.tanh();
                        1.0 - (v * v)
                    });
                }
                Activation::Sigmoid => {
                    // derivation of sigmoid function is a*(1-a)
                    cur_layer_output.mapv_inplace(|val| val * (1.0 - val));
                }
                Activation::ReLu => {
                    // derivation of ReLu function is {0 if val<0, 1 if val>=0}
                    cur_layer_output.mapv_inplace(|val| {
                        if val < 0.0 {
                            return 0.0;
                        }
                        1.0
                    });
                }
                Activation::Softmax => {
                    panic!("Sofmax should be used only at the ouput layer");
                }
            }

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
            let db = cur_dz.sum_axis(Axis(1)) * (1.0 / no_of_examples as f64);
            //sum_axis returns the 1 dimensional arrar. but we actually want column vector.
            let u = self
                .units
                .get(cur_layer)
                .expect("Expecting no of units in current layer");
            let db = Array2::from_shape_vec((*u, 1), db.to_vec())
                .expect("Expecting reshaped bias matrix");

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

            if cur_layer == 0 {
                break;
            }
            cur_layer -= 1;
        }
        updated_weights.reverse();
        update_biases.reverse();
        self.weights = updated_weights;
        self.biases = update_biases;
    }
}

fn sigmoid(val: f64) -> f64 {
    let e = std::f64::consts::E;
    1.0 / (1.0 + e.powf(-1.0 * val))
}

fn relu(v1: f64, v2: f64) -> f64 {
    if v1 > v2 {
        return v1;
    }
    v2
}
