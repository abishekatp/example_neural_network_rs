use ndarray::{Array2, Axis};
use ndarray_rand::{rand_distr::Uniform, RandomExt};

#[derive(Clone)]
pub struct Dense {
    output_size: usize,
    weights: Array2<f64>,
    biases: Array2<f64>,
    input: Option<Array2<f64>>,
}

// Ref: https://youtu.be/pauPCy_s0Ok
// for no of input examples = 1. for m input examples we will sum and find the average of all input examples.
// dE/dB = dE/dY.
// dE/dW = dE/dY x X^T dimension (ouput_size,input_size).
// dE/dX = W^T x dE/dY.
// dE/dY dimension (ouput_size, no of examples)
// X dimension (input_size, no of examples)

impl Dense {
    pub fn new(input_size: usize, output_size: usize) -> Dense {
        Dense {
            weights: Array2::random((output_size, input_size), Uniform::new(0.01, 0.09)),
            biases: Array2::zeros((output_size, 1)),
            input: None,
            output_size,
        }
    }
    // input dimension (input_size, no of examples)
    // ouput dimension (output_size, no of examples)
    pub fn forward(&mut self, input: Array2<f64>) -> Array2<f64> {
        self.input = Some(input.clone());
        self.weights.dot(&input) + &self.biases
    }

    pub fn backward(&mut self, output_grad: Array2<f64>, learning_rate: f64) -> Array2<f64> {
        if let Some(input) = self.input.clone() {
            let weight_grad = output_grad.dot(&input.clone().reversed_axes());
            self.weights = &self.weights - weight_grad * learning_rate;

            let average_bias_grad = output_grad.sum_axis(Axis(1)) * (1.0 / input.ncols() as f64);
            let reshaped_bias_grad =
                Array2::from_shape_vec((self.output_size, 1), average_bias_grad.to_vec())
                    .expect("Expecting reshaped bias matrix");
            self.biases = &self.biases - reshaped_bias_grad * learning_rate;
        }
        let reversed_weight = self.weights.clone().reversed_axes();
        reversed_weight.dot(&output_grad)
    }
}
