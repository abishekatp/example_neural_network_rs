use convolutions_rs::convolutions::*;
use convolutions_rs::Padding;
use ndarray::{Array1, Array3, Array4};
use ndarray_rand::{rand_distr::Uniform, RandomExt};

pub struct Convolution {
    depth: usize,
    input_depth: usize,
    // (input_depth, height, width)
    input_shape: (usize, usize, usize),
    // (depth, height, width)
    output_shape: (usize, usize, usize),
    // (depth, input_depth,kernal_height,kernal_widht)
    kernal_shape: (usize, usize, usize, usize),
    kernals: Array4<f64>,
    biases: Array3<f64>,
}

impl Convolution {
    // input_shape: (input_depth, height, width)
    fn new(input_shape: (usize, usize, usize), kernal_size: usize, depth: usize) -> Convolution {
        let (input_depth, input_height, input_width) = input_shape;
        let kernal_shape = (depth, input_depth, kernal_size, kernal_size);
        let output_shape = (
            depth,
            input_height - kernal_size + 1,
            input_width - kernal_size + 1,
        );
        Convolution {
            depth,
            input_depth,
            input_shape,
            output_shape,
            kernal_shape,
            kernals: Array4::random(kernal_shape, Uniform::new(0.01, 0.09)),
            biases: Array3::zeros(output_shape),
        }
    }
    fn forward(&self, input: Array3<f64>) -> Array3<f64> {
        let conv = ConvolutionLayer::new(self.kernals.clone(), None, 1, Padding::Valid);
        let output = conv.convolve(&input);
        let output = output + &self.biases;
        output
    }

    fn backward(&self, output_grad: Array3<f64>, learning_rate: f64) {
        let kernal_grad: Array4<f64> = Array4::zeros(self.kernal_shape);
        let input_grad: Array3<f64> = Array3::zeros(self.input_shape);
    }
}
