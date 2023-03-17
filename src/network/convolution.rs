use convolutions_rs::convolutions::*;
use convolutions_rs::Padding;
use ndarray::Axis;
use ndarray::{Array3, Array4};
use ndarray_rand::{rand_distr::Uniform, RandomExt};

// Ref: https://youtu.be/Lakz2MoHy6o
// How one layer looks like?
// [Y1, Y2, Y3...Yd] = [B1,B2,B3...Bd] + [[K11,K12,K13...K1n],[K21,K22,K23...K2n]...[Kd1,Kd2,Kd3...Kdn]] .|* [X1,X2,X3...Xn].
// d - no of kernals.
// Yd - 2D matrix which is ouputed by d'th kernal or filter.
// Bd - 2D bias matrix correcpond to d'th kernal or filter. This matrix is same for all input channels.
// Wdn - 2D weight matrix correspond to d'th kernal and n'th input channel.
// Xn - 2D input matrix corresponds to n'th input channel.
// .|* - dummy operation correlation 2D multiplicaion for each pair of matrices and sum them like normal matrix multiplication.
// for example output of channel 1 -> Y1 = B1 + K11*X1 + K12*X2 + ... + K1n*Xn.
// note: Y = B + K*X is the simplified form wher Y,B,K,X are all 2D matrix.
// It is just 1 kernal(filter), 1 input channel, 1 ouput channel(1 filter), 1 bias matrix for 1 kernal

// Backward Propagation:
// In all these cases we are given derivative of L with respect to ouput Y.

//
// dL/dK(ij) = X(j) * dL/dY(i). here  * - cross correlation, Y-output of the layer, L - loss.
// K(ij) - is Weight for i'th filter which corresponds to i'th output so we use dY(i)
// K(ij) - is Weight for j'th channel which corresponds to j'th input channel so we use X(j).
// K(11) - intutively will affect the 1'st output 2D matrix and 1's input channel.
// X(j) - is the j'th input channel

// dL/dB(i) = dL/dY(i) x dY(i)/dB(i) = dL/dY(i). since dY(i)/dB(i) = 1.
// B(i) - is bias matrix for i'th kernal or filter. It will affect all the channels in the input.

// dL/dX(j) += dL/dY(i) (*full) rot180(K(ij)) => for i from 1 to d
// X(j) - 2D matrix of j'th input channel
// Y(i) - is ouput produced by i'th kernal or filter
// K(ij) - is weight matrix for i'th kernal and j'th input channel.
// Same: output dimension will be equal to the input.
// Valid: output dimension will be less than the input.
// full: output dimension will be higher than input.

#[derive(Clone)]
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
    input: Option<Array3<f64>>,
    output: Option<Array3<f64>>,
}

impl Convolution {
    // input_shape: (input_depth, height, width)
    pub fn new(
        input_shape: (usize, usize, usize),
        kernal_size: usize,
        depth: usize,
    ) -> Convolution {
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
            input: None,
            output: None,
        }
    }
    pub fn forward(&mut self, input: Array3<f64>) -> Array3<f64> {
        self.input = Some(input.clone());
        let convolver = ConvolutionLayer::new(self.kernals.clone(), None, 1, Padding::Valid);
        let output = convolver.convolve(&input);
        let output = output + &self.biases;
        self.output = Some(output.clone());
        output
    }

    pub fn backward(&self, output_grad: Array3<f64>, learning_rate: f64) {
        let kernels_gradient: Array4<f64> = Array4::zeros(self.kernal_shape);
        let input_gradient: Array3<f64> = Array3::zeros(self.input_shape);
        let input = self.input.clone().expect("Expecting input matrix");
        for i in 0..self.depth {
            for j in 0..self.input_depth {
                // finding correlation dL/dK(ij) = X(j) * dL/dY(i)
                // note: each ouput channel we are convolving with each input channel of the layer to get kernal_gradient.
                let output_grad_single_channel = output_grad.index_axis(Axis(0), i);
                let output_grad_reshape = output_grad_single_channel
                    .into_shape((1, 1, self.output_shape.1, self.output_shape.2))
                    .expect("Expecting 4 dimensional array");
                let covolver =
                    ConvolutionLayer::new(output_grad_reshape.to_owned(), None, 1, Padding::Valid);
                let input_single_channel = input.index_axis(Axis(0), j);
                let input_reshape = input_single_channel
                    .into_shape((1, self.input_shape.1, self.input_shape.2))
                    .expect("Expecting 4 dimensional array");
                let output = covolver
                    .convolve(&input_reshape.to_owned())
                    .index_axis(Axis(0), 0);

                // todo: find a way to 180 degree rotated correlation
                todo!()
                // finding full correlation with 180 degree rotation dL/dX(j) += dL/dY(i) (*full) rot180(K(ij)).
                // let convolver = TransposedConvolutionLayer::new();
            }
        }

        let input_grad: Array3<f64> = Array3::zeros(self.input_shape);
    }
}
