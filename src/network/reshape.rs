use ndarray::{Array2, Array3};

#[derive(Clone)]
pub struct Reshape {
    input_shape: (usize, usize, usize),
    output_shape: (usize, usize),
}

impl Reshape {
    // input_shape: (channels, height, width) for convolution layer
    // output_shape: (no of features, no of examples) for dense layer
    pub fn new(input_shape: (usize, usize, usize), output_shape: (usize, usize)) -> Reshape {
        Reshape {
            input_shape,
            output_shape,
        }
    }

    pub fn forward(&self, input: Array3<f64>) -> Array2<f64> {
        input
            .into_shape(self.output_shape)
            .expect("Expecting the reshaped matrix")
    }

    pub fn backward(&self, output_gradient: Array2<f64>) -> Array3<f64> {
        output_gradient
            .into_shape(self.input_shape)
            .expect("Expecting the reshaped matrix")
    }
}
