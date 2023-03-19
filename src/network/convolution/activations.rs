use ndarray::Array3;
use std::f64::consts::E as e;

// todo: just a quick impl. we can use generics for this.
// dE/dX = dE/dY . f'(X) here . is element wise multiplication.
#[derive(Clone)]
pub struct Tanh {
    input: Option<Array3<f64>>,
}
impl Tanh {
    pub fn new() -> Tanh {
        Tanh { input: None }
    }
    pub fn forward(&mut self, input: Array3<f64>) -> Array3<f64> {
        self.input = Some(input.clone());
        input.mapv(|val| val.tanh())
    }

    pub fn backward(&self, output_grad: Array3<f64>) -> Array3<f64> {
        let input = self.input.clone().expect("Expecting the input matrix");
        //element wise multiplication.
        output_grad * input.mapv(|val| 1.0 - val.tanh().powf(2.0))
    }
}

#[derive(Clone)]
pub struct Sigmoid {
    input: Option<Array3<f64>>,
}
impl Sigmoid {
    pub fn new() -> Sigmoid {
        Sigmoid { input: None }
    }
    fn sigmoid(&self, input: Array3<f64>) -> Array3<f64> {
        input.mapv(|val| 1.0 / (1.0 + e.powf(-1.0 * val)))
    }
    pub fn forward(&mut self, input: Array3<f64>) -> Array3<f64> {
        self.input = Some(input.clone());
        self.sigmoid(input)
    }

    pub fn backward(&self, output_grad: Array3<f64>) -> Array3<f64> {
        let input = self.input.clone().expect("Expecting the input matrix");
        let input = self.sigmoid(input);
        let input = &input * (1.0 - &input);
        output_grad * input
    }
}
