use ndarray::Array2;

// dE/dX = dE/dY . f'(X) here . is element wise multiplication.
pub struct Tanh {}
impl Tanh {
    pub fn new() -> Tanh {
        Tanh {}
    }
    pub fn forward(mut input: Array2<f64>) -> Array2<f64> {
        input.mapv_inplace(|val| val.tanh());
        input
    }

    pub fn backward(mut input: Array2<f64>) -> Array2<f64> {
        input.mapv_inplace(|val| 1.0 - val.tanh().powf(2.0));
        input
    }
}

pub struct ReLu {}
impl ReLu {
    pub fn new() -> ReLu {
        ReLu {}
    }
    pub fn forward(mut input: Array2<f64>) -> Array2<f64> {
        input.mapv_inplace(|val| {
            if val > 0.0 {
                return val;
            }
            0.0
        });
        input
    }

    pub fn backward(mut input: Array2<f64>) -> Array2<f64> {
        input.mapv_inplace(|val| {
            if val < 0.0 {
                return 0.0;
            }
            1.0
        });
        input
    }
}
