use ndarray::Array2;
use std::f64::consts::E as e;

// dE/dX = dE/dY . f'(X) here . is element wise multiplication.
#[derive(Clone)]
pub struct Tanh {}
impl Tanh {
    pub fn new() -> Tanh {
        Tanh {}
    }
    pub fn forward(&mut self, mut input: Array2<f64>) -> Array2<f64> {
        input.mapv_inplace(|val| val.tanh());
        input
    }

    pub fn backward(&self, mut input: Array2<f64>) -> Array2<f64> {
        input.mapv_inplace(|val| 1.0 - val.tanh().powf(2.0));
        input
    }
}

#[derive(Clone)]
pub struct Sigmoid {}
impl Sigmoid {
    pub fn new() -> Sigmoid {
        Sigmoid {}
    }
    pub fn forward(&mut self, mut input: Array2<f64>) -> Array2<f64> {
        input.mapv_inplace(|val| 1.0 / (1.0 + e.powf(-1.0 * val)));
        input
    }

    pub fn backward(&self, input: Array2<f64>) -> Array2<f64> {
        &input * (1.0 - &input)
    }
}

#[derive(Clone)]
pub struct Softmax {
    output: Option<Array2<f64>>,
}
impl Softmax {
    pub fn new() -> Softmax {
        Softmax { output: None }
    }

    // ref: https://youtu.be/AbLvJVwySEo
    // y(i) = e^x(i) / (sum over j=1 to n (e^x(j)))
    // n is number of features in the input
    // input with three features and two examples = [[1,2],[1,2],[1,2]]
    // output will be of same dimension.
    pub fn forward(&mut self, mut input: Array2<f64>) -> Array2<f64> {
        for j in 0..input.ncols() {
            let mut sum = 0.0;
            for i in 0..input.nrows() {
                sum += input[[i, j]].exp();
            }
            for i in 0..input.nrows() {
                input[[i, j]] = &input[[i, j]].exp() / sum;
            }
        }
        self.output = Some(input.clone());
        input
    }

    // there are two cases when we differntiate above forward implementation.
    // dE/dx(k) = sum over i=1 to n (dE/dy(i) x dy(i)/dx(k))
    // i = k we will get dy(i)/dx(k) = y(i)(1- y(i)).
    // i != k we will get dy(i)/dx(k) = y(i)*y(k).
    // if you differenctiate y(i) you will actually get these two equations.
    // for single input example we can convert this to matrix form as below
    // (Y . (I - Y)) x dE/dY. here . - element wise multiplication, x - normal matrix multiplication.
    // I dimension is (no of features, no of features)
    pub fn backward(&self, input: Array2<f64>) -> Array2<f64> {
        let mut gradient = Array2::zeros(input.dim());
        if let Some(output) = self.output.clone() {
            let rows = output.nrows();
            for i in 0..output.ncols() {
                let out_col = output.column(i);
                let in_col = input
                    .column(i)
                    .into_shape((rows, 1))
                    .expect("Expecting to reshape");
                let m1 = out_col.into_shape((1, rows)).expect("Expecting to reshape");
                let m2 = out_col.into_shape((rows, 1)).expect("Expecting to reshape");
                let identity: Array2<f64> = Array2::eye(rows);
                //dimension is (rows,1)
                let grad = ((identity - m1) * m2).dot(&in_col);
                let mut j = 0;
                for val in grad.iter() {
                    gradient[[j, i]] = val.clone();
                    j += 1;
                }
            }
        }
        gradient
    }
}
#[derive(Clone)]
pub struct ReLu {}
impl ReLu {
    pub fn new() -> ReLu {
        ReLu {}
    }
    pub fn forward(&mut self, mut input: Array2<f64>) -> Array2<f64> {
        input.mapv_inplace(|val| {
            if val > 0.0 {
                return val;
            }
            0.0
        });
        input
    }
    pub fn backward(&self, mut input: Array2<f64>) -> Array2<f64> {
        input.mapv_inplace(|val| {
            if val < 0.0 {
                return 0.0;
            }
            1.0
        });
        input
    }
}
