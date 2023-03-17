use ndarray::Array2;
pub fn mse(predicted: Array2<f64>, actual: Array2<f64>) -> f64 {
    let mut loss = &predicted - &actual;
    loss.mapv_inplace(|val| val.powf(2.0));
    loss.mean().expect("Expecting mean")
}

pub fn mse_grad(predicted: Array2<f64>, actual: Array2<f64>) -> Array2<f64> {
    let mut loss_grad = &predicted - &actual;
    let rows = actual.nrows();
    loss_grad.mapv_inplace(|val| (val * 2.0) / rows as f64);
    loss_grad
}

pub fn binary_cross_entropy(predicted: Array2<f64>, actual: Array2<f64>) -> f64 {
    let log_pred = predicted.mapv(|val| val.ln());
    let one_minus_log_pred = predicted.mapv(|val| (1.0 - val).ln());
    let first = (&actual * log_pred * -1.0) - (1.0 - &actual) * one_minus_log_pred;
    first.mean().expect("Expecting mean")
}

pub fn binary_cross_entropy_grad(predicted: Array2<f64>, actual: Array2<f64>) -> Array2<f64> {
    let loss_grad = (1.0 - &actual / 1.0 - &predicted) - (&actual / &predicted);
    let rows = actual.nrows();
    loss_grad * (1.0 / rows as f64)
}
