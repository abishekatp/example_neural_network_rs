use dnnrs::{
    Activation::{Sigmoid, Tanh},
    DNN,
};

use ndarray::Array2;
use ndarray_rand::rand::SeedableRng;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use rand_isaac::isaac64::Isaac64Rng;

fn main() {
    println!("I will train the neural network here");
    let mut dnn = DNN::new(3, vec![3, 1], vec![Tanh, Sigmoid], 0.01);
    let mut rng = Isaac64Rng::seed_from_u64(42);
    let input_sample = Array2::random_using((3, 6), Uniform::new(0., 10.), &mut rng);
    let output = Array2::random_using((1, 6), Uniform::new(0., 10.), &mut rng);
    // dnn.train(input_sample, output);
}
