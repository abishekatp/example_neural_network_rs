mod cnn;
mod utils;

pub use cnn::{
    Activation::{Sigmoid, Softmax, Tanh},
    CNN,
};
use utils::read_csv;

// is_number_one outputs the probability of input image being 1
pub fn _is_number_one() {}
