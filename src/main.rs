use dnnrs::{
    Activation::{Sigmoid, Tanh},
    DNN,
};

fn main() {
    println!("I will train the neural network here");
    let dnn = DNN::new(3, vec![3, 1], vec![Tanh, Sigmoid]);
    dbg!(dnn);
}
