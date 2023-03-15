mod neural_network;
mod utils;

pub use neural_network::{
    Activation::{Sigmoid, Softmax, Tanh},
    CNN,
};
use utils::read_csv;

// is_number_one outputs the probability of input image being 1
pub fn _is_number_one() {
    let train_count = 55000;
    let test_count = 20;
    let high_val = vec![1.0];
    let (train_data, train_label) =
        read_csv("./archive/mnist_train.csv", train_count, high_val, true);
    //if you set learning rate too high weight will never converge properly.
    //if you set it too low it will converge to the local minimas. Here 2.5 works better.
    let mut dnn = CNN::new(784, vec![5, 5, 1], vec![Tanh, Tanh, Sigmoid], 2.5);
    //training the same data 20 times or more works better for me.
    for _i in 0..20 {
        dnn.train(train_data.clone(), train_label.clone());
    }

    //evaluating the model
    let (test_data, test_label) =
        read_csv("./archive/mnist_test.csv", test_count, vec![1.0], false);
    let output = dnn.evaluate(test_data);
    let mut i = 0;
    while i < test_count {
        println!(
            "Act: {:.3} , Pre: {:.3}",
            test_label[[0, i]],
            output[[0, i]]
        );
        i += 1;
    }
}
