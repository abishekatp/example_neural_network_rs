mod biclas_network;
mod mini_batch;
mod utils;

use biclas_network::{
    Activation::{Sigmoid, Tanh},
    BIDNN,
};
pub use mini_batch::_train_using_minibatch;
use ndarray::s;
use utils::read_csv;

// is_number_one_or_five outputs the probability of input image beign either 1 or 5
pub fn _is_number_one() {
    let train_count = 55000;
    let test_count = 20;
    let high_val = vec![1.0];
    let (train_data, train_label) =
        read_csv("./archive/mnist_train.csv", train_count, high_val, true);
    //if you set learning rate too high weight will never converge properly.
    //if you set it too low it will converge to the local minimas. Here 2.5 works better.
    let mut dnn = BIDNN::new(784, vec![5, 5, 1], vec![Tanh, Tanh, Sigmoid], 2.5);
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

// is_number_one_or_five outputs the probability of input image being 6.
pub fn _is_number_six() {
    let train_count = 55000;
    let test_count = 300;
    let high_val = vec![6.0];
    let (train_data, train_label) = read_csv(
        "./archive/mnist_train.csv",
        train_count,
        high_val.clone(),
        true,
    );
    //Here learning rate of 0.8 works better.
    let mut dnn = BIDNN::new(784, vec![6, 5, 1], vec![Tanh, Tanh, Sigmoid], 0.79);
    //training the same data 40 times or more works better for me.
    for _i in 0..40 {
        dnn.train(train_data.to_owned(), train_label.to_owned());
    }

    //evaluating the model
    let (test_data, test_label) = read_csv("./archive/mnist_test.csv", test_count, high_val, false);
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

pub fn is_number_match() {
    let train_count = 10000;
    let test_count = 200;
    let high_val = vec![8.0];
    let (train_data, train_label) = read_csv(
        "./archive/mnist_train.csv",
        train_count,
        high_val.clone(),
        true,
    );
    //Here learning rate of 0.8 works better.
    let mut dnn = BIDNN::new(784, vec![9, 5, 8, 1], vec![Tanh, Tanh, Tanh, Sigmoid], 1.1);
    //training the same data 40 times or more works better for me.
    for i in 0..100 {
        let log_loss = dnn.train(train_data.to_owned(), train_label.to_owned());
        println!("Iteration: {} Log loss: {}", i + 1, log_loss);
        if log_loss < 0.22 {
            break;
        }
    }

    //evaluating on the train data itself.
    let output = dnn.evaluate(train_data.slice(s![..784, 0..test_count]).to_owned());
    let mut i = 0;
    let label = train_label.slice(s![..1, 0..test_count]);
    while i < test_count {
        println!("Act: {:.3} , Pre: {:.3}", label[[0, i]], output[[0, i]]);
        i += 1;
    }

    //evaluating the model
    let (test_data, test_label) = read_csv("./archive/mnist_test.csv", test_count, high_val, false);
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
