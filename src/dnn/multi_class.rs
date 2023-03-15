use crate::dnn::utils::read_csv_multi_class;
use crate::dnn::{Softmax, Tanh, DNN};
use ndarray::s;

// Batch Gradient Descent(For training set size < 2000.): In this method we will be using the whole data set as a single batch where each column of the matrix is one input example.
// Mini Batch Gradient Descent(For training set size >2000.Usual mini batch size = 64 to 512): better than batch gradient descent for large data set. We will split our training examples into mini batches and use them to train the model.
// Stochastic Gradient Descent: In contrast to mini batch this approach uses each input example as 1 batch. In this approch convergence will be bit noisy. This is quite opposite to Batch gradient descent.
pub fn _train_multi_class() {
    let train_count = 60000;
    let test_count = 200;
    let no_of_epoches = 5;
    let mini_batch_size = 500;
    let no_of_mini_batches = train_count / mini_batch_size;
    let (train_data, train_label) = read_csv_multi_class("./archive/mnist_train.csv", train_count);
    //Here learning rate of 0.01 works better.
    let mut dnn = DNN::new(
        784,
        vec![50, 80, 60, 90, 100, 40, 10],
        vec![Tanh, Tanh, Tanh, Tanh, Tanh, Tanh, Softmax],
        0.1,
    );
    //training the whole data as mini batches.
    // we are splitting 60000 training examples into mini batches.
    for epoch in 0..no_of_epoches {
        let mut sum = 0.0;
        for j in 0..no_of_mini_batches {
            let s = j * mini_batch_size;
            let e = (j + 1) * mini_batch_size;
            let a = train_data.slice(s![.., s..e]).to_owned();
            let b = train_label.slice(s![.., s..e]).to_owned();
            let log_loss = dnn.train(a, b).expect("Expecting loss value");
            sum += log_loss;
            println!(
                "Iteration: {} Epoch: {} Log loss: {}",
                epoch + 1,
                j,
                log_loss
            );
        }
        // checking average loss of current epoch rather than checking directly.
        if sum / 120.0 < 0.2 {
            break;
        }
    }

    //evaluating on the train data itself.
    let output = dnn.evaluate(train_data.slice(s![..784, 0..test_count]).to_owned());
    let mut i = 0;
    let label = train_label.slice(s![.., 0..test_count]);
    while i < test_count {
        println!("Act: {:.3} , Pre: {:.3}", label.column(i), output.column(i));
        i += 1;
    }

    //evaluating the model
    let (test_data, test_label) = read_csv_multi_class("./archive/mnist_test.csv", test_count);
    let output = dnn.evaluate(test_data);
    let mut i = 0;
    while i < test_count {
        println!(
            "Act: {:.3} , Pre: {:.3}",
            test_label.column(i),
            output.column(i)
        );
        i += 1;
    }
}
