use crate::bi_class::utils::read_csv;
use crate::bi_class::{Sigmoid, Tanh, BIDNN};
use ndarray::s;

// Batch Gradient Descent(For training set size < 2000.): In this method we will be using the whole data set as a single batch where each column of the matrix is one input example.
// Mini Batch Gradient Descent(For training set size >2000.Usual mini batch size = 64 to 512): better than batch gradient descent for large data set. We will split our training examples into mini batches and use them to train the model.
// Stochastic Gradient Descent: In contrast to mini batch this approach uses each input example as 1 batch. In this approch convergence will be bit noisy. This is quite opposite to Batch gradient descent.
pub fn _train_using_minibatch() {
    let train_count = 60000;
    let test_count = 200;
    let high_val = vec![8.0];
    let (train_data, train_label) = read_csv(
        "./archive/mnist_train.csv",
        train_count,
        high_val.clone(),
        true,
    );
    //Here learning rate of 0.01 works better.
    let mut dnn = BIDNN::new(
        784,
        vec![50, 80, 60, 1],
        vec![Tanh, Tanh, Tanh, Sigmoid],
        0.01,
    );
    //training the whole data as mini batches.
    // we are splitting 60000 training examples into mini batches.
    for i in 0..5 {
        let mut log_loss = 0.0;
        for epoch in 0..120 {
            let s = epoch * 500;
            let e = (epoch + 1) * 500;
            let a = train_data.slice(s![.., s..e]).to_owned();
            let b = train_label.slice(s![.., s..e]).to_owned();
            log_loss = dnn.train(a, b);
            println!(
                "Iteration: {} Epoch: {} Log loss: {}",
                i + 1,
                epoch,
                log_loss
            );
        }
        if log_loss < 0.1 {
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
