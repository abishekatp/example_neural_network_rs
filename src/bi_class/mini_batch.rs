use crate::bi_class::utils::read_csv;
use crate::bi_class::{Sigmoid, Tanh, BIDNN};
use ndarray::s;

// this function kind of implements minibatch gradient descent.
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
    //Here learning rate of 0.8 works better.
    let mut dnn = BIDNN::new(784, vec![6, 5, 8, 1], vec![Tanh, Tanh, Tanh, Sigmoid], 0.7);
    //training the whole data as mini batches.
    // we are splitting 60000 training examples as 12 mini batches and train the model for 10 iterations.
    for i in 0..10 {
        let mut log_loss = 0.0;
        for epoch in 0..12 {
            let s = epoch * 5000;
            let e = (epoch + 1) * 5000;
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
        if log_loss < 0.25 {
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
