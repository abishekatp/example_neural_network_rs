mod biclas_network;

use biclas_network::{
    Activation::{Sigmoid, Tanh},
    BIDNN,
};
use csv;
use ndarray::Array2;

pub fn is_number_one() {
    let (train_data, train_label) = read_csv("./archive/mnist_train.csv", 55000);
    //if you set learning rate too high weight will never converge properly.
    //if you set it too low it will converge to the local minimas. Here 2.5 works better.
    let mut dnn = BIDNN::new(784, vec![5, 5, 1], vec![Tanh, Tanh, Sigmoid], 2.5);
    //training the same data 20 times or more works better for me.
    for _i in 0..20 {
        dnn.train(train_data.clone(), train_label.clone());
    }

    //evaluating the model
    let (test_data, test_label) = read_csv("./archive/mnist_test.csv", 15);
    let output = dnn.evaluate(test_data);
    let mut i = 0;
    while i < 15 {
        println!(
            "Act: {:.3} , Pre: {:.3}",
            test_label[[0, i]],
            output[[0, i]]
        );
        i += 1;
    }
}

fn read_csv(file_path: &str, no_of_records: usize) -> (Array2<f64>, Array2<f64>) {
    let mut reader = csv::Reader::from_path(file_path).expect("Expecting to read the file");
    let mut data = Array2::zeros((784, no_of_records));
    let mut label = Array2::zeros((1, no_of_records));
    let mut i = 0;
    for record in reader.records() {
        if i == no_of_records {
            break;
        }
        let val = record.expect("Expecting numerical value");
        let mut iter = val.iter();
        let mut lab = iter
            .next()
            .expect("Expecting label")
            .parse::<f64>()
            .unwrap();
        // as of now we are going to predict if the input is number 1 or not.
        // since we have implemented output layer for binary classification.
        if lab != 1.0 {
            lab = 0.0;
        }
        label[[0, i]] = lab;
        let mut j = 0;
        while let Some(val) = iter.next() {
            if j == 784 {
                break;
            }
            let p = val.parse::<f64>().unwrap();
            data[[j, i]] = p;
            j += 1;
        }
        i += 1;
    }
    (data, label)
}

// below are the first 15 predicted values from testing data.
// we can consider these predicted values as the probability of image being 1.
// Act: 0.000 , Pre: 0.007
// Act: 0.000 , Pre: 0.007
// Act: 1.000 , Pre: 0.832
// Act: 0.000 , Pre: 0.007
// Act: 0.000 , Pre: 0.007
// Act: 1.000 , Pre: 0.837
// Act: 0.000 , Pre: 0.007
// Act: 0.000 , Pre: 0.007
// Act: 0.000 , Pre: 0.007
// Act: 0.000 , Pre: 0.007
// Act: 0.000 , Pre: 0.007
// Act: 0.000 , Pre: 0.007
// Act: 0.000 , Pre: 0.007
// Act: 0.000 , Pre: 0.007
// Act: 1.000 , Pre: 0.837
