use csv;
use ndarray::Array2;

pub fn read_csv(
    file_path: &str,
    no_of_records: usize,
    hight_values: Vec<f64>,
    train: bool,
) -> (Array2<f64>, Array2<f64>) {
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
        if train {
            if hight_values.contains(&lab) {
                lab = 1.0;
            } else {
                lab = 0.0;
            }
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
