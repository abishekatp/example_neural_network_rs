# Example Neural Network in Rust
- I have implemented this Neural Network for just learning purpose.(This is only for learning purpose)
- I was going through the [Deep Learning and Neural Network lectures](https://www.youtube.com/playlist?list=PLkDaE6sCZn6Ec-XTbcX1uRg2_u4xOEky0) by Andrew Ng and can't resist the thought of implementing it myself as a barebone model.
## Binary Classification with Neural Networks
-`src/bi_class` directory contains the basic neural network implementation for binary classification. This network accepts arbitrary number units in all the arbitrary number of hidden layers and single unit in output layer(since it is binary classification).
- I am using the MNIST data set for handwritten numbers from [kagle](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv?resource=download) for training and testing. I am using csv format data. where each row first value is label and remaining 784 values are pixels of the image. these 784 values represent 28 x 28 matrix.

### Initialization function
- This function will initialize the Neural Network with weights assigned to random values. If we assign 0 to all, then the result will be equivalent to using single layer with single unit.This is happening because all the units in all the layers will learn the same thing instead of learning diffent features of the input image.
- We need to assign small random values so that when we use sigmoid or tanh activation function slope(derivative) will not be nearly zero.for large values of weights when we calculate derivative or slope, it will be zero for sigmoid and tanh functions. becasue these functions are flat line for large x axis values. we can assign all 0's to the biases because it will not create any of the above problems.

### Forward Propagation
- Forward_propagate function will propagate through each layer of the network one by one
and store the predicted output of each layer in the the cache. Each neural network unit will have learning parameters W and B assoiciated with them. In forward propagate we will use these parameters W(self.weights) and B(self.biases) to predict the output.
- Second layer will predict the output based on first layers output. At the end we will have single predicted output from the output layer.

### Backward Propagation
- Backward_propagate will propagate through each layer of the network from the last layer to the first layer.
- In the backward propagation we will calculate derivative of L with respect each learning parameter of the each neural network unit. Intutively these derivative values will help us to minimize this loss value.
- In the process it will comput he dw[i],db[i] for each layer i and update their corresponding weights weights[i] and bias biases[i]. Here we will calculate derivative of L with respect to each learning parameter associated with each neural network unit. These derivatives are dw = dL/dw and db = dL/dB.
- For example to calculate dw we will use da=dL/dA and dA/dz and dz/dw. But for simplification we have calculated these derivatives already as formulas which we are using to calculate these derivatives.
- Why do we use derivatives? Because derivatives will indicate us how much each parameter will contributed to the increase or decrease in the Loss value L. For example the value dw = dL/dw will say how much weight w contributed for the loss value created by current iteration.
- Since this is binary classification we use the formula L = - y * log(a) - (1-y) * log(1-a) where `a` is the predicted output and y is actual output. If you look deeply into this function when actual output y=0, we will get L = -log(1-a) and when actual output is y=1, we will get L = -log(a). `a=0.2` and `y=1` means actual output is 1 but our model predicted 0 in this case L = - log(a) = -log(0.2) = 0.69. This value indicates us there is loss in the prediction. Based on this loss value we will calculate the derivative with respect to each of the learning parameters.Finally apply the change to the learning parameters based on these derivatives. we will continue this process many times to improve the learnings.
- Below is the result of the testing I have done. These are all the actual output and predicted output values. These predicted outputs indicate probability of image being 1.

```
Act: 0.000 , Pre: 0.007
Act: 0.000 , Pre: 0.007
Act: 1.000 , Pre: 0.832
Act: 0.000 , Pre: 0.007
Act: 0.000 , Pre: 0.007
Act: 1.000 , Pre: 0.837
Act: 0.000 , Pre: 0.007
Act: 0.000 , Pre: 0.007
Act: 0.000 , Pre: 0.007
Act: 0.000 , Pre: 0.007
Act: 0.000 , Pre: 0.007
Act: 0.000 , Pre: 0.007
Act: 0.000 , Pre: 0.007
Act: 0.000 , Pre: 0.007
Act: 1.000 , Pre: 0.837
```
- One more model I have tried to train to predict the image which contains number 6. It took me some reasonable amount of time to figure out values for the hyper parameters like learning rate(this was the main culprit), no of units in each layer of network. Eventhough I got some false possitives and false negatives, It predicted the images of the 6 correctly most of the time.
```
Act: 6.000 , Pre: 0.700
Act: 9.000 , Pre: 0.011
Act: 0.000 , Pre: 0.011
Act: 3.000 , Pre: 0.011
Act: 6.000 , Pre: 0.011
Act: 5.000 , Pre: 0.011
Act: 5.000 , Pre: 0.011
Act: 7.000 , Pre: 0.011
Act: 2.000 , Pre: 0.011
Act: 8.000 , Pre: 0.011
Act: 2.000 , Pre: 0.700
Act: 6.000 , Pre: 0.700
Act: 7.000 , Pre: 0.011
Act: 5.000 , Pre: 0.011
Act: 9.000 , Pre: 0.011
Act: 2.000 , Pre: 0.011
Act: 6.000 , Pre: 0.700
Act: 4.000 , Pre: 0.011
Act: 1.000 , Pre: 0.011
Act: 8.000 , Pre: 0.011
Act: 2.000 , Pre: 0.011
Act: 9.000 , Pre: 0.011
```


