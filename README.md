# Example Neural Network in Rust
- I have implemented this Neural Network for just learning purpose.
- I was going through the [Deep Learning and Neural Network lectures](https://www.youtube.com/playlist?list=PLkDaE6sCZn6Ec-XTbcX1uRg2_u4xOEky0) by Andrew Ng and can't resist the thought of implementing it myself as a barebone model.
- First I have implemented directly looping through all the layers. This old implementation you can find in this same repo in `old_approach` branch. But then I have learned that there is better way to do it [here](https://youtu.be/Lakz2MoHy6o). I have refered the similar kind of implementation in python [here](https://github.com/TheIndependentCode/Neural-Network). This implementation seems very self explanatory and easy to expand.
- But the main difference is in this repo I have implemented the vectorized implementation. By vectorized implementation I mean passing multiple input examples as single 2D array to the deep neural network(dnn). 
- Still in convolution neural network(cnn) we have to loop through each input example separately.  Also I have to implement the convolution back propagation.
