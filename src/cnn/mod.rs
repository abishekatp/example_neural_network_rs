mod convolution;
mod utils;

// is_number_one outputs the probability of input image being 1
pub fn _is_number_one() {}

// Ref: https://youtu.be/Lakz2MoHy6o
// How one layer looks like?
// [Y1, Y2, Y3...Yd] = [B1,B2,B3...Bd] + [[W11,W12,W13...W1n],[W21,W22,W23...W2n]...[Wd1,Wd2,Wd3...Wdn]] .|* [X1,X2,X3...Xn].
// d - no of kernals.
// Yd - 2D matrix which is ouputed by d'th kernal or filter.
// Bd - 2D bias matrix correcpond to d'th kernal or filter. This matrix is same for all input channels.
// Wdn - 2D weight matrix correspond to d'th kernal and n'th input channel.
// Xn - 2D input matrix corresponds to n'th input channel.
// .|* - dummy operation correlation 2D multiplicaion for each pair of matrices and sum them like normal matrix multiplication.
// for example output of channel 1 -> Y1 = B1 + K11*X1 + K12*X2 + ... + K1n*Xn.
// note: Y = B + K*X is the simplified form wher Y,B,K,X are all 2D matrix.
// It is just 1 kernal(filter), 1 input channel, 1 ouput channel(1 filter), 1 bias matrix for 1 kernal

// Backward Propagation:
// In all these cases we are given derivative of L with respect to ouput Y.

//
// dL/dW(ij) = X(j) * dL/dY(i). here  * - cross correlation, Y-output of the layer, L - loss.
// W(ij) - is Weight for i'th filter which corresponds to i'th output so we use dY(i)
// W(ij) - is Weight for j'th channel which corresponds to j'th input channel so we use X(j).
// W(11) - intutively will affect the 1'st output 2D matrix and 1's input channel.
// X(j) - is the j'th input channel

// dL/dB(i) = dL/dY(i) x dY(i)/dB(i) = dL/dY(i). since dY(i)/dB(i) = 1.
// B(i) - is bias matrix for i'th kernal or filter. It will affect all the channels in the input.

// dL/dX(j) += dL/dY(i) (*full) rot180(K(ij)) => for i from 1 to d
// X(j) - 2D matrix of j'th input channel
// Y(i) - is ouput produced by i'th kernal or filter
// K(ij) - is weight matrix for i'th kernal and j'th input channel.
// Same: output dimension will be equal to the input.
// Valid: output dimension will be less than the input.
// full: output dimension will be higher than input.
