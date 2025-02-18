# MULTI LAYER PERCEPTRON (MLP)

## Introduction

As you know, some algorithms such as linear regression, logistic regression, softmax classification, or Perceptron learning algorithms (PLA) require classes to be linearly separable (which is difficult in real life). To solve this, a multi-layer perceptron is used.

## Problem Description
Suppose you have datasets like this:
<img src="Image/EX.png" width=800><br/>
The classes are not linearly separable, so we will use MLP to tackle this. We will add one hidden layer between the input layer and the output layer as shown below:
<img src="Image/HiddenLayer.png" width=800><br/>
Your task is to train a model that classifies the data correctly.

## Behind the scenes

This image shows more details about MLP:
<img src="Image/MoreDetailHiddenlayer.png" width=800><br/>

Note: **W**<sup>(l)</sup> are the weights corresponding to **z**<sup>(l)</sup>. The activation function f(.) is often the ReLU function or another activation function. **b**<sup>(l)</sup> is the bias in the l-th layer.

Suppose J is the loss function. To improve the model, we aim to minimize J. J can be the mean squared error or another loss function.

The derivative of the loss function with respect to a single component of the weight matrix of the final layer is:

\[
\frac{\partial J}{\partial w_{ij}^{(L)}}
= \frac{\partial J}{\partial z_j^{(L)}} \times \frac{\partial z_j^{(L)}}{\partial w_{ij}^{(L)}}
= e_j^{(L)} a_i^{(L-1)} \quad (1)
\]

where \( e_j^{(L)} \) is defined as:

\[
e_j^{(L)} = \frac{\partial J}{\partial z_j^{(L)}}
\]

which is often an easily computable quantity, and

\[
\frac{\partial z_j^{(L)}}{\partial w_{ji}^{(L)}} = a_i^{(L-1)}
\]

since:

\[
z_j^{(L)} = \mathbf{w}_j^{(L)T} \mathbf{a}^{(L-1)} + b_j^{(L)}
\]

Similarly, the derivative of the loss function with respect to the bias of the final layer, using the chain rule, is:

\[
\frac{\partial J}{\partial b_j^{(L)}}
= \frac{\partial J}{\partial z_j^{(L)}} \cdot \frac{\partial z_j^{(L)}}{\partial b_j^{(L)}}
= e_j^{(L)}
\]

Continuing with the expression from (1) and the picture showing more details about MLP:

\[
\frac{\partial J}{\partial w_{ij}^{(l)}}
= \frac{\partial J}{\partial z_j^{(l)}} \cdot \frac{\partial z_j^{(l)}}{\partial w_{ij}^{(l)}}
= e_j^{(l)} a_i^{(l-1)}
\]

Where:

\[
e_j^{(l)} = \frac{\partial J}{\partial z_j^{(l)}}
= \frac{\partial J}{\partial a_j^{(l)}} \cdot \frac{\partial a_j^{(l)}}{\partial z_j^{(l)}}
\]

\[
= \left( \sum_{k=1}^{d^{(l+1)}} \frac{\partial J}{\partial z_k^{(l+1)}} \cdot \frac{\partial z_k^{(l+1)}}{\partial a_j^{(l)}} \right) f'(z_j^{(l)})
\]

\[
= \left( \sum_{k=1}^{d^{(l+1)}} e_k^{(l+1)} w_{jk}^{(l+1)} \right) f'(z_j^{(l)})
\]

\[
= \left( \mathbf{w}_{j:}^{(l+1)} \cdot \mathbf{e}^{(l+1)} \right) f'(z_j^{(l)})
\]

In which:

\[
\mathbf{e}^{(l+1)} = \left[ e_1^{(l+1)}, e_2^{(l+1)}, \dots, e_{d^{(l+1)}}^{(l+1)} \right]^T \in \mathbb{R}^{d^{(l+1)} \times 1}
\]
and:

\[
\mathbf{w}_j^{(l+1)}
\]
is understood as the row \( j-th \) of matrix \( \mathbf{W}^{(l+1)} \) (Note the colon; when the colon is absent, I assume it represents a column vector).

The sigma notation sums up in the second row of the operation appearing as \( a_j^{(l)} \) contributing to the calculation of all \( z_k^{(l+1)} \), \( k = 1, 2, \dots, d^{(l+1)} \). The derivative expression outside the big parentheses is due to \( a_j^{(l)} = f(z_j^{(l)}) \). Up to this point, we can see that having a simple activation function with simple derivatives will be very helpful for computations.

## Result
<img src='Image/Result.png'>

We have 99.33% accuracy on classifying the dataset mentioned earlier using 20 hidden units and one additional hidden layer.

## Requirements
- **Python**
- **Softmax regression**
- **CrossEntropy**
- **Gradient Descent**

## References
* [Machine Learning Cơ Bản](https://machinelearningcoban.com/2017/02/24/mlp/)
* [CS231n Neural Networks Case Study](https://cs231n.github.io/neural-networks-case-study/)

