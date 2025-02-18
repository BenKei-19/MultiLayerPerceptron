# MULTI-LAYER PERCEPTRON (MLP)

## Introduction

As you know, some algorithms such as linear regression, logistic regression, softmax classification, or Perceptron learning algorithms (PLA) require classes to be linearly separable (which is difficult in real life). To solve this, a multi-layer perceptron is used.

## Problem Description
Suppose you have datasets like this:

![Dataset Example](Image/EX.png)

The classes are not linearly separable, so we will use MLP to tackle this. We will add one hidden layer between the input layer and the output layer as shown below:

![MLP Hidden Layer](Image/HiddenLayer.png)

Your task is to train a model that classifies the data correctly.

## Behind the scenes

This image shows more details about MLP:

![More Detail MLP Hidden Layer](Image/MoreDetailHiddenlayer.png)

Note: **W**<sup>(l)</sup> are the weights corresponding to **z**<sup>(l)</sup>. The activation function f(.) is often the ReLU function or another activation function. **b**<sup>(l)</sup> is the bias in the l-th layer.

Suppose J is the loss function. To improve the model, we aim to minimize J. J can be the mean squared error or another loss function.

### Derivative of the Loss Function
The derivative of the loss function with respect to a single component of the weight matrix of the final layer is:

\[
\frac{\partial J}{\partial w_{ij}^{(L)}} = \frac{\partial J}{\partial z_j^{(L)}} \times \frac{\partial z_j^{(L)}}{\partial w_{ij}^{(L)}} = e_j^{(L)} a_i^{(L-1)}
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

### Derivative of the Bias
Similarly, the derivative of the loss func
