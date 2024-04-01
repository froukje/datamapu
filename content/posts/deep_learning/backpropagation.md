+++
title = 'Backpropagation Step by Step'
date = 2024-03-31T20:21:50-03:00
draft = true
tags = ["Data Science", "Machine Learning", "Artificial Intelligence"]
categories = ["Data Science", "Machine Learning", "Artificial Intelligence"]
keywords = ["Data Science", "Machine Learning", "Artificial Intelligence"]
images = ['images/backpropagation/']
+++

## Introduction

A neural network consists of a set  of parameters - the weights and biases - that define the outcome of the network, that is the predictions. When training a neural network we aim to adjust these weights and biases such that the predictions improve. In order to achieve that *Backpropagation* is used. In this post we discuss how backpropagation works, and explain it in detail for different examples. We additionally show the general formulation, but without going into details.

**Main Concepts of Training a Neural Net**

Before starting with the first example, let's quickly go through the main ideas of the training process of a neural net. The first thing we need, when we want to train a neural net is the *training data*. The training data consists of pairs of *inputs* and *labels*. The inputs are also called *features* and usually written as $X = (x_1, \dots, x_n)$, with $n$ the number of data samples. The labels are the expected outcomes - or true values - they are usually denoted as $y = (y_1, \dots, y_n)$. Training a neural net is an iterative process over a certain number of *epochs*. In each epoch the training data is processed through the network in a so-called *forward pass*, which results in the model output. Then the error - *loss* - of model output compared to the true values is calculated to evaluate the model. Finally, in the backward pass - the *backpropagation* - [gradient descent]({{< ref "gradient_descent">}} "gradient_descent") is used to update the model parameters and reduce the loss. For a general and more detailed introduction to Deep Learning terms and concepts, please refer to [Introduction to Deep Learning]({{< ref "/posts/deep_learning/intro_dl.md">}} "intro_dl").

If not mentioned differently, we use the following data, activation function, and loss throughout the examples of this post.

**Training Data**

We consider the most simple situation with one dimensional input data and just one sample $x = 0.5$ and labels $y = 1$.

**Activation Function**

As activation function, we use the *Sigmoid function*

$$\sigma(x) = \frac{1}{1 + e^{-x}}.$$

**Loss Function**

As loss function we use the *Sum of the Squared Error*, defined as 

$$L(y, \hat{y}) = \frac{1}{2}\sum_{i=1}^n(y_i - \hat{y}_i)^2,$$

with $y_i = (y_1, \dots, y_n)$ the labels and $\hat{y} = (\hat{y}_1, \dots, \hat{y}_n)$ the predicted labels, and $n$ the number of samples. In our case $n=1$ and the formula simplifies to

$$L(y, \hat{y}) = \frac{1}{2} (y - \hat{y})^2.$$

## 1. Example

To illustrate how backpropagation works, we start with the most simple neural network, which only constists of one single neuron. 

![one_neuron](/images/backpropagation/one_neuron.png)
*Illustration of a Neural Network consisting of a single Neuron.*

In this simple neural net, $z(x) = w\cdot x + b$ represents the linear part of the neuron, $a$ the activation function, which we chose to be the sigmoid function, i.e. $a = \sigma(z) = \frac{1}{1 + e^{-z}}$. For the following calculations, we assume the initial weight $w = 0.3$ and the initial bias $b = 0.1$. Further the learning rate is set to $\alpha = 0.1$. These values are chosen arbitrary for illustration purposes.

**The Forward Pass**

We can calculate the forward pass through this network as

$$\hat{y} = \sigma(z)$$
$$\hat{y} = \sigma(wx + b),$$
$$\hat{y} = \frac{1}{1 + e^{-(wx+b)}}$$.

Using the data, weight, and bias defined above, we get for $x = 0.5$

$$\hat{y} = \frac{1}{1 + e^{-(0.3\cdot 0.5 + 0.1)}} = \frac{1}{1 + e^{-0.25}} \approx 0.56$$

The error after this forward pass can be calculated as 

$$L(1.5, 0.56) = \frac{1}{2}(1.5 - 0.56)^2 = 0.44.$$

![one_neuron_forward](/images/backpropagation/one_neuron_forward.png)
*Forward pass through the Neural Net.*

**The Backward Pass**

To update the weight and the bias we use [gradient descent]({{< ref "gradient_descent">}} "gradient_descent"), that is

$$w_{new} = w - \alpha \frac{\delta L}{\delta w}$$
$$b_{new} = b - \alpha \frac{\delta L}{\delta b},$$

with $\alpha = 0.1$ the learning rate. That is we need to calculate the partial derivatives of $L$ with respect to $w$ and $b$ to get the new weight and bias. This can be done using the chain rule and is illustrated in the plots below.

$$\frac{\delta L}{\delta w} = \frac{\delta L }{\delta \hat{y}} \frac{\delta \hat{y}}{\delta z} \frac{\delta z}{\delta w}$$
$$\frac{\delta L}{\delta b} = \frac{\delta L }{\delta \hat{y}} \frac{\delta \hat{y}}{\delta z} \frac{\delta z}{\delta b}$$

![one_neuron](/images/backpropagation/one_neuron_back.png)
![one_neuron](/images/backpropagation/one_neuron_back1.png)
*Illustration of Backpropagation in a Neural Network consisting of a single Neuron.*

We can calculte the individual derivatives as

$$\frac{\delta L}{\delta \hat{y}} = \frac{\delta}{\delta \hat{y}} \frac{1}{2} (y - \hat{y})^2 = - (y - \hat{y}),$$
$$\frac{\delta \hat{y}}{\delta z} = \frac{\delta}{\delta z} \sigma(z) = \sigma(z)\cdot \big(1 - \sigma(z)\big),$$
$$\frac{\delta z}{\delta w} = \frac{\delta}{\delta w} (w\cdot x + b) = x,$$ 
$$\frac{\delta z}{\delta b} = \frac{\delta}{\delta b} (w\cdot x + b) = 1.$$

Please find the detailed calculation of the derivative of the sigmoid function in the appendix of this post.

For the data we are considering, we get for the first equation

$$\frac{\delta L}{\delta \hat{y}} = - (y - \hat{y}) = - (1.5 - 0.56) = -0.94.$$

The second equation leads to
$$\frac{\delta \hat{y}}{\delta z} = \sigma(z)\cdot \big(1 - \sigma(z)\big)$$
$$\frac{\delta \hat{y}}{\delta z} = \frac{1}{1 + e^{-0.25}}\Big( 1 - \frac{1}{1 + e^{-0.25}}\Big) = 0.56\cdot 0.44 = 0.25,$$

and finally
$$\frac{\delta z}{\delta w} = x = 0.5,$$ 
$$\frac{\delta z}{\delta b} = 1.$$

Putting the equations back together, we get

$$\frac{\delta L}{\delta w} = -0.94 \cdot 0.25 \cdot 0.5 = -0.118$$
$$\frac{\delta L}{\delta b} = -0.94 \cdot 0.25 \cdot 1 = -0.235$$

The calculation for $\frac{\delta L }{\delta w}$ is illustrated in the plot below.

![one_neuron](/images/backpropagation/one_neuron_back2.png)
*Backpropagation for the weight $w$.*

The weight and the bias then update to

$$w_{new} = 0.3 - 0.1 \cdot (-0.118) = 0.312,$$
$$b_{new} = 0.1 - 0.1 \cdot (-0.235) = 0.125.$$

**Note**

With this simple example we illustrated one forward and one backward pass. It is a good example to understand the calculations, in real projects, however, data and neural nets are much more complex. We not only considered a simple dataset because it is one dimensional data, but also because we just have one data sample. In reality one forward pass would consists of processing all the $n$ datasamples through the network, and accordingly the backward pass. The next example is slightly more complex, but still a very simplified neural net, which is used for illustration purposes.

## 2. Example

The second example we consider is a Neural Net, that consists of two neurons after each other, as illustrated in the following plot.

![two_neurons](/images/backpropagation/two_neurons.png)
*A Neural Net with two layers, each consisting of one neuron.*

## Example: Shallow Network

## General Formulation

## Summary

## Appendix

### Derivative of the Sigmoid Function

The *Sigmoid Function* is defined as

$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

The derivative can be derived using the [chain rule](https://en.wikipedia.org/wiki/Chain_rule)

$$\sigma\prime(x) = \frac{\delta}{\delta x} \sigma(x) = \frac{\delta}{\delta x} (1 + e^{-x})^{-1} = -(1 + e^{-x})^{-2} \frac{\delta}{\delta x}(1 + e^{-x}).$$

In the last expression we applied the outer derivative, calculating the inner derivative again needs the chain rule.

$$\sigma\prime(x) = -(1 + e^{-x})^{-2} e^{-x} \cdot (-1).$$

This can be reformulated to

$$\sigma\prime(x) = \frac{e^{-x}}{(1 + e^{-x})^2}$$
$$\sigma\prime(x) = \frac{e^{-x}}{(1 + e^{-x})(1 + e^{-x})}$$
$$\sigma\prime(x) = \frac{1}{1 + e^{-x}} \cdot \frac{e^{-x}}{1 + e^{-x}}$$
$$\sigma\prime(x) = \frac{1}{1 + e^{-x}} \cdot \frac{e^{-x} + 1 - 1}{1 + e^{-x}}$$
$$\sigma\prime(x) = \frac{1}{1 + e^{-x}} \cdot \big(\frac{1 + e^{-x}}{1 + e^{-x}} - \frac{1}{1 + e^{-x}} \big)$$
$$\sigma\prime(x) = \frac{1}{1 + e^{-x}} \cdot \big(1 - \frac{1}{1 + e^{-x}}\big).$$

That is, we can write the derivative as follows
$$\sigma\prime(x) = \sigma(x)\cdot(1 - \sigma(x)).$$

![sigmoid_function](/images/loss_functions/sigmoid_function.png)
*Illustration of the Sigmoid function and its derivative.*


---
If this blog is useful for you, I'm thankful for your support!
{{< bmc-button slug="pumaline" >}}

