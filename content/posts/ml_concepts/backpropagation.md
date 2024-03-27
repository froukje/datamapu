+++
title = 'Backpropagation - Step by Step'
date = 2024-03-20T22:49:06-03:00
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

Throughout the examples of this post, we use the following training data, activation function and loss.

**Training Data**

We use the following data with $x = (x_1, x_2)$ the inputs and $y = (y_1, y_2)$ the labels.

$$x_1 = 0.5, x_2 = 1$$
$$y_1 = 1.5, y_2 = 2$$
 
**Activation Function**

As activation function, we use the *Sigmoid function*

$$\sigma(x) = \frac{1}{1 + e^{-x}}.$$

**Loss Function**

As loss function we use the *Mean Squared Error*

$$L(y, \hat{y}) = \frac{1}{2}\big((y_1 - \hat{y}_1) + (y_2 - \hat{y}_2)\big),$$

with $y = (y_1, y_2)$ the labels and $\hat{y} = (\hat{y}_1, \hat{y}_2)$ the predicted labels.

## Example: 1 Neuron

To illustrate how backpropagation works, we start with the most simple neural network, which only constists of one single neuron. 

![one_neuron](/images/backpropagation/one_neuron.png)
*Illustration of a Neural Network consisting of a single Neuron.*

For the following calculations, we assume the initial weight to be $w = 0.3$ and the initial bias to be $b = 0.1$. Further the learning rate is set to$\alpha = 0.1$. This values are chosen arbitrary for illustration purposes.

We can calculate the forward pass through this network as

$$\hat{y} = \sigma(wx + b),$$
$$\hat{y} = \frac{1}{1 + e^{-(wx+b)}}$$.

Using the dataset defined above, we get for $x_1 = 0.5$

$$\hat{y}_1 = \frac{1}{1 + e^{-0.3\cdot 0.5 - 0.1}} = \frac{1}{1 + e^{-0.05}} = \frac{1}{1 + e^{0.05}} \approx 0.51$$

![one_neuron](/images/backpropagation/one_neuron_back.png)
*Illustration of Backpropagation in a Neural Network consisting of a single Neuron.*

## Example: 2 Neurons

* in a row
* parallel

## Example: Small Network

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

