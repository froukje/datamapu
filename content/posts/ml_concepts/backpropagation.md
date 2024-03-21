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

## Example: 1 Neuron

## Example: 2 Neurons

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

