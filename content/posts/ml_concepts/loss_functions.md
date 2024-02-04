+++
title = 'Loss Functions'
date = 2024-02-04T18:57:51-03:00
draft = true
tags = ["Data Science", "Machine Learning", "Deep Learning"]
categories = ["Data Science", "Machine Learning", "Deep Learning"]
keywords = ["Data Science", "Machine Learning", "Deep Learning"]
images = ['/images/']
+++

## Introduction

In Machine Learning Loss Functions are used to evaluate the model. It is used to compare the true target values with the predicted once and measures the error of the predictions. During the training of a model the Loss Function is aimed to be optimized to monimize the error of the predictions. The specific choice of Loss Function depends on the problem we want to solve, e.g. whether a regression or a classification task is considered. In this article we will discuss the most common once, but we can create custom Loss Functions for specific problems. We will also give an example for a custom Loss Function later in this post. The idea of using custom Loss Functions is that they focus on the specific errors we aim to minimize, the only condition they need to safisfy is that they need to be differentiable.  

**Terminology**

The term *Loss Function* is most commonly used, however in some contexts they are also called *Cost Function*, *Objective Function*, or *Error Function*. The different naming can be a bit confusing, but they usually refer to the exact same thing.

### How are Loss Functions optimized


In Deep Learning the Loss Function is a function depending on the weigths and the biases and we ain to minimize it with respect to them. 

## Examples

### Loss Functions for Regression Tasks

**Mean Squared Error**

**Mean Absolite Error**

**Huber Loss**

### Loss Functions for Classification Tasks

**Cross Entropy**

### Example for a custom Loss Function

## Summary

If this blog is useful for you, I'm thankful for your support!
{{< bmc-button slug="pumaline" >}}

