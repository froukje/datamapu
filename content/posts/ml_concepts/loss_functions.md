+++
title = 'Loss Functions in Machine Learning'
date = 2024-02-04T18:57:51-03:00
draft = true
tags = ["Data Science", "Machine Learning", "Deep Learning"]
categories = ["Data Science", "Machine Learning", "Deep Learning"]
keywords = ["Data Science", "Machine Learning", "Deep Learning"]
images = ['/images/']
+++

## Introduction

In Machine Learning Loss Functions are used to evaluate the model. They are used to compare the true target values with the predicted once and are directly related the error of the predictions. During the training of a model the Loss Function is aimed to be optimized to minimize the error of the predictions. The specific choice of Loss Function depends on the problem we want to solve, e.g. whether a regression or a classification task is considered. In this article we will discuss the most common once, which work very well for a lot of tasks. We can, however, also create custom Loss Functions adapted for specific problems. Custom Loss Functions help to focus on the specific errors we aim to minimize, the only condition they need to safisfy is that they need to be differentiable. We will give an example of a custom Loss function later is this post. 

**Terminology**

The term *Loss Function* is most commonly used, however in some contit is also called *Error Function*.  The Loss Function is applied to each sample of the dataset, it is related to the *Cost Function* (sometimes also called *Objective Function*), which is the average of all Loss Function values. The Cost Function is therefore a measure of how the model performs on the entire dataset. The outcome of the Loss Function is called the *Loss*.

## Which ML Models use Loss Functions?

Loss Functions are used in a set of Machine Learning Models to train and improve them.

Yes: Gradient Boosting, Neural Net

No: Decision Trees, Random Forest, AdaBoost

## How are Loss Functions used during Training?

Loss Functions are functions depending on the model's parameters, because it includes the predictions, which are calculated by the model. These parameters are different depending on the model type. For example, in XGBoost ...  . In a neural network these parameters are the weigths and the biases. During the training of a model we aim to change these paramters in such a way that the error (or the Loss) between the true values and the predicted values is minimized. That is we aim to minimize the Loss Function. Minimizing the Loss function is an iterative process. To approximate the minimum of a function numerically different optimization techniques exist. The most popular one is [Gradient Descent]() or a variant of it. In this post, we are not going into detail, for a detailed explanation of [Gradient Descent](), please refer to the separate article. The main idea is to use the negative of the gradient of a function at a specific point to find the direction of the steepest descent in order to move int othe direction of the minimum. This is why the Loss Function needs to be differentiable. The small steps into this direction are taken in each training step. The parameters of the model are then updated using the gradient of the Loss Function. The process is illustrated in the following plot. 


During the training process the Loss is calculated after each training step. If the Loss is decreasing, we know that the model is improving, while when it is increasing we know that the training is not. The Loss thus guides the model into the correct direction. 

## Examples

The choice of the Loss Function used, depends on the problem we are considering. Especially, we can divide them into two types. Loss Functions for regression tasks and Loss Functions for classification tasks. In a regression task we aim to predict continous values as close as possible (e.g. a price), while in a classification task we aim to predict the probability of a category (e.g. a grade). In the following we will discuss the most commly used Loss Functions for each case, and also define a customized Loss Function.

### Loss Functions for Regression Tasks

**Mean Squared Error / L2-Loss**

**Mean Absolute Error / L1-Loss**

**Huber Loss**

**Log-Cosh-Loss**

### Loss Functions for Classification Tasks

**(Binary-)Cross Entropy**

**Hinge Loss**


### Example for a custom Loss Function

## Summary

---
If this blog is useful for you, I'm thankful for your support!
{{< bmc-button slug="pumaline" >}}

