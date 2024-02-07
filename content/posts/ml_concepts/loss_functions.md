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

In Machine Learning Loss Functions are used to evaluate the model. They are used to compare the true target values with the predicted once and are directly related to the error of the predictions. During the training of a model the Loss Function is aimed to be optimized to minimize the error of the predictions. It is general convention to define a Loss Function such that it is minimized rather than maximized. The specific choice of Loss Function depends on the problem we want to solve, e.g. whether a regression or a classification task is considered. In this article we will discuss the most common once, which work very well for a lot of tasks. We can, however, also create custom Loss Functions adapted for specific problems. Custom Loss Functions help to focus on the specific errors we aim to minimize. We will give an example of a custom Loss Function later is this post. 

**Terminology**

The term *Loss Function* is most commonly used, however sometimes it is also called *Error Function*.  The outcome of the Loss Function is called *Loss*. The Loss Function is applied to each sample of the dataset, it is related to the *Cost Function* (sometimes also called *Objective Function*), which is the average of all Loss Function values. The Cost Function is therefore a measure of how the model performs on the entire dataset, while the Loss Function evaluates the Loss for each sample. 

## How are Loss Functions used in Machine Learning?

Loss Functions can be used in different ways for **training and evaluating** a Machine Learning Model. All Machine Learning models need to be evaluated with a metric to check how well the predictions fit the true values. These metrics can be considered as Cost Functions, because they measure the performance of the entire dataset. Examples for such evaluation metrics are e.g. the [Mean Squared Error]() for a regression task or the [Accuracy]() for a classification task. More examples for common metrics can be found in the separate articles [Metrics for Classification Problems]() and [Metrics for Regression Problems](). These metrics however, are not necessarily based on the same Loss Function that is used during the training of a model. Depending on the underlying algorithm Loss Functions are used in different way. 

For [Decision Trees](), Loss Functions are used to guide the construction of the tree. For classification usually the Gini-Impurity or Entropy are used as Loss Function and for regression tasks the Sum of Squared Errors. These Losses are minimized as each split of the tree. We can therefore say, that in a Decision Tree at each split the local minimum of the Loss Function is calculated. Decision Trees follow a so-called [greedy search]() and assume that the sequence of locally optimal solutions lead to a globally optimal solution. In other words they assume by choosing the lowest Loss (error) at each split, they assume the overall Loss (error) of the model is also minimized. This assumption, however, does not always hold true.

Other Machine Learning models, like e.g. [Gradient Boosting]() or Neural Networks, use a global Loss Function to optimize the results. The Loss Functions are depending on the model's parameters, because the predictions are calculated based on these parameters. In a neural network these parameters are the weigths and the biases. During the training of such models we aim to change these parameters in such a way that the Loss (error) between the true values and the predicted values is minimized. That is we aim to minimize the Loss Function. This is an iterative process. To approximate the minimum of a function numerically different optimization techniques exist. The most popular one is [Gradient Descent]() or a variant of it. For a detailed explanation of [Gradient Descent](), please refer to the separate article. The main idea is to use the negative of the gradient of a function at a specific point to find the direction of the steepest descent in order to move into the direction of the minimum. That is why for such these type of models, the Loss Function needs to be differentiable. The small steps into this direction are taken in each training step. The parameters of the model are then updated using the gradient of the Loss Function. The process is illustrated in the following plot. 

< IMAGE >

During the training process the Loss is calculated after each training step. If the Loss is decreasing, we know that the model is improving, while when it is increasing we know that the training is not. The Loss thus guides the model into the correct direction. Note, in contrast to Decision Trees the Loss is not calculated locally for a specific region, but globally. 

## Examples

The choice of the Loss Function used, depends on the problem we are considering. Especially, we can divide them into two types. Loss Functions for regression tasks and Loss Functions for classification tasks. In a regression task we aim to predict continous values as close as possible (e.g. a price), while in a classification task we aim to predict the probability of a category (e.g. a grade). In the following we will discuss the most commly used Loss Functions for each case, and also define a customized Loss Function. 

### Loss Functions for Regression Tasks

**Mean Squared Error / L2-Loss**

**Mean Absolute Error / L1-Loss**

**Huber Loss**

**Log-Cosh-Loss**

< IMAGE WITH DIFFERENT LOSS FUNCTIONS >

### Loss Functions for Classification Tasks

**(Binary-)Cross Entropy**

**Hinge Loss**

< IMAGE WITH DIFFERENT LOSS FUNCTIONS >

### Example for a custom Loss Function

### Specific Machine Learning Models and their Loss Functions

Some Machine Learning Algorithm have a fixed Loss Function, while others are flexible and the Loss Function can be adapted to the specific task. The following table gives an (non-exaustive) overview about some popular Machine Learning Algorithms and their corresponding Loss Functions.

< IMAGE TABLE >

## Summary

---
If this blog is useful for you, I'm thankful for your support!
{{< bmc-button slug="pumaline" >}}

