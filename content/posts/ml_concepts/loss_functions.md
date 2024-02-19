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

Loss Functions can be used in different ways for **training and evaluating** a Machine Learning Model. All Machine Learning models need to be evaluated with a metric to check how well the predictions fit the true values. These metrics can be considered as Cost Functions, because they measure the performance of the entire dataset. Examples for such evaluation metrics are e.g. the [Mean Squared Error]() for a regression task or the [Accuracy]() for a classification task. More examples for common metrics can be found in the separate articles [Metrics for Classification Problems]() and [Metrics for Regression Problems](). These metrics however, are not necessarily based on the same Loss Function that is used during the training of a model. Depending on the underlying algorithm Loss Functions are used in different ways. 

For [Decision Trees](), Loss Functions are used to guide the construction of the tree. For classification usually the Gini-Impurity or Entropy are used as Loss Function and for regression tasks the Sum of Squared Errors. These Losses are minimized as each split of the tree. We can therefore say, that in a Decision Tree at each split the local minimum of the Loss Function is calculated. Decision Trees follow a so-called [greedy search]() and assume that the sequence of locally optimal solutions lead to a globally optimal solution. In other words they assume by choosing the lowest Loss (error) at each split, they assume the overall Loss (error) of the model is also minimized. This assumption, however, does not always hold true.

Other Machine Learning models, like e.g. [Gradient Boosting]() or [Neural Networks]({{< ref "/posts/deep_learning/intro_dl.md">}}), use a global Loss Function to optimize the results. The Loss Functions are depending on the model's parameters, because the predictions are calculated based on these parameters. In a Neural Network these parameters are the weigths and the biases. During the training of such models we aim to change these parameters in such a way that the Loss (error) between the true values and the predicted values is minimized. That is we aim to minimize the Loss Function. This is an iterative process. To approximate the minimum of a function numerically different optimization techniques exist. The most popular one is [Gradient Descent]() or a variant of it. For a detailed explanation of [Gradient Descent](), please refer to the separate article. The main idea is to use the negative of the gradient of a function at a specific point to find the direction of the steepest descent in order to move into the direction of the minimum. That is why for such these type of models, the Loss Function needs to be differentiable. The small steps into this direction are taken in each training step. The parameters of the model are then updated using the gradient of the Loss Function. The process is illustrated in the following plot. 

![ai_ml_dl](/images/20231102_ai_ml_dl/gradient_descent.gif)
*Ilustration of Gradient Descent.*

During the training process the Loss is calculated after each training step. If the Loss is decreasing, we know that the model is improving, while when it is increasing we know that the training is not. The Loss thus guides the model into the correct direction. Note, in contrast to Decision Trees the Loss is not calculated locally for a specific region, but globally. 

## Examples

The choice of the Loss Function used, depends on the problem we are considering. Especially, we can divide them into two types. Loss Functions for regression tasks and Loss Functions for classification tasks. In a regression task we aim to predict continous values as close as possible (e.g. a price), while in a classification task we aim to predict the probability of a category (e.g. a grade). In the following we will discuss the most commly used Loss Functions for each case, and also define a customized Loss Function. 

### Loss Functions for Regression Tasks

**Mean Absolute Error**

The [Mean Absolute Error (MAE)]({{< ref "/posts/ml_concepts/regression_metrics.md#metrics">}}) or also called *L1-Loss*, is defined as

$$L(y, \hat{y}) = \frac{1}{N} \sum_{i=1}^N |y_i - \hat{y}_i|,$$

with $N$ the number of data samples, $y_i = (y_1, y_2, \dots, y_n)$ the true observation values, and $\hat{y}_i = (\hat{y}_1, \hat{y}_2, \dots, \hat{y}_N)$ the predicted values. The MAE is summarized in the following plot.

![msa](/images/20231001_regression_metrics/mae.jpg)

**Mean Squared Error**

The [Mean Squared Error (MSE)]({{< ref "/posts/ml_concepts/regression_metrics.md#metrics">}}) or also called *L2-Loss*, is defined as

$$L(y, \hat{y} = \frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2,$$

with $N$ the number of data samples, $y_i = (y_1, y_2, \dots, y_n)$ the true observation values, and $\hat{y}_i = (\hat{y}_1, \hat{y}_2, \dots, \hat{y}_N)$ the predicted values. The MSE is summarized in the following plot.

![mse](/images/20231001_regression_metrics/mse.jpg)

**Mean Absolute Percentage Error**

The [Mean Absolute Percentage Error (MAPE)]({{< ref "/posts/ml_concepts/regression_metrics.md#metrics">}}), is defined as

$$L(y, \hat{y}) = \frac{1}{N} \sum_{i=1}^N \frac{|y_i - \hat{y}_i|}{|y_i|} \cdot 100,$$

with $N$ the number of data samples, $y_i = (y_1, y_2, \dots, y_n)$ the true observation values, and $\hat{y}_i = (\hat{y}_1, \hat{y}_2, \dots, \hat{y}_N)$ the predicted values. The MAPE is summarized in the following plot.

![mse](/images/20231001_regression_metrics/mape.jpg)


**Huber Loss**

The Huber Loss is defined as

![huber_loss](/images/loss_functions/huber_loss.png)

with $\delta$ a hyperparameter, that specifies from which point on the loss should follow a linear curve instead a quadratic curve. The Huber Loss is summarized and illustrated for different parameters in the following plot.

![huber_loss](/images/loss_functions/huber_loss.gif)
*The Huber Loss.*

**Log-Cosh-Loss**

The Log-Cosh-Loss is very similar to the Huber loss. It also combines the advantages of both MSE and MAE. From the formular this is not as obvious as fo rthe Huber Loss, but it can be shown, that the Logcosh approximates a quadratic function, when the independent variable goes to zero and a linear function, when it goes to infinity [1].

$$L(y, \hat{y}) = \sum_{i=1}^N \log (\cosh (\hat{y}_i - y_i))$$

![logcosh_loss](/images/loss_functions/logcosh_loss.png)
*The Logcosh Loss.*

There are of course much more loss functions for regression tasks, the ones listed above are just a selection. They are compared in the below plot.

![loss_regression](/images/loss_functions/loss_functions_regression.png)
*Ilustration of different loss functions for regression tasks.*


### Loss Functions for Classification Tasks

As for regression tasks, in classification we use Loss Functions to meassure the error our model makes. The difference however is, that is this case we don't have continuous target values, but categorical classes and the predictions of our model are probabilities.

**(Binary-)Cross Entropy**

To explain *Cross Entropy*, we start with the special case of having two classes, i.e. a binary classification. The Cross Entropy, then turns into *Binary Cross Entropy (BCE)*, which is also often called *Log Loss*. 

The mathematical formulation is as follows 

$$L(y, \hat{y}) = -\frac{1}{N}\sum_{i=1}^N{\Big(y_i\log(\hat{y}_i) + (1 - y_i)\log(1 - \hat{y}_i)\Big)},$$

with $y = (y_1, \dots, y_N)$ the true label ($0$ or $1$) and $\hat{y} = (\hat{y}_1, \dots, \hat{y}_N)$ the predicted probability. To understand how Binary Cross Entropy works, let's consider just one sample. That means we can forget the outer sum over $i$ and have the Cost Function

$$L(y_i, \hat{y}_i) = -y_i\log(\hat{y}_i) - (1 - y_i)\log(1 - \hat{y}_i.$$

Let's consider two possible target outcomes. For the case $y_i = 0$, the first part of the sum vanishes and only

$$L(y_i, \hat{y}_i) = - \log(1-\hat{y}_i)$$ 

remains. On the other hand, for the case $y_i = 1$, the second part of the sum vanishes and only

$$L(y_i, \hat{y}_i) = - \log(\hat{y_i})$$

remains. These two functions are shown in the following plot.

![logcosh_loss](/images/loss_functions/bce.png)
*The Binary Cross Entropy Loss.*

We know that $\hat{y}_i$ is a probability and thus can only take values between $0$ and $1$. We can see that for the case $y_i = 0$, the loss is close to $0$, when $\hat{y}_i$ is close to $0$ and it increases if $\hat{y}_i$ approaches $1$. That means the Binary Cross Entropy penalizes more the further away the predicted probability is from the true value. The same holds true if $y_i = 1$. In this case the loss is high if $\hat{y}_i$ is close to $0$ and low if it approaches $1$.

The more general formulation for $M>2$ classes of the Cross Entropy is

$$L(y, \hat{y}) = -\frac{1}{N}\sum_{j=1}^M\sum_{i=1}^N{y_{i,j}\log(\hat{y}_{i,j})}.$$


**Hinge Loss**

< IMAGE WITH DIFFERENT LOSS FUNCTIONS >

### Example for a custom Loss Function

## Specific Machine Learning Models and their Loss Functions

Some Machine Learning Algorithm have a fixed Loss Function, while others are flexible and the Loss Function can be adapted to the specific task. The following table gives an (non-exaustive) overview about some popular Machine Learning Algorithms and their corresponding Loss Functions.

< IMAGE TABLE >

## Which Loss Function to Choose?

Choosing an appropriate Loss Function is very importan, because it is used to evaluate and improve the model performance. It should thus reflect well the metric that is important for the project, so that errors are minimized accordingly. 

## Summary

## Further Reading

[1] ["Log Hyperbolic Cosine Loss Improves Variational Auto-Encoder"](https://openreview.net/pdf?id=rkglvsC9Ym), anonymous authors, 2019. 

---
If this blog is useful for you, I'm thankful for your support!
{{< bmc-button slug="pumaline" >}}

