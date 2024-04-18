+++
title = 'Loss Functions in Machine Learning'
date = 2024-02-04T18:57:51-03:00
draft = true
tags = ["Data Science", "Machine Learning", "Deep Learning"]
categories = ["Data Science", "Machine Learning", "Deep Learning"]
keywords = ["Data Science", "Machine Learning", "Deep Learning"]
images = ['/images/loss_functions/loss_function.png']
+++

## Introduction

In Machine Learning Loss Functions are used to evaluate the model. They compare the true target values with the predicted ones and are directly related to the error of the predictions. During the training of a model, the Loss Function is aimed to be optimized to minimize the error of the predictions. It is a general convention to define a Loss Function such that it is minimized rather than maximized. The specific choice of a Loss Function depends on the problem we want to solve, e.g. whether a regression or a classification task is considered. In this article, we will discuss the most common ones, which work very well for a lot of tasks. We can, however, also create custom Loss Functions adapted for specific problems. Custom Loss Functions help to focus on the specific errors we aim to minimize. We will have a look at examples of custom Loss Functions later in this post. 

![loss_function](/images/loss_functions/loss_function.png)

**Terminology**

The term *Loss Function* is most commonly used, however, sometimes it is also called *Error Function*.  The outcome of the Loss Function is called *Loss*. The Loss Function is applied to each sample of the dataset, it is related to the *Cost Function* (sometimes also called *Objective Function*), which is the average of all Loss Function values. The Cost Function is therefore a measure of how the model performs on the entire dataset, while the Loss Function evaluates the Loss for each sample. In practice, however, the terms *Loss Function* and *Cost Function* are often used interchangeably. 

## How are Loss Functions used in Machine Learning?

Loss Functions can be used in different ways for **training and evaluating** a Machine Learning model. All Machine Learning models need to be evaluated with a metric to check how well the predictions fit the true values. These metrics can be considered as Cost Functions because they measure the performance of the entire dataset. Examples of such evaluation metrics are e.g. the [Mean Squared Error]({{< ref "regression_metrics#metrics" >}} "regression_metrics") for a regression task or the [Accuracy]({{< ref "classification_metrics#metrics" >}} "classification_metrics") for a classification task. More examples of common metrics can be found in the separate articles [Metrics for Classification Problems]({{< ref "classification_metrics">}} "classification_metrics") and [Metrics for Regression Problems]({{< ref "regression_metrics">}} "regression_metrics"). These, metrics however, are not necessarily based on the same Loss Function that is used during the training of a model. Depending on the underlying algorithm Loss Functions are used in different ways. 

For [Decision Trees]({{< ref "decision_trees">}} "decision_trees"), Loss Functions are used to guide the construction of the tree. For classification usually the Gini-Impurity or Entropy is used as Loss Function and for regression tasks the Sum of Squared Errors. These Losses are minimized at each split of the tree. We can therefore say, that in a Decision Tree at each split the local minimum of the Loss Function is calculated. Decision Trees follow a so-called [greedy search](https://en.wikipedia.org/wiki/Greedy_algorithm) and assume that the sequence of locally optimal solutions leads to a globally optimal solution. In other words, they assume by choosing the lowest Loss (error) at each split, the overall Loss (error) of the model is also minimized. This assumption, however, does not always hold.

Other Machine Learning models, like e.g. [Gradient Boosting]({{< ref "gradient_boosting_regression">}} "gradient_boosting") or [Neural Networks]({{< ref "/posts/deep_learning/intro_dl.md">}} "neural_net"), use a global Loss Function to optimize the results. The Loss Functions are depending on the model's parameters because the predictions are calculated based on these parameters. In a Neural Network, these parameters are the weights and the biases. During the training of such models, we aim to change these parameters in such a way that the Loss (error) between the true values and the predicted values is minimized. That is we aim to minimize the Loss Function. This is an iterative process. To approximate the minimum of a function numerically different optimization techniques exist. The most popular one is [Gradient Descent]({{< ref "/posts/ml_concepts/gradient_descent.md">}} "gradient_descent") or a variant of it. The main idea is to use the negative of the gradient of a function at a specific point to find the direction of the steepest descent to move into the direction of the minimum. That is why for such types of models, the Loss Function needs to be differentiable. Small steps in the direction of the minimum are taken in each training step. The parameters of the model are then updated using the gradient of the Loss Function. The process is illustrated in the following plot. For a more detailed explanation, please refer to the separate article about [Gradient Descent]({{< ref "gradient_descent">}} "gradient_descent").

![ai_ml_dl](/images/20231102_ai_ml_dl/gradient_descent.gif)
*Ilustration of Gradient Descent.*

During the training process the Loss is calculated after each training step. If the Loss is decreasing, we know that the model is improving, while when it is increasing we know that is not. The Loss thus guides the model into the correct direction. Note, in contrast to Decision Trees the Loss is not calculated locally for a specific region, but globally. 

## Examples

The choice of the Loss Function used depends on the problem we are considering. Especially, we can divide them into two types. Loss Functions for regression tasks and Loss Functions for classification tasks. In a regression task, we aim to predict continuous values as closely as possible (e.g. a price), while in a classification task, we aim to predict the probability of a category (e.g. a grade). In the following, we will discuss the most commonly used Loss Functions for each case, and also define a customized Loss Function. 

### Loss Functions for Regression Tasks{#loss_reg}

**Mean Absolute Error**

The [Mean Absolute Error (MAE)]({{< ref "/posts/ml_concepts/regression_metrics.md#metrics">}}) or also called *L1-Loss*, is defined as

$$L(y_i, \hat{y}_i) = |y_i - \hat{y}_i|,$$

and accordingly ththe cost function over all samples

$$f(y, \hat{y}) = \frac{1}{N} \sum_{i=1}^N |y_i - \hat{y}_i|,$$

with $N$ the number of data samples, $y_i = (y_1, y_2, \dots, y_n)$ the true observation values, and $\hat{y}_i = (\hat{y}_1, \hat{y}_2, \dots, \hat{y}_N)$ the predicted values. The MAE is summarized in the following plot.

![msa](/images/20231001_regression_metrics/mae.jpg)

**Mean Squared Error**

The [Mean Squared Error (MSE)]({{< ref "/posts/ml_concepts/regression_metrics.md#metrics">}}) or also called *L2-Loss*, is defined as

$$L(y_i, \hat{y}_i) = (y_i - \hat{y}_i)^2,$$

and accordingly th ecost function over all samples

$$f(y, \hat{y}) = \frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2,$$

with $N$ the number of data samples, $y_i = (y_1, y_2, \dots, y_n)$ the true observation values, and $\hat{y}_i = (\hat{y}_1, \hat{y}_2, \dots, \hat{y}_N)$ the predicted values. The MSE is summarized in the following plot.

![mse](/images/20231001_regression_metrics/mse.jpg)

**Mean Absolute Percentage Error**

The [Mean Absolute Percentage Error (MAPE)]({{< ref "/posts/ml_concepts/regression_metrics.md#metrics">}}), is defined as

$$L(y, \hat{y}) = \frac{|y_i - \hat{y}_i|}{|y_i|} \cdot 100,$$

and accordingly the cost function over all data samples

$$f(y, \hat{y}) = \frac{1}{N} \sum_{i=1}^N \frac{|y_i - \hat{y}_i|}{|y_i|} \cdot 100,$$

with $N$ the number of data samples, $y_i = (y_1, y_2, \dots, y_n)$ the true observation values, and $\hat{y}_i = (\hat{y}_1, \hat{y}_2, \dots, \hat{y}_N)$ the predicted values. The MAPE is summarized in the following plot.

![mse](/images/20231001_regression_metrics/mape.jpg)


**Huber Loss**

The Huber Loss is a possibility to combine the advantages of both the MSE and MAE. It is defined as

![huber_loss](/images/loss_functions/huber_loss.png)

with $\delta$ a hyperparameter, that specifies from which point the loss should follow a linear curve instead of a quadratic curve. The Huber Loss is summarized and illustrated for different parameters in the following plot.

![huber_loss](/images/loss_functions/huber_loss.gif)
*The Huber Loss.*

**Log-Cosh-Loss**

The Log-Cosh-Loss is very similar to the Huber loss. It also combines the advantages of both MSE and MAE. From the formula this is not as obvious as for the Huber Loss, but it can be shown, that the Logcosh approximates a quadratic function, when the independent variable goes to zero and a linear function, when it goes to infinity [1].

$$L(y, \hat{y}) = \sum_{i=1}^N \log (\cosh (\hat{y}_i - y_i))$$

![logcosh_loss](/images/loss_functions/logcosh_loss.png)
*The Logcosh Loss.*

There are of course much more loss functions for regression tasks, the ones listed above are just a selection. They are compared in the below plot.

![loss_regression](/images/loss_functions/loss_functions_regression.png)
*Ilustration of different loss functions for regression tasks.*


### Loss Functions for Classification Tasks{#loss_class}

As for regression tasks, in classification, we use Loss Functions to measure the error our model makes. The difference however is, that in this case, we don't have continuous target values, but categorical classes and the predictions of our model are probabilities.

**(Binary-)Cross Entropy**

*(Binary-)Cross Entropy* is the most used Loss for classification problems. To explain *Cross Entropy*, we start with the special case of having two classes, i.e. a binary classification. The Cross Entropy then turns into *Binary Cross Entropy (BCE)*, which is also often called *Log Loss*. 

The mathematical formulation of the Cost Function is as follows 

$$L(y, \hat{y}) = -\frac{1}{N}\sum_{i=1}^N{\Big(y_i\log(\hat{y}_i) + (1 - y_i)\log(1 - \hat{y}_i)\Big)},$$

with $y = (y_1, \dots, y_N)$ the true label ($0$ or $1$) and $\hat{y} = (\hat{y}_1, \dots, \hat{y}_N)$ the predicted probability. To understand how Binary Cross Entropy works, let's consider just one sample. That means we can forget the outer sum over $i$ and get the Loss Function

$$L(y_i, \hat{y}_i) = -y_i\log(\hat{y}_i) - (1 - y_i)\log(1 - \hat{y}_i).$$

Let's consider two possible target outcomes. For the case $y_i = 0$, the first part of the sum vanishes and only

$$L(y_i, \hat{y}_i) = - \log(1-\hat{y}_i)$$ 

remains. On the other hand, for the case $y_i = 1$, the second part of the sum vanishes and only

$$L(y_i, \hat{y}_i) = - \log(\hat{y}_i)$$

remains. These two functions are shown in the following plot.

![logcosh_loss](/images/loss_functions/bce.png)
*The Binary Cross Entropy Loss.*

We know that $\hat{y}_i$ is a probability and thus can only take values between $0$ and $1$. We can see that for the case $y_i = 0$ the loss is close to $0$, when $\hat{y}_i$ is close to $0$ and it increases if $\hat{y}_i$ approaches $1$. That means the Binary Cross Entropy penalizes more the further away the predicted probability is from the true value. The same holds if $y_i = 1$. In this case, the loss is high if $\hat{y}_i$ is close to $0$ and low if it approaches $1$.

The more general formulation for $M>2$ classes of the Cross Entropy is

$$L(y, \hat{y}) = -\frac{1}{N}\sum_{j=1}^M\sum_{i=1}^N{y_{i,j}\log(\hat{y}_{i,j})}.$$


**Hinge Loss**

The *Hinge Loss* is especially used by Support Vector Machines (SVM). SVM is a type of Machine Learning model, which aims to create hyperplanes (or in two dimensions decision boundaries), which can be used to separate the data points into classes. The Hinge Loss is used to measure the distance of points to the hyperplane or decision boundary.  For binary classification, it is defined as

$$L(y_i, \hat{y}_i) = max(0, 1 - y_i \cdot \hat{y}_i),$$

with $\hat{y}_i$ the predicted value and $y_i$ the true target value for $i = 1, 2, \dots N$. The convention is that the true values have values $-1$ and $1$. The Hinge Loss is zero, when $y_i\cdot\hat{y}_i >= 1$, which is the case for $y_i = 1$ and $\hat{y}_i >= 1$ or $y_i = -1$ and $\hat{y}_i <= -1$. In both cases, the data point was correctly classified, thus the loss of $0$ means we are not penalizing our model. For the cases $y_i = 1$ and $\hat{y}_i < 1$ and $y_i = -1$ and $\hat{y}_i > -1$, $y_i \cdot \hat{y}_i < 1$ and the Loss is therefore positive. The further away $\hat{y}_i$ is from the true value the more increases the loss.

The Hinge Loss can be extended to multi-class classification. This is, however, not in the scope of this article. An explanation can be found on [Wikipedia](https://en.wikipedia.org/wiki/Hinge_loss).

![hinge_loss](/images/loss_functions/hinge_loss.png)
*The Hinge Loss illustrated.*

### Example for a custom Loss Function

The above discussed examples are the most common ones, however, we can define a Loss Function specific to our needs. We learned that the MSE penalizes outliers more than the MAE. If we want our Loss to penalize even stronger the outliers, we could for example define loss of degree $4$ as follows

$$L(y_i, \hat{y}_i) = (y_i - \hat{y}_i)^4.$$

We can also define an assymentric Loss that penalizes more negative values than positive values

![asym_loss](/images/loss_functions/asym_loss_klein.png)

in both cases with $\hat{y} = (y_1, \dots, y_N)$ the predicted value and $y = (y_1, \dots, y_N)$ the true values. Customized Loss Functions may help our model to learn. These were two examples for regression tasks, custom Loss Functions can however also be created for classification tasks.

![custom_loss](/images/loss_functions/custom_loss.png)
*Two examples of custom Loss Functions.*

## Specific Machine Learning Models and their Loss Functions

Some Machine Learning algorithms have a fixed Loss Function, while others are flexible and the Loss Function can be adapted to the specific task. The following table gives a (non-exhaustive) overview of some popular Machine Learning Algorithms and their corresponding Loss Functions. Important to keep in mind is that the Loss Function needs to be differentiable if some form of [Gradient Descent]() is performed, e.g. in [Neural Networks]({{< ref "/posts/deep_learning/intro_dl.md">}}) or [Gradient Boosting]({{< ref "gradient_boosting_regression">}} "gradient_boosting").

![loss_functions_examples](/images/loss_functions/loss_functions_examples.png)
*Examples for Machine Learning models and their Loss Functions.*

## Summary

Loss Functions are used to evaluate a model and to analyze if it is learning. We discussed typical Loss Functions for regression and classification tasks and also saw two examples of customized Loss Functions. Choosing an appropriate Loss Function is very important because it is used to evaluate and improve the model performance. It should thus reflect well the metric that is important for the project so that errors are minimized accordingly. 

## Further Reading

[1] ["Log Hyperbolic Cosine Loss Improves Variational Auto-Encoder"](https://openreview.net/pdf?id=rkglvsC9Ym), anonymous authors, 2019. 

---
If this blog is useful for you, please consider supporting.

{{< bmc-button slug="pumaline" >}}

