+++
title = 'Gradient Boost for Classification - Explained'
date = 2024-04-14T20:45:19-03:00
draft = true
tags = ["Data Science", "Machine Learning", "Gradient Boosting", "Ensemble", "Boosting", "Tree Methods", "Regression", "Classification"]
categories = ["Data Science", "Machine Learning", "Gradient Boosting", "Ensemble", "Boosting", "Tree Methods", "Classification", "Regression"]
keywords = ["Data Science", "Machine Learning", "Gradiend Boosting", "Ensemble", "Classification", "Tree Methods", "Regression"]
images = ['/images/gradient_boosting/gb_intro.png']
+++

---
## Introduction

Gradient Boosting is an [ensemble]({{< ref "/posts/ml_concepts/ensemble">}}) machine learning model, that - as the name suggests - is based on [boosting]({{< ref "/posts/ml_concepts/ensemble#boosting" >}}). An ensemble model based on boosting refers to a model that sequentially builds models, and the new model depends on the previous model. In Gradient Boosting these models are built such that they improve the error of the previous model. These individual models are so-called weak learners, which means models that have a low predictive skill. The ensemble of these weak learners build the final model - a strong learner with a high predictive skill. In this post, we will go through the algorithm of Gradient Boosting in general and then concretise the individual steps for a classification tasks using [Decision Trees]({{< ref "/posts/classical_ml/decision_trees" >}}) as weak learners and the log-loss function. There will be some overlapping with the article [Gradient Boosting for Regression - Explained]({{< ref "/posts/classical_ml/gradient_boosting_regression.md" >}}), where a detailed explanation of Gradient Boosting is given, which is then applied to a regression problem. However, in order to have the complete guide to a classification problem is one articel, we will repeat some of the underlying formulations. If you are interested in a concrete example with the detailed calculations, please refer to [Gradient Boosting for Regression - Example]({{< ref "/posts/classical_ml/gradient_boosting_regression_example.md" >}}) for a regression problem and [Gradient Boosting for Classification - Example]() for a classification problem.

## The Algorithm

Gradient Boosting is a [boosting]({{< ref "/posts/ml_concepts/ensemble#boosting" >}}) algorithm that aims to build a series of weak learners, which together act as a strong learner. In Gradient Boosting the objective is to improve the error of the preceeding model by minimizing its loss function using [Gradient Descent]({{< ref "/posts/ml_concepts/gradient_descent.md">}}). That means the weak learners are build up on the error and not up on the targets themselves as in other boosting algorithm like [AdaBoost]({{< ref "/posts/classical_ml/adaboost.md" >}})

In the following the algorithm is described for the general case. The notation is adapted from [Wikipedia](https://en.m.wikipedia.org/wiki/Gradient_boosting).

!["gradient boosting algorithm"](/images/gradient_boosting/gradient_boosting_algorithm.png)
*Gradient Boosting Algorithm. Adapted from [Wikipedia](https://en.wikipedia.org/wiki/Gradient_boosting).*

Let's have a look at the individual steps. The general case is explained in [Gradient Boosting for Regression - Explained]({{< ref "/posts/classical_ml/gradient_boosting_regression.md" >}}), we will apply them to the special case of a binary classification using Decision Trees as weak learners and the log-loss as [loss function]({{< ref "/posts/ml_concepts/loss_functions.md">}}). The [log-loss]({{< ref "/posts/ml_concepts/loss_functions.md#log_class">}}) is defined as

$$L\big(y_i, F(x_i)\big) = - y_i\cdot log\big(F(x_i)\big) - (1 - y_i)\cdot log\big(1 - F(x_i)\big),$$ 

with $y_i$ the true values and $F(x_i)$ the predicted propabilies. To make sure that $F(x_i)$ represent propabilities we use the [sigmoid]({{< ref "/posts/classical_ml/logistic_regression.md#sigmoid" >}}) function convert the output to values between $0$ and $1$

$$L\big(y_i, \gamma\big) = - y_i\cdot log\big(\sigma(\gamma)\big) - (1 - y_i)\cdot log\big(1 - \sigma(\gamma)\big),$$ 

with $\gamma$ the predicted values and $\sigma(\gamma) = \frac{1}{1 + e^{-\gamma}}$. 

With $\{(x_i, y_i)\}_{i=1}^n = \{(x_1, y_1), \dots, (x_n, y_n)\}$ be the training data, with $x = x_0, \dots, x_n$  the input features and $y = y_0, \dots, y_n$ the target values, the algorithm is as follows.

**Step 1 - Initialize the model with a constant value**

The first initialization of the model is given by

$$F_0(x) = \underset{\gamma}{\text{argmin}}\sum_{i=1}^n L(y_i, \gamma). $$

Using the log-loss as formulated above, this turns into

$$F_0(x) = \underset{\gamma}{\text{argmin}}\sum_{i=1}^n \big(- y_i\cdot log\big(\sigma(\gamma)\big) - (1 - y_i)\cdot log\big(1 - \sigma(\gamma)\big)\big), $$

The expression $\underset{\gamma}{\textit{argmin}}$ refers to finding the value for $\gamma$ which minimizes the equation. To find a minimum, we need to set the derivative equal to $0$. Let's calculaye the derivative with respect to $\gamma$.

$$\frac{\delta}{\delta \gamma} \sum_{i=1}^n L = \frac{\delta}{\delta \gamma}\sum_{i=1}^n\Big( - y_i\cdot log\big(\sigma(\gamma)\big) - (1 - y_i)\cdot log\big(1 - \sigma(\gamma)\big) \Big).$$

To calculate this derivative, we need to use the [chain rule](https://en.wikipedia.org/wiki/Chain_rule) and we need to remember the [derivative of the logarithm](https://www.cuemath.com/calculus/derivative-of-log-x/), which is.

$$\frac{d}{dz} log(z) = \frac{1}{z}$$

and using the chain rule

$$\frac{d}{dz} log(f(z)) = \frac{1}{f(z)} f'(z).$$

Note, that this is the derivative for the natural logarithm. If the logarithm is to a different base the derivative changes. The derivative of the sigmoid function is

$$\sigma\prime(z) = \sigma(z)\cdot(1 - \sigma(z)).$$

The derivation of this equation can be found [here]({{< ref "/posts/deep_learning/backpropagation.md#appendix" >}}). 

With this we get

$$\frac{\delta}{\delta \gamma} \sum_{i=1}^n L = \sum_{i=1}^n\Big(- y_i \frac{1}{\sigma(\gamma)}\sigma(\gamma)\big(1 - \sigma(\gamma)\big) - (1 - y_i)\frac{1}{1 - \sigma(\gamma)}\big(-\sigma(\gamma)(1 - \sigma(\gamma) \big)\Big).$$

This can be simplified to

$$\frac{\delta}{\delta \gamma} \sum_{i=1}^n L = \sum_{i=1}^n\Big(- y_i(1 - \sigma(\gamma)) + (1 - y_i) \sigma(\gamma)\Big)$$
$$\frac{\delta}{\delta \gamma} \sum_{i=1}^n L = \sum_{i=1}^n\Big(- y_i + y_i \sigma(\gamma)) + \sigma(\gamma) - y_i \sigma(\gamma)\Big)$$
$$\frac{\delta}{\delta \gamma} \sum_{i=1}^n L = \sum_{i=1}^n\Big(\sigma(\gamma) - y_i\Big).$$

Now we set the derivative to $0$ to find the minimum

$$0 = \sum_{i=1}^n\Big(\sigma(\gamma) - y_i\Big).$$

This can be transformed to

$$\sum_{i=1}^n\sigma(\gamma) = \sum_{i=1}^n y_i.$$

$$n\sigma(\gamma) = \sum_{i=1}^n y_i.$$

$$\sigma(\gamma) = \frac{1}{n}\sum_{i=1}^n y_i.$$

The right-hand site of this equation corresponds to the propapility of the positive class $p = \frac{1}{n}\sum_{i=1}^n y_i$. Using $\sigma(\gamma) = \frac{1}{1 + e^{-\gamma}}$, we get

$$p = \frac{1}{1 + e^{-\gamma}}$$
$$\frac{1}{p} = 1 + e^{-\gamma}$$
$$e^{-\gamma} = \frac{1}{p} - 1.$$

Applying the logarithm this leads to

$$-\gamma = log\Big(\frac{1}{p} - 1\Big)$$
$$-\gamma = log\Big(\frac{1-p}{p}\Big).$$

Using logarithmic transformations we get

$$\gamma = log\Big(\frac{p}{1-p}\Big).$$

The last transformation is explained in more detail in the appendix. 

This expression refers to *log of the odds* of the target variable, which is used to initialze the model for the specific case of a binary classification.

The next step is performed $M$ times, where $M$ refers to the number of weak learners used.

**Step 2 - for $m = 1$ to $M$**

**2A. Compute (pseudo-)residuals of the predictions and the true values.**

**2B. Fit a model (weak learner) closed after scaling $h_m(x)$.**

**2C. Find an optimized solution $\gamma_m$ for the loss function.**

**2D. Update the model.**

**Step 3 - Output final model $F_M(X)$.**

## Gradient Boosting in Python

## Summary

## Appendix

Derive $-log\big(\frac{x}{y}\big) = log\big(\frac{y}{x}\big)$:

$$-log \big(\frac{x}{y}\big) = - \big(log(x) - log(y)\big) = log(y) - log(x) = log\big(\frac{y}{x}\big)$$

If this blog is useful for you, please consider supporting.

{{< bmc-button slug="pumaline" >}}

