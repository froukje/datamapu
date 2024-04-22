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

In the following the algorithm is described for the general case. The notation is adapted from [Wikipedia](https://en.m.wikipedia.org/wiki/Gradient_boosting). The general case is explained in [Gradient Boosting for Regression - Explained]({{< ref "/posts/classical_ml/gradient_boosting_regression.md" >}}), , in this post we apply them to the special case of a binary classification using Decision Trees as weak learners and the log-loss as [loss function]({{< ref "/posts/ml_concepts/loss_functions.md">}}).

!["gradient boosting algorithm"](/images/gradient_boosting/gradient_boosting_algorithm.png)
*Gradient Boosting Algorithm. Adapted from [Wikipedia](https://en.wikipedia.org/wiki/Gradient_boosting).*

Let's have a look at the individual steps. The [log-loss]({{< ref "/posts/ml_concepts/loss_functions.md#log_class">}}), which we use for the special case of a binary classification is defined as

$$L\big(y_i, p_i\big) = - y_i\cdot \log p_i - (1 - y_i)\cdot \log\big(1 - p_i\big),$$ 

with $y_i$ the true values and $p_i$ the predicted propabilities. To assure that $p$ represent propabilities, we use the [sigmoid]({{< ref "/posts/classical_ml/logistic_regression.md#sigmoid" >}}) function convert the model output to values between $0$ and $1$, i.e. $p_i = \sigma\big(F_{m-1}(x_i)\big)$.

$$L\big(y_i, \gamma\big) = - y_i\cdot \log\big(\sigma(\gamma)\big) - (1 - y_i)\cdot \log\big(1 - \sigma(\gamma)\big),$$ 

with $\gamma$ the predicted values and $p = \sigma(\gamma) = \frac{1}{1 + e^{-\gamma}}$. 

With $\{(x_i, y_i)\}_{i=1}^n = \{(x_1, y_1), \dots, (x_n, y_n)\}$ be the training data, with $x = x_0, \dots, x_n$  the input features and $y = y_0, \dots, y_n$ the target values, the algorithm is as follows.

**Step 1 - Initialize the model with a constant value**

The first initialization of the model is given by

$$F_0(x) = \underset{\gamma}{\text{argmin}}\sum_{i=1}^n L(y_i, \gamma). $$

Using the log-loss as formulated above, this turns into

$$F_0(x) = \underset{\gamma}{\text{argmin}}\sum_{i=1}^n \big(- y_i\cdot log\big(\sigma(\gamma)\big) - (1 - y_i)\cdot log\big(1 - \sigma(\gamma)\big)\big), $$

The expression $\underset{\gamma}{\textit{argmin}}$ refers to finding the value for $\gamma$ which minimizes the equation. To find a minimum, we need to set the derivative equal to $0$. Let's calculaye the derivative with respect to $\gamma$.

$$\frac{\delta}{\delta \gamma} \sum_{i=1}^n L = \frac{\delta}{\delta \gamma}\sum_{i=1}^n\Big( - y_i\cdot log\big(\sigma(\gamma)\big) - (1 - y_i)\cdot log\big(1 - \sigma(\gamma)\big) \Big).$$

To calculate this derivative, we need to use the [chain rule](https://en.wikipedia.org/wiki/Chain_rule) and we need to remember the [derivative of the logarithm](https://www.cuemath.com/calculus/derivative-of-log-x/), which is.

$$\frac{d}{dz} \log(z) = \frac{1}{z}$$

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

$$-\gamma = \log\Big(\frac{1}{p} - 1\Big)$$
$$-\gamma = \log\Big(\frac{1-p}{p}\Big).$$

Using logarithmic transformations we get

$$\gamma = log\Big(\frac{p}{1-p}\Big).$$

The last transformation is explained in more detail in the appendix. 

This expression refers to *log of the odds* of the target variable, which is used to initialze the model for the specific case of a binary classification.

The next step is performed $M$ times, where $M$ refers to the number of weak learners used.

**Step 2 - for $m = 1$ to $M$**

**2A. Compute (pseudo-)residuals of the predictions and the true values.**

The (pseudo-) residuals are defined as
![pseudo_residual](/images/gradient_boosting/pseudo_residual.drawio.png)

for  $i = 1, \dots, n$. 

That is we need to calculate the derivative of the loss function with respect to the predictions. 

With the loss function $L(y_i, p_i)$ defined above, where $p$ are the propabilities after applying the sigmoid function, $p_i= \sigma\big(F_{m-1}(x_i)\big) = \sigma(y_i)$, this turns into

$$r_{im} = -\frac{\delta L(y_i, p_i)}{\delta \gamma}$$

To calculate this derivative we need to use the chain-rule

$$\frac{\delta L}{\delta \gamma} = \frac{\delta L}{\delta p_i}\frac{\delta p_i}{\delta \gamma}.$$

We calculate the first derivative $\frac{\delta L}{\delta p_i}$.

$$\frac{\delta L}{\delta p_i} = \frac{\delta\Big(-y_i\cdot \log p_i - (1 - y_i)\cdot \log(1 - p_i)\Big)}{\delta p_i},$$
$$\frac{\delta L}{\delta p_i} = \frac{-y_i\cdot\delta \log p_i}{\delta p_i} - \frac{(1 - y_i)\cdot\delta \log\big(1 - p_i\big)}{\delta p_i}$$

Using the derivative of the logarithm as above, this leads to

$$\frac{\delta L}{\delta p_i} = - \frac{y_i}{p_i} + \frac{1 - y_i}{1 - p_i}$$

For the second part of the derivative we again need the [derivative of the sigmoid function]({{< ref "/posts/deep_learning/backpropagation.md#appendix" >}}).

$\frac{\delta p_i}{\delta \gamma} = \frac{\delta \sigma (\gamma)}{\delta \gamma} = \sigma (\gamma) \cdot \big(1 - \sigma (\gamma)\big) = p_i\cdot(1-p_i)$$

Now, we can calculate the derivative $\frac{\delta L}{\delta \gamma}$ as

$$\frac{\delta L}{\delta \gamma} = -\Big(\frac{y_i}{p_i} - \frac{1 - y_i}{1 - p}\Big)\cdot p_i \cdot (1 - p_i)$$
$$\frac{\delta L}{\delta \gamma} = -\big(y_i (1 - p_i) - (1 - y_i) p_i\big)$$
$$\frac{\delta L}{\delta \gamma} = p_i - y_i$$

That is the (pseudo-)residuals are given as

$$r_{im} = - (p_i^{m-1} - y_i),$$
$$r_{im} = y_i - p_i^{m-1}$$

with $p_i^{m-1}$ the propabilities of the previous weak learner.

**2B. Fit a model (weak learner) closed after scaling $h_m(x)$.**

In this step we fit a weak model to the input features and the residuals $(x_i, r_{im})_{i=1}^n$. The weak model in our special case is a Decision Tree for classification, which is pruned by the number of trees or leaves.

**2C. Find an optimized solution $\gamma_m$ for the loss function.**

In this step the optimization problem

$$\gamma_m = \underset{\gamma}{\text{argmin}}\sum_{i=1}^nL(y_i, F_{m-1}(x_i) + \gamma h_m(x_i)), (2a)$$

where $h_m(x_i)$ is the just fitted model (weak learner) at $x_i$ needs to be solved. The derivation for the special case of Decision Trees this can be rewritten as

$$h(x_i) = \sum_{j=1}^{J_m} b_{jm} 1_{R_{jm}}(x),$$

where $J_m$ is the number of leaves or terminal nodes of the tree, and $R_{1m}, \dots R_{J_{m}m}$ are so-called *regions*. The regions refer to the terminal nodes of the Decision Tree. We use a pruned Decision Tree, that is the terminal nodes will contain several different predictions. The prediction for each region is calculated denoted as $b_{jm}$ in the above equation. The formulation is illustrated in the following plot, which should make the concept of the *regions* clear.

![Gradient Boosting Terminology](/images/gradient_boosting/gb_terminology.png)
*Terminology for Gradient Boosting with Decision Trees.*

The final prediction for each leaf is a bit more complicated than in the case of a regression, where it is the mean of all possible outcomes.

$b_{jm} =\frac{\sum r}{\sum p_i^{m-i}\cdot(1 - p_i^{m-1})}$

...

As explained on [Wikipdia](https://en.wikipedia.org/wiki/Gradient_boosting) [1] and [Friedman (1999)](https://statweb.stanford.edu/~jhf/ftp/trebst.pdf) [2] for a Decision Tree as underlying model, this step is a bit modifed. Individual optimal values $\gamma_{jm}$ for each region $j$ is chosen, instead of a single $\gamma_{m}$ for the whole tree. The coefficients $b_{jm}$ can be then discarded and the previous equation can be rewritten as

$$\gamma_m = \underset{\gamma}{\text{argmin}}\sum_{x_i \in R_{jm}} L(y_i, F_{m-1}(x_i) + \gamma).$$

Note, that the sumation is only over the elements of the region, which simplifies the equation. Using the log-loss $L\big(y_i, p\big) = - y_i\cdot \log p - (1 - y_i)\cdot \log\big(1 - p\big)$, this convert into

$$\gamma_m = \underset{\gamma}{\text{argmin}}\sum_{x_i \in R_{jm}} L\big(y_i, p_i\big) = \underset{\gamma}{\text{argmin}}\sum_{x_i \in R_{jm}} \Big(- y_i\cdot \log p_i - (1 - y_i)\cdot \log\big(1 - p_i\big)\Big),$$

with $p_i = \sigma \big(F_{m-1}(x_i)\big).$

**2D. Update the model.**

**Step 3 - Output final model $F_M(X)$.**

## Gradient Boosting in Python

## Summary

## Appendix

Derive $-log\big(\frac{x}{y}\big) = log\big(\frac{y}{x}\big)$:

$$-log \big(\frac{x}{y}\big) = - \big(log(x) - log(y)\big) = log(y) - log(x) = log\big(\frac{y}{x}\big)$$

## Further Reading

* [1] Friedman, J.H. (1999), ["Greedy Function Approximation: A Gradient Boosting Machine"](https://statweb.stanford.edu/~jhf/ftp/trebst.pdf)
* [2] Wikipedia, ["Gradient boosting"](https://en.wikipedia.org/wiki/Gradient_boosting), date of citation: January 2024

If this blog is useful for you, please consider supporting.

{{< bmc-button slug="pumaline" >}}

