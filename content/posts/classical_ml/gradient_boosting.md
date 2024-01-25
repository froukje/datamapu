+++
title = 'Gradient Boosting - Explained'
date = 2024-01-12T09:21:46-03:00
draft = true
tags = ["Data Science", "Machine Learning", "Gradient Boosting", "Ensemble", "Boosting", "Tree Methods", "Regression", "Classification"]
categories = ["Data Science", "Machine Learning", "Gradient Boosting", "Ensemble", "Boosting", "Tree Methods", "Classification", "Regression"]
keywords = ["Data Science", "Machine Learning", "Gradiend Boosting", "Ensemble", "Classification", "Tree Methods", "Regression"]
images = ['/images/']
+++

## Introduction

*Gradient Boosting*, also called *Gradient Boostimg Machine (GBM)* is a type of [supervised](supervised_unsupervised.md#supervised) Machine Learning algorthm that is based on [ensemble learning]({{< ref "/posts/ml_concepts/ensemble">}}). It consists of a sequential series of models, each one trying to improve the errors of the previous one. It can be used for both regression and classification tasks. In this post we will give a general introduction and explain the algorithm. There are some differences in the details depending on whether a regression or a classification problem is considered. Please find the details explained on a simplified example in the separate articles [Gradient Boosting for Regression - Explained]() and [Gradient Boosting for Classification - Explained]().

## The Algorithm

Gradient Boosting is, as the same suggests, an ensemble model that is based on [boosting]({{< ref "/posts/ml_concepts/ensemble#boosting">}}). In boosting, an initial model is fit to the data. Then a second model is built on the results of the first one, trying to improve the inaccurate results of the first one, and so on until a series of additive models is built, which together are the ensemble model. The individual models are so-called weak learners, which means that they are simple models with low predictive skill, that is only a bit better than random chance. The idea is to combine a set of weak learners to achieve one strong learner, i.e. a model with high predictive skill. 

< IMAGE BOOSTING >

The most popular underlying models in Gradient Boosting are [Decision Trees]({{ ref "/posts/classical_ml/decision_trees">}}), however using other models, is also possible. When a Decision Tree is used as a base model the algorithm is called *Gradient Boosted Trees*, and a shallow tree is used as a weak learner. Gradient Boosing is a [supervised]() Machine Learning algorithm, that means we aim to find a mapping that approximates the target data as good as possible. This is done by minimizing a [Loss Function](), that meassures the error between the true and the predicted values. Common choices for Loss functions in the context of Gradient Boosting are the [Mean Squared Error (MSE)]() for a regression task and the [logarithmic loss]() for a classification task. It can however be any differentiable function. 

< INTUITIVE EXPLANATION > add residuals

For the explanation of the algorithm, we will follow the notations used in [Wikipedia](https://en.m.wikipedia.org/wiki/Gradient_boosting). Let $(x, y) = {(x_1, y_1), \dots, (x_N, y_N)} be the training data, with $x$ being the input features and $y$ the target values and $F(x)=\hat{y}$ be the mapping we want to determine to approximate the target data. The algorithm is then describes as follows.

1. **Make an initial constant prediction.** The initial prediction depends on the Loss function ($L$) we choose. Mathematically this initial prediction is defined as $$F_0(x) = \hat{y}_0(x) = argmin\lim_{\gamma}\sum_{i=10}^n L(y_i, \gamma}$$

	**Case 1: Regression.** When we are considering a regression task and use the MSE as Loss Function, we have $L(y_i, \gamma) = (y_i - \gamma)^2 $ this expression reduces to the  mean of the target values $F_{0}(x) = \bar{y}$. That means, the initial prediction is simply the mean of the target data. Please find a detailed derivation in the separate articel [Gradient Boosting for Regression - Example]().

	**Case 2: Classification.** In the case we are considering a classification task and use the logarithmic loss as Loss Function, we have $L(y_i, \gamma) = -\frac{1}{N}\sum_{i=1}^N\sum_{i=1}^M x_{ij} \cdot log(p_{ij})$ for a dataset of $N$ samples and $M$ classes. Accordingly for a binary classification the binary logarithmic loss is $L(y_i, \gamma) = -\frac{1}{N}\sum_{i=1}^N y_i\cdot log(p(y_i)) + (1-y_i)\cdot log(1-p(y_i)), which reduces to $F_{0}(x) = -y + p(y_i)$, with $p(y_i)$ the probabilty of $y_i$. Please find a detailed derivation in the separate article [Gradient Boosting for Classification](). 

2. **Make predictions ($F(x)=\hat{y}_i$).** 

3. **Calculate the (pseudo-)residuals of the preditions and the true observations.** The residuals are defined as follows. 

$$r_i = - \Big[\frac{\delta L(y_i, F(x_i))}{\delta F(x_i)}\Big]$$
$$r_i = - \Big[\frac{\delta L(y_i, \hat{y}_i))}{\delta \hat{y}_i}\Big]$$

	**Case 1: Regression.** For a regression task with MSE loss this term simplifies to ... . 
	**Case 2: Classification.** For a binary classification task using binary logarithmic loss this expression simplifies to ...
For a detailed derivation please refer to the articles [Gradient Boosting for Regression - Example](), and [Gradient Boosting for Classification - Example](), respectively.

3. **Fit a model (weak learner) to the residuals.** That is train a model with the residuals as target values.
4. **Calculate improved predictions.** The improved predictions are $\hat{y} + \alpha \cdot F_{res}$, with $\alpha$ being the learning rate, which is a hyperparamter between $0$ and $1$ that needs to be chosen. The idea behind this hyperparamter is that more small changes in the predictions lead to better results than a few large changes.


Repeat 2 and 4 $d$ times.
The final prediction is $\hat{y} + \alpha \cdot r_1 + r_2 + \cdots + \alpha \cdot r_d$, with $r_1, r_2, \dots, r_d$ the predicted residuals from the $d$ weak learner.

< IMAGE FOR GRADIENT BOOSTING (REGRESSION + CLASSIFICATION) > 

The main difference between these two algorithms is that Gradient boosting has a fixed base estimator i.e., Decision Trees whereas in AdaBoost we can change the base estimator according to our needs.


gradient boosting hekps to reduce the bias

The algorithm was first described by Friedman (1999). 


## Gradient Boosting vs. AdaBoost

## Advantages & Disadvantages Gradient Boosted Trees

**Pros**

* Can deal with missing data and outlier
* Can deal with umerical and categorical data

**Cons**

## Gradient Boosting in Python

## Summary

fast & accurate

If this blog is useful for you, I'm thankful for your support!
{{< bmc-button slug="pumaline" >}}

