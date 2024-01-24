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

Gradient Boosting is, as the same suggests, a ensemble model that is based on [boosting]({{< ref "/posts/ml_concepts/ensemble#boosting">}}). In boosting, an initial model is fit to the data. Then a second model is built on the results of the first one, trying to improve the inaccurate results of the first one, and so on until a series of additive models is built, which together are the ensemble model. The individual models are so-called weak learners, which means that they are simple models with low predictive skill, that is only a bit better than random chance. The idea is to combine a set of weak learners to achieve one strong learner, i.e. a model with high predictive skill. 

< IMAGE BOOSTING >

The most popular underlying models in Gradient Boosting are [Decision Trees]({{ ref "/posts/classical_ml/decision_trees">}}), however using other models, is also possible. When a Decision Tree is used as a base model the algorithm is called *Gradient Boosted Trees*, and only the tree stump or a shallow tree is used as a weak learner.  

< IMAGE EXAMPLE OF STUMPS >

More specifically the steps to perform Gradient Boosting are as follows.

1. Choose a loss function
 * must be differentible
 * For regression: e.g. MSE
 * For classification: e.g. logarithmic loss
	
2. Choose a model (weak learner)
 * e.g. Decision Tree (stumps)

The model is then built as follows.

1. Fit a model (weak learner) to the original dataset (input: X, y) with the chosen loss function.
2. Make predictions and calculate the residuals (errors) between the preditions and the true observations.
3. The improved predictions are $\hat{y} + res$
3. Fit a model to the residuals of the previous model. (input: X, res)

Repeat 2 and 3 $d$ times.

The final prediction is $\hat{y} + r_1 + r_2 + \cdots + r_d$, with $r_1, r_2, \dots, r_d$ the residuals from the $d$ weak learner.

< IMAGE FOR GRADIENT BOOSTING (REGRESSION + CLASSIFICATION) > 

The main difference between these two algorithms is that Gradient boosting has a fixed base estimator i.e., Decision Trees whereas in AdaBoost we can change the base estimator according to our needs.


gradient boosting hekps to reduce the bias

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

