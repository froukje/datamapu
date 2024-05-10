+++
title = 'Gradient Boosting Variants'
date = 2024-05-08T20:55:43-03:00
draft = true
tags = ["Data Science", "Machine Learning", "Boosting", "Tree Methods"]
categories = ["Data Science", "Machine Learning", "Ensemble", "Boosting", "Tree Methods"]
keywords = ["Data Science", "Machine Learning", "Ensemble", "Boosting", "Tree Methods"]
images = ['/images/']

+++

## Introduction

Gradient Boosting is an ensemble model which is built of a sequential series of shallow [Decision Trees]({{< ref "/posts/classical_ml/decision_trees.md" >}}). The single trees are weak learners and have little predictive skill, that is only a higher than random guessing. Together, they form a strong learner with high predictive skill. In this article, we discuss the different implementations of [Gradient Boosting]({{< ref "/posts/classical_ml/gradient_boosting_regression.md" >}}). We give a high-level overview of the differences for a more in depth understanding, further literature is given.


## Gradient Boosting with sklearn

## XGBoost

"XGBoost (eXtreme Gradient Boosting) is a machine learning algorithm that focuses on computation speed and model performance. It was introduced by Tianqi Chen and is currently a part of a wider toolkit by DMLC (Distributed Machine Learning Community). The algorithm can be used for both regression and classification tasks and has been designed to work with large and complicated datasets."

The model supports the following kinds of boosting:

Gradient Boosting as controlled by the learning rate

Stochastic Gradient Boosting that leverages sub-sampling at a row, column or column per split levels

Regularized Gradient Boosting using L1 (Lasso) and L2 (Ridge) regularization 


Some of the other features that are offered from a system performance point of view are:

Using a cluster of machines to train a model using distributed computing

Utilization of all the available cores of a CPU during tree construction for parallelization

Out-of-core computing when working with datasets that do not fit into memory

Making the best use of hardware with cache optimization


In addition to the above the framework:

Accepts multiple types of input data

Works well with sparse input data for tree and linear booster

Supports the use of customized objective and evaluation functions

< explain histogram based algorithm >

* Level-wise tree growth

< image level-wise growth>

## LightGBT

developed by Microsoft (reference)

histogram-based algorithm that performs bucketing of values (also requires lesser memory)

Also compatible with large and complex datasets but is much faster during training
Support for both parallel learning and GPU learning

"In contrast to the level-wise (horizontal) growth in XGBoost, LightGBM carries out leaf-wise (vertical) growth that results in more loss reduction and in turn higher accuracy while being faster. But this may also result in overfitting on the training data which could be handled using the max-depth parameter that specifies where the splitting would occur. Hence, XGBoost is capable of building more robust models than LightGBM."

< imgare leave-wise growth >

https://neptune.ai/blog/xgboost-vs-lightgbm#:~:text=In%20contrast%20to%20the%20level,higher%20accuracy%20while%20being%20faster.

## CatBoost

https://catboost.ai/

https://www.geeksforgeeks.org/catboost-ml/

## Summary

---
If this blog is useful for you, please consider supporting.

{{< bmc-button slug="pumaline" >}}


