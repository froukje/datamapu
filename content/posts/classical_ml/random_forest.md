+++
title = 'Random Forests - Explained'
date = 2023-12-26T10:57:13+01:00
draft = true
featured_image = ''
tags = ["Data Science", "Machine Learning", "Random Forest", "Tree Methods", "Regression", "Classification"]
categories = ["Data Science", "Machine Learning", "Random Forest", "Tree Methods", "Classification", "Regression"]
keywords = ["Data Science", "Machine Learning", "Random Forest", "Tree Methods", "Classification", "Regression"]
+++

## Introduction

Random Forests are a supervised Machine Learning model, that is build on Decision Trees. A main disadvantage of [Decision Trees]() is that they tend to overfit and often have difficuilties to generalize to new data. Random Forests try to overcome this weakness. They are build of a set of Decision Trees, which are combined to an ensemble model and their outcomes are converted into a single result. As Decision Trees, they can be used for classification and regression tasks.

## Build a Random Forest

A Random Forest is an example of an *ensemble learning method*. More precisely it is an example for a [*Bagging* or *Bootstrap Aggregation*]() method. The individual models in the ensemble are [Decision Trees](), which are powerful supervised Machine Learning models, but prone to overfit on the training data. Decision Trees are models that have a [low bias, but a high variance](), especially when they are trained without pruning. This is aimed to be improved by using Bagging / Bootstrap Aggregation. Consider a dataset of $N$ samples and $M$ features. A Random Forest consists of $s$ Decision Trees, each of which is build as follows.

1. Draw $N$ samples without replacement. This is also called [bootstrapping](). For these samples a Decision Tree is build considering the following condisions. This is the step where the variance is reduced. The variance is reduced in the underlying dataset for the Decision Tree and therewith also in the Decision Tree itself.
2. At each node draw a subset of features randomly. That means, if at at node $i$, $M_i$ features are available, draw $m_i<<M_i$ features randomly. 
3. From these $m_i$ drawn features select the one with the best split. To learn how to find the best split, check [Decision Trees - Explaine]() for a detailed explanation.
4. Build each tree completely without [pruning]()

Doing that results in $d$ Decision Tree models, where $d$ is a hyperparameter that we need to choose. Each tree is build independingly and may therefore give different results. The final decision is then taken by considering the results of all trees developed. In a classification problem, the class that was predicted by most of the trees is the final decision. In a regression task, the mean of all prediction is taken as the final prediction. By not only choosing a subset of the dataset, but also a subset of the possible features a second randomness is introduced, which reduces the correlation between the individual trees. The number of features used is another hyperparamters that need to be set.

< IMAGE with different trees >

## Advantages & Disadvantages


## Random Forests in Python

In Python we can use the [sklearn]() library. It provides Methode for both regression and classification tasks. Below you can find an example for a simplified example for classification. 

## Summary
In summary a bootstrapping is used to draw a random subset from the entire dataset, then a number of models is build, which results are combined using an aggregationfunction, such as mean or majority vote, which explains the name *Bootstrap Aggregation*.
