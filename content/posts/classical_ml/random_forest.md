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

A Random Forests is a supervised Machine Learning model, that is build on Decision Trees. In order to understand how a Random Forest works, you should be familar with Decision Trees. You can find an introduction in a separate article: [Decision Trees - Explained](). A major disadvantage of Decision Trees is that they tend to overfit and often have difficuilties to generalize to new data. Random Forests try to overcome this weakness. They are build of a set of Decision Trees, which are combined to an ensemble model and their outcomes are converted into a single result. As Decision Trees, they can be used for classification and regression tasks.

## Build a Random Forest

A Random Forest is an example of an *ensemble learning method*. More precisely it is an example for a [*Bagging* or *Bootstrap Aggregation*]() method. The individual models in the ensemble are [Decision Trees](), which are powerful supervised Machine Learning models, but prone to overfit to the training data. Decision Trees are models that have a [low bias, but a high variance](), especially when they are trained without pruning. This is aimed to be improved by using Bagging / Bootstrap Aggregation. Consider a dataset of $N$ samples and $M$ features. 

1. Draw $N$ samples without replacement. This is also called [bootstrapping]().
2. Draw a subset $m<<M$ of all possible features.
3. Build a [Decision Tree]() using the data samples drawn and the $m$ features.
4. Repeat 1-3 $d$ times.

This will result in $d$ Decision Trees, where $d$ is a hyperparamter that we need to choose. The variance compared to a single Decision Tree is reduced by using a subsample, that is drawn with replacement. This reduces the variance of the underlying dataset and therewith for the Decision Tree itself. Each tree is build independingly and may give different results. The final decision is then taken by considering the results of all trees developed and applying an aggregation function. In a classification problem, this aggregation function is the majority class, that is the class that was predicted by most of the trees is the final decision. In a regression task, the aggregation function is the mean of all predictions, that is taken as the final prediction. In order to improve the decision taken by a Random Forest compared to a single Decision Tree, it is important that the individual trees are as uncorrelated as possible. By not only choosing a subset of the dataset, but also a subset of the possible features a second randomness is introduced, this reduces the correlation between the individual trees. The number of features $m$ used is another hyperparamters that need to be set.

< IMAGE with different trees >

## Advantages & Disadvantages


## Random Forests in Python

In Python we can use the [sklearn]() library. It provides Methode for both regression and classification tasks. Below you can find an example for a simplified example for classification. 

## Summary
In summary a bootstrapping is used to draw a random subset from the entire dataset, then a number of models is build, which results are combined using an aggregation function, such as mean or majority vote, which explains the name *Bootstrap Aggregation*.
