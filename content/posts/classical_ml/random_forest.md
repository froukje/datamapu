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

A Random Forest is an example of an ensemble method. More precicely it is an example for *Bagging* or *Bootstrap Aggregation*. Consider a dataset of N samples and M features. The algorithm to build a Random Forest is as follows:

1. Draw $n<=N$ samples without replacement (bootstrapping)
2. At each node draw $m<<M$ of all possible features randomly. 
3. From these $m$ drawn features select the one with the best split (see [Decision Trees - Explaine]() for a detailed explanation.)
4. Build each tree completely without [pruning]()

The final decision is then Tagen by considering the results of all trees developed. In a classification problem, the class that was predicted by most of the trees is the final decision. In a regression task, the mean of all prediction is Tagen as the final prediction. 

Each tree is build independingly and the trees are uncorrelated.
 
Random Forest consists of a number of Decision Trees. However, when building a [Decision Tree]() all possible features are considered, in a Random Forest each Tree is build of a randomly drawn subset of all features. The number of features used is one of the hyperparamters that need to be set.

< link to ensemble methods >

## Advantages & Disadvantages


## Random Forests in Python

In Python we can use the [sklearn]() library. It provides Methode for both regression and classification tasks. Below you can find an example for a simplified example for classification. 
## Summary
