+++
title = 'Decision Trees - Explained'
date = 2023-12-16T12:33:55+01:00
draft = true
+++

## Introduction

A *Decision Tree* is a [supervised Machine Learning]({{< ref "supervised_unsupervised#supervised">}}) algorithm that can be used for both regression and classification problems. It is a non-parametric model, which means there is no specific mathematical function underlying to fit the data (as e.g. in Linear Regression or Logistic Regression), but the algorithm only learns from the data itself. Decision Trees learn rules for decision making and used to be drawn manually, before Machine Learning came up. Decision trees are hierarchical models, that have a flow-chart tree structure as the name suggests.

![example for a decision tree](/images/decision_tree/dt_example.png)
*Example for a Decision Tree.*

## Terminology

Before diving into the details how to build a Decision Tree, let's have a look at some important terms.

**Root Node.** The Root Node is the top-level node. It contains the entire dataset and the first decision is taken.

**Node.** A node is also called *internal node* or *decision node*. It represents a split into further (child) nodes or leafs. 

**Parent Node.** A parent node is a node which precedes a (child) node.

**Child Node.** A child node is a node following another (parent) node.

**Leaf.** A leaf is also called a terminal node. It is a node at the end of a branch and has no following nodes. It represents a possible outcome of the tree, i.e. a class label or a numerical value.

**Splitting.** The process of dividing a node into two child nodes depending on a criterion and a selected feature.

**Branches.** A branch is a subset of a tree, starting at an (internal) node until the leafs.

**Pruning.** Removing a branch from a tree is called pruning. This is usually done to avoid overfitting.

![terminology decision tree](/images/decision_tree/dt_terminology.png)
*Illustration of the terminology of a Decision Tree.*

## Build a Tree

Each node is split based on a feature and a splitting criterion.

### Splitting criteria

### Classification

### Regression

## Decision Trees in Python

To determine a Decision Tree for a given dataset in Python, we can use the [sklearn](https://scikit-learn.org/stable/modules/tree.html) library. Both, Decision Trees for classification and regression tasks are supported. Here is a simple example for a classification problem. The task is to decide whether a person should go rock climbing or not, depending on whether the person likes height, goats, and their age, as illstrated in the beginning of this article.

```Python
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree

data = {'age': [23, 31, 35, 35, 42, 43, 45, 46, 46, 51], 
        'likes goats': [0, 1, 0, 0, 0, 1, 1, 1, 0, 1], 
        'likes height': [0, 1, 1, 0, 0, 1, 0, 1, 1, 1], 
        'go rock climbing': [0, 1, 1, 0, 0, 1, 0, 1, 0, 1]}

df = pd.DataFrame(data)
X = df[['age', 'likes goats', 'likes height']].values
y = df[['go rock climbing']].values.reshape(-1,)
clf = DecisionTreeClassifier()
clf = clf.fit(X, y)
y_hat = clf.predict(X)
y_pred_proba = clf.predict_proba(X)
```

The class used to determine the Decision Tree is [DecisionTreeClassifier](https://scikit-learn.org/stable/modules/tree.html#tree-classification). To get the predicted categories we can use the *predict* method. In this example the result is *y_hat = [0 1 1 0 0 1 0 1 0 1]*. Accordingly, *predict_proba* gives the probability of each category. The Decision Tree can be illustrated in Python using the [plot_tree](https://scikit-learn.org/stable/modules/generated/sklearn.tree.plot_tree.html) class from sklearn.

```Python
plot_tree(clf, feature_names=['age', 'likes goats', 'likes height'], fontsize=10)
```
![example in python](/images/decision_tree/dt_python.png)
*Illustration of a Decision Tree in Python.*

## Advantages & Disadvantages

**Advantages.**

* Decision Trees are intuitive, easy to implement, and interpret.
* Decision trees are not effected by outliers and missing values.
* Can be used with numerical and categorical data
* The data doesn’t need to be scaled.
* As a non-parametric algorithms Decision Trees are very flexible.

**Disadvantages.**

* Decision Trees tend to overfit. To overcome this, pruning the tree may help.
* Decision Trees cannot predict continous variables. That is also when applied to a regression problem the predictions muct be separated into categories.
* As a non-parametric algorithm, the training of a Decision Tree may be expensive if the dataset is large.

## Summary

In the articel [Decision Tree Example for Classification]() you can find a detailed calculation by hand of the above developed Decision Tree using scikit-learn. 

Regression example?

For a more realistic example with a larger dataset you can find a notebook on [kaggle]().
