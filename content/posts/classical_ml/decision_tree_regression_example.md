+++
title = 'Decision Trees for Regression - Example'
date = 2023-12-19T17:46:29+01:00
draft = true
featured_image = ''
tags = ["Data Science", "Machine Learning", "Regression", "Decision Trees", "Tree Methods"]
categories = ["Data Science", "Machine Learning", "Decision Trees", "Tree Methods", "Regression"]
keywords = ["Data Science", "Machine Learning", "Decision Trees", "Tree Methods", "Regression"]
+++

## Introduction

Decision Trees are a simple model that can be used for both regression and classification tasks. In [Decision Trees for Classification - Example]({{< ref "decision_tree_classification_example">}}) a Decision Tree for a classification problem is developed in detail. In this post, we consider a regression problem and build a Decision Tree step by step for a simplified dataset. Additionally we use [sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html) to fit a model to the data and compare the results. To learn about Decision Trees in a more general setup, please refer to [Decision Trees - Explained]({{< ref "decision_trees">}})


## Data

We use a dataset that contains only 10 samples. We are predicting the number of meters climbed by a person, depending on their age, whether they like goats, and whether they like height. That is we have three input features of which one is numerical and two are categorical. The two categorical features consists each of two classes and are therefore even binary. The target variable is numerical.

![data](/images/decision_tree/dt_data_regression.png)
*Data used to build a Decision Tree.*

## Build the Tree

The essential part of building a Decision Tree is finding the best split of the data to grow the tree. The split is done by a certain criterion, which depends on whether the target data is numerical or categorical. In this example we use the *Sum of Squared Errors (SSE)*, which is also the default choice in the [sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html) framework. The SSE for a dataset $D$, that is split into two subsets $D_1$ and $D_2$ is defined as

$$SSE(D) = SSE(D_1) + SSE(D_2),$$

with

$$SSE(D_i) = \sum_{j=1}^N(x_j-\bar{x}_i)^2.$$


## Fit a Model in Python

```Python
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

data = {'age': [23, 31, 35, 35, 42, 43, 45, 46, 46, 51], 
        'likes goats': [0, 1, 0, 0, 0, 1, 1, 1, 0, 1], 
        'likes height': [0, 1, 1, 0, 0, 1, 0, 1, 1, 1], 
        'climbed meters': [200, 700, 600, 300, 200, 700, 300, 700, 600, 700]}

df = pd.DataFrame(data)

X = df[['age', 'likes goats', 'likes height']].values
y = df[['climbed meters']].values.reshape(-1,)

reg = DecisionTreeRegressor()
reg = reg.fit(X, y)
```

We can visualize the fitted tree using [sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.tree.plot_tree.html).

```Python
from sklearn.tree import plot_tree

plot_tree(reg, feature_names=['age', 'likes goats', 'likes height'], fontsize=6)
```


![python example](/images/decision_tree/dt_regression_sklearn.png)
*Data used to build a Decision Tree.*


## Summary
