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

A Random Forest is a supervised Machine Learning model, that is build on Decision Trees. In order to understand how a Random Forest works, you should be familar with Decision Trees. You can find an introduction in the separate article [Decision Trees - Explained]({{< ref "decision_trees">}}). A major disadvantage of Decision Trees is that they tend to overfit and often have difficuilties to generalize to new data. Random Forests try to overcome this weakness. They are build of a set of Decision Trees, which are combined to an ensemble model and their outcomes are converted into a single result. As Decision Trees, they can be used for classification and regression tasks.

![random forest](/images/random_forest/rf_overview.png)
*Illustration of a Random Forest.*

## Build a Random Forest - The Algorithm 

A Random Forest is an example for an [*ensemble learning method*]({{< ref "ensemble">}}). Multiple Machine Learning models are combined to obtain a better model ('the wisdom of crowds'). More precisely it is an example for the [*Bagging* or *Bootstrap Aggregation*]({{< ref "ensemble#bagging">}}) method. The underlying models in a Random Forest are [Decision Trees]({{< ref "decision_trees">}}), which individual outcomes are combined to one single prediction. Decision Trees are powerful Machine Learning models, which are easy to interpret. They have, however, one severe disadvantage, which is that they are prone to [overfit]({{< ref "bias_variance">}}). Decision Trees are an example for models that have [a low bias, but a high variance]({{< ref "bias_variance">}}), especially when they are trained without pruning. This is aimed to be improved by using Bagging / Bootstrap Aggregation. The algorithm to build a Random Forest is as follows. Consider a dataset of $N$ samples and $M$ features. 

1. Draw $N$ samples without replacement. This is also called bootstrapping.
2. At each node use a subset $m<<M$ of all possible features.
3. Build a [Decision Tree]({{< ref "decision_trees">}}) using the data samples drawn and the $m$ features until the end, i.e. without pruning.
4. Repeat 1-3 $d$ times.

This will result in $d$ Decision Trees, where $d$ is a hyperparamter that we need to choose. The variance compared to a single Decision Tree is reduced by using a subsample, that is drawn with replacement. This reduces the variance of the underlying dataset and therewith for the Decision Tree itself. Each tree is build independingly and may give different results. The final decision is then taken by considering the results of all trees developed and applying an aggregation function. In a classification problem, this aggregation function is the majority class, that is the class that was predicted by most of the trees is the final decision. In a regression task, the aggregation function is the mean of all predictions, which is the final prediction. 

![random forest](/images/random_forest/random_forest2.png)
*Simplified example for a Random Forest.*

In order to improve the decision taken by a Random Forest compared to a single Decision Tree, it is important that the individual trees are as uncorrelated as possible. By not only choosing a subset of the dataset, but also a subset of the possible features a second randomness is introduced, this reduces the correlation between the individual trees. The number of features $m$ used is another hyperparamters that need to be set.

## Advantages & Disadvantages

**Pros**

* Random Forests can be used for regression and classification problems.
* Random Forests are able to learn non-linear relationships.
* Random Forests can combine categorical and numerical variables.
* Random Forests are not sensitive to outliers and missing data. 
* Scaling the data is not neccessary before fitting a Random Forest.
* Random Forests balance the [bias-variance tradeoff]({{< ref "bias_variance#tradeoff">}}).
* Random Forests reduce overfitting compared to Decision Trees.
* Random Forests reduce the variance in the predictions compared to Decision Trees.
* Random Forests can provide information about the feature importance.
* The trees in a Random Forest are independent of each other and can be created in parallel, which makes the training faster. 

**Cons**

* Random Forests are less interpretable than a single Decision Tree.
* Random Forests need a lot of memory, because several trees are stored in parallel. 
* For a large number of trees and / or a large dataset they are expensive to train.
* Although Random Forests are less prone to overfitting, they still may overfit if too many trees are used or the trees are too deep.

## Random Forests in Python

In Python we can use the sklearn library. It provides Methode for both regression and classification tasks. Below you can find an example for a simplified example for classification. 

```Python
from sklearn
```

In this example the dataset is very small and it is only used to illustrate a Random Forest. In [sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) a lot of hyperparameters exist to optimize a Random Forest. In the above example we set *n_estimators=3*, which is the number of Decision Trees used in the Random Forest. For real world examples, this number will be chosen higher, the default value in sklearn is $100$. Other important hyperparamters in sklearn, are:

**Hyperparamters**
* **n_estimators**: Number of Decision Trees used to create the Random Forest (default: 100).
* **criterion**: Function used to define the best split (default: Gini Impurity).
* **max_depth**: Maximum depth of the trees (default: None, i.e. the trees are expanded until all leaves are pure).
* **min_sample_split**: Minimum number of samples to split a node (default: 2).
* **min_samples_leaf**: Minimum number of samples required to become a leaf node (default: 1).
* **max_features**: Maximum number of features considered to find the best split (default: the squareroot of the total number of features).
* **bootstrap**: Whether bootstrapping is used or not (default: True). If set to False, the entire dataset is used for each tree.

A complete list with detailed explanations of all possible hyperparamters can be found in the [sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) documentation. For a detailed example, of how to develop a Random Forest, check the separate article [Random Forests for Classification - Example]() and for a more realistic example with a larger dataset, you can find an example on [kaggle](). For regression tasks the Procedere is Analogie, the used method in sklearn is called *Random Forest Regressor* and a detailed list of all hyperparamters can be found in the [sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html) documentation.

## Summary

In this article we learned about Random Forests, how they are created, and there main advantages and disadvantages. Random Forests are an [ensemble Machine Learning]({{< ref "ensemble">}}) model consisting of multiple [Decision Trees]({{< ref "decision_trees">}}). They use the strength of Decision Trees and at the same time overcome their tendency to overfit. Compared to Decision Trees, Random Forests are more robust, flexible and accurate, they loose however interpretability and are more expensive to train. For a deatiled Version of the above example derived by hand, please refer to the article [Random Forests for Classification - Example]() and for a more realistic example with a larger dataset, you can find an example on [kaggle]().
