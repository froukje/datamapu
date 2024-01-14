+++
title = 'AdaBoost - Explained'
date = 2024-01-14T09:22:00-03:00
draft = true
tags = ["Data Science", "Machine Learning", "Random Forest", "Ensemble", "Boosting", "Tree Methods", "Regression", "Classification"]
categories = ["Data Science", "Machine Learning", "Random Forest", "Ensemble", "Boosting", "Tree Methods", "Classification", "Regression"]
keywords = ["Data Science", "Machine Learning", "Random Forest", "Ensemble", "Boosting", "Tree Methods", "Classification", "Regression"]
images = ['/images/ada_boost/']
+++

## Introduction

AdaBoost is an example for an [ensemble]({{< ref "/posts/ml_concepts/ensemble.md">}}) [supervised]({{< ref "/posts/ml_concepts/supervised_unsupervised#supervised">}}) Machine Learning model. It consists of a sequential series of models, each one focussing on the errors of the previous one, trying to improve them. The most common underlying model is the [Decision Tree]({{< ref "/posts/classical_ml/decision_trees.md">}}), other models are however possible. In this post, we will introduce the algorithm of AdaBoost and have a detailed look into a simplified example, using Decision Trees as base model. If you are not familiar with Decision Trees, please check the separate article [Decision Trees - Explained]({{< ref "/posts/classical_ml/decision_trees.md">}}). 

## The Algorithm

1. fit model
2. make predictions
3. weight dataset/predictions
4. fit model to weighted dataset

## AdaBoost vs. Random Forest

As mentioned earlier the most common way of constructing AdaBoost is using [Decision Trees]({{< ref "/posts/classical_ml/decision_trees.md">}}) as underlying models. Comparing to [Random Forests]({{< ref "/posts/classical_ml/random_forest.md">}}), which are also an ensemble model based on Decision Trees, the weak learner associated are very short trees or even only the root node and the first two leaves, which is called the tree *stump*, whereas in a Random Forest all trees are built until the end. Stumps and very shallow trees are not using the entire information available from the data and are therefore not as good in making correct decisions. Also in Random Forest all included Decision Trees are built independently, while in AdaBoost they build upon eachother and each new tree tries to reduce the errors of the previous one. This is why Random Forest is an ensemble model based on [Bagging](), while AdaBoost is based on [Boosting](). Finally, in a Random Forest all trees are equally important, while in AdaBoost, the shallow trees / stumps are weighted differently.

< IMAGE tree stumps >

## Ada Boost in Python

* example for classification [sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html)
* plot trees / stumps

## Summary

If this blog is useful for you, I'm thankful for your support!
{{< bmc-button slug="pumaline" >}}

