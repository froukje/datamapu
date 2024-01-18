+++
title = 'AdaBoost for Classification - Example'
date = 2024-01-17T22:08:14-03:00
draft = true
tags = ["Data Science", "Machine Learning", "Ensemble", "Boosting", "Tree Methods", "Classification"]
categories = ["Data Science", "Machine Learning", "Ensemble", "Boosting", "Tree Methods", "Classification"]
keywords = ["Data Science", "Machine Learning", "Ensemble", "Boosting", "Tree Methods", "Classification"]
images = ['/images/adaboost/']
+++

## Introduction

AdaBoost is an ensemble model that is based on [Boosting]({{< ref "/posts/ml_concepts/ensemble.md#boosting">}}). The individual models are sequentially built to improve the errors of the previous one. A detailed description of the Algorithm can be found in the separate article [AdaBoost - Explained]({{< ref "/posts/classical_ml/adaboost.md">}}). In this post, we will focus on a concrete example for a classification task and develop the final ensemble model in detail. We will use [Decision Trees]({{< ref "/posts/classical_ml/decision_trees.md">}}) as a base model. A detailed example for a regression task is given in the article [AdaBoost for Regression - Example]().

## Data

The dataset used in this example contains of only 10 samples, to make the calculations by hand more feasible. It describes the problem whether a person should go rock climbing or not, depending on their age, and whether the person likes height and goats. This dataset was also in the article [Decision Trees for Classification - Example]({{< ref "/posts/classical_ml/decision_tree_classification_example.md">}}), which makes comparisons easier. The data is described in the plot below.  

![adaboost_data_clf](/images/adaboost/adaboost_data.png)

## Build the Model

We build an AdaBoost model, constructed of [Decision Trees]({{< ref "/posts/classical_ml/decision_trees.md">}}), as this is the most common application. The underlying trees have depth $1$, that is only the *stump* of each tree is used as a weak learner. This is also the default configuration in [sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html), which you can use to fit a model in Python.

The first step in building an AdaBoost model is asigning weights to the individual data points. In the beginning, for the inital model, all datapoints get the same weight asigned, which is $\frac{1}{N}$, with $N$ the dataset size.

![adaboost_data_clf](/images/adaboost/ab_clf_data_first_stump.png)


## Fit a Model in Python



## Summary

If this blog is useful for you, I'm thankful for your support!
{{< bmc-button slug="pumaline" >}}
