+++
title = 'Gradient Boost for Classification - Explained'
date = 2024-04-14T20:45:19-03:00
draft = true
tags = ["Data Science", "Machine Learning", "Gradient Boosting", "Ensemble", "Boosting", "Tree Methods", "Regression", "Classification"]
categories = ["Data Science", "Machine Learning", "Gradient Boosting", "Ensemble", "Boosting", "Tree Methods", "Classification", "Regression"]
keywords = ["Data Science", "Machine Learning", "Gradiend Boosting", "Ensemble", "Classification", "Tree Methods", "Regression"]
images = ['/images/gradient_boosting/gb_intro.png']
+++

---
## Introduction

Gradient Boosting is an [ensemble]({{< ref "/posts/ml_concepts/ensemble">}}) machine learning model, that - as the name suggests - is based on [boosting]({{< ref "/posts/ml_concepts/ensemble#boosting" >}}). An ensemble model based on boosting refers to a model that sequentially builds models, and the new model depends on the previous model. In Gradient Boosting these models are built such that they improve the error of the previous model. These individual models are so-called weak learners, which means models that have a low predictive skill. The ensemble of these weak learners build the final model - a strong learner with a high predictive skill. In this post, we will go through the algorithm of Gradient Boosting in general and then concretise the individual steps for a classification tasks using [Decision Trees]({{< ref "/posts/classical_ml/decision_trees" >}}) as weak learners and the logarithmic logarithm as loss function. There will be some overlapping with the article [Gradient Boosting for Regression - Explained]({{< ref "/posts/classical_ml/gradient_boosting_regression.md" >}}), where a detailed explanation of Gradient Boosting is given, which is then applied to a regression problem. However, in order to have the complete guide to a classification problem is one articel, we will repeat some of the underlying formulations. If you are interested in a concrete example with the detailed calculations, please refer to [Gradient Boosting for Regression - Example]({{< ref "/posts/classical_ml/gradient_boosting_regression_example.md" >}}) for a regression problem and [Gradient Boosting for Classification - Example]() for a classification problem.

## The Algorithm

Gradient Boosting is a [boosting]({{< ref "/posts/ml_concepts/ensemble#boosting" >}}) algorithm that aims to build a series of weak learners, which together act as a strong learner. In Gradient Boosing the objective is to improve the error of the preceeding model by minimizing its loss function using [Gradient Descent]({{< ref "/posts/ml_concepts/gradient_descent.md">}}). That means the weak learners are build up on the error and not up on the targets themselves as in other boosting algorithm like [AdaBoost]({{< ref "/posts/classical_ml/adaboost.md" >}})

## Gradient Boosting in Python


## Summary


If this blog is useful for you, please consider supporting.

{{< bmc-button slug="pumaline" >}}

