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

AdaBoost is an ensemble model that is based on [Boosting]({{< ref "/posts/ml_concepts/ensemble.md#boosting">}}). The individual models are so-called weak learners, which means that they have only little predictive skill, and they are sequentially built to improve the errors of the previous one. A detailed description of the Algorithm can be found in the separate article [AdaBoost - Explained]({{< ref "/posts/classical_ml/adaboost.md">}}). In this post, we will focus on a concrete example for a classification task and develop the final ensemble model in detail. We will use [Decision Trees]({{< ref "/posts/classical_ml/decision_trees.md">}}) as a base model. A detailed example for a regression task is given in the article [AdaBoost for Regression - Example]().

## Data

The dataset used in this example contains of only 10 samples, to make the calculations by hand more feasible. It describes the problem whether a person should go rock climbing or not, depending on their age, and whether the person likes height and goats. This dataset was also in the article [Decision Trees for Classification - Example]({{< ref "/posts/classical_ml/decision_tree_classification_example.md">}}), which makes comparisons easier. The data is described in the plot below.  

![adaboost_data_clf](/images/adaboost/adaboost_data.png)
*Dataset used to develop an AdaBoost model.*

## Build the Model

We build an AdaBoost model, constructed of [Decision Trees]({{< ref "/posts/classical_ml/decision_trees.md">}}) as weak learners, because this is the most common application. The underlying trees have depth $1$, that is only the *stump* of each tree is used as a weak learner. This is also the default configuration in [sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html), which you can use to fit a model in Python. We will consider three weak learners to build the ensemble Adaboost model.

The first step in building an AdaBoost model is asigning weights to the individual data points. In the beginning, for the inital model, all datapoints get the same weight asigned, which is $\frac{1}{N}$, with $N$ the dataset size.

![adaboost_data_clf](/images/adaboost/ab_clf_data_first_stump.png)
*Initial dataset and weights for the example.*

We now start with fitting a [Decision Tree]({{< ref "/posts/classical_ml/decision_trees.md">}}) to this data, because this is our underling model. As stated earlier we will use decision stumps, that is we are only interested in the first split. As this exact same data was used in [Decision Trees for Classification - Example]({{< ref "/posts/classical_ml/decision_tree_classification_example.md">}}) to develop a Decision Tree by hand, please check there for the datails how to develop a Decision Tree and how to find the best split. Following the steps from the mentioned article the first stump is.

![adaboost_first_stump](/images/adaboost/ab_example_clf_first_stump.png)
*The first stump, i.e. the first weak learner for our AdaBoost algorithm.*

With the *influence* $\alpha$ calculated, we can determine the new weights for the next iteration.

$$w_{new} = w_{old}\cdot e^{\pm\alpha}.$$

The sign in this equation depends on whether a sample was correctly classified or not. For correctly classified samples, we get

$$w_{new} = 0.1\cdot e^{-1.099} = 0.0333,$$ 

and for wrongly classified samples

$$w_{new} = 0.1\cdot e^{1.099} = 0.3,$$ 

accordingly. These weights still need to be normalized, so that their sum equals $1$. This is done, by dividing by their sum. The next plot shows the dataset with the new weights.

![adaboost_data_new_weights1](/images/adaboost/ab_example_clf_new_weights1.png)
*The dataset with updated weights based on the influence $\alpha$.*

The weights are used to create bins. The bin for the first sample is $[0, 0.056]$, for the second $[0.056,0.112]$, etc..

![adaboost_data_bins1](/images/adaboost/ab_example_clf_bins1.png)
*The dataset with bins based on the weights.*

Now, some randomness comes into play. Random numbers between $0$ and $1$ are drawn, then we check in which bin the random number falls, and the according data sample is selected for the new modified dataset. We draw as many numbers as the length of our dataset is, that is in this example we draw $10$ numbers. Due to the higher weight of the misclassified example, this example has a larger bin and the probabilty to draw it is higher. Let's assume we draw the numbers $[0.1, 0.15, 0.06, 0.5, 0.65, 0.05, 0.8, 0.7, 0.95, 0.97]$, which leads to the selection of the samples $[1, 2, 1, 8, 8, 0, 8, 8, 9, 9]$. The modified dataset, then has the following form.

![adaboost_data2](/images/adaboost/ab_clf_data_first_second_stump.png)
*Modified dataset to build the second stump.*

We now use this modified dataset to create the second stump. Following the steps described in [Decision Trees for Classification - Example]({{< ref "/posts/classical_ml/decision_tree_classification_example.md">}}), we achive the following decision stump.

![adaboost_first_stump](/images/adaboost/ab_example_clf_second_stump.png)
*The second stump, i.e. the first weak learner for our AdaBoost algorithm.*

As in the first stump, one sample is misclassified, so we get the same value for alpha as for the first stump. Accordingly, the weights are the same. The following plot shows the data together with their new weights and the normalized weights.

![adaboost_data_new_weights2](/images/adaboost/ab_example_clf_new_weights2.png)
*The dataset with updated weights based on the influence $\alpha$.*

We convert the weights into bins.

![adaboost_data_new_bins2](/images/adaboost/ab_example_clf_second_stump_bins.png)
*The modified dataset with bins based on the weights.*

We repeat the bootstrapping and draw $10$ random numbers between $0$ and $1$. Let's assume we draw the numbers $[0.15, 0.95, 0.55, 0.65, 0.85, 0.04, 0.45, 0.35]$, which refer to the samples $[2, 9, 6, 9, 6, 6, 6, 0, 6, 6]$, then we get the following modified dataset.

![adaboost_data_modified](/images/adaboost/ab_example_clf_modified_data_stump3.png)
*Modified dataset based on the weights.*

## Fit a Model in Python

We now fit the example in Python using [sklearn](). Note, that the results are not exactly the same due to the randomness in the bootstrapping and because slightly different hyperparamters are used in sklearn. We chose the base model to be a Decision Tree with max_depth $1$, which is the same as in sklearn. A difference, however is that in sklearn "max_features" is set to "sqrt", which means ... In our example we used all the features.

difference: max_features='sqrt'
in our example: None

## Summary

If this blog is useful for you, I'm thankful for your support!
{{< bmc-button slug="pumaline" >}}
