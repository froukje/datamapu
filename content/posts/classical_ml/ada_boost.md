+++
title = 'AdaBoost - Explained'
date = 2024-01-14T09:22:00-03:00
draft = true
tags = ["Data Science", "Machine Learning", "Random Forest", "Ensemble", "Boosting", "Tree Methods", "Regression", "Classification"]
categories = ["Data Science", "Machine Learning", "Random Forest", "Ensemble", "Boosting", "Tree Methods", "Classification", "Regression"]
keywords = ["Data Science", "Machine Learning", "Random Forest", "Ensemble", "Boosting", "Tree Methods", "Classification", "Regression"]
images = ['/images/adaboost/adaboost.png']
+++

## Introduction

AdaBoost is an example for an [ensemble]({{< ref "/posts/ml_concepts/ensemble.md">}}) [supervised]({{< ref "/posts/ml_concepts/supervised_unsupervised#supervised">}}) Machine Learning model. It consists of a sequential series of models, each one focussing on the errors of the previous one, trying to improve them. The most common underlying model is the [Decision Tree]({{< ref "/posts/classical_ml/decision_trees.md">}}), other models are however possible. In this post, we will introduce the algorithm of AdaBoost and have a look at a simplified example for a classification task using [sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html). For a more detailed exploration of this example please refer to [AdaBoost for Classification - Example](). A more realistic example with a larger dataset is provided on [kaggle](). Accordingly, if you are interested how AdaBoost is developed for a regression task, please check the article [AdaBoost for Regression - Example](). 

![adaboost](/images/adaboost/adaboost.png)
*AdaBoost illustrated.*


## The Algorithm

The name *AdaBoost* is short for *Adaptive Boosting*, which already explains the main ideas of the algorithm. AdaBoost is a [Boosting]({{< ref "/posts/ml_concepts/ensemble.md#boosting">}}) algorithm, which means that the ensemble model is built sequentially and each new model builds on the results of the previous one, trying to improve its errors. The developed models are all *weak-learners*, that is they have low predictive skill, that is only slightly higher than random guessing. The word *adaptive* refers to the adaption of the weights that are asigned to each sample before fitting the next model. The weights are determined in such way that the wrongly predicted samples get higher weights than the correctly predicted samples. In more detail, the algorithm works as follows.

1. **Fit a model to the initial dataset with equal weights.** A weight is assigned to each sample of the dataset. The initial weight is $\frac{1}{N}$, with $N$ being the number of data points. The weights always sum up to $1$. This means in this first step the weights can be ignored, because they are all equal. If the base model is a Decision Tree, the  weak learner is a very shallow tree or even only the *stump*, which refers to the the tree that consists only of the root node and only the first two leaves. How deep the tree is developed is a hyperparameter, that needs to be set. If you fit an AdaBoost algorithm in Python and use [sklearn](https://scikit-learn.org/stable/) the default setting depends on whether a classification or a regression is considered. For [classification](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html) the default setting is to use stumps and for [regression](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostRegressor.html#sklearn.ensemble.AdaBoostRegressor) the default value is a maximal depth of $3$.
2. **Make predictions and calculate the *influence ($\alpha$)* of the fitted model.** Not only the dataset gets weights asigned, but also each model. We call the weights that are associated with each model their *influence*. The influence of a model to the final prediction, depends on its error and is calculated as 
$$\alpha =  \frac{1}{2} \ln\Big(\frac{1 - TotalError}{TotalError}\Big).$$ 
The *Total Error* is the sum of the sample weights for all wrongly predicted data points, which is always between $0$ and $1$. With $0$ meaning the model predicts all samples wrongly and $1$ meaning all samples are correctly predicted. How the influence $\alpha$ evolves depending on the Total Error is illustrated below. For values lower that $0.5$ the model gets a negative influence, for values higher than $0.5$ it gets a positive influence, for exactly $0.5$ the influence is $0$.
3. **Adjust the weights of the data samples.** The new weight ($w_{new}$) for each data sample is calculated as
$$w_{new} = w_{old} * e^{\pm\alpha},$$
with $w_{old}$ the old or previous weight of that sample and $\alpha$ the influence of the model calculated in the previous step. The sign in the exponent changes depending on the sample was correctly predicted or not. If it was correctly predicted, the sign is negative, so that the weight decsreases. On the other hand, if it was wrongly predicted, the sign is positive, so that the weight increases. Because the sum of all weights must be $1$, we normalize the weights, by dividing it by their sum. 
4. **Create a weighted dataset.** The calculated weights are now used to create a new dataset. For that the calculated weights are used as bins for each sample. Let's assume we calculated the weights $w_1, w_2, \dots, w_N$, the the bin for the first sample is $0$ to $w_1$, the bin for the second sample is $w_1$ to $w_1+w_2$ and so on. Data samples are now selected by choosing $N$ random numbers between $0$ and $1$, with $N$ being the number of data samples. The for each of these random numbers, the sample is chosen in which bin the random number falls. Since wrongly predicted samples have higher weight than correctly predicted samples, there bins are larger. Therefore the probability of drawing a wrongly predicted sample is higher. The newly created dataset consists again of $N$ samples, but there are likely duplicates from the wrongly predicted samples.
5. **Fit a model to the new dataset.** Now we start again and fit a model, equally to the first step, but this time using the modified dataset. 

Repeat steps 2 to 5 $d$ times, where $d$ is the number of final weak learners of which the ensemble model is composed. It is a hyperparamter that needs to be chosen. You can find an example of these stepswith detailed calculations in the articles [AdaBoost for Classification - Example]() or [AdaBoost for Regression - Example]().

![influence_error](/images/adaboost/influence_error.png)
*The influence of an individual model to the final ensemble model, depending on its total error.*

## AdaBoost vs. Random Forest

As mentioned earlier the most common way of constructing AdaBoost is using [Decision Trees]({{< ref "/posts/classical_ml/decision_trees.md">}}) as underlying models. Another important ensemble machine learning model based on Decision Trees is the [Random Forest]({{< ref "/posts/classical_ml/random_forest.md">}}). While Decision Trees are powerful machine learning algorithms, one of their major disadvanges is that they tend to [overfit]({{< ref "/posts/ml_concepts/bias_variance.md">}}). Both, Random Forest and AdaBoost try to improve this while maintaining the advantages of Decision Trees, such as their robustness towards outliers and missing values. Both algorithms, however, differ substantially. In Adaboost, the weak learners associated are very short trees or even only the root node and the first two leaves, which is called the tree *stump*, whereas in a Random Forest all trees are built until the end. Stumps and very shallow trees are not using the entire information available from the data and are therefore not as good in making correct decisions. Also in Random Forest all included Decision Trees are built independently, while in AdaBoost they build upon each other and each new tree tries to reduce the errors of the previous one. In other words, Random Forests are an ensemble model based on [Bagging]({{< ref "/posts/ml_concepts/ensemble.md#bagging">}}), while AdaBoost is based on [Boosting]({{< ref "/posts/ml_concepts/ensemble.md#boosting">}}). Finally, in a Random Forest all trees are equally important, while in AdaBoost, the individual shallow trees / stumps have different influence, because they are weighted differently. The following table summarizes the differences between Random Forests and AdaBoost based on Decision Trees.

![adaboost_vs_random_forest](/images/adaboost/adaboost_rf.png)
*Main differences between AdaBoost and a Random Forest.*

![adaboost_vs_random_forest](/images/adaboost/adaboost_rf_illustrated.png)
*AdaBoost and  Random Forest illustrated.*

## Ada Boost in Python

* example for classification [sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html)
* plot trees / stumps
* hyperparamters: base_estimator - default Decision Tree with deoth 1, what other hyperparamters are importrnat?
* link to regression
* First stump is the start of the Decision Tree build in DT classification article
* show staged predictions
* calculate the next stumps by hand, if too long -> extra post

## Summary

If this blog is useful for you, I'm thankful for your support!
{{< bmc-button slug="pumaline" >}}

