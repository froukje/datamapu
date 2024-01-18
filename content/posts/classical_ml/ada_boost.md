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

AdaBoost is an example for an [ensemble]({{< ref "/posts/ml_concepts/ensemble.md">}}) [supervised]({{< ref "/posts/ml_concepts/supervised_unsupervised#supervised">}}) Machine Learning model. It consists of a sequential series of models, each one focussing on the errors of the previous one, trying to improve them. The most common underlying model is the [Decision Tree]({{< ref "/posts/classical_ml/decision_trees.md">}}), other models are however possible. In this post, we will introduce the algorithm of AdaBoost and have a look at a simplified example for a classification task using [sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html). For a more detailed exploration of this example - deriving it by hand - please refer to [AdaBoost for Classification - Example](). A more realistic example with a larger dataset is provided on [kaggle](). Accordingly, if you are interested how AdaBoost is developed for a regression task, please check the article [AdaBoost for Regression - Example](). 

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

## AdaBoost in Python

The [sklearn](https://scikit-learn.org/stable/) library offers a method to fit AdaBoost in Python for both [classification](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html) and [regression](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostRegressor.html#sklearn.ensemble.AdaBoostRegressor) problems. We will consider a simplified example for a classification task, using the following data.

```Python
import pandas as pd

data = {'age': [23, 31, 35, 35, 42, 43, 45, 46, 46, 51], 
        'likes goats': [0, 1, 0, 0, 0, 1, 1, 1, 0, 1], 
        'likes height': [0, 1, 1, 0, 0, 1, 0, 1, 1, 1], 
        'go rock climbing': [0, 1, 1, 0, 0, 1, 0, 1, 0, 1]}

df = pd.DataFrame(data)
```
![adaboost_data](/images/adaboost/adaboost_data.png)
*Example dataset to illustrate AdaBoost in Python.*

We use this data to fit an AdaBoost Classifier. As this is a very simplified dataset, we use only three models to build the ensemble model. This is done by setting the hyperparamter *n_estimators=3*. The other hyperparameters are left as the default values. That means as base models the stumps of Decision Trees are used.

```Python
from sklearn.ensemble import AdaBoostClassifier

X = df[['age', 'likes goats', 'likes height']].values
y = df[['go rock climbing']].values.reshape(-1,)

clf = AdaBoostClassifier(n_estimators=3, random_state=42)
clf.fit(X, y)
```

To make predictions we can use the *predict* method, and then we can print the score, which is defined as the mean accuracy.

```Python
y_hat = clf.predict(X)
print(f"predictions: {y_hat}")
print(f"score: {clf.score(X, y)}")
```

The predictions are $[0, 1, 1, 0, 0, 1, 0, 1, 0, 1]$ and the score is $1.0$. That means our model predicts all samples correctly. We can also print the predictions and scores after each boosting iteration, to see how they evolve.

```Python
staged_predictions = [p for p in clf.staged_predict(X)]
staged_score = [p for p in clf.staged_score(X, y)]
```

The predictions for the three stages are 

stage 1: $[0, 1, 1, 0, 0, 1, 0, 1, 1, 1]$, 

stage 2: $[0, 1, 0, 0, 0, 1, 0, 1, 0, 1]$, and

stage 3 $[0, 1, 1, 0, 0, 1, 0, 1, 0, 1]$. 

Accordingly the scores are

stage 1: $0.9$,

stage 2: $0.9$, and

stage 3: $1.0$.

 This shows how the predictions and the score improve over the three iterations. To illustrate the model we can plot the three stumps created. We can access them using the *estimators_* attribute. Note that creating the stumps contains randomness, when the modified dataset is constructed, as described above. In order to make the results reproducible the *random_seed* is set, when fitting the model.

```Python
from sklearn import tree

tree.plot_tree(clf.estimators_[0], 
	feature_names=['age', 'likes goats', 'likes height'], fontsize=10) 
```

![adaboost_stumps](/images/adaboost/adaboost_stumps.png)
*The three stumps for the AdaBoost model of the example.*

Comparing the first stump to the calculations in the article [Decision Trees for Classification - Example]({{<ref "/posts/classical_ml/decision_tree_classification_example.md">}}), in which the same dataset is used, we can see that this is exactly the beginning of the Decision Tree developed. Please find a detailed derivation of the above example, calculating it by hand in the separate article [AdaBoost for Classification - Example](). In a real project, we would of course divide our data in training, validation and test data, and then fit the model to the training data only and evaluate on validation and finaly on the test data. A more realistic example is provided on [kaggle]() 

## Summary

In this article, we learned about AdaBoost, a sequential ensemble model, in which a sequential series of models is developed. Sequentially the errors of the developed models are evaluated and the dataset modified such that a higher focus lies on the wrongly predicted samples for the next iteration. In Python, we can use [sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html) to fit a AdaBoost model, which also offers some methods to explore created models and their predictions. The example used in this article, however, was very simplified and only for illustration purposes. For a more developed example for AdaBoost in Python, please refer to [kaggle](). 

If this blog is useful for you, I'm thankful for your support!
{{< bmc-button slug="pumaline" >}}

