+++
title = 'Adaboost for Regression - Example'
date = 2024-01-19T23:05:44-03:00
draft = true
tags = ["Data Science", "Machine Learning", "Regression", "Ensemble", "Boosting", "Tree Methods", "Regression", "Classification"]
categories = ["Data Science", "Machine Learning", "Regression", "Ensemble", "Boosting", "Tree Methods", "Classification", "Regression"]
keywords = ["Data Science", "Machine Learning", "Regression", "Ensemble", "Boosting", "Tree Methods", "Classification", "Regression"]
images = ['/images/adaboost/']
+++

## Introduction

AdaBoost is an ensemble model, that sequentially builds new models based on the errors of the previous model to improve the predictions. The most common case is to use Decision Trees as base models. Very often the examples explained are for classification tasks. AdaBoost can however also be used for Regression problems, on what we will focus in this post.

## Data



## Build the Model

We will build a AdaBoost model from scratch using the above dataset. We use the default values, that are used in [sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html), that is we use Decision Trees as underlying models with a maximum depth of three. 

## Fit a Model in Python

After developing a model by hand, we will now see how to fit a AdaBoost for a regression task in Python. We can use the [sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html) method *AdaBoostRegressor*. **Note: The fitted model in sklearn differs from our developed model, due to some randomness in the algorithm.** Randomness occurss in the underlying  *DecisionTreeRegressor* algorithm and in the boosting used in the *AdaBoostRegressor*. 

We first create a dataframe for our data

```Python
import pandas as pd

data = {'age': [23, 31, 35, 35, 42, 43, 45, 46, 46, 51], 
        'likes goats': [0, 1, 0, 0, 0, 1, 1, 1, 0, 1], 
        'likes height': [0, 1, 1, 0, 0, 1, 0, 1, 1, 1], 
        'climbed meters': [200, 700, 600, 300, 200, 700, 300, 700, 600, 700]}

df = pd.DataFrame(data)
```

Now, we can fit the model. Because this example is only for illustration purposes and the dataset is very small, we limit the number of underlying Decision Trees to $3$, by setting the hyperparameter *n_estimators=3*. Note, that in a real world project, this number would usually be much higher, the default value in [sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html) is $100$.  

```Python
from sklearn.ensemble import AdaBoostRegressor

X = df[['age', 'likes goats', 'likes height']].values
y = df[['climbed meters']].values.reshape(-1,)
reg = AdaBoostRegressor(n_estimators=3, random_state=42)
reg.fit(X, y)
```

We can then make predictions using the *predict* method and print the *score*, which is defined as the mean accuracy.

```Python
y_hat = reg.predict(X)
print(f"predictions: {y_hat}")
print(f"score: {reg.score(X, y)}")
```

This leads to the predictions $[250. 700. 600. 300. 200. 700. 300. 700. 700. 700.]$ and a score of $0.97$. Additionally, we can also print the predictions of the individual models for the three stages using the method *staged_predict*.

```Python
staged_predictions = [p for p in reg.staged_predict(X)]
```

This yields to

stage 1: $[250., 700., 700., 250., 200., 700., 300., 700., 700., 700.]$,

stage 2: $[300., 700., 600., 300., 200., 700., 300., 700., 600., 700.]$, and

stage 3: $[250., 700., 600., 300., 200., 700., 300., 700., 700., 700.]$
 
which shows that all three models yield to different predictions for some samples. The influences are called *estimator_weights_* in sklearn and can also be printed.

```Python
clf.estimator_weights_
```

For this example the weights are $[1.38629436, 1.14072377, 1.26274917]$. These weights are used for the final prediction, which is achieved by calculating the weighted mean of the individual predictions, with the weights being the influences of the underlying models. Let´s consider a concrete example and make predictions for one sample of the dataset.

|Feature     | Value|
|:----------:|:----:|
|age         | 45   |
|likes height| 0    |
|likes goats | 1    |

We can visualize the underlying Decision Trees and follow the decision paths. For the first tree the visualization is achieved as follows.

```Python
from sklearn import tree

tree.plot_tree(reg.estimators_[0], 
	feature_names=['age', 'likes goats', 'likes height'], fontsize=8)
```
![adaboost_first_tree](/images/adaboost/ab_example_reg_tree1.png)
*Prediction of the first tree.*

![adaboost_first_tree](/images/adaboost/ab_example_reg_tree2.png)
*Prediction of the second tree.*

![adaboost_first_tree](/images/adaboost/ab_example_reg_tree3.png)
*Prediction of the third tree.*

Combining these three predictions with the influences (estimator_weights_) leads to the final prediction

$$\hat{y} = \frac{\sum w_i \cdot \hat{y}_i}{\sum w_i}$$

Filling in the according numbers, this gives

$$\hat{y} = \frac{1.38629436 * 300 + 1.14072377 * 300 + 1.26274917 * 300}{1.38629436 + 1.14072377 + 1.26274917} = 300,$$

which coincides with the prediction, we printed above for this sample.

## Summary

In this article we developed an AdaBoost model for a Regression task by hand following the steps described in the separate articel [AdaBoost - Explained]({{< ref "/posts/classical_ml/adaboost.md">}}). Additionally a model was developed using [sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html). Although both models were derived for the same dataset, the final models differ due to some randomness in the algorithm. This post focused on the application of the algorithm to a simplified regression example. For a detailed example for a classification task, please refer to the article [AdaBoost for Classification - Example]({{< ref "/posts/classical_ml/adaboost_example_clf.md">}}). 


If this blog is useful for you, I'm thankful for your support!
{{< bmc-button slug="pumaline" >}}
                                     
