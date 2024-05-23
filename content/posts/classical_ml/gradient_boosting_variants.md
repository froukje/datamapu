+++
title = 'Gradient Boosting Variants - Sklearn vs. XGBoost vs. LightGBM vs. CatBoost'
date = 2024-05-08T20:55:43-03:00
draft = true
tags = ["Data Science", "Machine Learning", "Boosting", "Tree Methods"]
categories = ["Data Science", "Machine Learning", "Ensemble", "Boosting", "Tree Methods"]
keywords = ["Data Science", "Machine Learning", "Ensemble", "Boosting", "Tree Methods"]
images = ['/images/']

+++

## Introduction

Gradient Boosting is an ensemble model which is built of a sequential series of shallow [Decision Trees]({{< ref "/posts/classical_ml/decision_trees.md" >}}). The single trees are weak learners and have little predictive skill, that is only a higher than random guessing. Together, they form a strong learner with high predictive skill. In this article, we discuss the different implementations of [Gradient Boosting]({{< ref "/posts/classical_ml/gradient_boosting_regression.md" >}}). We give a high-level overview of the differences for a more in depth understanding, further literature is given.

## Background

**Gradient Boosting in sklearn**

Gradient Boosting is one of many available Machine Learning algorithms available in [sklearn](https://scikit-learn.org/stable/), which is short for scikit-learn. It started as a Google summer of code project started by David Cournapeau and was originally called scikits.learn. The first version was published in 2010 by the contributors Fabian Pedregosa, Gaël Varoquaux, Alexandre Gramfort and Vincent Michel, from the French Institute for Research in Computer Science and Automation in Saclay, France. Nowadays it is one of the most extensive and used libraries in Machine Learning. 

**XGBoost**

XGBoost stands for eXtreme Gradient Boosting and is an algorithm optimizing computation speed and model performance. XGBoost was first developed by Tianqi Chen. It became popular and famous for many winning solutions in Machine Learning competitions. It can be used as a separated library, but nowadays also an integration in sklearn exists. XGBoosts algorithm is capable to handle large datasets.

**LightGBM**

LightGBM is shirt for Light Gradient-Boosting Machine and was developed by Microsoft. It has similar advantages as XGBoost and also able to handle large datasets. The main difference between LighGBM and XGBoost is the way the trees are build. In LightGBM the trees are not grown level-wise, but leave-wise. We will discuss this in more detail later.

**CatBoost**

CatBoost was developed by Yandex, a russion technology company and has a special focus on how categorical values are treated in Gradient Boosting. It was developed in 2016 and was based on previous projects focussing on Gradient Boosting algorithms. It was first released open-source in 2017.

## Performance

**Gradient Boosting in sklearn**

**XGBoost**

**LightGBM**

## Feature Handling

**Gradient Boosting in sklearn**

**XGBoost**

**LightGBM**

**CatBoost**

## Tree Growths

**Gradient Boosting in sklearn**

* mention hist-based model 

histogram-based algorithm that performs bucketing of values (also requires lesser memory)

**XGBoost**


**LightGBM**

"In contrast to the level-wise (horizontal) growth in XGBoost, LightGBM carries out leaf-wise (vertical) growth that results in more loss reduction and in turn higher accuracy while being faster. But this may also result in overfitting on the training data which could be handled using the max-depth parameter that specifies where the splitting would occur. Hence, XGBoost is capable of building more robust models than LightGBM."

**CatBoost**

## Accuracy

**Gradient Boosting in sklearn**

**XGBoost**

**LightGBM**

**CatBoost**

## Ease of Use

**Gradient Boosting in sklearn**

**XGBoost**

**LightGBM**

**CatBoost**

## Scalability and Deployment

**Gradient Boosting in sklearn**

**XGBoost**

**LightGBM**

**CatBoost**

## Code Examples

In this section a brief example of each method is given for a classification task, however all models have variants for regression problems. The dataset used is described in the plot below. The example used here is very simple and only for illustration purposes. More detailed examples on a larger dataset can be found on [kaggle](https://www.kaggle.com/work/code).
 
!["data"](/images/gradient_boosting/gb_class_data.png)
*The data used for the following examples.*

We read this data into a Pandas dataframe

```Python
import pandas as pd

data = {'age': [23, 31, 35, 35, 42, 43, 45, 46, 46, 51],
        'likes goats': [0, 1, 0, 0, 0, 1, 1, 1, 0, 1],
        'likes height': [0, 1, 1, 0, 0, 1, 0, 1, 1, 1],
        'go rock climbing': [0, 1, 1, 0, 0, 1, 0, 1, 0, 1]}

df = pd.DataFrame(data)
```

**Gradient Boosting in sklearn**

For the **installation** of sklearn please check the official [documentation](https://scikit-learn.org/stable/install.html). You have different options using conda or pip, that may differ depending on your operating system. You can find all details there.
To use the *GradientBoostingClassifier*, we need to import it from sklearn. With the input features $X$ and the target data $y$ defined, we can fit the model. First we need to instatiate it and then following the concept of training machine Learning models in sklearn, we can use the *fit* method. The process is illustrated in the following code snipped. In this example, three hyperparameters are set: *n_estimators*, which describes the number of trees used, *max_depth*; which is the depth of the single trees; and the *learning_rate*, which defines the weight of each tree. A full list of hyperparameters, that can be changed to optimize the model can be found in the documentation of the [GradientBoostingRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html) and the [GradientBoostingClassification](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html) methods.


```Python
from sklearn.ensemble import GradientBoostingClassifier

X = df[['age', 'likes goats', 'likes height']].values
y = df[['go rock climbing']].values.reshape(-1,)
clf = GradientBoostingClassifier(
    n_estimators=2,
    max_depth=2,
    learning_rate=0.3
    )
clf.fit(X, y)
```

We can use the *predict* method  to get the predictions and the *score* method to calculate the accuracy.

```Python
y_pred = clf.predict(X)
score = clf.score(X, y)
```

A more detailed example of applying Gradient Boosting in Python to a Regression task can be found on [kaggle](https://www.kaggle.com/code/pumalin/gradient-boosting-tutorial).

**XGBoost**

The **installation** is described in detail in the [XGBboost](https://xgboost.readthedocs.io/en/stable/install.html) documentaion.  	

To use the *XGBoostClassifier*, we need to import this method. The procedure is the same as for the *sklearn* model. We instantiate the model and then use the *fit* and *predict* method. A detailed list of all possible parameters can be found in the [XGBoost documentation](https://xgboost.readthedocs.io/en/stable/parameter.html). There are three types of parameters: general parameters, booster parameters and task parameters. The following gives a brief explanation of each category, but is not an exaustive list.

* *General parameters* define which booster is used. By default this is a *Gradient Boosting Tree*; the same as in the sklearn method, other options are *dart* and *gblinear*. The methods *gbtree* and *dart* are tree based models, *gblinear* is a linear model. Other parameters set here refer to the device used (cpu oder gpu), the number of threads used for training, verbosity and several more.
* *Booster parameters* depend on which booster method is used. For a Tree Booster, we can e.g. choose the number of estimators (*n_estimators*), the learning rate (*learning_rate* or *eta*), the maximal depth of the trees (*max_depth*), and many more. Further the type of tree method to use, which can be changed to optimize the speed. 
* *Learning task parameters* depend on the task to be solved. The objective function and the evaluation metrics are defined here, which depend on whether we consider a regression or a clissification task. There is a wide variety of pre-defined objective functions and metrics for both types of problems are available, but we can also define custom functions. For more details, please refer to this [post](https://medium.com/@pumaline/loss-functions-in-xgboost-c89885b57346).

```Python
from xgboost import XGBClassifier

X = df[['age', 'likes goats', 'likes height']].values
y = df[['go rock climbing']].values.reshape(-1,)

bst = XGBClassifier(
	n_estimators=2, 
	max_depth=2, 
	learning_rate=1, 
	objective='binary:logistic')
bst.fit(X, y)
preds = bst.predict(X)
```

*XGBoost* also allows to use callbacks, such as *early stopping*, or customized callbacks. A more detailed example using XGBoost with different parameters and callbacks on a larger dataset can be found on [kaggle](). 

< kaggle NOTEBOOK >

**LightGBM**

A guide for the **installation** can be found on the [LightGBM](https://lightgbm.readthedocs.io/en/stable/Python-Intro.html) documentation. 

As in the previous models, we define the input features $X$ and the target value $y$, then we instantiate the model, fit it to the data, and make predictions. A full list of possible parameters can be found in the [documentation](https://lightgbm.readthedocs.io/en/stable/Parameters.html). We can define a large set of different types of parameters, considering the core functionality, the control of the learning, the dataset, the metrics and more. Similar to *XGBoost*, *LightGBM* allows us to choose between different boosting methods. The default method are Gradient Boosting Trees In this example we set the number of estimators (*n_estimators*), the maximal depth of the Decision Trees (*max_depth*), the learning rate (*learning_rate*), and the maximal number of leaves (*num_leaves*).
 
```Python
import lightgbm as lgb

X = df[['age', 'likes goats', 'likes height']].values
y = df[['go rock climbing']].values.reshape(-1,)

clf = lgb.LGBMClassifier(n_estimators=2,
                         max_depth=5,
                         learning_rate=1,
                         num_leaves=3)
clf.fit(X, y)
clf.predict(X)
```

*LightGBM* allows an easy implementation of e.g. early stopping and cross validation using callbacks. For a more detailed example on a larger dataset, please check this [kaggle]() notebook.

< kaggle NOTEBOOK >

**CatBoost**

Using *CatBoost*, the procedure of training and predicting is the same as in the previous examples: instantiate the model, fit it to the data, and make predictions. A full list of all possible parameters is in the [documentation](https://catboost.ai/en/docs/concepts/python-reference_catboostclassifier). For this example, we set *iterations*, which refers to the number of trees trained, the *learning_rate*, and the *depth*, which defines the maximal depth of the Decision Trees.  

```Python
from catboost import CatBoostClassifier

X = df[['age', 'likes goats', 'likes height']].values
y = df[['go rock climbing']].values.reshape(-1,)

clf = CatBoostClassifier(iterations=2,
                          learning_rate=1,
                          depth=2)
clf.fit(X, y)
clf.predict(X)
```

For a more realistic example on a larger dataset, please refer to this [kaggle]() notebook.

< kaggle NOTEBOOK >

## XGBoost


The model supports the following kinds of boosting:

Gradient Boosting as controlled by the learning rate

Stochastic Gradient Boosting that leverages sub-sampling at a row, column or column per split levels

Regularized Gradient Boosting using L1 (Lasso) and L2 (Ridge) regularization 


Some of the other features that are offered from a system performance point of view are:

Using a cluster of machines to train a model using distributed computing

Utilization of all the available cores of a CPU during tree construction for parallelization

Out-of-core computing when working with datasets that do not fit into memory

Making the best use of hardware with cache optimization


In addition to the above the framework:

Accepts multiple types of input data

Works well with sparse input data for tree and linear booster

Supports the use of customized objective and evaluation functions

## LightGBT


https://neptune.ai/blog/xgboost-vs-lightgbm#:~:text=In%20contrast%20to%20the%20level,higher%20accuracy%20while%20being%20faster.

## CatBoost

https://catboost.ai/

https://www.geeksforgeeks.org/catboost-ml/

## Summary

---
If this blog is useful for you, please consider supporting.

{{< bmc-button slug="pumaline" >}}


