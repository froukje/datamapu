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

Gradient Boosting is an ensemble model which is built of a sequential series of shallow [Decision Trees]({{< ref "/posts/classical_ml/decision_trees.md" >}}). The single trees are weak learners and have little predictive skill, that is only slightly higher than random guessing. Together, they form a strong learner with high predictive skill. For a more detailed explanatione, please refer to the post [Gradient Boosting for Regression - Explained]({{< ref "/posts/classical_ml/gradient_boosting_regression.md" >}}). In this article, we will discuss differet implementations of Gradient Boosting. The focus is to give a high-level overview of different implementations and discuss the differences. For a more in depth understanding of each framework, further literature is given.

## Background

### Gradient Boosting in sklearn

Gradient Boosting is one of many available Machine Learning algorithms available in [sklearn](https://scikit-learn.org/stable/), which is short for scikit-learn, which started as a Google summer of code project by David Cournapeau and was originally called scikits.learn. The first version was published in 2010 by the contributors Fabian Pedregosa, Gaël Varoquaux, Alexandre Gramfort and Vincent Michel, from the French Institute for Research in Computer Science and Automation in Saclay, France. Nowadays it is one of the most extensive and used libraries in Machine Learning. 

### XGBoost

XGBoost stands for eXtreme Gradient Boosting and is an algorithm focussing on optimizing computation speed and model performance. XGBoost was first developed by Tianqi Chen. It became popular and famous for many winning solutions in Machine Learning competitions. It can be used as a separated library, but also an integration in sklearn exists. The XGBoosts algorithm is capable to handle large and complex datasets.

### LightGBM

LightGBM is short for Light Gradient-Boosting Machine and was developed by Microsoft. It has similar advantages as XGBoost and also able to handle large and complex datasets. The main difference between LightGBM and XGBoost is the way the trees are built. In LightGBM the trees are not grown level-wise, but leave-wise.

### CatBoost

CatBoost was developed by Yandex, a Russian technology company and has a special focus on how categorical values are treated in Gradient Boosting. It was developed in 2016 and was based on previous projects focussing on Gradient Boosting algorithms. It was first released open-source in 2017.

## Feature Handling

### Gradient Boosting in sklearn

1. [GradientBoostingRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html) / [GradientBoostingClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html)

**Numerical Features:** Gradient Boosting in sklearn expects numerical input. 

**Categorical Features:** Gradient Boosting in sklearn does not natively support categorical features. They need to be encoded using e.g. one-hot encoding or label-encoding.

**Missing Values:** Gradient Boosting in sklearn does not handle missing values. Missing values need to be removed or imputed prior to training the model.

2. [HistGradientBoostingRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html) / [HistGradientBoostingClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html)

Hostgram-based Gradient Boosting by sklearn was inspired by LightGBM.

**Numerical & Categorical Features:** The Histogram-based Gradient Booster natively supports numerical and categorical features. For more details, please check the [documentation](https://scikit-learn.org/stable/auto_examples/ensemble/plot_gradient_boosting_categorical.html).

**Missing Values:** The Histogram-based Gradient Booster is able to handle missing values natively. More explanations can be found in the [documentation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html). 

### XGBoost

**Numerical Features:** XGBoost supports directly numerical data.

**Categorical Features:** Both the native environment and the sklearn interface support categorical features using the parameter *enable_categorical*. Examples for both interfaces can be found in the [documentation](https://xgboost.readthedocs.io/en/stable/tutorials/categorical.html). 

**Missing Values:** XGBoost natively supports missing values. Note, that the treatment of missing values differ with the booster type used. For more explanataions, please refer to the [documentation](https://xgboost.readthedocs.io/en/stable/faq.html#how-to-deal-with-missing-values). 

### LightGBM

**Numerical Features:** LightGBM supports numerical data.

**Categorical Features:** LightGBM has built-in support for categorical features. Categorical features can be specified using the parameter *categorical_feature*. For more details, please refer to the [documentation](https://lightgbm.readthedocs.io/en/stable/Advanced-Topics.html#categorical-feature-support).

**Missing Values:** LightGBM natively supports the handling of missing values. It can be disabled using the parameter *use_missing=false*. More information is available in the [documentation](https://lightgbm.readthedocs.io/en/stable/Advanced-Topics.html#missing-value-handle).

### CatBoost

**Numerical Features:**  CatBoost supports numerical data.

**Categorical Features:** CatBoost is specifically designed to handle categorical features without needing to preprocess them into numerical formats. How this transformation is done, is described in the [documentation](https://catboost.ai/en/docs/concepts/algorithm-main-stages_cat-to-numberic)

**Missing Values:** CatBoost has robust handling for missing values, treating them as a separate category or using specific strategies for imputation during training. How missing values are treated depends on the feature type. More details can be found in the [documentation](https://catboost.ai/en/docs/concepts/algorithm-missing-values-processing)

![feature handling](/images/gradient_boosting/gb_variants_1.png)
*Feature handling in the different implementations.*


## Tree Growths

### Gradient Boosting in sklearn

1. [GradientBoostingRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html) / [GradientBoostingClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html)

Level-wise Growth: Trees are grown level by level, where all nodes at a given depth are expanded before moving to the next level.
Splitting Criterion: Uses criteria like Mean Squared Error (MSE) for regression tasks or log-loss for classification tasks to determine the best splits.
Depth Control: Tree depth is typically controlled by parameters like max_depth or max_leaf_nodes.

Characteristics:

Produces balanced trees.
Can be slower and more memory-intensive due to examining many potential splits across the entire dataset.
Requires careful tuning of parameters to balance bias and variance.

2. [HistGradientBoostingRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html) / [HistGradientBoostingClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html)

Histogram-based Growth: Utilizes histograms to approximate the data distribution, which speeds up the process of finding optimal splits.
Binning: Continuous features are binned into discrete bins, which reduces the number of split points to evaluate.
Level-wise Growth: Similar to traditional gradient boosting in scikit-learn, it grows trees level by level.

Characteristics:

Faster training compared to traditional gradient boosting due to reduced computational complexity.
Handles large datasets more efficiently by reducing the number of potential split points.
Balances between speed and accuracy by approximating continuous data with histograms.

histogram-based algorithm that performs bucketing of values (also requires lesser memory)

### XGBoost

Level-wise Growth: Trees are grown level by level, ensuring balanced tree structures.
Regularization: Includes L1 and L2 regularization to prevent overfitting.
Optimal Split Finding: Utilizes advanced algorithms to efficiently find the best splits.
Pruning: Implements a technique called "pruning" where splits are undone if they do not result in a positive gain.

Characteristics:

Balanced between complexity and computational efficiency.
Strong regularization techniques to improve generalization.
Supports parallel processing, enhancing training speed.

### LightGBM

Leaf-wise Growth: Instead of growing level by level, LightGBM grows the tree by expanding the leaf with the highest potential for reducing the loss function.
Gradient-based One-Side Sampling (GOSS): Prioritizes instances with larger gradients to improve efficiency.
Exclusive Feature Bundling (EFB): Combines mutually exclusive features to reduce the number of features considered for splits.

Characteristics:

Generally faster and more efficient, especially on large datasets.
Produces deeper and more complex trees, which can improve accuracy but also risk overfitting.
Highly optimized for performance with techniques like GOSS and EFB.

"In contrast to the level-wise (horizontal) growth in XGBoost, LightGBM carries out leaf-wise (vertical) growth that results in more loss reduction and in turn higher accuracy while being faster. But this may also result in overfitting on the training data which could be handled using the max-depth parameter that specifies where the splitting would occur. Hence, XGBoost is capable of building more robust models than LightGBM."


### CatBoost

Symmetric Tree Growth: All leaves at a given depth are split in the same way, resulting in a symmetric tree structure.
Ordered Boosting: Uses ordered boosting to reduce overfitting and improve the accuracy of the model.
Categorical Feature Handling: Handles categorical features natively and efficiently without the need for preprocessing.

Characteristics:

Produces balanced and symmetric trees, leading to efficient training and inference.
Robust against overfitting with ordered boosting and other regularization techniques.
Performs well with default parameters, requiring less hyperparameter tuning.

## Accuracy

**Gradient Boosting in sklearn**

**XGBoost**

**LightGBM**

**CatBoost**

## Performance

**Gradient Boosting in sklearn**

**XGBoost**

**LightGBM**


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


