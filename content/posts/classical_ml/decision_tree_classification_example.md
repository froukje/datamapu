+++
title = 'Decision Trees for Classification - Example'
date = 2023-12-19T09:11:46+01:00
draft = false
featured_image = ''
tags = ["Data Science", "Machine Learning", "Classification", "Decision Trees", "Tree Methods"]
categories = ["Data Science", "Machine Learning", "Decision Trees", "Tree Methods", "Classification"]
keywords = ["Data Science", "Machine Learning", "Decision Trees", "Tree Methods", "Classification"]
+++

## Introduction

Decision Trees are a powerful, yet simple Machine Learning Model. An advantage of their simplicity is that we can build and understand them step by step. In this post, we are looking at a simplified example to build an entire Decision Tree by hand for a classification task. After calculating the tree, we will use the [sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html) package and compare the results. To learn how to build a Decision Tree for a regression problem, please refer to the article [Decision Trees for Regression - Example]({{< ref "decision_tree_regression_example">}}). For a general introduction to Decision Trees and how to build them please check the article [Decision Trees - Explained]({{< ref "decision_trees">}}). 

## Data

The Dataset we use in this post contains only 10 samples. We want to decide whether a person should go rock climbing or not, depending on whether they like height, like goats, and their age. That is the dataset contains three input features, of which two are categorical, as both have exactly two classes they are even binary and one is numerical. The target variable is also categorical.  

![data](/images/decision_tree/dt_data_classification.png)
*Data used to build a Decision Tree.*

## Build the Tree

Our target data is categorical, that is we are building a Decision Tree for a classification problem. The main step in building a Decision Tree is splitting the data according to a splitting criterion. There exist different splitting criteria. We will use the *Gini Impurity*, which is the most common criterion and also used in the [sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html) package as the default criterion. The Gini Impurity for a Dataset $D$ that is split into two Datasets $D_1$ and $D_2$, is defined as

$$Gini(D) = \frac{n_1}{n} \cdot Gini(D_1) + \frac{n_2}{n} \cdot Gini(D_2),$$

with $n = n_1 + n_2$ the size of the dataset $D$, its subsets $D_1$, $D_2$, and

$$Gini(D_i) = 1 - \sum_{j=1}^c p_j^2.$$

With $p_j$ being the probability that a randomly drawn sample from this node belongs to class $j$ and $c$ the number of classes. Starting with the root node, which contains the entire dataset to make the first split, we calculate the Gini Impurity for all three features in the dataset ('age', 'likes goats', and 'likes height') and then choose the feature that gives the lowest Gini Impurity. We will start with the categorical features. 

![split_goat](/images/decision_tree/dt_classification_goat.png)
*Gini Impurity if the split is done with the feature 'likes goats'.*

![split_height](/images/decision_tree/dt_classification_height.png)
*Gini Impurity if the split is done with the feature 'likes height'.*

From these two features, we can see that 'likes height' has a lower Gini Impurity and would therefore be preferred to 'likes goats'. For numerical features, the calculation is a bit more complex. The strategy is the following

1. Order the numerical feature in an ascending way.
2. Calculate the mean of neighboring items. These are all possible splits.
3. Determine the Gini Impurity for all possible splits.
4. Choose the lowest of these Gini Impurities as the Gini Impurity of this feature.

For the feature 'age' the values are already ordered, but we still need to calculate the means to find all possible splits.

![possiple splits age](/images/decision_tree/dt_splits_age.png)
*Possible splits for the numerical feature 'age'.*

Now, let's calculate the Gini Impurity for each of these splits.


![all possiple splits age](/images/decision_tree/dt_classification_age_all_splits.drawio.png)
*All possible splits for 'age' and their corresponding Gini Impurity.*

From the above calculations, we see that all Gini Impurities for the feature 'age' are higher than the one for 'likes height', which was our previous best feature. That is 'likes height' is the feature that results in the lowest Gini Impurity of all three features and we will use it for the first split of the tree. 

![first split](/images/decision_tree/dt_classification_first_split.png)
*First split of the Decision Tree.*
 
After this first split, one of the resulting nodes is already pure, that is no further split is possible and we have the first leaf of our tree. The second node is not pure and will be split using the remaining dataset. We calculate the Gini Impurity for the features 'likes goats' and 'age' exactly as we did for the entire dataset.

![second splits all possibilities](/images/decision_tree/dt_classification_second_split_all.drawio.png)
*All possible splits for the second split.*

From the above plot we see that the feature with the lowest Gini Impurity is 'likes goats'. This will thus be our second split.

![second split](/images/decision_tree/dt_classification_second_split.png)
*Second split of the Decision Tree.*

Now there is just one node remaining that we need to split. The final Decision Tree has the following form.

![example](/images/decision_tree/dt_example2.png)
*Illustration of the final Decision Tree.*
 
## Fit a Model in Python

In Python we can use [sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html) to fit a Decision Tree. For a classification task, we use the *DecisionTreeClassifier* Class.

```Python
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

data = {'age': [23, 31, 35, 35, 42, 43, 45, 46, 46, 51],
        'likes goats': [0, 1, 0, 0, 0, 1, 1, 1, 0, 1],
        'likes height': [0, 1, 1, 0, 0, 1, 0, 1, 1, 1],
        'go rock climbing': [0, 1, 1, 0, 0, 1, 0, 1, 0, 1]}

df = pd.DataFrame(data)
X = df[['age', 'likes goats', 'likes height']].values
y = df[['go rock climbing']].values.reshape(-1,)
clf = DecisionTreeClassifier()
clf = clf.fit(X, y)
```
We can visualize the fitted tree also using [sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.tree.plot_tree.html). The tree build using sklearn is exactly the same we build by calculating the splits by hand.

```Python
from sklearn.tree import plot_tree
plot_tree(clf, feature_names=['age', 'likes goats', 'likes height'], fontsize=10)
```
![example in python](/images/decision_tree/dt_python.png)
*Illustration of the final Decision Tree built in Python.*

## Summary

In this article, we analyzed in detail how to build a Decision Tree for a classification task, especially how to choose the best split step by step. A more realistic example of how to fit a Decision Tree to a dataset using sklearn can be found on [kaggle](https://www.kaggle.com/code/pumalin/decision-trees-tutorial).
