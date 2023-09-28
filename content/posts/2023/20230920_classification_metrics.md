+++
title = 'Metrics for Classification Problems'
date = 2023-09-20T00:08:49+02:00
draft = true
featured_image = '/images/20230920_classification_metrics/classification_metrics.gif'
tags = [ "Data Science", "Classification", "Metrics"]
categories = [ "Data Science" ]
+++

## Metrics for Classification Problems  
### An Overview

![classification metrics](/images/20230920_classification_metrics/classification_metrics.gif)

### Classification Problems

Supervised Machine Learning projects can be divided into [regression](https://en.wikipedia.org/wiki/Regression_analysis) and [classification](https://en.wikipedia.org/wiki/Statistical_classification) problems. In regression problems, we predict a continuous variable (e.g. temperature), while in classification, we classify the data into discrete classes (e.g. classify cat and dog images). A subset of classification problems is the so-called [binary classification](https://en.wikipedia.org/wiki/Binary_classification), where only two classes are considered. An example of this is classifying e-mails as spam and no-spam or cat images versus dog images. When we consider more than two classes, we speak of [multi-class classification](https://en.wikipedia.org/wiki/Multiclass_classification). An example of this is the [MNIST](https://en.wikipedia.org/wiki/MNIST_database) dataset, where images of handwritten digits are classified, i.e. we have 10 different classes. Depending on the problem we consider, we need different metrics to evaluate our Machine Learning model. Within each of these two types of Machine Learning problems, we have to choose which of the metrics fits our needs best. This post will give an overview of the most common metrics for **binary classification** problems, what they mean, and when to use them.

![regression_classification](/images/20230920_classification_metrics/regression_classification.gif "Regression, binary classification and multiple classifiction illustrated.")
*Regression, binary classification and multiple classifiction illustrated.*

### Confusion Matrix
A good way to start evaluating a classification problem is the so-called *confusion matrix*. It gives an overview of how many predictions were correct and how many were not for each class. Let's imagine we want to classify cat and dog images and our model gives the following result on 10 test images. 

![classification table](/images/20230920_classification_metrics/cats_dogs.gif "Example true and predicted values.")
*Example true and predicted values converted to 0s and 1s.*

From this table, we can see when our model was correct and when not. The confusion matrix is a way to illustrate these results more clearly. We can use the [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html) package to display the confusion matrix in Python. 

We can then display the confusion matrix as follows:

```
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=["cat", "dog"])
disp.plot();
```

![confusion matrix](/images/20230920_classification_metrics/confusion_matrix.jpg "Confusion matrix.")
*Confusion matrix example.*

The confusion matrix shows the predicted labels versus the true labels, and the numbers in the matrix give the number of images that were predicted for each case. The first column, first row of the matrix (cat - cat) shows how many results predicted a cat and the real image was also a cat. The first column, second row (cat-dog) shows how many results predicted a cat, but the true label was a dog and so on. Note, that this not only divides into correct and incorrect, but also in the type of error we make. There are four possible outcomes:

* *TP (True Positive)*: The prediction is a dog (positive) and the true image is also a dog (positive)
* *FN (False Negative)*: The prediction is a cat (negative), but the true image is a dog (positive)
* *FP (False Positive)*: The prediction is a dog (positive), but the true image is a cat (negative)
* *TN (True Negative)*: The prediction is a cat (negative) and the true image is a cat (negative).

More general this looks like this.
![confusion matrix](/images/20230920_classification_metrics/confusion_matrix2.jpg "Confusion matrix.")
*Confusion matrix.*

### Metrics
With the above concepts of TP, FN, FP, and TN we can define several metrics that evaluate binary classification problems in different aspects. Let's have a look at the most common metrics for binary classification problems, how to calculate them and when to use them.


![accuracy](/images/20230920_classification_metrics/accuracy.jpg "Accuracy.")
![recall](/images/20230920_classification_metrics/recall.jpg "Recall.")
![precision](/images/20230920_classification_metrics/precision.jpg "Precision.")
![TNR](/images/20230920_classification_metrics/true_negative_rate.jpg "True Negative Rate.")
![f1_score](/images/20230920_classification_metrics/f1score.jpg "f-1-score.")
![f_beta_score](/images/20230920_classification_metrics/fbetascore.jpg "f-beta-score.")
![FPR](/images/20230920_classification_metrics/false_positive_rate.jpg "f-beta-score")

To classify into positive and negative predictions, we need to define a threshold. Predictions below this threshold are classified as 0 (cat) and predictions above are clssified as 1 (dog). By default this value is 0.5, but we can change it. when we change the theshold, the number of TP, FN, FP, and TN change and with that the above defined metrics. When we compare Recall (True-Positive-Rate) and False-Positive-Rate for different thresholds and plot their values against each other, we get the **Receiver-Operator-Characteristics (ROC) Curve**. The ROC-Curve can help us to fnd a suited threshold for our task.

![ROC](/images/20230920_classification_metrics/roc.jpg "ROC.")

Using the ROC curve we can compare different classifiers. The higher the curve, the better a classifier. If the ROC curve would be the diagonal, this would equal the classifer to make random guesses. The **Area Under the Curve (AUC)** is used to meassure this value. It lies between 0 and 1. The closer it is to 1, the better the classifier. In the case of the diagonal, the AUC would equal 0.5.

![AUC](/images/20230920_classification_metrics/auc.jpg "AUC.")


Note: These metrics can be extended for multi-class classification. An overview can be found in this [blog](https://medium.com/r/?url=https%3A%2F%2Fwww.analyticsvidhya.com%2Fblog%2F2021%2F07%2Fmetrics-to-evaluate-your-classification-model-to-take-the-right-decisions%2F).

### Further Reading

* https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc
* https://www.analyticsvidhya.com/blog/2021/07/metrics-to-evaluate-your-classification-model-to-take-the-right-decisions/
