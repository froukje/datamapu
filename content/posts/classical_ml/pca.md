+++
title = 'Understanding Principal Component Analysis (PCA)'
date = 2024-07-28T19:04:00+02:00
draft = true
tags = ["Data Science", "Machine Learning", "unsupervised", "dimensionality reduction"]
categories = ["Data Science", "Machine Learning", "unsupervised"]
keywords = ["Data Science", "Machine Learning", "unsupervised", "dimensionality reduction"]
images = ['/images/pca/']
+++

## Introduction

Principal Component Analysis (PCA) is an unsupervised Machine Learning algorithm used for dimensionality reduction. In Data Science and Machine Learning, often large datasets with numerous features are often analyzed. PCA simplifies these complex datasets by retaining their essential information while reducing their dimensionality. It transforms a large set of correlated variables into a smaller set of uncorrelated variables known as *principal components*. These principal components capture the maximum variance in the data. They are ordered in decreasing order of explaining variance. This makes it easier to identify patterns, reduce noise, and enhance the efficiency of Machine Learning models.

## Why Use PCA?

When a dataset contains a large number of features, PCA can be a valuable tool for reducing its dimensionality. This **dimensionality reduction** simplifies the dataset while preserving most of the essential information. This approach can be beneficial for several reasons:

1. *Avoid overfitting*: High-dimensional data can lead to overfitting in Machine Learning models. Reducing the dimensionalty may help to avoid the [curse of dimensionality](https://en.wikipedia.org/wiki/Curse_of_dimensionality)

2. *Remove noise:* PCA identifies and eliminates features that contribute little to the overall variance in the data and therefore removes reduntant features.

3. *Visualization:* High-dimensional data can be difficult to visualize. PCA allows to reduce the dimensions to two or three, enabling clear visual representations that help in identifying patterns, clusters, or outliers.

4. *Generate new features:* PCA generates new features (the principal components) that are linear combinations of the original features. These components can be used as inputs for Machine Learning models.

5. *Data compression:* For large datasets PCA can be beneficial to compress data by reducing the number of dimensions. This lowers storage needs, computational power and time to process the data without significant loss of information.

6. *Handling multicollinearity:* When features are highly correlated, it can cause issues in regression models (e.g., multicollinearity). PCA transforms the correlated features into a set of uncorrelated principal components.


## How Does PCA Work? - The Algorithm

The following steps describe how to achieve the transformed dataset.

### 1. Standardize the Data

Standardization ensures that each variable contributes equally to the analysis. If the variables have large differences in their ranges, the variables with larger scales could dominate the principal components. Standardization makes sure that all variables have the same contribution.

To standardize the data, for each variable the mean is subtracted and divided is by the standard deviation, so that each feature has a mean of 0 and a standard deviation of 1. Mathematically this is formalated as

$$Z = \frac{X−\mu}{\sigma},$$

where $X = (x_1, \dots, x_n)$ is the original variable, $\mu$ is the mean, and $\sigma$ is the standard deviation.

### 2. Compute the Covariance Matrix

The covariance matrix captures how features vary together, which is essential for identifying the directions in which the data varies the most.

The covariance matrix is calculated as follows:

$$C = \frac{1}{n}Z^TZ,$$ 

where $C$ is the covariance matrix, $Z$ is the standardized data, and $n$ is the number of observations.

### 3. Compute and sort the Eigenvalues and Eigenvectors of the covariance matrix

The objective of PCA is to rotate the coordinate system of the feature space such that the new axes point in the direction of the maximum variance. To achieve that we use the [Eigenvectors]({{< ref "#appendix">}} "appendix") of the covariance matrix. 


### 4. Sort the Eigenvalues

The first Eigenvector, which is the one with the largest eigenvalue, points into the direction of the highest variance, the second in the direction of the second large variance and so on. That is, if we sort the Eigenvectors accoring to the magnitude in descending order we get the new feature space sorted by the magnitude of the Eigenvalues, we get the new feature space ordered by the magnitude of the variance. Each Eigenvector represents a feature in the new feature space and since the Eigenvectors are orthogonal, the new features are uncorrelated.


### 5. Select the Principal Components

If we use all the Eigenvectors from the covariance matrix we only rotate the coordinate system of the feature space. To reduce the dimensionality, we need to reduce the number of Eigenvalues and Eigenvectors. Since we sorted the Eigenvalues by their magnitude, we can choose the top $k$ Eigenvectors and get a dataset that respresents the maximum amount of the explained variance in the data. The exact number $k$ we choose depend on the amount of variance of the data to be present in the data.  The sum of the top $k$ eigenvalues divided by the sum of all eigenvalues gives the proportion of variance explained by the selected components.

### 6. Transform the Data

To represent the original data in the new lower-dimensional space, we multiply the standardized data by the matrix of the selected eigenvectors. This projects the original data onto the new feature space defined by the principal components. Mathematically, this can be formulated as

$$Z_{new} = Z \times W,$$ 

where $Z$ is the standardized data, and $W$ is the matrix of the selected eigenvectors.

## 2-D Example

calculations and plots

## PCA in Python

## Summary:

Given a dataset $X$:

1. **Standardize** the dataset to get $Z$.
2. **Compute the covariance matrix** $C$ of $Z$.
3. **Perform eigenvalue decomposition** on $C$ to get eigenvalues and eigenvectors.
4. **Sort the eigenvalues** and their corresponding eigenvectors in descending order.
5. **Select the top $k$ eigenvectors** to form a matrix $W$.
6. **Transform the data** to the new space: $Z_{new} Z \times W$.

Practical Implications:

* PCA simplifies the data by reducing the number of variables, making patterns and trends easier to detect.
* The principal components are orthogonal, ensuring that they are uncorrelated, which can improve the performance of machine learning algorithms.

< image- illustration >

The data is transformed into a new coordinate system to maximize the variance of the components. 

## Practical Implementation

Example Dataset: Introduce a simple dataset for demonstration.
Step-by-Step Implementation: Provide code snippets (in Python, R, etc.) showing how to perform PCA.
Interpretation: Show how to interpret the results, including the explained variance and principal components.
Use Cases and Examples
Real-World Examples: Provide examples of PCA applications in different domains (e.g., image compression, finance, biology).
Visualization: Show plots or visualizations that illustrate the impact of PCA.

## Appendix

*Eigenvector:* A non-zero vector that, when multiplied by a matrix, only changes in magnitude (not direction). Mathematically, for a matrix $A$ and eigenvector 
$𝑣$:

$$𝐴 𝑣= \lambda v,$

where $\lambda$ is the eigenvalue corresponding to that eigenvector $v$.

*Eigenvalue:* A scalar that indicates how much the eigenvector is stretched or compressed when multiplied by the matrix.

---
If this blog is useful for you, please consider supporting.

{{< bmc-button slug="pumaline" >}}

