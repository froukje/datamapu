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

Principal Component Analysis (PCA) is an unsupervised Machine Learning algorithm used for dimensionality reduction. In Data Science and Machine Learning, large datasets with numerous features are often analyzed. PCA simplifies these complex datasets by retaining their essential information while reducing their dimensionality. It transforms a large set of correlated variables into a smaller set of uncorrelated variables known as *principal components*. These principal components capture the maximum variance in the data, making it easier to identify patterns, reduce noise, and enhance the efficiency of Machine Learning models.

## Why Use PCA?

When a dataset contains a large number of features, PCA can be a valuable tool for reducing its dimensionality. This **dimensionality reduction** simplifies the dataset while preserving most of the essential information. This approach can be beneficial for several reasons:

1. *Avoid overfitting*: High-dimensional data can lead to overfitting in Machine Learning models. Reducing the dimensionalty may help to avoid the [curse of dimensionality](https://en.wikipedia.org/wiki/Curse_of_dimensionality)

2. *Remove noise:* PCA identifies and eliminates features that contribute little to the overall variance in the data and therefore removes reduntant features.

3. *Visualization:* High-dimensional data can be difficult to visualize. PCA allows to reduce the dimensions to two or three, enabling clear visual representations that help in identifying patterns, clusters, or outliers.

4. *Generate new features:* PCA generates new features (the principal components) that are linear combinations of the original features. These components can be used as inputs for Machine Learning models.

5. *Data compression:* For large datasets PCA can be beneficial to compress data by reducing the number of dimensions. This lowers storage needs, computational power and time to process the data without significant loss of information.

6. *Handling multicollinearity:* When features are highly correlated, it can cause issues in regression models (e.g., multicollinearity). PCA transforms the correlated features into a set of uncorrelated principal components.


## How Does PCA Work? - The Algorithm


Principal Component Analysis (PCA) is a linear algebra-based technique that transforms a dataset into a set of linearly uncorrelated variables called principal components. Here's a step-by-step explanation of how the PCA algorithm works:

### 1. Standardize the Data

Why?: Standardization ensures that each feature contributes equally to the analysis. Features with larger scales could dominate the principal components if not standardized.

How?: Subtract the mean and divide by the standard deviation for each feature, so that each feature has a mean of 0 and a standard deviation of 1.

Formula: $Z = \frac{X−\mu}{\sigma}$, where $X$ is the original data, $\mu$ is the mean, and $\sigma$ is the standard deviation.

### 2. Compute the Covariance Matrix

Why?: The covariance matrix captures how features vary together, which is essential for identifying the directions in which the data varies the most.

How?: Calculate the covariance matrix of the standardized data.

Formula: $C = \frac{1}{n}Z^TZ$, where $C$ is the covariance matrix, $Z$ is the standardized data, and $n$ is the number of observations.

### 3. Compute the Eigenvalues and Eigenvectors

Why?: Eigenvalues indicate the magnitude of variance in the data along the direction of their corresponding eigenvectors. Eigenvectors determine the directions (principal components) in which the data varies.

How?: Perform eigenvalue decomposition on the covariance matrix to obtain the eigenvalues and eigenvectors.

Interpretation:
* Eigenvalues: Represent the amount of variance captured by each principal component.
* Eigenvectors: Represent the directions of the principal components.

### 4. Sort Eigenvalues and Eigenvectors

Why?: To prioritize the principal components that capture the most variance in the data.

How?: Sort the eigenvalues in descending order, and rearrange the corresponding eigenvectors accordingly. The eigenvector with the highest eigenvalue is the first principal component, and so on.

### 5. Select the Principal Components

Why?: To reduce dimensionality while retaining most of the variance in the data.

How?: Choose the top $k$ eigenvectors corresponding to the largest eigenvalues. These $k$ eigenvectors form the new feature space.
Variance Explained: The sum of the top $k$ eigenvalues divided by the sum of all eigenvalues gives the proportion of variance explained by the selected components.

### 6. Transform the Data

Why?: To represent the original data in the new lower-dimensional space.

How?: Multiply the standardized data by the matrix of the selected eigenvectors. This projects the original data onto the new feature space defined by the principal components.

Formula: $Z_{new} = Z \times W$, where $Z$ is the standardized data, and $W$ is the matrix of the selected eigenvectors.

### 7. Analyze the Results

Principal Components: The transformed data now consists of the principal components, which are uncorrelated and ordered by the amount of variance they explain.

Explained Variance: Examine the proportion of the total variance explained by each principal component to assess how much information is retained.

Visualization: The reduced dimensionality allows for easier visualization and analysis, often in 2D or 3D space.

## Summary:

Given a dataset $X$:

1. **Standardize** the dataset to get $Z$.
2. **Compute the covariance matrix** $C$ of $Z$.
3. **Perform eigenvalue decomposition** on $C$ to get eigenvalues and eigenvectors.
4. **Sort the eigenvalues** and their corresponding eigenvectors in descending order.
5. **Select the top $k$ eigenvectors** to form a matrix $W$.
6. **Transform the data** to the new space: $Z_{new} Z \times W$.
7. **Analyze** the principal components and their explained variance.

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

---
If this blog is useful for you, please consider supporting.

{{< bmc-button slug="pumaline" >}}

