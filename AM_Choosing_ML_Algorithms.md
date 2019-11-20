

```python
%matplotlib inline
import matplotlib
import seaborn as sns
sns.set()
matplotlib.rcParams['figure.dpi'] = 144
```

# Choosing the Correct Machine Learning Algorithm
<!-- requirement: images/ml_map.png -->

It's the interview question many aspiring data scientists dread: what algorithm should you choose and why? 

On the one hand, it's a bit of an unfair question. We rarely have all of the answers _before_ analyzing the data. If we did, we wouldn't need Machine Learning algorithms to begin with. In practice, you shouldn't be afraid to try different options and use cross validation to determine which is best. 

On the other hand, this is a great opportunity to show off what you know, and your ability to give a practical answer has the potential to make you stand out from other candidates. The key is to know the relative strengths and weaknesses of different approaches and to recommend those that best match the data being considered. This will also help you to explore options more efficiently in a practical setting. There are endless ways to modify and combine the techniques we've learned, so 'just try everything' isn't really a practical option.

The goal of this notebook is to explore which algorithms and techniques are most likely to be effective in a variety of circumstances. It can be thought of as a cheat sheet or shortcut towards answering the question posed above, but we challenge you to internalize the material and form your own opinions about how Machine Learning problems ought to be approached. You may find it helpful to try answering each question on your own (or discussing with a friend) before reading our answers. 

Not all answers are clean-cut; _edge cases are marked with an asterisk_.  

## Few Features

What algorithms and techniques should we favor (or avoid) when data has few features? (e.g. $1-3$ features)

**Favor**:

* **K-Nearest Neighbors** - This algorithm should be thought of as a form of interpolation, which works better when training data thoroughly covers the area where we want to make predictions. Data has less room to spread out in low dimensions.  
* **Support Vector Machine** - SVM with radial kernel can be thought of as a continuous version of K-Nearest Neighbors which only uses as a subset of the training data (its support vectors) for interpolation.
* **Visualization** - Human interpretable plots are limited to two or three dimensions, so it's much easier to visualize data with few features. 

**Avoid**:

* **Linear Regression*** - The flexibility of Linear Regression scales with the number of features, so there is a significant danger of underfitting when the number of features is small. However, this may still be an appropriate choice if we know that the behavior we want to model is simple, are able to do intelligent feature engineering, or _want_ to err on the side of underfitting. 

**Other ideas**:

* **Feature Engineering** - Artificial features can effectively increase model complexity as well as expose data relationships that your Machine Learning algorithm wouldn't have found on its own.
* **Get Better Data** - The features you have may simply lack sufficient information to model the relationship you are interested in. However, that does not mean that information does not exist. Part of being a data scientist is thinking about the big picture and making sure you have good data, even if that means pushing back or asking for help.  

<span style="color:blue">**#ML#**</span>

**KNN**: Easy to train (memorize data) but hard to predict.

**`ploynomial`**: Feature Engineering.

## Many Features

What algorithms and techniques should we favor (or avoid) when data has many features? (e.g. $1000$+ features)

**Favor**:

* **Linear Models** - The flexibility of linear models (`LinearRegression`, `Ridge`, `LogisticRegression`, etc.) scales with the number of features, so you should not underestimate these algorithms when many features are available (e.g. when working with NLP data). They may also offer computational advantages.
* **Dimension Reduction/Feature Selection/Regularization** - These are all reasonable methods to combat overfitting, which may occur if too many features are used. Mild regularization is almost always a good idea, but premature dimension reduction or feature selection may result in _underfitting_; you should check how your algorithm performs without them first (unless you need to reduce the number of features for computational reasons). 
* **Naive Bayes** - A large number of features can compensate for the simplicity of this algorithm. It is most commonly used with NLP data. 
* **Neural Networks*** - Neural networks can model very complicated behaviors, but this often comes at a high computational cost. Network architecture should be chosen for the type of data being analyzed. For example, convolutional neural networks should be used with image data.

**Avoid**:

* **Decision Trees** - Decision trees are a poor choice since each branch can use only one feature at a time. However, Random Forests are more resilient and can still be considered. 
* **K-Nearest Neighbors/Support Vector Machine** - See the previous section.


<span style="color:blue">**#ML#**</span>

**Feature reduction**: Sometimes can perform better not just speed up.

## Few Observations

What algorithms and techniques should we favor when data has few observations? (e.g. $~30$ observations)

* **Simple Models** - K-Nearest Neighbors, Support Vector Machine, and Linear Regression are all reasonable attempts as long as you don't have too many features. With few observations it's very easy to overfit.
* **Get More Data** - It's difficult to have confidence in models that are trained on small data sets. Even data for a related problem can be better than nothing. (e.g. transfer learning approaches for image recognition) 
* **Data Set Augmentation** - Another option is to augment data using synthetic observations. Common strategies are to average pairs of observations (e.g. in [SMOTE](https://jair.org/index.php/jair/article/view/10302)) or to apply transformations to individual data points (e.g. adding noise, cropping, or rotating image data).

## Many Observations

Having many observations (e.g. $100,000$+) means that we have more information and can potentially build better models, but it can also be a computational burden. What algorithms and techniques should we favor if we want to emphasize prediction quality (or keep computations manageable)?

**Best Predictions**:

* **Neural Networks** - Neural Networks are the best option when the behavior you are trying to model is truly complicated. They may require significant setup and computational investment, but they are able to take advantage of large data sets (as opposed to something like linear regression that will quickly converge, or learn its best coefficient values, and then cease to improve in any meaningful way).
* **Random Forest** - This is one of the easiest algorithms to use, and its performance increases when more data is available. It also has the advantage of being highly parallelizable.
* **Gradient Boosted Trees** - Gradient boosting was invented to combat underfitting. Overfitting is less of a concern with larger data sets, so this is a natural option to consider, although it's more difficult computationally since its steps can't be parallelized.

**Keeping Computation Manageable**: 

* **Subsampling** - When dealing with overwhelming quantities of data, many algorithms will perform nearly as well when trained on a smaller random subset. For ensemble approaches like Random Forest, we can give each component algorithm a different subset of the data.
* **Dimension Reduction/Feature Selection** - Reducing the number of features used also reduces computation, but exercise caution. This often reduces prediction quality more dramatically than subsampling.
* **Stochastic Gradient Descent/Minibatch Training** - When we have too many observations, it becomes impractical to use the entire data set for each step of gradient descent, but we can get a good approximation of the gradient using a relatively small number of observations. (In Scikit-learn, minibatch training is implemented by the `.partial_fit` method of [`SGDRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDRegressor.html) and [`SGDClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html).)

**Avoid:**
* **Support Vector Machines** - The SVM algorithm scales inefficiently; its time complexity is [between $O(pn^2)$ and $O(pn^3)$](http://scikit-learn.org/stable/modules/svm.html#complexity) for data sets with $p$ features and $n$ observations. It may be necessary to use [kernel approximation](https://scikit-learn.org/stable/modules/kernel_approximation.html) when attempting to train SVMs on large data sets.  

## Underfitting

What are some causes of underfitting? What can be done in each case?

* **Non-linear Relationships or Feature Interaction** - These sorts of data relationships can be missed by simpler models like Linear Regression. They can sometimes be identified using exploratory visualization (e.g. feature vs target plots) and then incorporated into your model using feature engineering. To a limited extent they are handled automatically by tree-based methods.
* **Complicated Underlying Behavior** - For harder problems, we need to use fancier models. See 'Best Predictions' in the previous section. Feature engineering can also help when we have some insight about what the model ought to be looking for.  In some cases, we can try combining the predictions of many (usually simple) models and passing them as features to a final regressor or classifier (using the `FeatureUnion` method from Scikit-learn). Using a linear model is equivalent to taking a weighted average of the contributors, whereas a more complex final estimator is capable of combining the component models nonlinearly. 
* **Bad Encodings** - Data needs to be represented in a format that your algorithm can exploit. For example, if you're using Linear Regression, it would be a bad idea to represent ice cream flavors using the numbers $1-10$. There's no order relation between flavors and we'd like to be able to treat each as a separate case, so one-hot encoding is likely a better option. This is a very general concept; you should always think about what representations are appropriate when working with new data types. 
* **Missing Information** - We may simply lack the information we need to make good predictions. Additional features are sometimes the answer, but you need to be realistic about what you can accomplish. If your team is expecting you to predict what the stock market will do tomorrow, then you probably have some expectation setting to do.    

## Overfitting

What algorithms and techniques should we favor when we're worried about overfitting?

* **Simple Models** - Models with less flexibility are less likely to overfit. Favor options like linear/logistic regression over neural networks or boosted trees. Be conservative with hyperparameters like the degree of a polynomial model or the maximum depth of a decision tree.
* **Regularization** - This penalty for large coefficients is designed to combat overfitting. Try playing around with `alpha` when using [`Ridge`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html) and [`Lasso`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html) or `C` when using [`LogisticRegression`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html).
* **Ensemble Methods** - Ensemble methods do a great job of avoiding overfitting. Random Forest is the most common choice, but there is also something called an Extremely Random Forest ([`ExtraTreesRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesRegressor.html) and [`ExtraTreesClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html)), and Scikit-learn has the general [`BaggingClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingClassifier.html). 
* **Feature Selection/Dimension Reduction** - Reducing the number of features used can help combat overfitting, but exercise caution. Since we're essentially throwing away information, this approach sometimes hurts more than it helps.
* **Data Augmentation** - It's harder to overfit when we have more data, so one strategy is to add artificial observations at the cost of lower data quality. We can add noise to existing observations or randomly drop feature values. This prevents the model from becoming too sensitive (but may increase underfitting).

## Explicability

What algorithms and techniques should we favor (or avoid) when explicability is our top priority?

**Favor**:

* **Linear Models** - Linear/Logistic regression models are easy to understand because their decision functions are simple and explicit. We can also interpret their coefficients as a measure of feature importance (provided we have done appropriate feature scaling beforehand).
* **Decision Trees** - Decision trees follow simple and explicit logic and are easy to represent visually.
* **K-Nearest Neighbors** - We can explain each prediction by providing the nearest neighbors of the point in question. We can also plot the decision function in low dimensions.
* **Visualization** - Pictures are easier to understand than equations. Plots are indispensable both for exploring data and for presenting data insights eloquently.
* **Feature Selection** - Limiting our attention to a few key variables often makes a business case more compelling; models with fewer features are easier to understand. 
* **Naive Bayes*** - (Multinomial and Bernoulli) Naive Bayes methods are relying on the assumption of conditional independence of underlying features given the class variable, and therefore model fitting reduces to counting to predict these conditional probabilities.  Of course, this explanation relies on the target audience having a good understanding of Bayes' Theorem.  Also, while Naive Bayes methods are generally decent classifiers, they are typically poor estimators, so the probability outputs from these methods should not be taken seriously.  
* **Support Vector Machine*** - SVM is easy to understand when the decision function can be plotted, but may be a poor choice for problems with more than two or three features.
 
**Avoid**:

* **Neural Networks** - This is the quintessential black box model. Meaning can be ascribed to neural network outputs (and their gradients), but most hidden activation values are inscrutable.
* **Boosted Trees/Random Forest*** - Although the component trees may be easy to understand individually, there's no good way to summarize the entire decision process in a way that provides human intuition. However, it is possible to give a numerical measure of feature importance based on which features control the splits that sort the most data.     
* **Dimension Reduction*** - The new features created by dimension reduction are rarely easy to interpret. However, dimension reduction is often necessary to facilitate data visualization, which can help us to find patterns and groupings in data. Dimension reduction (specifically [non-negative matrix factorization](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.NMF.html)) is also useful in topic modeling. 

## Prediction Speed

What algorithms and techniques should we favor (or avoid) when we need to be able to make predictions quickly?

**Favor**:

* **Linear Models** - The decision function is simple and quick to evaluate.
* **Decision Tree** - We only need to perform one comparison for each level in the decision tree.
* **Support Vector Machines*** - Very fast for linear kernel, otherwise scales with the number of support vectors.
* **K-Nearest Neighbors*** - Some work is done ahead of time (during the `.fit` method in Scikit-learn), but finding the nearest neighbors can be slow for high-dimensional data.

**Avoid**:

* **Neural Networks** - These should not be your first choice, but they are not categorically slow. Prediction time scales with the number of connections.
* **Gradient Boosted Trees** - Prediction time scales with the number of estimators. Component predictions must be made in sequence, so we can't take advantage of parallel processing to save time (which might be possible for Random Forest).


## Parallelization

What algorithms parallelize well?

* **Ensemble Methods** - For example, Random Forest. Each component tree can be trained independently. 
* **Naive Bayes** - Uses conditional probabilities based on counts. Counting is trivially parallelizable.
* **Neural Networks*** - Training steps must be performed in sequence, but individual steps rely heavily on matrix multiplication, which can be parallelized. This is why GPUs are able to dramatically improve the speed of neural network training.

## Online Learning

What algorithms and techniques support online learning (can learn incrementally from new data without needing to be refit on the entire training set)?

* **Stochastic Gradient Descent/Minibatch Training** - Each step of gradient descent incrementally changes parameters instead of restarting the training process. In Scikit-learn, this is the difference between `.partial_fit` and `.fit`.
    * **Neural Networks** are most commonly trained using minibatch gradient descent.
    * **Linear Models** including Logistic Regression and linear SVMs have SGD/minibatch variants implemented by [`SGDRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDRegressor.html) and [`SGDClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html) in Scikit-learn.
    
* **Random Forest** - Additional trees can be fit to new data without refitting trees trained in earlier steps. We can discard each batch of data after using it to train a new tree, so we don't need to store the entire data set (although we may want to discard the oldest trees once the forest grows too large). Note that the `RandomForest` estimators in Scikit-learn do _not_ have `.partial_fit` methods; this online version of Random Forest would require a custom implementation.  

## Feature Scaling

What algorithms and techniques require feature scaling? Which don't?

**Require Feature Scaling**: 

* **K-Nearest Neighbors/Support Vector Machine** - If feature scaling is not used with distance-based algorithms, they will be biased to emphasize the features that vary the most.
* **Feature Importance** - Models like Linear/Logistic regression don't technically require feature scaling, but we need to scale so that features are treated fairly if we want to use coefficient size as a measure of feature importance.
* **Regularization** - Similarly, we need to apply feature scaling to ensure that features are treated fairly if we want to use regularization.
* **Stochastic Gradient Descent/Minibatch Gradient Descent** - Gradient descent is sometimes less efficient when features have different scales. This is a more serious issue for the SGD/minibatch variants, which are inherently noisier.
* **Principal Component Analysis*** - It's usually correct to scale prior to dimension reduction, but this can destroy useful information when features have the same units to begin with.

**Don't Require Feature Scaling**:

* **Tree-Based Algorithms** - Decision Trees, Random Forest, etc. search for split points in a manner that is scale independent.

## Outlier Detection/Novelty Detection

What algorithms can be used for outlier detection (finding observations in a training set that are "far" from the majority of other observations) and novelty detection (classifying **new** observations as "inliers" or "outliers" based on the original training data)?  

Scikit-learn implements these methods as unsupervised learning algorithms.  Training data is fit using the `.fit` method, and `.predict` is used to label data as inliers or outliers.  Two examples of these methods implemented in Scikit-learn include:

* **Isolation Forests** -  An efficient way of performing outlier detection in high-dimensional data sets using random forests, by randomly selecting a feature and then randomly selecting a split value between the maximum and minimum values of the selected feature.  The idea is that the path length for an outlier (the number of splittings required to isolate a sample, averaged over a forest of random trees) is shorter for outliers/anomalies in the data set.  
* **One-class Support Vector Machine** - Typically used for novelty detection where we are given a "clean" training data set.  A one-class SVM can learn a boundary that includes all/most of the training data and can then be used to detect novel observations.  

## Comparing ML Algorithms

It's important to be able to clearly understand and explain the theoretical and practical differences between machine learning algorithms. We hope this notebook helps to provide some clarity with respect to factors to consider when choosing a machine learning algorithm.  Don't limit yourself solely to the topics we have discussed here, and consider solutions taking into account the type of question you're attempting to answer and the particular data that you have to work with.  

This flowchart ([original interactive source](http://scikit-learn.org/stable/tutorial/machine_learning_map/)) can also be a good starting point when deciding which machine learning algorithm to use.
![Machine learning flowchart from the Scikit-Learn documentation](images/ml_map.png)

*Copyright &copy; 2019 The Data Incubator.  All rights reserved.*
