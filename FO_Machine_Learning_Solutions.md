

```python
%matplotlib inline
import matplotlib
import seaborn as sns
sns.set()
matplotlib.rcParams['figure.dpi'] = 144
```

# Machine Learning Interview Question Solutions

## ML Fundamentals

#### 1. What’s the trade-off between bias and variance?

Bias is a measure of under-fitting, how poorly the model captures the underlying patterns in the data (i.e. it does bad on any sample set).  Variance is a measure of over-fitting, how much the model’s predictions are different for different sub-samples of the data and measures how much the model has fit the noise in the data


#### 2. What is the difference between supervised and unsupervised machine learning?

Supervised learning is telling the model to look for a specific pattern - we are providing it a target (also called a label) and asking it to figure out how to map from data to label.  Unsupervised learning is looking for a more general pattern.  We give the system data without any correct answers, so we can’t expect a “correct” answer to come out
    

#### 3. When should you use classification over regression?

Regression is for continuous values, specifically ones where order matters - saying 2 when you mean 4 is worse than saying 2 when you mean 3.  Classification is for discrete values when order doesn’t matter.  Saying 2 when you mean 3 is just as bad as saying 2 when you mean 4 (think animal vs vegetable vs mineral)


#### 4. What are hyperparameters, and how do they differ from parameters?

Hyperparameters are not parameters of our model, but parameters that select between models in a family.  The parameters are the internal state of the model that the fitting process adjusts to do the learning, while the hyperparameters are things that we fix ahead of time to control the system.  The classic example is a decision tree - our hyperparameter would be the depth of the tree or the leaf size, while the system fits to get the decision values (the parameters) of each split. 

#### 5. What’s the difference between Type I and Type II error?

Type I: False Positive, Type II: False Negative.  False positives are when we mark something when we shouldn’t, and vice-versa for False Negative.  For example, let’s say we’re looking at email and deciding if it’s spam.  A false negative would be letting a spam through to regular mail, and a false positive would be marking a normal email as spam.  Very, very different consequences - the false negative here is a mild annoyance and a false positive is an email from your boss that you never get.  This is a high precision situation - we want to be very sure the things we catch are indeed spam.  On the other end, consider detecting cancer.  Here a false positive (seeing cancer when there is none) means someone gets extra tests, and a false negative (missing cancer that is present) is potentially life threatening.  This is a high recall situation - we want to be sure we catch all the actual cases of cancer.

#### 6. Explain over- and under-fitting and how to combat them?

See question 1 for the first part.  You combat them by changing your model - either actually switching models, or by altering the hyperparameters of your model.  If you are under-fitting, you need a more powerful model (a bigger decision tree, a smaller alpha in regularization, a bigger neural network).  If you’re overfitting, you need either a weaker model or a specific overfitting-fighting technique such as regularization, bagging, dropout, etc

#### 7. What is regularization, and why do we use it?  Give some examples of common methods?

Regularization is a method for controlling overfitting.  We use it most often with linear models, but it can be applied to any model with continuous-valued parameters (such as a neural net).  With it, we penalize the system’s cost function (error term) for the size of the parameters - i.e. we now have a penalty both for getting the wrong answer and for making large parameters, creating a tension between the two that steers the system away from doing “too good” of a fit on the data, which usually means fitting noise.  The most common examples are L1 and L2 regularization (see the next question)

#### 8. Explain the difference between L1 and L2 regularization.
In L2 regularization we are looking at penalizing the cost function with the square of the size of the parameters (𝛽^2), while L1 penalizes on the absolute value of the parameters (|𝜷|).  These have different properties - L2 very heavily penalizes “outliers” (particularly large values), but can’t send anything to zero due to the properties of the derivative.  L1 penalizes linearly with size of parameter, but because the derivative remains non-zero all the way down to zero, it can set some of the parameters exactly to zero (thus eliminating them from the fit)

#### 9. What is the concern with an imbalanced data set? How would you deal with this?

The problem with an imbalanced set is that the normal classification metrics won’t work.  As an example, if your sample is 99% one category and 1% another, your model always guessing the first will be 99% accurate.  We have a few possible ways to fight this.  One way is to re-weight out metric to balance the classes, or to choose an F score that emphasizes false positives or false negatives as is appropriate.  But this isn’t usually the route that is gone down.  Instead, usually we either downsample the larger set (randomly throwing away things from that set until we have about as many in the bit one as the small one), or we upsample the small set.  This can be done by oversampling the smaller set explicitly, so we get multiple copies of the data points in it - this can be very prone to overfitting, but often works.  Another option is to upsample via synthetic data creation such as SMOTE (there are many options), where we use the existing data in the smaller set to create new data points that look like the existing ones.

#### 10. What evaluation approaches would you work to gauge the effectiveness of a machine learning model?

This can be a tricky question, as it depends highly on what our use case is.  But at a fundamental level, we want to test our model on a set of data it has never seen before.  Given that, we’ll want to compare our model’s performance in a technical sense to our business objective.  Ideally we’re comparing this to an existing model, or an existing method, and we use both methods and see which one performs better (we’ll need to think about complexity, cost, etc as a separate issue, hopefully later).  Often this can be done pretty easily by comparing metrics directly (MSE, accuracy, etc).  If this model is the first model addressing the problem, then we need to figure out how good “good enough” is.  This is a business decision, ultimately - we may decide that a 30% accurate model of potential customers is fine, since sending a flier is cheap.  But ultimately, we’re mapping technical metrics to business metrics and measuring them on a set of data we set aside from the beginning to evaluate on.

#### 11. Explain the benefit and drawback of making an ML model compared to an expert-created model to a non-technical person.

The biggest benefit, of course, is that you don’t need a topic expert for each topic, just a lot of relevant data.  As long as we have a enough data, we can let our learning system learn from it regardless.  But that’s our biggest weakness as well: we need lots of data.  A somewhat mixed thing is what happens when the world changes - our expert will need to redo their model, which may or may not be accurate depending on what assumptions are made, but we will need to get new data. With a ML model, we can often find patterns that are present that the expert didn’t think of or didn’t know about, but the expert may have external knowledge that can help.  Ideally, of course, we’ll consult an expert about things to look for when we make our model.


## Handling Data
#### 1. What is data normalization and why do we need it?

Data normalization is the process of making our data's features have a more consistent range of values - for each feature, we subtract out the mean and divide by the standard deviation (or (max - min), depending on circumstance).  This gets all of our data to be on the same scale, so that if one feature was originally numbers between 1000 and 4000, and another was numbers between 0.01 and 0.02, these wildly different sizes don’t confuse our learner, as they’re all now roughly between -1 and 1.  This is very helpful for many learners, as it keeps the scale from dominating the calculations and keeps the learner from finding a rather large number of local minima. 

#### 2. How do you handle missing or corrupted data in a data set?

This answer depends heavily on the size of our data set, how much is bad, and what we’re doing.  If it’s a very small percentage of bad data, it’s usually safe to just filter them out, unless they would bias our sample.  If not, we need to fill in our missing values.  First, we need to decide what is actually bad data.  Many things can be done here, and it’s beyond this question.  Once that’s done, we’ll need to fill in the missing values with some sort of default.  I really should be saying “field” here - we normally don’t fill in whole missing data points, just features that are missing from particular data values.  One method is to do exactly that - fill in with some sort of average value or user-chosen reasonable default.  Another method is “imputation” - we use more clever methods to look at the data we do have and get a best guess as to what the missing value should be.  At their simplest, you can think of these as either clustering data to get an average or building an interpolator or learner linking existing fields with missing fields.

#### 3. Explain dimensionality reduction, where it’s used, and its benefits?

Dimensionality reduction is a way to reduce the number of features as the name suggests.  Most methods are meant to do this in a smart way, so that we keep most of the information present in the data and throw away features that aren’t going to be helpful, but not all of them do.  This is done as part of the final steps of altering our data to feed in to our learner.  We use this when we have too many features - it’s not uncommon when you’re working with text to have thousands of features, or to get similar numbers from some data sets.  This can be too much for many learners to handle, and is the leading cause of overfitting in linear fitters. Also, by selecting out high-value combinations of features, we often improve the fit our learner comes up with, since we are doing some of its work up-front for it.

#### 4. Explain Principal Component Analysis (PCA)?

PCA is a form of dimensional reduction.  It’s a linear algebra technique related to eigenvectors and Singular value decomposition that finds combinations of features that have the most variance and are (relatively, linearly) uncorrelated.  Generally only the ones accounting for the most variance are kept (which is a sliding scale that the user gets to pick - how many to keep), but sometimes all the new, transformed features are used.  This is also considered an unsupervised learning technique, as it doesn’t look at the labels for data (if they exist) and instead is looking for a pattern directly in the data.  


## More Specifics

#### 1. What are bagging and boosting?  What are they used for?

Bagging is an way of making ensemble models.  A collection of (usually identical) models are trained on sub-samples of the data and their estimates combined.  The key step is in how the sub-samples are made - they are random draws from the original data with replacement, so that the same point can appear multiple times.  This is a powerful technique to fight overfitting.  Boosting is in some sense the opposite - it’s a technique for fighting underfitting, where a series of learners are trained in series.  Each one takes the output of the previous one and is trained on data re-weighted to focus on the data the previous one(s) got wrong.  The predictions are then combined in a weighted average at the end.

#### 2. What is the interpretation of regularization from a Bayesian point of view?

Regularization is imposing a prior on our system, particularly that our parameters are in some distribution with mean zero (i.e. we expect parameters to be small).  The hyperparameters control the width the distribution.  For L2, this distribution is Gaussian.  For L1, this is an exponential-like distribution

#### 3. What are the strengths and weaknesses of linear regression vs decision trees vs neural networks?

The biggest trades you usually make with algorithms are between power (how complex of a model it can handle), interpretability (how well can we understand why it’s making the decisions it makes), and computation needed.  In those terms, linear regression has very low power, high interpretability, and cheap computation.  Decision trees have medium power, medium to high interpretability (less so for random forests), and moderate computation needs.  Neural nets have extremely high power, no interpretability, and very high computation cost.  In addition, decision trees have an added benefit of being non-parametric, so they do not assume a model shape, and neural networks have the benefits of being able to do their own feature engineering if given enough computing power and being mathematically proven to be able to model any arbitrary function.

#### 4. How do you tell what features are important in a linear regression?  In a random forest?

In a linear regression, we can retrieve the fit parameters.  Since they are just coefficients in a linear model, their relative size tells us the relative importance of the features, but only if all the features were of the same “size” (i.e. we scaled and centered them before doing the model) - if we don’t do that, then the scale of the data is mixed in with the importance.  For random forests (or regular decision trees), there are algorithms that look at how much the Gini coefficients or the variance change at each decision, and do a weighted sum for the decisions involving each feature.  This is implemented in sklearn as “.feature_importance”, so it’s just a single function call.

#### 5. What is a support vector in a SVM?

These are data points that are either on the edge of the margin or in the margin, depending on your slack variables.  They actually define the margin.  In a fully separable data set, these will be on the edges of the data set.  Either way, these are the vectors that control the fit.

#### 6. What is deep learning, and how does it contrast with other machine learning algorithms?

Deep learning, simply put, is learning with a lot of parameters.  This is generally taken to mean large neural networks, which can easily have thousands or even millions of parameters.  It offers several advantages.  First, they have enormous flexibility and power and can find nearly arbitrarily subtle patterns in data, in ways we humans can’t discern.  They can keep learning from as the amount of data increases, nearly without bound, while “normal” learners eventually saturate and stop improving with more data.  With Neural Nets specifically, their architecture can be altered to suit the problem, and they can perform “feature learning” where they automatically learn the best way to manipulate the input features so we don’t have to do feature engineering.  They come with a few major drawbacks, however.  They require large amounts of data and lots and lots of computation power, as well as special techniques to handle them.

#### 7. What is online learning and when do you use it?

Online learning is updating a fit with new data, rather than re-fitting the whole model.  It has two main uses: when you are getting in new data and want to incrementally adjust your model to accommodate this (say when you’re predicting user behavior, and taking what people actually do as adjustments to your model), and when your data is too large to train on all at once, where this is effectively a form of mini-batching.  Then only a portion of your data needs to be in memory or operated on at any given time.

#### 8. What cross-validation technique would you use on a time series data set?

You can’t use the traditional test-train split or cross-fold validation on a time series, since order matters in your data.  So a random selection of points will wind up training on future data and testing on past data at some point, which isn’t acceptable (we’re predicting, not interpolating).  Also, these splits traditionally shuffle the data as well, as there is a built-in assumption that data is independently identically distributed (i.i.d.), i.e. that our data points are random draws from the same distribution.  Instead, we will need to preserve order, and only train on the past when we’re testing data.  This is usually done with one of two methods: “sliding window” or “forward chaining”.  First, we break our data in to some number of chunks by simply slicing in time (so we maintain order in and between chunks).  In sliding window, we train on a chunk, then test on the next.  Then we train on the next, and test on the one after that (so if we have 4 chunks, we do train 1 - test 2, train 2 - test 3, train 3 - test 4).  In forward chaining, we train on progressively larger sets of data and test on the next one (so with the 4 chunks, it would be train 1 - test 2, train 1+2 - test 3, train 1+2+3 - test 4).

#### 9. What’s your favorite algorithm, and can you explain it to me in less than a minute?

Let them talk.  Hopefully they’re coherent.


*Copyright &copy; 2018 The Data Incubator.  All rights reserved.*
