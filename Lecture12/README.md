STAT406 - Lecture 12 notes
================
Matias Salibian-Barrera
2017-10-15

#### LICENSE

These notes are released under the "Creative Commons Attribution-ShareAlike 4.0 International" license. See the **human-readable version** [here](https://creativecommons.org/licenses/by-sa/4.0/) and the **real thing** [here](https://creativecommons.org/licenses/by-sa/4.0/legalcode).

Lecture slides
--------------

The lecture slides are [here](STAT406-17-lecture-12-preliminary.pdf).

Bagging
-------

One strategy to obtain more stable predictors is called **Bootstrap AGGregatING** (bagging). It can be applied to many predictors (not only trees), and it generally results in larger improvements in prediction quality when it is used with predictors that are flexible (low bias), but highly variable.

The justification and motivation were discussed in class. Intuitively we are averaging the predictions obtained from an estimate of the "average prediction" we would have computed had we had access to several (many?) independent training sets (samples).

There are several (many?) `R` packages implementing bagging for different predictors, with varying degrees of flexibility (the implementations) and user-friendliness. However, for pedagogical and illustrative purposes, in these notes I will *bagg* by hand.

### Bagging by hand

Again, to simplify the discussion and presentation, in order to evaluate prediction quality I will split the data (`Boston`) into a training and a test set. We do this now:

``` r
library(rpart)
data(Boston, package='MASS')
set.seed(123456)
n <- nrow(Boston)
ii <- sample(n, floor(n/4))
dat.te <- Boston[ ii, ]
dat.tr <- Boston[ -ii, ]
```

I will now train *N* = 5 trees and average their predictions. Note that, in order to illustrate the process more clearly, I will compute and store the *N* × *n*<sub>*e*</sub> predictions, where *n*<sub>*e*</sub> denotes the number of observations in the test set. This is not the best (most efficient) way of implementing *bagging*, but the main purpose here is to understand **what** we are doing. Also note that an alternative (better in terms of reusability of the ensamble, but maybe still not the most efficient option) would be to store the *N* trees directly. This would also allow for more elegant and easy to read code. Once again, this approach will be sacrificed in the altar of clarity of presentation and pedagogy (but do try it yourself!)

First create an array where we will store all the predictions:

``` r
N <- 5
myps <- array(NA, dim=c(nrow(dat.te), N))
con <- rpart.control(minsplit=3, cp=1e-3, xval=1)
```

The last object (`con`) contains my options to train large (potentially overfitting) trees.

``` r
n.tr <- nrow(dat.tr)
set.seed(123456)
for(j in 1:N) {
  ii <- sample(n.tr, replace=TRUE)
  tmp <- rpart(medv ~ ., data=dat.tr[ii, ], method='anova', control=con)
  myps[,j] <- predict(tmp, newdata=dat.te, type='vector')
}
pr.bagg <- rowMeans(myps)
with(dat.te, mean( (medv - pr.bagg)^2 ) )
```

    ## [1] 14.81517

And compare with predictions from the pruned tree, and the ones from other predictors discussed in the previous note:

``` r
myc <- rpart.control(minsplit=3, cp=1e-8, xval=10)
set.seed(123)
bos.to <- rpart(medv ~ ., data=dat.tr, method='anova',
                control=myc)
b <- bos.to$cptable[which.min(bos.to$cptable[,"xerror"]),"CP"]
bos.t3 <- prune(bos.to, cp=b)
pr.t3 <- predict(bos.t3, newdata=dat.te, type='vector')
with(dat.te, mean((medv - pr.t3)^2) )
```

    ## [1] 18.96988

What if we *bagg* *N* = 10 trees?

    ## [1] 14.80446

or *N* = 100 trees?

    ## [1] 12.71124

or *N* = 1000 trees?

    ## [1] 12.566

Should we consider higher values of *N*? How about other training / test splits? Should we use CV instead?

Another split:

    ## [1]  5.00000 20.32676
    ## [1] 10.00000 20.52832
    ## [1] 100.00000  18.24876
    ## [1] 1000.00000   17.93342

Similar conclusion: increasing *N* helps, but the improvement becomes smaller, while the computational cost keeps increasing.

### Bagging a regression spline

Bagging does not provide much of an advantage when applied to linear predictors (can you explain why?) Nevertheless, let us try it on the `lidar` data, which, as we did before, we randomly split into a training and test set:

``` r
data(lidar, package='SemiPar')
set.seed(123456)
n <- nrow(lidar)
ii <- sample(n, floor(n/5))
lid.te <- lidar[ ii, ]
lid.tr <- lidar[ -ii, ]
```

Now fit a cubic spline, and estimate the MSPE using the test set:

``` r
library(splines)
a <- lm(logratio ~ bs(x=range, df=10, degree=3), data=lid.tr) 
oo <- order(lid.tr$range)
pr.of <- predict(a, newdata=lid.te)
```

    ## Warning in bs(x = range, degree = 3L, knots = structure(c(430, 475, 513, :
    ## some 'x' values beyond boundary knots may cause ill-conditioned bases

``` r
mean( (lid.te$logratio - pr.of)^2 )
```

    ## [1] 0.008443088

We build an ensemble of 10 fits and estimate the corresponding MSPE using the test set:

``` r
N <- 10 # 5 500 1500
myps <- matrix(NA, nrow(lid.te), N)
set.seed(123456)
n.tr <- nrow(lid.tr)
for(i in 1:N) {
  ii <- sample(n.tr, replace=TRUE)
  a.b <- lm(logratio ~ bs(x=range, df=10, degree=3), data=lid.tr[ii,]) 
  myps[,i] <- predict(a.b, newdata=lid.te)
}
pr.ba <- rowMeans(myps)# , na.rm=TRUE)
mean( (lid.te$logratio - pr.ba)^2 )
```

    ## [1] 0.008040338

Classification
==============

As we discussed in class, what is commonly referred to as *classification* can be thought of as prediction, when the responses are classes and we use a particular loss function (the *0-1 loss* we discussed in class). Furthermore, it is easy to show (which we did in class) that the optimal classifier (in terms of minimizing the expected misclassification error) is the one that assigns an observation to the class with the highest probability of occuring, conditional to the value of the observed explanatory variables.

Most (if not all) classification methods we will cover in this course can be simply thought of as different approaches to estimate the conditional probability of each class, conditional on the value of the explanatory variables. In symbols: *P*(**G** = **g**|**X** = **x**<sub>0</sub>). The obvious parallel with what we have done before in this class, is that many (all?) regression methods we discussed in class are different ways of estimating the conditional mean of the response variable (conditional on the value of the explanatory variables).

As in the regression case, there are different ways to estimate this optimal predictor / classifier. Some will be model-based, some will be non-parametric in nature. And some can be considered "restricted" non-parametric methods (without relying on a model, but imposing some other type of constrain on the shape of the classifier). The equivalent methods for regression with continuous responses are: linear or non-linear regression as model-based methods; kernel or local regression as non-parametric methods; and splines or regression trees as "constrained" (regularized?) non-parametric methods.

Below we first discuss model-based methods (Linear and Quadratic Discriminant Analysis) and later introduce non-parametric methods (nearest-neighbours and classification trees).

Linear Discriminant Analysis
----------------------------

Probably the "second easiest to explain" approach (what would "*the* easiest to explain" be?) to estimate the above probability is to model the distribution of the explanatory variables *within* each class. This will then uniquely determine the probabilities we need to estimate, as discussed above and in class. In particular, one the simplest models we can use is a normal (Gaussian) one. As we saw in class, if we assume that the distribution of the features for each class is Gaussian with a common covariance matrix across clases, then it easy to show (*and I strongly suggest that you do it*) that the optimal classifier (using the 0-1 loss function m.entioned above) is a linear function of the explanatory variables. The coefficients of this linear function depend on the parameters of the assumed Gaussian distributions. The latter can be estimated using the training set and plugging them in provides a natural estimator of the optimal classifier.

The function `lda` in the `MASS` library implements this simple classifier. We illustrate it here on the rather simple and well-known vaso constriction data, available in the `robustbase` package. More details, as usual, can be found on its help page. The response variable takes two values (represented below as **blue** and **red**), and there are only two explanatory variables (which allows us to visualize our methods and results).

``` r
data(vaso, package='robustbase')
plot(Volume ~ Rate, data=vaso, pch=19, cex=1.5, 
       col=c('red', 'blue')[Y+1])
```

![](README_files/figure-markdown_github-ascii_identifiers/lda1-1.png)

To train the LDA classifier we use the function `lda` as follows (note the *model-like* syntax to indicate the response and explanatory variables):

``` r
library(MASS)
a.lda <- lda(Y ~ Volume + Rate, data=vaso)
```

Now, given any value of the explanatory variables `(Volume, Rate)` we can use the method `predict` on the object returned by `lda()` to estimate the conditional probabilities of **blue** and **red**. Next we visualize which regions of the feature space will be predicted to contain **blue** points (and then obviously which areas will be predicted to correspond to **red** responses). To do this we build a relatively fine grid of posible `(Volume, Rate)` values and plot the corresponding predictions:

``` r
# build a grid of points of (Volume, Rate),
# and obtain predictions for each of them
# 40,000 in total
xvol <- seq(0, 4, length=200)
xrat <- seq(0, 4, length=200)
xx <- expand.grid(xvol, xrat)
names(xx) <- c('Volume', 'Rate')

pr.lda <- predict(a.lda, newdata=xx)$posterior[,2]
image(xrat, xvol, matrix(pr.lda, 200, 200), col=terrain.colors(100),
      ylab='Volume', xlab='Rate', main='LDA')
points(Volume ~ Rate, data=vaso, pch=19, cex=1.5,
       col=c('red', 'blue')[Y+1])
```

![](README_files/figure-markdown_github-ascii_identifiers/lda1.2-1.png)

This model-based approach to classification can be relaxed slightly if we assume instead that the features have a Gaussian distribution within each class, but that the covariance matrix may be different across classes. The corresponding optimal classifier is now a quadratic function of the predictors (**prove it!**). The function `qda` in the `MASS` library implements this classifier.

This approach can be used with any number of classes. Any limitations?

Logistic regression (Review)
----------------------------

If we model the distribution of the features within each class using a multivariate Gaussian distribution, then it is easy to see that the boundaries between classes are linear (functions of the features) (*check!*) Furthermore, the log of the odds ratio between classes is a linear function. It is interesting to note that one can start with this last assumption (instead of the full Gaussian model) and arrive at a fully parametric model for the conditional distibution of the classes given the features (see the class slides). The parameters can be estimated using maximum likelihood. For two classes this is the logistic regression model, which you may have seen in previous courses.

We illustrate this on the `vaso` data as before. Since this is a 2-class problem, we just need to fit a logistic regression model. The function `glm` in `R` does it for us, we specify that we want to fit such a model using the argument `family=binomial`. Once we obtain parameter estimators (in the `glm` object `a` below), we use the `predict` method to obtain predicted conditional probabilities on the same grid we used before:

``` r
a <- glm(Y ~ ., data=vaso, family=binomial)
pr <- predict(a, newdata=xx, type='response')
```

We now plot the data and the *surface* of predicted probabilities for blue points (higher probabilites are displayed with lighter colors).

``` r
image(xrat, xvol, matrix(pr, 200, 200), col=terrain.colors(100),
      ylab='Volume', xlab='Rate', main='Logistic')
points(Volume ~ Rate, data=vaso, pch=19, cex=1.5,
       col=c('red', 'blue')[Y+1])
```

![](README_files/figure-markdown_github-ascii_identifiers/logistic2-1.png)
