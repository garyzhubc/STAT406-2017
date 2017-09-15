STAT406 - Lecture 4 notes
================
Matias Salibian-Barrera
2017-09-15

Lecture slides
--------------

The lecture slides are [here](STAT406-17-lecture-4-preliminary.pdf).

Estimating MSPE with CV when the model was built using the data
---------------------------------------------------------------

Last week we learned that one needs to be careful when using cross-validation (in any of its flavours--leave one out, K-fold, etc.) Misuse of cross-validation is, unfortunately, not unusual. For [one example](https://doi.org/10.1073/pnas.102102699) see:

> Ambroise, C. and McLachlan, G.J. (2002). Selection bias in gene extraction on the basis of microarray gene-expression data, PNAS, 2002, 99 (10), 6562-6566. DOI: 10.1073/pnas.102102699

In particular, for every fold one needs to repeat **everything** that was done with the training set (selecting variables, looking at pairwise correlations, AIC values, etc.)

Correlated covariates
---------------------

Technological advances in recent decades have resulted in data being collected in a fundamentally different manner from the way it was done when most "classical" statistical methods were developed (early to mid 1900's). Specifically, it is now not at all uncommon to have data sets with an abundance of potentially useful explanatory variables (for example with more variables than observations). Sometimes the investigators are not sure which of the collected variables can be expected to be useful or meaningful.

A consequence of this "wide net" data collection strategy is that many of the explanatory variables may be correlated with each other. In what follows we will illustrate some of the problems that this can cause both when training and interpreting models, and also with the resulting predictions.

### Variables that were important may suddenly "dissappear"

Consider the air pollution data set we used earlier, and the **reduced** linear regression model discussed in class:

``` r
# Correlated covariates
x <- read.table('../Lecture1/rutgers-lib-30861_CSV-1.csv', header=TRUE, sep=',')
reduced <- lm(MORT ~ POOR + HC + NOX + HOUS + NONW, data=x)
round( summary(reduced)$coef, 3)
```

    ##             Estimate Std. Error t value Pr(>|t|)
    ## (Intercept) 1172.831    143.241   8.188    0.000
    ## POOR          -4.065      2.238  -1.817    0.075
    ## HC            -1.480      0.333  -4.447    0.000
    ## NOX            2.846      0.652   4.369    0.000
    ## HOUS          -2.911      1.533  -1.899    0.063
    ## NONW           4.470      0.846   5.283    0.000

Note that all coefficients seem to be significant based on the individual tests of hypothesis (with `POOR` and `HOUS` maybe only marginally so). In this sense all 5 explanatory varibles in this model appear to be relevant.

Now, we fit the **full** model, that is, we include all available explanatory variables in the data set:

``` r
full <- lm(MORT ~ ., data=x)
round( summary(full)$coef, 3)
```

    ##             Estimate Std. Error t value Pr(>|t|)
    ## (Intercept) 1763.981    437.327   4.034    0.000
    ## PREC           1.905      0.924   2.063    0.045
    ## JANT          -1.938      1.108  -1.748    0.087
    ## JULT          -3.100      1.902  -1.630    0.110
    ## OVR65         -9.065      8.486  -1.068    0.291
    ## POPN        -106.826     69.780  -1.531    0.133
    ## EDUC         -17.157     11.860  -1.447    0.155
    ## HOUS          -0.651      1.768  -0.368    0.714
    ## DENS           0.004      0.004   0.894    0.376
    ## NONW           4.460      1.327   3.360    0.002
    ## WWDRK         -0.187      1.662  -0.113    0.911
    ## POOR          -0.168      3.227  -0.052    0.959
    ## HC            -0.672      0.491  -1.369    0.178
    ## NOX            1.340      1.006   1.333    0.190
    ## SO.            0.086      0.148   0.585    0.562
    ## HUMID          0.107      1.169   0.091    0.928

In the **full** model there are many more parameters that need to be estimated, and while two of them appear to be significantly different from zero (`NONW` and `PREC`), all the others appear to be redundant. In particular, note that the p-values for the individual test of hypotheses for 4 out of the 5 regression coefficients for the variables of the **reduced** model have now become not significant.

``` r
round( summary(full)$coef[ names(coef(reduced)), ], 3)
```

    ##             Estimate Std. Error t value Pr(>|t|)
    ## (Intercept) 1763.981    437.327   4.034    0.000
    ## POOR          -0.168      3.227  -0.052    0.959
    ## HC            -0.672      0.491  -1.369    0.178
    ## NOX            1.340      1.006   1.333    0.190
    ## HOUS          -0.651      1.768  -0.368    0.714
    ## NONW           4.460      1.327   3.360    0.002

In other words, the coeffficients of explanatory variables that appeared to be relevant in one model may turn to be "not significant" when other variables are included. This could pose some challenges for interpreting the estimated parameters of the models.

### Why does this happen?

Recall that the covariance matrix of the least squares estimator involves the inverse of (X'X), where X' denotes the transpose of the n x p matrix X (that contains each vector of explanatory variables as a row). It is easy to see that if two columns of X are linearly dependent, then X'X will be rank deficient. When two columns of X are "close" to being linearly dependent (e.g. their linear corrleation is high), then the matrix X'X will be ill-conditioned, and its inverse will have very large entries. This means that the estimated standard errors of the least squares estimator will be unduly large, resulting in non-significant test of hypotheses for each parameter separately, even if the global test for all of them simultaneously is highly significant.

### Why is this a problem if we are interested in prediction?

Although in many applications one is interested in interpreting the parameters of the model, even if one is only trying to fit / train a model to do predictions, highly variable parameter estimators will typically result in a noticeable loss of prediction accuracy. This can be easily seen from the bias / variance factorization of the mean squared prediction error (MSPE) mentioned in class. Hence, better predictions can be obtained if one uses less-variable parameter (or regression function) estimators.

### What can we do?

A commonly used strategy is to remove some explanatory variables from the model, leaving only non-redundant covariates. However, this is easier said than done. You will have seen some strategies in previous Statistics courses (e.g. stepwise variable selection). In coming weeks we will investigate other methods to deal with this problem.

Comparing models -- General strategy
------------------------------------

Suppose we have a set of competing models from which we want to choose the "best" one. In order to properly define our problem we need the following:

-   a list of models to be considered;
-   a numerical measure to compare any two models in our list;
-   a strategy (algorithm, criterion) to navigate the set of models; and
-   a criterion to stop the search.

For example, in stepwise methods the models under consideration in each step are those that differ from the current model only by one coefficient (variable). The numerical measure used to compare models could be AIC, or Mallow's Cp, etc. The strategy is to only consider submodels with one fewer variable than the current one, and we stop if either none of these "p-1" submodels is better than the current one, or we reach an empty model.

Comparing models -- What is AIC?
--------------------------------

One intuitively sensible quantity that can be used to compare models is a distance measuring how "close" the distributions implied by these models are from the actual stochastic process generating the data (here "stochastic process" refers to the random mechanism that generated the observations). In order to do this we need:

1.  a distance / metric (or at least a "quasimetric") between models; and
2.  a way of estimating this distance when the "true" model is unknown.

AIC provides an unbiased estimator of the Kullback-Leibler divergence between the estimated model and the "true" one. See the lecture slides for more details.

Shrinkage methods / Ridge regression
------------------------------------

Stepwise methods are highly variable, and thus their predictions may not be very accurate (high MSPE). A different way to manage correlated explanatory variables (to "reduce" their presence in the model without removing them) is...

### Selecting the amount of shrinkage
