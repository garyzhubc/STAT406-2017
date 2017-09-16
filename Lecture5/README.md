STAT406 - Lecture 5 notes
================
Matias Salibian-Barrera
2017-09-16

Lecture slides
--------------

The lecture slides are [here](STAT406-17-lecture-5-preliminary.pdf).

Ridge regression
----------------

Variable selection methods like stepwise can be highly variable. To illustrate this issue consider the following simple experiment. As in the previous lecture, we apply stepwise on 5 randomly selected folds of the data, and look at the models selected in each of them.

``` r
airp <- read.table('../lecture1/rutgers-lib-30861_CSV-1.csv', header=TRUE, sep=',')
library(MASS)
k <- 5
n <- nrow(airp)
set.seed(123456)
ii <- sample( (1:n) %% k + 1 )
for(j in 1:k) {
  x0 <- airp[ii != j, ]
  null0 <- lm(MORT ~ 1, data=x0)
  full0 <- lm(MORT ~ ., data=x0) # needed for stepwise
  step.lm0 <- stepAIC(null0, scope=list(lower=null0, upper=full0), trace=FALSE)
  print(formula(step.lm0)[[3]])
}
```

    ## NONW + JANT + EDUC + SO. + PREC + JULT
    ## NONW + PREC + SO. + DENS + HOUS + WWDRK
    ## NONW + EDUC + SO. + PREC + JANT + JULT
    ## NONW + JANT + SO. + PREC + DENS + JULT
    ## NONW + EDUC + SO. + JANT + HUMID + POPN

Although many variables appear in more than one model, only `NONW` and `SO.` are in all of them, and `JANT` and `PREC` in 4 out of the 5. There are also several that appear in only one model (`HOUS`, `WWDRK` and `POPN`). <!-- `EDUC` 3 --> <!-- `JULT` in 3,  --> <!-- `DENS` in 2 --> <!-- and  --> This variability may in turn impact (negatively) the accuracy of the resulting predictions.

A different approach to dealing with potentially correlated explanatory variables (with the goal of obtaining less variable / more accurate predictions) is to "regularize" the parameter estimates. In other words we modify the optimization problem that defines the parameter estimators (in the case of linear regression fits we tweak the least squares problem) to limit their size (in fact restricting them to be in a bounded and possibly small subset of the parameter space).

The first proposal for a regularized / penalized estimator for linear regression models is Ridge Regression. We will use the function `glmnet` in package `glmnet` to compute the Ridge Regression estimator. Note that this function implements a larger family of regularized estimators, and in order to obtain a Ridge Regression estimator we need to set the argument `alpha = 0` of `glmnet()`. <!-- We use Ridge Regression with the air pollution data to obtain a --> <!-- more stable predictor. --> We also specify a range of possible values of the penalty coefficient (below we use a grid of 50 values between exp(-3) and exp(10)).

``` r
airp <- read.table('../lecture1/rutgers-lib-30861_CSV-1.csv', header=TRUE, sep=',')
library(glmnet)
# alpha = 0 - Ridge
# alpha = 1 - LASSO
y <- as.vector(airp$MORT)
xm <- as.matrix(airp[, -16])
lambdas <- exp( seq(-3, 10, length=50))
a <- glmnet(x=xm, y=y, lambda=rev(lambdas),
            family='gaussian', alpha=0)
```

The returned object contains the estimated regression coefficients for each possible value of the regularization parameter. We can look at them using the `plot` method for objects of class `glmnet` as follows:

``` r
plot(a, xvar='lambda', label=TRUE, lwd=6, cex.axis=1.5, cex.lab=1.2, ylim=c(-20, 20))
```

![](README_files/figure-markdown_github-ascii_identifiers/ridge.plot-1.png)

5-fold CV

``` r
# run 5-fold CV
set.seed(123)
tmp <- cv.glmnet(x=xm, y=y, lambda=lambdas, nfolds=5, alpha=0, family='gaussian')
plot(tmp, lwd=6, cex.axis=1.5, cex.lab=1.2)
```

![](README_files/figure-markdown_github-ascii_identifiers/ridge.cv-1.png)

5-fold CV again

``` r
set.seed(23)
tmp <- cv.glmnet(x=xm, y=y, lambda=lambdas, nfolds=5, alpha=0, family='gaussian')
plot(tmp, lwd=6, cex.axis=1.5, cex.lab=1.2)
```

![](README_files/figure-markdown_github-ascii_identifiers/ridge.cv2-1.png)

What is the optimal lambda?. Average over several runs?

``` r
set.seed(123)
op.la <- 0
for(j in 1:20) {
  tmp <- cv.glmnet(x=xm, y=y, lambda=lambdas, nfolds=5, alpha=0, family='gaussian')
  op.la <- op.la + tmp$lambda.min # tmp$lambda.1se
}
(op.la <- op.la / 20)
```

    ## [1] 10.39156

``` r
log(op.la)
```

    ## [1] 2.340994

Effective degrees of freedom

``` r
# compute EDF
xm.svd <- svd(xm) #  scale(xm, scale=FALSE))
(est.edf <- sum( xm.svd$d^2 / ( xm.svd$d^2 + op.la ) ))
```

    ## [1] 13.15941

``` r
xm.svd <- svd(scale(xm, scale=FALSE))
(est.edf <- sum( xm.svd$d^2 / ( xm.svd$d^2 + op.la ) ))
```

    ## [1] 13.05737

Compare the MSPE of the different models

``` r
library(MASS)
n <- nrow(xm)
k <- 5
ii <- (1:n) %% k + 1
set.seed(123)
N <- 100
mspe.st <- mspe.ri <- mspe.f <- rep(0, N)
for(i in 1:N) {
  ii <- sample(ii)
  pr.f <- pr.ri <- pr.st <- rep(0, n)
  for(j in 1:k) {
    tmp.ri <- cv.glmnet(x=xm[ii != j, ], y=y[ii != j], lambda=lambdas,
                        nfolds=5, alpha=0, family='gaussian')
    null <- lm(MORT ~ 1, data=airp[ii != j, ])
    full <- lm(MORT ~ ., data=airp[ii != j, ])
    tmp.st <- stepAIC(null, scope=list(lower=null, upper=full), trace=0)
    pr.ri[ ii == j ] <- predict(tmp.ri, s='lambda.min', newx=xm[ii==j,])
    pr.st[ ii == j ] <- predict(tmp.st, newdata=airp[ii==j,])
    pr.f[ ii == j ] <- predict(full, newdata=airp[ii==j,])
  }
  mspe.ri[i] <- mean( (airp$MORT - pr.ri)^2 )
  mspe.st[i] <- mean( (airp$MORT - pr.st)^2 )
  mspe.f[i] <- mean( (airp$MORT - pr.f)^2 )
}
boxplot(mspe.ri, mspe.st, mspe.f, names=c('Ridge', 'Stepwise', 'Full'), 
        col=c('gray80', 'tomato', 'springgreen'), cex.axis=1.5, cex.lab=1.5, 
        main='Credit - 10 runs 5-fold CV', cex.main=2, ylim=c(1300, 3000))
mtext(expression(hat(MSPE)), side=2, line=2.5)
```

![](README_files/figure-markdown_github-ascii_identifiers/ridge.mspe-1.png)
