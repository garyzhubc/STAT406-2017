STAT406 - Lecture 12 notes
================
Matias Salibian-Barrera
2017-10-12

#### LICENSE

These notes are released under the "Creative Commons Attribution-ShareAlike 4.0 International" license. See the **human-readable version** [here](https://creativecommons.org/licenses/by-sa/4.0/) and the **real thing** [here](https://creativecommons.org/licenses/by-sa/4.0/legalcode).

Lecture slides
--------------

The lecture slides are [here](STAT406-17-lecture-12-preliminary.pdf).

Logistic regression (Review)
----------------------------

``` r
data(vaso, package='robustbase')

plot(Volume ~ Rate, data=vaso, pch=19, cex=1.5, col=c('red', 'blue')[Y+1],
     xlim=c(0, 4), ylim=c(0,4))
```

![](README_files/figure-markdown_github-ascii_identifiers/logistic1-1.png)

``` r
a <- glm(Y ~ ., data=vaso, family=binomial)

# build a grid of points of (Volume, Rate),
# and obtain predictions for each of them 
# 40,000 in total
xvol <- seq(0, 4, length=200)
xrat <- seq(0, 4, length=200)
xx <- expand.grid(xvol, xrat)
names(xx) <- c('Volume', 'Rate')
pr <- predict(a, newdata=xx, type='response')

# display them
image(xrat, xvol, matrix(pr, 200, 200), col=terrain.colors(100),
      ylab='Volume', xlab='Rate', main='Logistic')
points(Volume ~ Rate, data=vaso, pch=19, cex=1.5, 
       col=c('red', 'blue')[Y+1])
```

![](README_files/figure-markdown_github-ascii_identifiers/logistic1-2.png)

``` r
# Y = 1 corresponds to blue points
# higher probabilities are displayed with lighter colors
```

Linear Discriminant Analysis
----------------------------

``` r
library(MASS)

a.lda <- lda(Y ~ Volume + Rate, data=vaso)
pr.lda <- predict(a.lda, newdata=xx)$posterior[,2]
image(xrat, xvol, matrix(pr.lda, 200, 200), col=terrain.colors(100),
      ylab='Volume', xlab='Rate', main='LDA')
points(Volume ~ Rate, data=vaso, pch=19, cex=1.5, 
       col=c('red', 'blue')[Y+1])
```

![](README_files/figure-markdown_github-ascii_identifiers/lda1-1.png)
