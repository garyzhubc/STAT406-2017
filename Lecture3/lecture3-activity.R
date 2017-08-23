
library(MASS)
dat <- read.table('fallacy.dat', header=TRUE, sep=',')
n <- nrow(dat)
p <- ncol(dat)
null <- lm(Y~1, data=dat)
full <- lm(Y~., data=dat) # needed for stepwise
step.lm <- stepAIC(null, scope=list(lower=null, upper=full), trace=FALSE)

# improper way of using CV to estimate MSPE
# compare stepwise and "null" linear models
# 5-fold CV
k <- 5
ii <- rep(1:k, each=n/k) # bad programming! (1:n) %% k + 1
# make it reproducible
set.seed(17)
# 10 runs
N <- 10
mspe.n <- mspe.st <- rep(0, N)
for(i in 1:N) {
  # shuffle indices
  ii <- sample(ii)
  pr.n <- pr.st <- rep(0, n)
  for(j in 1:k) {
    tmp.st <- update(step.lm, data=dat[ii != j, ])
    # tmp.f <- update(full, data=dat[ii != j, ])
    pr.st[ ii == j ] <- predict(tmp.st, newdata=dat[ii==j,])
    pr.n[ ii == j ] <- mean((dat$Y)[ii != j])
  }
  mspe.st[i] <- mean( (dat$Y - pr.st)^2 )
  mspe.n[i] <- mean( (dat$Y - pr.n)^2 )
  # print(c(mean(mspe.st[1:i]), mean(mspe.n[1:i])))
}
boxplot(mspe.st, mspe.n, names=c('Stepwise', 'NULL'), col=c('gray60', 'hotpink'), main='Wrong')
summary(mspe.st)
summary(mspe.n)



# proper way
k <- 5
ii <- rep(1:k, each=n/k) # bad programming! (1:n) %% k + 1
set.seed(17)
N <- 10
mspe.n <- mspe.st <- rep(0, N)
for(i in 1:N) {
  ii <- sample(ii)
  pr.n <- pr.st <- rep(0, n)
  for(j in 1:k) {
    dat0 <- dat[ii != j, ]
    null0 <- lm(Y~1, data=dat0)
    full0 <- lm(Y~., data=dat0) # needed for stepwise
    step.lm0 <- stepAIC(null0, scope=list(lower=null0, upper=full0), trace=FALSE)
    pr.st[ ii == j ] <- predict(step.lm0, newdata=dat[ii==j,])
    pr.n[ ii == j ] <- mean((dat$Y)[ii != j])
  }
  mspe.st[i] <- mean( (dat$Y - pr.st)^2 )
  mspe.n[i] <- mean( (dat$Y - pr.n)^2 )
  # print(c(mean(mspe.st[1:i]), mean(mspe.n[1:i])))
}
boxplot(mspe.st, mspe.n, names=c('Stepwise', 'NULL'), col=c('gray60', 'hotpink'), main='Correct')
summary(mspe.st)
summary(mspe.n)



# Check the "actual MSPE" of both methods
# using a real new data sets, and verify
# that the proper CV gives a better
# estimate for MSPE

# re-fit the models with the original fallacy data
library(MASS)
dat <- read.table('fallacy.dat', header=TRUE, sep=',')
n <- nrow(dat)
p <- ncol(dat)
null <- lm(Y~1, data=dat)
full <- lm(Y~., data=dat)
step.lm <- stepAIC(null, scope=list(lower=null, upper=full), trace=FALSE)


# an actual new data set
n <- 150 # try 100000
p <- 110
set.seed(123456)
y <- rnorm(n)
x <- matrix( rnorm(n*p), n, p)
dat1 <- data.frame(list(y=y, x=x))
names(dat1) <- names(dat)
# rm(x,y)
y.st <- predict(step.lm, newdata=dat1)
(pr.st <- mean( (dat1$Y - y.st)^2 ) )
(pr.n <- mean( (dat1$Y - coef(null))^2 ) )
# (pr.n <- var(dat1$Y)*(n-1)/n )

# note that pr.st are slightly lower than our CV-estimates
# can you explain why?
#

###
### Correlations

set.seed(123)
x1 <- rnorm(506)
x2 <- rnorm(506, mean=2, sd=1)
x3 <- rexp(506, rate=1)
x4 <- x2 + rnorm(506, sd=.1)
x5 <- x1 + rnorm(506, sd=.1)
x6 <- x1 - x2 + rnorm(506, sd=.1)
x7 <- x1 + x3 + rnorm(506, sd=.1)
y <- x1*3 + x2/3 + rnorm(506, sd=2.2)

x <- data.frame(y=y, x1=x1, x2=x2,
                x3=x3, x4=x4, x5=x5, x6=x6, x7=x7)

# Variables $X_1$ and $X_2$ are clearly important. But they are
# also highly correlated to $X_4$, $X_5$, $X_6$ and $X_7$.

# nothing is significant
summary(lm(y~., data=x))

# But...
summary(lm(y~x1+x2, data=x))


# Even worse...
summary(lm(y~x1+x2+x4, data=x))

null <- lm(y ~ 1, data=x)
full <- lm(y ~ ., data=x)
MASS::stepAIC(null, scope=list(lower=null, upper=full), trace=FALSE)


