
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


