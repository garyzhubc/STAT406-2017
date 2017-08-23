
# Load prostate cancer data
x <- read.table('prostate.data', header=TRUE, row.names=1)

# leave-one-out CV
n <- nrow(x)
y.hat1 <- y.hat2 <- rep(NA, n)
for(j in 1:n) {
  tmp1 <- lm(lpsa ~ lcavol + lweight + age + lbph, data=x[-j,])
  y.hat1[j] <- predict(tmp1, newdata=x[j,])
  tmp2 <- lm(lpsa ~ lcavol + lweight + svi + lcp, data=x[-j,])
  y.hat2[j] <- predict(tmp2, newdata=x[j,])
}
mean( (x$lpsa - y.hat1)^2 )
mean( (x$lpsa - y.hat2)^2 )


# 5-fold CV
n <- nrow(x)
k <- 5
ii <- rep(1:5, each=floor(n/k)) 
if( length(ii) < n ) ii <- c(ii, 1:(n - length(ii)))
set.seed(123)
ii <- sample(ii)
pr.1 <- pr.2 <- rep(0, n)
for(j in 1:k) {
  tmp.1 <- lm(lpsa ~ lcavol + lweight + age + lbph, data=x[ii != j, ])
  tmp.2 <- lm(lpsa ~ lcavol + lweight + svi + lcp, data=x[ii != j, ])
  pr.1[ ii == j ] <- predict(tmp.1, newdata=x[ii==j,])
  pr.2[ ii == j ] <- predict(tmp.2, newdata=x[ii==j,])
}
mean( (x$lpsa - pr.1)^2 )
mean( (x$lpsa - pr.2)^2 )

# run again, and again

# Use 10 runs of 5-fold CV
N <- 10
mspe1 <- mspe2 <- rep(0, N)
ii <- rep(1:5, each=floor(n/k)) 
if( length(ii) < n ) ii <- c(ii, 1:(n - length(ii)))
set.seed(123)
for(i in 1:N) {
  ii <- sample(ii)
  pr.1 <- pr.2 <- rep(0, n)
  for(j in 1:k) {
    tmp.1 <- lm(lpsa ~ lcavol + lweight + age + lbph, data=x[ii != j, ])
    tmp.2 <- lm(lpsa ~ lcavol + lweight + svi + lcp, data=x[ii != j, ])
    pr.1[ ii == j ] <- predict(tmp.1, newdata=x[ii==j,])
    pr.2[ ii == j ] <- predict(tmp.2, newdata=x[ii==j,])
  }
  mspe1[i] <- mean( (x$lpsa - pr.1)^2 )
  mspe2[i] <- mean( (x$lpsa - pr.2)^2 )
}  
boxplot(mspe1, mspe2, names=c('Model 1', 'Model 2'), 
        col=c('gray70', 'lightpink'), 
        main='Prostate Cancer - 10 runs 5-fold CV')
mtext(expression(hat(MSPE)), side=2, line=2.5)


# Add the full model to the comparison
N <- 10
mspe3 <- mspe1 <- mspe2 <- rep(0, N)
ii <- rep(1:5, each=floor(n/k)) 
if( length(ii) < n ) ii <- c(ii, 1:(n - length(ii)))
set.seed(123)
for(i in 1:N) {
  ii <- sample(ii)
  pr.3 <- pr.1 <- pr.2 <- rep(0, n)
  for(j in 1:k) {
    tmp.3 <- lm(lpsa ~ ., data=x[ii != j, ])
    tmp.1 <- lm(lpsa ~ lcavol + lweight + age + lbph, data=x[ii != j, ])
    tmp.2 <- lm(lpsa ~ lcavol + lweight + svi + lcp, data=x[ii != j, ])
    pr.1[ ii == j ] <- predict(tmp.1, newdata=x[ii==j,])
    pr.2[ ii == j ] <- predict(tmp.2, newdata=x[ii==j,])
    pr.3[ ii == j ] <- predict(tmp.3, newdata=x[ii==j,])
  }
  mspe1[i] <- mean( (x$lpsa - pr.1)^2 )
  mspe2[i] <- mean( (x$lpsa - pr.2)^2 )
  mspe3[i] <- mean( (x$lpsa - pr.3)^2 )
}  
boxplot(mspe1, mspe2, mspe3, names=c('Model 1', 'Model 2', 'Model 3'), 
        col=c('gray80', 'lightpink1', 'tomato3'), 
        main='Prostate Cancer - 10 runs 5-fold CV')
mtext(expression(hat(MSPE)), side=2, line=2.5)



# Correlated covariates
x <- read.table('pollution.dat', header=TRUE, sep=',')
reduced <- lm(MORT ~ POOR + HC + NOX + HOUS + NONW, data=x)
full <- lm(MORT ~ ., data=x)

# significant variables in "reduced" dissappear in "full"
round( summary(reduced)$coef, 3)
round( summary(full)$coef[ names(coef(reduced)), ], 3)


# reduced gives better predictions than full
# check, using 10 runs of 5-fold CV
n <- nrow(x)
N <- 10
k <- 5
ii <- rep(1:5, each=floor(n/k)) 
(length(ii) == n ) # if not, we need to fill-in some labels
set.seed(123)
mspe.f <- mspe.r <- rep(0, N)
for(i in 1:N) {
ii <- sample(ii)
pr.f <- pr.r <- rep(0, n)
for(j in 1:k) {
  tmp.f <- lm(MORT ~ ., data=x[ii != j, ])
  tmp.r <- lm(MORT ~ POOR + HC + NOX + HOUS + NONW, data=x[ii != j, ])
  pr.f[ ii == j ] <- predict(tmp.f, newdata=x[ii==j,])
  pr.r[ ii == j ] <- predict(tmp.r, newdata=x[ii==j,])
}
mspe.f[i] <- mean( (x$MORT - pr.f)^2 )
mspe.r[i] <- mean( (x$MORT - pr.r)^2 )
}
boxplot(mspe.f, mspe.r, names=c('Full', 'Reduced'), 
        col=c('gray80', 'tomato'), 
        main='Air Pollution - 10 runs 5-fold CV')
mtext(expression(hat(MSPE)), side=2, line=2.5)


