# Read training set
x.tr <- read.table('pollution-train.dat', header=TRUE, sep=',')

# fit a linear regression model with all available
# predictors
full <- lm(MORT ~ . , data=x.tr)

# now fit a smaller linear model
# using only 5 predictors
reduced <- lm(MORT ~ POOR + HC + NOX + HOUS + NONW, data=x.tr)

# which model fit the data better?
# (in terms of residual sum of squares)
sum( resid(reduced)^2 )
sum( resid(full)^2 )

# no surprises there

# which model produces better predictions
# on the test set?
x.te <- read.table('pollution-test.dat', header=TRUE, sep=',')

# obtain predicted values for the test set
# with the full and reduced models
x.te$pr.full <- predict(full, newdata=x.te)
x.te$pr.reduced <- predict(reduced, newdata=x.te)

# compute the mean squared prediction error
# (on the test set) obtained with each of the
# models
with(x.te, mean( (MORT - pr.full)^2 ))
with(x.te, mean( (MORT - pr.reduced)^2 ))

# repeat with different partitions
x <- read.csv('rutgers-lib-30861_CSV-1.csv')
set.seed(123)
n <- nrow(x)
tra <- sample(n, 45, repl=FALSE)
x.tr <- x[tra, ]
x.te <- x[-tra, ]
full <- lm(MORT ~ . , data=x.tr)
reduced <- lm(MORT ~ POOR + HC + NOX + HOUS + NONW, data=x.tr)
x.te$pr.full <- predict(full, newdata=x.te)
x.te$pr.reduced <- predict(reduced, newdata=x.te)
with(x.te, mean( (MORT - pr.full)^2 ))
with(x.te, mean( (MORT - pr.reduced)^2 ))


# Leave-one-out CV
pr.full <- pr.reduced <- rep(0, n)
for(i in 1:n) {
  full <- lm(MORT ~ . , data=x[-i, ])
  reduced <- lm(MORT ~ POOR + HC + NOX + HOUS + NONW, data=x[-i, ])
  pr.full[i] <- predict(full, newdata = x[i, ])
  pr.reduced[i] <- predict(reduced, newdata = x[i, ])
}
mean( (x$MORT - pr.full)^2 )
mean( (x$MORT - pr.reduced)^2 )


# K-fold CV
n <- nrow(x)
k <- 5
pr.full <- pr.reduced <- rep(0, n)
# This is bad, bad coding!
inds <- rep(1:k, each=n/k) # I know n/k is an integer!
# shuffle x, this is bad coding practice!!
set.seed(123)
xs <- x[ sample(n, repl=FALSE), ]
# loop through the folds
for(j in 1:k) {
  x.tr <- xs[inds != j, ]
  x.te <- xs[inds == j, ]
  full <- lm(MORT ~ . , data=x.tr)
  reduced <- lm(MORT ~ POOR + HC + NOX + HOUS + NONW, data=x.tr)
  pr.full[ inds== j] <- predict(full, newdata=x.te)
  pr.reduced[ inds==j ] <- predict(reduced, newdata=x.te)
}
# compare predictions
mean( (xs$MORT - pr.full)^2 )
mean( (xs$MORT - pr.reduced)^2 )

