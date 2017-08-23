
# load data
x <- read.csv('rutgers-lib-30861_CSV-1.csv')

# full model
a <- lm(MORT ~ . , data=x)
summary(a)

# try to find apparently non-significant
# covariates that become significant when
# other variables are removed from the model

# a "reverse engineering" approach, in some sense

b <- lm(MORT ~ POOR + HC + NOX + SO. + HUMID, data=x)
summary(b)
b <- lm(MORT ~ POOR + HC + NOX + HUMID, data=x)
summary(b)
summary(a)
b <- lm(MORT ~ POOR + HC + NOX + JULT, data=x)
summary(b <- lm(MORT ~ POOR + HC + NOX + JULT, data=x))
summary(b <- lm(MORT ~ POOR + HC + NOX + JANT, data=x))
names(x)
summary(b <- lm(MORT ~ POOR + HC + NOX + OVR65, data=x))

summary(b <- lm(MORT ~ POOR + HC + NOX + HOUS + NONW, data=x))

# most of these variables are non-significant when others
# are added in the model
# compare with full model

summary(a)

# leave one out CV
pr1 <- pr2 <- pr3 <- rep(0, n)
n <- dim(x)[1]
pr1 <- pr2 <- pr3 <- rep(0, n)
with(x, mean( (MORT - pr1)^2 ))
with(x, mean( (MORT - pr2)^2 ))
n <- dim(x)[1]
pr1 <- pr2 <- pr3 <- rep(0, n)
for(j in 1:n) {
a <- lm(MORT ~ ., data=x[-j,])
pr1[j] <- predict(a, newdata=x[j,])
b <- lm(MORT ~ POOR + HC + NOX + HOUS + NONW, data=x[-j,])
pr2[j] <- predict(b, newdata=x[j,])
}

with(x, mean( (MORT - pr1)^2 ))
with(x, mean( (MORT - pr2)^2 ))

# training / test splits
set.seed(123)
ii <- sample(rep(1:6, each=10))
x.tr <- x[ii != 1, ]
x.te <- x[ii == 1, ]
a <- lm(MORT ~ . , data=x.tr)
b <- lm(MORT ~ POOR + HC + NOX + HOUS + NONW, data=x.tr)
pr1 <- predict(a, newdata=x.te)
pr2 <- predict(b, newdata=x.te)
with(x.te, mean( (MORT - pr1)^2 ))
with(x.te, mean( (MORT - pr2)^2 ))

x.tr <- x[ii != 2, ]
x.te <- x[ii == 2, ]
a <- lm(MORT ~ . , data=x.tr)
b <- lm(MORT ~ POOR + HC + NOX + HOUS + NONW, data=x.tr)
pr1 <- predict(a, newdata=x.te)
pr2 <- predict(b, newdata=x.te)
with(x.te, mean( (MORT - pr1)^2 ))
with(x.te, mean( (MORT - pr2)^2 ))

x.tr <- x[ii != 3, ]
x.te <- x[ii == 3, ]
a <- lm(MORT ~ . , data=x.tr)
b <- lm(MORT ~ POOR + HC + NOX + HOUS + NONW, data=x.tr)
pr1 <- predict(a, newdata=x.te)
pr2 <- predict(b, newdata=x.te)
with(x.te, mean( (MORT - pr1)^2 ))
with(x.te, mean( (MORT - pr2)^2 ))


a <- lm(MORT ~ . , data=x)
b <- lm(MORT ~ POOR + HC + NOX + HOUS + NONW, data=x)
summary(a)
summary(b)
sum( resid(a)^2 )
sum( resid(b)^2 )




# setwd("C:/Users/matias/Dropbox/STAT406")
# dat <- read.table('dwp-data.txt', header=TRUE)
# dat
# a <- lm(BSAAM ~ . - Year, data=dat)
# a
# summary(a)
# summary( a <- lm(BSAAM ~ APMAM, data=dat) )
# summary( a <- lm(BSAAM ~ . - Year, data=dat) )
# summary( a <- lm(BSAAM ~ OPBPC, data=dat) )
# summary( a.full <- lm(BSAAM ~ . - Year, data=dat) )
# summary( a <- lm(BSAAM ~ APSAB, data=dat) )
# summary( a <- lm(BSAAM ~ APSAB + OPBPC, data=dat) )
# summary( a.full <- lm(BSAAM ~ . - Year, data=dat) )
# summary( a <- lm(BSAAM ~ OPBPC, data=dat) )
# summary( a <- lm(BSAAM ~ OPRC + OPBPC, data=dat) )
# summary( a.full <- lm(BSAAM ~ . - Year, data=dat) )
# summary( a <- lm(BSAAM ~ APSLAKE, data=dat) )
# summary( a <- lm(BSAAM ~ APSLAKE + OPBPC, data=dat) )
# summary( a.full <- lm(BSAAM ~ . - Year, data=dat) )
# summary( a <- lm(BSAAM ~ OPSLAKE + APSLAKE + OPBPC, data=dat) )
# summary( a <- lm(BSAAM ~ APSLAKE + OPBPC, data=dat) )
# summary( a <- lm(BSAAM ~ OPSLAKE + APSLAKE + OPBPC, data=dat) )
# summary( a <- lm(BSAAM ~ APSLAKE + OPBPC, data=dat) )
# summary( a <- lm(BSAAM ~ OPSLAKE + APSLAKE + OPBPC, data=dat) )
# plot(a)
# library(robustbase)
# ?lmrob
# a.r <- lmrob(BSAAM ~ OPSLAKE + APSLAKE + OPBPC, data=dat)
# plot(a.r)
# pairs(~. OPSLAKE + APSLAKE + OPBPC, data=dat)
# ?pairs
# pairs(~ OPSLAKE + APSLAKE + OPBPC, data=dat)
# pairs(~ BSAAM + OPSLAKE + APSLAKE + OPBPC, data=dat)
# summary( a <- lm(BSAAM ~ OPSLAKE + APSLAKE + OPBPC, data=dat) )
# n <- nrow(dat)
# n
# m1 <- lm(BSAAM ~ APSLAKE + OPBPC, data=dat)
# m2 <- lm(BSAAM ~ OPSLAKE + APSLAKE + OPBPC, data=dat)
# predict(m1, newdata=dat[21,])
# predict(m1)[21]
# # leave-one-out CV
# n <- nrow(dat)
# pe1 <- pe2 <- rep(0, n)
# for(j in 1:n) {
# pe1[j] <- predict(lm(BSAAM ~ APSLAKE + OPBPC, data=dat[-j,]), newdata=dat[j,])
# pe2[j] <- predict(lm(BSAAM ~ OPSLAKE + APSLAKE + OPBPC, data=dat[-j,]), newdata=dat[j,])
# }
# mean( (dat$BSAAM - pe1)^2 )
# mean( (dat$BSAAM - pe2)^2 )
# # leave-one-out CV
# n <- nrow(dat)
# pe1 <- pe2 <- pe3 <- rep(0, n)
# for(j in 1:n) {
# pe1[j] <- predict(lm(BSAAM ~ APSLAKE + OPBPC, data=dat[-j,]), newdata=dat[j,])
# pe2[j] <- predict(lm(BSAAM ~ OPSLAKE + APSLAKE + OPBPC, data=dat[-j,]), newdata=dat[j,])
# pe3[j] <- predict(lm(BSAAM ~ . - Year, data=dat[-j,]), newdata=dat[j,])
# }
# mean( (dat$BSAAM - pe1)^2 )
# mean( (dat$BSAAM - pe2)^2 )
# mean( (dat$BSAAM - pe3)^2 )
# pe1
# dat$BSAAM
# pe3
# plot(abs(dat$BSAAM - pe3), abs(dat$BSAAM - pe1))
# abline(0,1)
# summary( a.full <- lm(BSAAM ~ . - Year, data=dat) )
# summary( a <- lm(BSAAM ~ APSLAKE + OPBPC + APMAM, data=dat) )
# summary( a <- lm(BSAAM ~ OPSLAKE + APSLAKE + OPBPC + APMAM, data=dat) )
# # leave-one-out CV
# n <- nrow(dat)
# pe1 <- pe2 <- pe3 <- rep(0, n)
# for(j in 1:n) {
# pe1[j] <- predict(lm(BSAAM ~ APSLAKE + OPBPC + APMAM, data=dat[-j,]), newdata=dat[j,])
# pe2[j] <- predict(lm(BSAAM ~ OPSLAKE + APSLAKE + OPBPC + APMAM, data=dat[-j,]), newdata=dat[j,])
# pe3[j] <- predict(lm(BSAAM ~ . - Year, data=dat[-j,]), newdata=dat[j,])
# }
# mean( (dat$BSAAM - pe1)^2 )
# mean( (dat$BSAAM - pe2)^2 )
# mean( (dat$BSAAM - pe3)^2 )
# 152512247 - 163556451
# 152512247 / 163556451
# m1 <- lm(BSAAM ~ APSLAKE + OPBPC, data=dat)
# summary(m1)
# m2 <- lm(BSAAM ~ APMAM+ APSLAKE + OPBPC, data=dat)
# summary(m2)
# # leave-one-out CV
# n <- nrow(dat)
# pe1 <- pe2 <- pe3 <- rep(0, n)
# for(j in 1:n) {
# pe1[j] <- predict(lm(BSAAM ~ APSLAKE + OPBPC, data=dat[-j,]), newdata=dat[j,])
# pe2[j] <- predict(lm(BSAAM ~ APSLAKE + OPBPC + APMAM, data=dat[-j,]), newdata=dat[j,])
# pe3[j] <- predict(lm(BSAAM ~ . - Year, data=dat[-j,]), newdata=dat[j,])
# }
# mean( (dat$BSAAM - pe1)^2 )
# mean( (dat$BSAAM - pe2)^2 )
# mean( (dat$BSAAM - pe3)^2 )
# m1 <- lm(BSAAM ~ APSLAKE + OPBPC, data=dat)
# m2 <- lm(BSAAM ~ APMAM+ APSLAKE + OPBPC, data=dat)
# summary(m1)
# summary(m2)
# m1 <- lm(BSAAM ~ APSLAKE + OPBPC, data=dat)
# m2 <- lm(BSAAM ~ APMAM + APSLAKE + OPBPC, data=dat)
# mean( resid(m1)^2 )
# mean( resid(m2)^2 )
# sqrt( mean( resid(m1)^2 ) )
# sqrt( mean( resid(m2)^2 ) )
# mean(resid(m1)^2) / mean(resid(m2)^2)
# # leave-one-out CV
# n <- nrow(dat)
# pe1 <- pe2 <- pe3 <- rep(0, n)
# for(j in 1:n) {
# pe1[j] <- predict(lm(BSAAM ~ APSLAKE + OPBPC, data=dat[-j,]), newdata=dat[j,])
# pe2[j] <- predict(lm(BSAAM ~ APSLAKE + OPBPC + APMAM, data=dat[-j,]), newdata=dat[j,])
# }
# mean( (dat$BSAAM - pe1)^2 )
# mean( (dat$BSAAM - pe2)^2 )
# dat
# dat <- read.table('dwp-data.txt', header=TRUE)
# dat2 <- dat
# dat$BSAAM <- dat$BSAAM / 100
# summary( a.full <- lm(BSAAM ~ . - Year, data=dat) )
# pairs(~ BSAAM + OPSLAKE + APSLAKE + OPBPC, data=dat)
# # leave-one-out CV
# n <- nrow(dat)
# pe1 <- pe2 <- pe3 <- rep(0, n)
# for(j in 1:n) {
# pe1[j] <- predict(lm(BSAAM ~ APSLAKE + OPBPC, data=dat[-j,]), newdata=dat[j,])
# pe2[j] <- predict(lm(BSAAM ~ APSLAKE + OPBPC + APMAM, data=dat[-j,]), newdata=dat[j,])
# }
# mean( (dat$BSAAM - pe1)^2 )
# mean( (dat$BSAAM - pe2)^2 )
# summary( m1 <- lm(BSAAM ~ APSLAKE + OPBPC, data=dat) )
# summary( m2 <- lm(BSAAM ~ APMAM + APSLAKE + OPBPC, data=dat) )
# sum(resid(m1)^2)
# sum(resid(m2)^2)
