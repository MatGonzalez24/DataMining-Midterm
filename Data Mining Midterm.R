library(caret)
library(glmnet)
library(e1071)
library(naivebayes)
library(MASS)

getwd()
# Change this to the path where you have both dftrain.csv and dftest.csv files.
setwd("/Users/matiasgonzalez/desktop")
traindf = read.csv("dftrain.csv")
testdf = read.csv("dftest.csv")

set.seed(23)

# Columns x14, x15 and x19 have string values of type "N" or "Y" so we change them to numbers. N = 0, Y = 1
traindf$x14 <- ifelse(traindf$x14 == "N", 0, 1)
traindf$x15 <- ifelse(traindf$x15 == "N", 0, 1)
traindf$x19 <- ifelse(traindf$x19 == "N", 0, 1)

# There are also some NA values in train. We will deal with them by removing them
traindf <- na.omit(traindf)

myControl <- trainControl( method="cv", number = 10, savePredictions="final", allowParallel=T)

grid <- expand.grid(alpha = seq(0,1,by = 0.02), lambda = seq(0, 100, length = 400))

netmodel <- train(y~., data = traindf, method = "glmnet", trControl = myControl, tuneGrid = grid)

print(netmodel)
# Set up the tuning grid with 
fitGrid <- expand.grid(alpha = 0.78, lambda = 3.759398)

fit = train(y~., data=traindf, method="glmnet", trControl = myControl, tuneGrid=fitGrid)

#Now we need to do the same conversions we did to the traindf in order to predict values
#First change N = 0 and Y = 1.
testdf$x14 <- ifelse(testdf$x14 == "N", 0, 1)
testdf$x15 <- ifelse(testdf$x15 == "N", 0, 1)
testdf$x19 <- ifelse(testdf$x19 == "N", 0, 1)

#Then remove NAs
testdf <- na.omit(testdf)

p = predict(fit, testdf[, -ncol(testdf)])
test_mse <- mean((testdf$y - p)^2)
print(test_mse)

#Train a model that uses linear regression to predict
linearfit <- lm(y ~ ., data = traindf)
linearp = predict(linearfit, testdf[, -ncol(testdf)])
linear_test_mse <- mean((testdf$y - linearp)^2)
print(linear_test_mse)

#--------------------- Part 2 ------------------------
traindf$y <- ifelse(traindf$y >= 750, "H", "L")

# logit model
logit_control <- trainControl( method="cv", number = 10,
                                    savePredictions="final", allowParallel=T)
logit_model <- train(y~., data = traindf, method = "glm", trControl = logit_control, family = "binomial")

# elastic net model
netgrid <- expand.grid(alpha = seq(0,1,by = 0.02), lambda = seq(0, 100, length = 50))
bin_net_fit = train(y~., data=traindf, method="glmnet", family="binomial",
            trControl = myControl, tuneGrid = netgrid)
print(bin_net_fit)

final_grid <- expand.grid(alpha = 0, lambda = 0)
final_net_fit = train(y~., data=traindf, method="glmnet", family="binomial",
                    trControl = myControl, tuneGrid = final_grid)

# lda
lda_control <- trainControl( method="cv", number = 10,
                                    savePredictions="final", allowParallel=T)

flda <- train(y~., data = traindf, method = "lda",
              trControl = lda_control)

# qda
qda_control <- trainControl( method="cv", number = 10,
                                    savePredictions="final", allowParallel=T)

fqda <- train(y~., data = traindf, method = "qda",
              trControl = qda_control)

# naivebayes
naivebayes_control <- trainControl( method="cv", number = 10,
                                 savePredictions="final", allowParallel=T)

nb_model <- train(y~., data = traindf, method = "naive_bayes",
                   trControl = naivebayes_control)


# knn model
bin_knn_control <- trainControl( method="cv", number = 10,
                                 savePredictions="final", allowParallel=T)

knn_model <- train(y~., data = traindf, method = "knn",
                   trControl = bin_knn_control, tuneLength = 10)

print(knn_model)

final_knn_model = train(y~., data=traindf, method="knn", trControl = bin_knn_control,
            tuneGrid = data.frame(k = 5))



testdf$y <- ifelse(testdf$y >= 750, "H", "L")

logit_p = predict(logit_model, testdf[, -ncol(testdf)])
net_p = predict(final_net_fit, testdf[, -ncol(testdf)])
lda_p = predict(flda, testdf[, -ncol(testdf)])
qda_p = predict(fqda, testdf[, -ncol(testdf)])
nb_p = predict(nb_model, testdf[, -ncol(testdf)])
knn_p = predict(final_knn_model, testdf[, -ncol(testdf)])

acc = function(x,y) mean(x==y)

acc(testdf$y, logit_p)
acc(testdf$y, net_p)
acc(testdf$y, lda_p)
acc(testdf$y, qda_p)
acc(testdf$y, nb_p)
acc(testdf$y, knn_p)










