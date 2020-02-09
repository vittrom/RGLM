library("randomGLM")
library("randomForest")
library("extraTrees")
library("e1071")

set.seed(12345)
data <- read.csv("Data/UKHousing/price_paid_records.csv", nrows = 30000)

cols_to_keep = c(4, 5, 6, 9)
X = data[, cols_to_keep]
y = data[, 2]

for (i in 1:ncol(X)){
  X[, i] = as.numeric(X[, i])
}

n_train = 20000

reps = 10
accuracy_rglm = rep(0, reps)
accuracy_rf_bigmtry = rep(0, reps)
accuracy_rf = rep(0, reps)
accuracy_xtree = rep(0, reps)
accuracy_svm = rep(0, reps)

for (i in 1:reps){
  idx = sample(length(y), size = n_train, replace = FALSE)
  
  X_train = X[idx, ]
  y_train = y[idx]
  X_test = X[-idx, ]
  y_test = y[-idx]
  
  bags = 100
  model_rglm = randomGLM(x=X_train, y=y_train, xtest = X_test, nBags = bags)
  model_rf_bigmtry = randomForest(x = X_train, y = y_train, ntree = 500, mtry = ncol(X_train))
  model_rf = randomForest(x = X_train, y = y_train, ntree = 500)
  model_xtree = extraTrees(x = X_train, y = y_train, ntree = 500)
  model_svm = svm(x = X_train, y = y_train)
  
  accuracy_rglm[i] = sqrt(mean((log(model_rglm$predictedTest) - log(y_test))^2))
  accuracy_rf_bigmtry[i] = sqrt(mean((log(predict(model_rf_bigmtry, newdata=X_test))- log(y_test))^2))
  accuracy_rf[i] = sqrt(mean((log(predict(model_rf, newdata=X_test))- log(y_test))^2))
  accuracy_xtree[i] = sqrt(mean((log(predict(model_xtree, newdata=X_test))- log(y_test))^2))
  accuracy_svm[i] = sqrt(mean((log(predict(model_svm, newdata=X_test))- log(y_test))^2))
}

mean(accuracy_rglm)
mean(accuracy_rf_bigmtry)
mean(accuracy_rf)
mean(accuracy_xtree)
mean(accuracy_svm)





