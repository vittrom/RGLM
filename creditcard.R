library("randomGLM")
library("randomForest")
library("extraTrees")
library("e1071")

set.seed(12345)
creditcard <- read.csv("Data/CreditCard/creditcard.csv")
X = creditcard[,2:30]
y = creditcard[, 31]

X_1 = X[y==1,]
y_1 = y[y==1]
X_0 = X[y==0,]
y_0 = y[y==0]

n_train_x = 3000

reps=10
accuracy_rglm = rep(0, reps)
accuracy_rf_bigmtry = rep(0, reps)
accuracy_rf = rep(0, reps)
accuracy_xtree = rep(0, reps)
accuracy_svm = rep(0, reps)

for (i in 1:reps){
  idx_0 = sample(length(y_0), size = n_train_x, replace = FALSE)
  idx_1 = sample(length(y_1), size = 300, replace = FALSE)
  
  X_train = rbind(X_0[idx_0, ], X_1[idx_1, ])
  y_train = c(y_0[idx_0], y_1[idx_1])
  
  X_test = X_0[-idx_0, ]
  y_test = y[-idx_0]
  
  idx_test_0 = sample(length(y_test), size = n_train_x, replace = FALSE)
  X_test = rbind(X_test[idx_test_0,], X_1[-idx_1,])
  y_test = c(y_test[idx_test_0], y_1[-idx_1])
  
  na_rows = which(is.na(X_test[, 1]))
  X_test = X_test[-na_rows,]
  y_test = y_test[-na_rows]
  
  bags = 100
  model_rglm = randomGLM(x=X_train, y=y_train, xtest = X_test, nBags = bags)
  model_rf_bigmtry = randomForest(x = X_train, y = as.factor(y_train),ntree = 500, mtry = ncol(X_train))
  model_rf = randomForest(x = X_train, y = as.factor(y_train), ntree = 500)
  model_xtree = extraTrees(x = X_train, y = as.factor(y_train), ntree = 500)
  model_svm = svm(x = X_train, y = as.factor(y_train))
  
  accuracy_rglm[i] = 1 - sum(abs(model_rglm$predictedTest - y_test))/length(y_test)
  accuracy_rf_bigmtry[i] = 1 - sum(abs(as.numeric(predict(model_rf_bigmtry, newdata=X_test)) - 1 - y_test))/length(y_test)
  accuracy_rf[i] = 1 - sum(abs(as.numeric(predict(model_rf, newdata=X_test)) - 1 - y_test))/length(y_test)
  accuracy_xtree[i] = 1 - sum(abs(as.numeric(predict(model_xtree, newdata=X_test)) - 1 - y_test))/length(y_test)
  accuracy_svm[i] =  1 - sum(abs(as.numeric(predict(model_svm, newdata=X_test)) - 1 - y_test))/length(y_test)
}

mean(accuracy_rglm)
mean(accuracy_rf_bigmtry)
mean(accuracy_rf)
mean(accuracy_xtree)
mean(accuracy_svm)


