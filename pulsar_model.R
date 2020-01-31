library("randomGLM")
library("randomForest")
library("extraTrees")
library("e1071")

set.seed(12345)
pulsar_stars <- read.csv("Data/PulsarStar/pulsar_stars.csv")
X = pulsar_stars[,1:8]
y = pulsar_stars[, 9]

n_train = 13000
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
                    
