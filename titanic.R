library("randomGLM")
library("randomForest")
library("extraTrees")
library("e1071")

set.seed(12345)
train <- read.csv("Data/Titanic/train.csv")
test <-  read.csv("Data/Titanic/test.csv")

cols_to_keep = c(3, 5, 6, 7, 8, 10, 12)
X_train = train[, cols_to_keep]
y_train = train$Survived

X_test = test[, cols_to_keep - 1]

# Transform columns dataset
X_train[, 1] = as.factor(X_train[, 1])
X_train[, 4] = as.factor(X_train[, 4])
X_train[, 5] = as.factor(X_train[, 5])

X_test[, 1] = as.factor(X_test[, 1])
X_test[, 4] = as.factor(X_test[, 4])
X_test[, 5] = as.factor(X_test[, 5])

X_train = cbind(X_train, y_train)
X_train = X_train[complete.cases(X_train), ]
y_train = X_train[, ncol(X_train)]
X_train = X_train[,-8]

for (i in 1:ncol(X_train)){
  X_train[, i] = as.numeric(X_train[, i])
  X_test[, i] = as.numeric(X_test[, i])
}

mean_age_train = mean(X_train$Age)
mean_fare_train = mean(X_train$Fare)
X_test[is.na(X_test$Age), 3] = mean_age_train
X_test[is.na(X_test$Fare), 6] = mean_fare_train


bags = 100
model_rglm = randomGLM(x=X_train, y=y_train, xtest = X_test, nBags = bags)
model_rf_bigmtry = randomForest(x = X_train, y = as.factor(y_train),ntree = 500, mtry = ncol(X_train))
model_rf = randomForest(x = X_train, y = as.factor(y_train), ntree = 500)
model_xtree = extraTrees(x = X_train, y = as.factor(y_train), ntree = 500)
model_svm = svm(x = X_train, y = as.factor(y_train))

pred_rglm = data.frame(test$PassengerId, model_rglm$predictedTest)
pred_rf_bigmtry = data.frame(test$PassengerId, predict(model_rf_bigmtry, newdata=X_test))
pred_rf = data.frame(test$PassengerId, predict(model_rf, newdata=X_test))
pred_xtree = data.frame(test$PassengerId, predict(model_xtree, newdata=X_test))
pred_svm = data.frame(test$PassengerId, predict(model_svm, newdata=X_test))

col_name = c("PassengerId", "Survived")
colnames(pred_rglm) = col_name
colnames(pred_rf_bigmtry) = col_name
colnames(pred_rf) = col_name
colnames(pred_xtree) = col_name
colnames(pred_svm) = col_name

write.csv(pred_rglm, file="Data/Titanic/Submission/sub_rglm.csv", row.names = FALSE)
write.csv(pred_rf_bigmtry, file="Data/Titanic/Submission/sub_rf_bigmtry.csv", row.names = FALSE)
write.csv(pred_rf, file="Data/Titanic/Submission/sub_rf.csv", row.names = FALSE)
write.csv(pred_xtree, file="Data/Titanic/Submission/sub_xtree.csv", row.names = FALSE)
write.csv(pred_svm, file="Data/Titanic/Submission/sub_svm.csv", row.names = FALSE)


