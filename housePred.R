library("randomGLM")
library("randomForest")
library("extraTrees")
library("e1071")

set.seed(12345)
train <- read.csv("Data/HousePred/train.csv")
test <-  read.csv("Data/HousePred/test.csv")

cols_to_keep = apply(apply(train, 2, is.na), 2, sum) == 0
X_train = train[, cols_to_keep]
X_train = X_train[, 2:(ncol(X_train) - 1)]
y_train = train$SalePrice

X_test = test[, cols_to_keep[-81]]
X_test = X_test[, 2:61]

convert_to_factor = c("YrSold", "YearRemodAdd", "YearBuilt")
for (c in convert_to_factor){
  X_train[, c] = as.factor(X_train[, c])
  X_test[, c] = as.factor(X_test[, c])
}

for (i in 1:ncol(X_train)){
  X_train[, i] = as.numeric(X_train[, i])
  X_test[, i] = as.numeric(X_test[, i])
  X_test[is.na(X_test[, i]), i] = floor(mean(X_train[, i]))
}


bags = 100
model_rglm = randomGLM(x=X_train, y=y_train, xtest = X_test, nBags = bags)
model_rf_bigmtry = randomForest(x = X_train, y = y_train,ntree = 500, mtry = ncol(X_train))
model_rf = randomForest(x = X_train, y = y_train, ntree = 500)
model_xtree = extraTrees(x = X_train, y = y_train, ntree = 500)
model_svm = svm(x = X_train, y = y_train)

pred_rglm = data.frame(test$Id, model_rglm$predictedTest)
pred_rf_bigmtry = data.frame(test$Id, predict(model_rf_bigmtry, newdata=X_test))
pred_rf = data.frame(test$Id, predict(model_rf, newdata=X_test))
pred_xtree = data.frame(test$Id, predict(model_xtree, newdata=X_test))
pred_svm = data.frame(test$Id, predict(model_svm, newdata=X_test))

col_name = c("Id", "SalePrice")
colnames(pred_rglm) = col_name
colnames(pred_rf_bigmtry) = col_name
colnames(pred_rf) = col_name
colnames(pred_xtree) = col_name
colnames(pred_svm) = col_name

write.csv(pred_rglm, file="Data/HousePred/Submission/sub_rglm.csv", row.names = FALSE)
write.csv(pred_rf_bigmtry, file="Data/HousePred/Submission/sub_rf_bigmtry.csv", row.names = FALSE)
write.csv(pred_rf, file="Data/HousePred/Submission/sub_rf.csv", row.names = FALSE)
write.csv(pred_xtree, file="Data/HousePred/Submission/sub_xtree.csv", row.names = FALSE)
write.csv(pred_svm, file="Data/HousePred/Submission/sub_svm.csv", row.names = FALSE)


