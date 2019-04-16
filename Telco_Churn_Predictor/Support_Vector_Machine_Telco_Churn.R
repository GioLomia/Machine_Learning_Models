library (xgboost)
library (magrittr)
library (dplyr)
library (Matrix)
library(tidyselect)
library(keras)
library(tidyverse)
library(recipes)
library(ROCR)
library(mlbench)
library(DataExplorer)
library(tidyverse)
library(polycor)
library(car)
library(broom)
library(dplyr)
library(rsample)
library(class)
library(caret)
library(ROSE)
library(randomForest)
library(glmnet)
library(gbm)
library(e1071)
library(kernlab)

data_raw<-HR_Churn%>%select(Gone,everything())

plot_missing(data_raw)
data_raw<-na.omit(data_raw)
data_raw[,"JobLevel"]<-as.factor(data_raw[,"JobLevel"])
data_raw[,"JobInvolvement"]<-as.factor(data_raw[,"JobInvolvement"])
data_raw[,"EnvironmentSatisfaction"]<-as.factor(data_raw[,"EnvironmentSatisfaction"])
data_raw[,"JobSatisfaction"]<-as.factor(data_raw[,"JobSatisfaction"])
data_raw[,"PerformanceRating"]<-as.factor(data_raw[,"PerformanceRating"])
data_raw[,"RelationshipSatisfaction"]<-as.factor(data_raw[,"RelationshipSatisfaction"])
data_raw[,"StockOptionLevel"]<-as.factor(data_raw[,"StockOptionLevel"])
data_raw[,"WorkLifeBalance"]<-as.factor(data_raw[,"WorkLifeBalance"])

glimpse(data_raw)

set.seed(1998)
train_test_split<-initial_split(data_raw, prop=0.8)

train_tbl<-training(train_test_split)
test_tbl<-testing(train_test_split)

rec_obj <- recipe(Gone ~ ., data = train_tbl) %>%
  step_center(all_numeric(), -all_outcomes()) %>%
  step_scale(all_numeric(), -all_outcomes()) %>%
  #step_BoxCox(all_numeric(),-all_outcomes())%>%
  step_dummy(all_nominal(),-all_outcomes(), one_hot = TRUE)%>%
  prep(data = train_tbl)

rec_obj

train_clean <- bake(rec_obj, new_data = train_tbl)
test_clean <- bake(rec_obj, new_data = test_tbl)
glimpse(train_clean)


ctrl <- trainControl(method = "repeatedcv", number=10, repeats=5, 
                     summaryFunction = twoClassSummary, classProbs = TRUE,
                     savePredictions = "final")

TC.log <- train(Gone ~ ., data = train_clean,
                method = "glm", family = "binomial", metric = "ROC",
                trControl = ctrl)

TC.log

TC.logpred = predict(TC.log, newdata=test_clean)

TC.logpred
confusionMatrix(data=TC.logpred, test_clean$Gone,positive = "Yes")

svm.ctrl<-trainControl(method="repeatedcv",
                       number=5,
                       repeats=2,
                       classProbs=T,
                       verboseIter=T,
                       savePredictions="final",
                       summaryFunction=twoClassSummary)

HR.svm.linear<-train(Gone~.,
                     data=train_clean,method='svmLinear',
                     trControl=svm.ctrl,
                     metric="ROC",tuneLength=3)

plot(HR.svm.linear)

svm.logpred = predict(HR.svm.linear, newdata=test_clean)

svm.logpred
confusionMatrix(data=svm.logpred, test_clean$Gone,positive = "Yes")


##########NONLIN#############
svm.ctrl<-trainControl(method="repeatedcv",
                       number=5,
                       repeats=2,
                       classProbs=T,
                       verboseIter=T,
                       savePredictions="final",
                       summaryFunction=twoClassSummary)

HR.svm.Rad<-train(Gone~.,
                     data=train_clean,method='svmRadial',
                     trControl=svm.ctrl,
                     metric="ROC",tuneLength=3)

plot(HR.svm.Rad)

svm.logpred = predict(HR.svm.Rad, newdata=test_clean)

svm.logpred
confusionMatrix(data=svm.logpred, test_clean$Gone,positive = "Yes")

##########POLY#############
svm.ctrl<-trainControl(method="repeatedcv",
                       number=5,
                       repeats=2,
                       classProbs=T,
                       verboseIter=T,
                       savePredictions="final",
                       summaryFunction=twoClassSummary)

HR.svm.Poly<-train(Gone~.,
                  data=train_clean,method='svmPoly',
                  trControl=svm.ctrl,
                  metric="ROC",tuneLength=3)

plot(HR.svm.Poly)

svm.logpred = predict(HR.svm.Poly, newdata=test_clean)

svm.logpred
confusionMatrix(data=svm.logpred, test_clean$Gone,positive = "Yes")
