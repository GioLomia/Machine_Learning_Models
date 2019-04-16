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

data_raw<-Telco_Churn%>%select(Churn,everything(),-customerID)


plot_missing(data_raw)
data_raw<-na.omit(data_raw)
data_raw[,"SeniorCitizen"]<-as.factor(data_raw[,"SeniorCitizen"])
glimpse(data_raw)
set.seed(1998)

train_test_split<-initial_split(data_raw, prop=0.4)

train_tbl<-training(train_test_split)
test_tbl<-testing(train_test_split)

rec_obj <- recipe(Churn ~ ., data = train_tbl) %>%
  step_center(all_numeric(), -all_outcomes()) %>%
  step_scale(all_numeric(), -all_outcomes()) %>%
  step_BoxCox(all_numeric(),-all_outcomes())%>%
  step_dummy(all_nominal(),-all_outcomes(), one_hot = TRUE)%>%
  prep(data = train_tbl)

rec_obj

train_clean <- bake(rec_obj, new_data = train_tbl)

glimpse(train_clean)

#Step 5 - Prepare data for lasso model
x_train <- model.matrix(Churn ~ ., train_clean)[,-1]
y_train <- as.factor(train_clean$Churn)

glimpse(x_train)
glimpse(y_train)

grid <- 10^seq(8,-3, by = -0.1)

churn.lasso <- cv.glmnet(x_train, y_train, family = "binomial", alpha = 0.5, lambda = grid)

plot(churn.lasso)

bestlam <- churn.lasso$lambda.min
bestlam

churn.lasso.coef=predict(churn.lasso, type="coefficients", s = bestlam)[1:47,]
churn.lasso.coef[churn.lasso.coef!=0]


reduced_data_raw<-Telco_Churn%>%select(
  Churn,
  tenure,
  TotalCharges,
  gender,
  SeniorCitizen,
  Partner,
  Dependents,
  MultipleLines,
  InternetService,
  OnlineSecurity,
  OnlineBackup,
  DeviceProtection,
  TechSupport,
  StreamingTV,
  StreamingMovies,
  Contract,
  PaperlessBilling,
  PaymentMethod)

reduced_data_raw<-na.omit(data_raw)
reduced_data_raw[,"SeniorCitizen"]<-as.factor(data_raw[,"SeniorCitizen"])

glimpse(reduced_data_raw)
set.seed(1998)

red_train_test_split<-initial_split(reduced_data_raw, prop=0.4)

red_train_tbl<-training(red_train_test_split)
red_test_tbl<-testing(red_train_test_split)

glimpse(red_train_tbl)

red_rec_obj <- recipe(Churn ~ ., data = red_train_tbl) %>%
  step_center(all_numeric(), -all_outcomes()) %>%
  step_scale(all_numeric(), -all_outcomes()) %>%
  step_BoxCox(all_numeric(),-all_outcomes())%>%
  step_dummy(all_nominal(),-all_outcomes(), one_hot = TRUE)%>%
  prep(data = red_train_tbl)

red_rec_obj

red_train_ready<-bake(red_rec_obj,red_train_tbl)
red_test_ready<-bake(red_rec_obj,red_test_tbl)
glimpse(red_train_ready)


ctrl <- trainControl(method = "repeatedcv", number=10, repeats=5, 
                     summaryFunction = twoClassSummary, classProbs = TRUE,
                     savePredictions = "final")

TC.log <- train(Churn ~ ., data = red_train_ready,
                       method = "glm", family = "binomial", metric = "ROC",
                       trControl = ctrl)


TC.log
print(TC.redmod.log)


TC.logpred = predict(TC.log, newdata=red_test_ready)

TC.logpred
confusionMatrix(data=TC.logpred, red_test_ready$Churn,positive = "Yes")




lda.fit = train(Churn ~ ., data=red_train_ready, method="lda",
                trControl = trainControl(method = "cv"))

plot(lda.fit)

AR.ldapred = predict(lda.fit, newdata=red_test_ready)

confusionMatrix(data=AR.ldapred, red_test_ready$Churn,positive = "Yes")


##############QDA############
qda.fit = train(Churn ~ ., data=red_train_ready, method="qda",
                trControl = trainControl(method = "cv"))

qda.qdapred = predict(qda.fit, newdata=red_test_ready)

# Step 11 - Generate a confusion matrix and summary report
confusionMatrix(data=AR.qdapred, red_test_ready$Churn, positive = "Yes")
#############RF to Find Best Trees#############
TC.rf.try <- randomForest(Churn ~ ., data = red_train_ready)

plot(TC.rf.try$importance)
plot(TC.rf.try)


################RANDOM FOREST###########################
ctrl <- trainControl(method = "repeatedcv", number=, repeats=2, verboseIter = TRUE,
                     summaryFunction = twoClassSummary, classProbs = TRUE,
                     savePredictions = "final")

TC.rf <- train(Churn ~ ., data = red_train_ready,
                method = "rf", family = "binomial", metric = "ROC",ntree=500,
                trControl = ctrl,tunelength=4)

imp<-varImp(TC.rf,scale = FALSE)
plot(imp)
plot(TC.rf)

AR.rfpred = predict(TC.rf, newdata=red_test_ready)

# Step 11 - Generate a confusion matrix and summary report
confusionMatrix(data=AR.rfpred, red_test_ready$Churn,positive = "Yes")





####################GBM MODEL##################

control <- trainControl(method = "repeatedcv", number = 3, repeats = 2, classProbs = TRUE, summaryFunction = twoClassSummary)

TC.gbm <- train(Churn ~., method = "gbm",data = red_train_ready, trControl = control,ntree=500, metric = "ROC", verbose = TRUE,tuneLength = 4)

y_2<-sum(red_train_ready$Contract_Two.year)
m_m<-sum(red_train_ready$Contract_Month.to.month)
y_2/m_m
ratio<-table(y_2,m_m)

barplot(ratio, 
        xlab=c("2 Year","Month to Month"))

TC.gbm
varImp(TC.gbm,scale = F)

AR.gbmpred = predict(TC.gbm, newdata=red_test_ready)

# Step 11 - Generate a confusion matrix and summary report
confusionMatrix(data=AR.gbmpred, red_test_ready$Churn,positive = "Yes")
