library(DataExplorer)
library(recipes)
library(MASS)
library(glmnet)
library(tidyverse)
library(caret)

# Import audit_risk.csv now



# Step 0 - Instantiate dataset as data_raw
data_raw <- audit_risk
data_raw[,"Risk"]<-as.factor(data_raw[,"Risk"])
data_raw[,"History"]<-as.factor(data_raw[,"History"])
#Step 1 - Inspect the dataset [Complete this]
glimpse(data_raw)


#Step 2 -  Check for missing data [Complete this]
plot_missing(data_raw)


#Step 3 -  Move response variable to first column
require(dplyr)
data_adj <- data_raw %>%
  dplyr::select(Risk, everything())

data_omit<-na.omit(data_adj)

#Step 4 - Recipe Here
cornbread <- recipe(Risk ~ ., data = data_omit) %>%
  step_center(all_numeric(), -all_outcomes()) %>%
  step_scale(all_numeric(), -all_outcomes()) %>%
  #insert normalization step here %>%
  step_BoxCox(all_numeric(),-all_outcomes())%>%
  step_dummy(all_nominal(),-all_outcomes())%>%
  step_nzv(all_predictors(), -all_outcomes()) %>%
  prep(data = data_omit)

cornbread

data_clean <- bake(cornbread, new_data = data_omit)


#Step 5 - Prepare data for lasso model
x_train <- model.matrix(Risk ~ ., data_omit)[,-1]
y_train <- as.factor(data_omit$Risk)


#Step 6 - Cross-validated LASSO Model
grid <- 10^seq(8,-2, by = -0.1)

AR.lasso <- cv.glmnet(x_train, y_train, family = "binomial", alpha = 1, lambda = grid)

plot(AR.lasso)

bestlam <- AR.lasso$lambda.min
bestlam

AR.lasso.coef=predict(AR.lasso, type="coefficients", s = bestlam)[1:68,]
AR.lasso.coef[AR.lasso.coef>10^-6]


# Revised dataset with reduced predictors from lasso

# Choose from the following:

# Sector_score   
# LOCATION_ID   !
# PARA_A      
# Score_A   !
# Risk_A     !  
# PARA_B       
# Score_B       
# Risk_B         
# numbers        
# Score_B.1      
# Risk_C        
# Money_Value    
# Score_MV    !   
# Risk_D         
# District_Loss  !
# PROB           
# RiSk_E         !
# History       
# Prob         
# Risk_F         
# Score          !
# Inherent_Risk  
# CONTROL_RISK   !
# Detection_Risk 

# Step 7 - Rebuild data frame with reduced features
AR.redmod <- data_raw %>%
  dplyr::select(Risk,LOCATION_ID, Score_A, Risk_A, Score_MV,District_Loss,RiSk_E,Score,CONTROL_RISK)




# Training/test split for final model
set.seed(0134)
samp <- sample(nrow(AR.redmod), .8*nrow(AR.redmod))

train <- AR.redmod[samp,]
test <- AR.redmod[-samp,]


# Recipe for final model here
spoonbread <- recipe(Risk ~ ., data = train) %>%
  # imputation step, if that's your strategy
  step_center(all_numeric(), -all_outcomes()) %>%
  step_scale(all_numeric(), -all_outcomes()) %>%
  step_BoxCox(all_numeric(),-all_outcomes())%>%
  #insert normalization here
  step_dummy(all_nominal(),-all_outcomes())%>%
  step_nzv(all_predictors(), -all_outcomes()) %>%
  prep(data = train)

# Print the recipe object


# Bake the data
train_clean2 <- bake(spoonbread, new_data = train)
test_clean  <- bake(spoonbread, new_data = test)


# Step 8 -  Build and run a cross-validated logistic regression model using caret package

ctrl <- trainControl(method = "repeatedcv", number=10, repeats=5, 
                     summaryFunction = twoClassSummary, classProbs = TRUE,
                     savePredictions = "final")

AR.redmod.log <- train(Risk ~ ., data = train_clean2,
                       method = "glm", family = "binomial", metric = "ROC",
                       trControl = ctrl)


# Step 9 - Generate logistic model output
print(AR.redmod.log)


# Step 10 - Create vector of class predictions
AR.logpred = predict(AR.redmod.log, newdata=test_clean)

AR.logpred
# Step 11 - Generate a confusion matrix and summary report
confusionMatrix(data=AR.logpred, test_clean$Risk)


# Step 12 - Repeat with cross validated LDA model here


# Step 11 - Generate a confusion matrix and summary report
lda.fit = train(Risk ~ ., data=train_clean2, method="lda",
                trControl = trainControl(method = "cv"))

AR.ldapred = predict(lda.fit, newdata=test_clean)

# Step 11 - Generate a confusion matrix and summary report
confusionMatrix(data=AR.ldapred, test_clean$Risk)

##################################
