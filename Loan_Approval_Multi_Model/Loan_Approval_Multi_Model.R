#Step 0: Import the data "train.csv" and run the code through line 12
library(recipes)
library(rsample)
library(car)
library(DataExplorer)
library(polycor)
library(tidyverse)
library(ROCR)
library(broom)

# load the data into an object
data_raw <- train


# STEP 1: check the dimension and structure of the data
dim(data_raw)
str(data_raw)
glimpse(data_raw)

# STEP 2: check for missing values by row and column in the data
plot_missing(data_raw)

# STEP 3: clean up training data by removing Loan ID and move the response variable to the first column
data_tbl <- data_raw %>% 
  select(-Loan_ID) %>%
  select(Loan_Status, everything())


# STEP 4: recognize Loan_status as a factor instead of text
data_tbl$Loan_Status <- ifelse(pull(data_tbl, Loan_Status) == "Y","Y","N")


# STEP 5: check the data for new structure
glimpse(data_tbl)


# STEP 6: create training/test split and check dimensions
set.seed(854)
train_test_split <- initial_split(data_tbl, prop = 0.8)

train <- training(train_test_split)
test <- testing(train_test_split)


# STEP 7: Check the dimension of the training and test data here
dim(train)
dim(test)


# STEP 8: Create recipe in correct order; 
# new feature-engineered variables are included code complete

rec_obj <- recipe(hf_score ~ ., data = train) %>%
  step_dummy(all_nominal(), -all_outcomes(),one_hot = FALSE)%>%
  step_bagimpute(all_predictors(),-all_outcomes())%>% 
  step_mutate(To = ApplicantIncome + CoapplicantIncome) %>%
  step_mutate(Monthly_Ratio = (LoanAmount/Loan_Amount_Term)/(Total_Income/12)) %>%
  step_log(Total_Income) %>%
  prep(data = train)

rec_obj

train_clean <- bake(rec_obj, new_data = train)
test_clean  <- bake(rec_obj, new_data = test)


# STEP 9: Remove any remaining missing rows and check the dimensions one last time
train_clean<-na.omit(train_clean)
test_clean<-na.omit(test_clean)
dim(train_clean)
dim(test_clean)

# STEP 10: Generate a logistic model
log.model<-glm(Loan_Status ~ ., data=train_clean, family=binomial(link="logit"),maxit=100)

summary(log.model)
#ploting the top 4 values 



# STEP 11: Locate four extreme observations and put into exam report
plot(log.model, which = 4, id.n = 4)

# STEP 12: Check for outliers; remove any outliers and rerun the model
outlierTest(log.model)

########################################
anova(log.model,test="Chisq")
vif(log.model) #delete predictors larger than 10
durbinWatsonTest(log.model)#if p value > 0.05 we fail to reject the null hyp which means we are good


# Rerun the logistic model with outliers removed (if necessary)


# STEP 13: Predict the probability of Loan Status against the test data; examine for imbalanced predictions
probabilities <- predict(log.model, newdata = test_clean, type = "response")
predicted.classes <- ifelse(probabilities > 0.5, "Y", "N")

table(predicted.classes, test_clean$Loan_Status)
mean(predicted.classes == test_clean$Loan_Status)


# STEP 14: Create the ROC chart
ROC.pred <- prediction(probabilities, test_clean$Loan_Status)

ROC.prf <- performance(ROC.pred, measure = "tnr", x.measure = "fnr")

plot(ROC.prf)

auc <- performance(ROC.pred, measure = "auc")
auc <- auc@y.values[[1]]
auc

# STEP 15: Create LDA and QDA models with confusion matrices and accuracy calculations
library(MASS)
################LDA###############
lda.fit<-lda(formula=Loan_Status~. , data=train_clean)
summary(lda.fit)
names(lda.fit)
plot(lda.fit)
lda.pred <-predict(lda.fit,test_clean)
names(lda.pred)
lda.class<-lda.pred$class
##########CONFUSION MATRIX LDA####
table(lda.class, test_clean$Loan_Status)
mean(lda.class== test_clean$Loan_Status)
################QDA###############
lda.fit<-qda(formula=Loan_Status~. , data=train_clean)
summary(qda.fit)
names(qda.fit)
plot(qda.fit)
lda.pred <-predict(qda.fit,test_clean)
names(qda.pred)
lda.class<-qda.pred$class
##########CONFUSION MATRIX LDA####
table(qda.class, test_clean$Loan_Status)
mean(qda.class== test_clean$Loan_Status)
