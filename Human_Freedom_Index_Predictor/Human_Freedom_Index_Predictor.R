#Step 0: Import the data "train.csv" and run the code through line 12
library(recipes)
library(rsample)
library(car)
library(DataExplorer)
library(polycor)
library(tidyverse)
library(ROCR)
library(broom)


raw_data<-Human_Freedom

glimpse(raw_data)
dim(raw_data)


data_tbl<-raw_data%>%select(hf_score,everything())
data_tbl_opt<-data_tbl%>%select(hf_score,
                                pf_ss_disappearances_disap,
                                pf_ss_disappearances_violent,
                                pf_ss_disappearances_organized,
                                pf_ss_disappearances_fatalities,
                                pf_score,
                                ef_legal_courts,
                                ef_legal_protection,
                                ef_legal_military,
                                ef_legal_integrity,
                                ef_legal_enforcement,
                                ef_legal_restrictions,
                                ef_legal_police,
                                ef_legal_crime,
                                ef_legal_gender,
                                ef_score)

plot_missing(data_tbl_opt)

set.seed(854)
train_test_split <- initial_split(data_tbl_opt, prop = 0.8)

glimpse(data_tbl_opt)
train_tbl <- training(train_test_split)
test_tbl <- testing(train_test_split)

rec_obj<-recipe(hf_score ~., data=train_tbl)%>%
  step_bagimpute(all_predictors(),-all_outcomes())%>%
  step_BoxCox(all_predictors(),-all_outcomes())%>%
  step_mutate(ef_total=ef_legal_courts+ ef_legal_protection+ef_legal_military+ef_legal_integrity+ef_legal_enforcement+ef_legal_restrictions+ef_legal_police+ef_legal_crime+ef_legal_gender)%>%
  step_log(ef_total)%>%
  step_sqrt(pf_score)%>%
  step_log(ef_score)%>%
  step_log(ef_legal_police)%>%
  prep(data=train_tbl)

rec_obj


train_clean <- bake(rec_obj, new_data = train_tbl)
test_clean  <- bake(rec_obj, new_data = test_tbl)


plot_density(train_clean)

hetcor(as.data.frame(train_clean))

na.omit(train_clean)
na.omit(test_clean)
plot_missing(train_clean)

plot_missing(test_clean)

lm.model<-lm(hf_score~.,data=train_tbl)
summary(lm.model)

par(mfrow=c(2,2))
plot(lm.model) #generates four plots (residuals vs fitted, QQ, residuals vs leverage)

#Non-Linearity of the Data
crPlots(lm.model)
ceresPlots(lm.model)

# qq plot for studentized resid
qqPlot(lm.model, main="QQ Plot")

durbinWatsonTest(lm.model)

#Non-Constant Variance of Error Terms (Heteroscedasticity)
ncvTest(lm.model)
spreadLevelPlot(lm.model)

#Outliers
outlierTest(lm.model) # Bonferonni p-value for most extreme obs

train_clean<-train_clean[-c(1151,1014,927,571,1063,920,848,1131),]

leveragePlots(lm.model) # leverage plots

lm.model<-lm(hf_score~.,data=train_clean)
summary(lm.model)

