library(Hmisc)
library(dplyr)
library(lavaan)
library(tidyverse)
library(lubridate)
library(gbm)
source('~/Babson/QTM Machine Learning/BabsonAnalytics.R')

# LOAD
df = read.csv('~/Babson/MSB Field Project/Data/Assignments/airbnb-recruiting-new-user-bookings/train_users_2.csv')
df = as.data.frame(df[sample(nrow(df), round(nrow(df)*0.1)), ]) 

# MANGAGE
#df$id = NULL
df$date_first_booking = NULL
df$date_account_created = NULL
df$timestamp_first_active = NULL

df = df %>% mutate(signup_flow = as.factor(signup_flow)) %>%
  #mutate(date_account_created = ymd(date_account_created)) %>%
  #mutate(date_first_booking = ymd(date_first_booking)) %>%
  mutate_if(is.character, as.factor) #%>% 
  #mutate_if(is.factor, ~fct_lump_prop(., 0.0001)) 

# age has a lot of outliers need to be removed
describe(df$age, na.rm = TRUE, interp=FALSE,skew = TRUE, ranges = TRUE,trim=.1,
         type=3,check=TRUE,fast=NULL,quant=NULL,IQR=FALSE,omit=FALSE,data=NULL)

describe(df, na.rm = TRUE, interp=FALSE,skew = TRUE, ranges = TRUE,trim=.1,
         type=3,check=TRUE,fast=NULL,quant=NULL,IQR=FALSE,omit=FALSE,data=NULL)


df$age = ifelse(df$age < 15, NA,
                ifelse(df$age > 105, NA,
                       df$age))
mean.age = round(mean(df$age, na.rm = TRUE))
df$age = ifelse(is.na(df$age), mean.age, df$age)

#NDF are non-booking accounts -> remove
#df_without_NDF = df[df$country_destination != "NDF", ]

#describe(df_without_NDF, na.rm = TRUE, interp=FALSE,skew = TRUE, ranges = TRUE,trim=.1,
        # type=3,check=TRUE,fast=NULL,quant=NULL,IQR=FALSE,omit=FALSE,data=NULL)

# PARTITION
# set.seed(1234)
training_size = round(nrow(df)*0.6)
training_cases = sample(nrow(df), training_size)
training = df[training_cases, ]
test = df[-training_cases, ] 


# TREE
library(rpart)
stopping_rules = rpart.control(minsplit = 2, minbucket = 1, cp=-1)
model_dt = rpart(country_destination ~ . -id, data=training, control=stopping_rules)
model_dt_pruned = easyPrune(model_dt)
#library(rpart.plot)
#rpart.plot(model_dt_pruned)
predictions_dt = predict(model_dt_pruned, test, type='class') 
errorRate_dt = sum(predictions_dt != test$country_destination)/nrow(test)
observations = test$country_destination
errorBench = benchmarkErrorRate(training$country_destination, test$country_destination)

# BOOST
model_boost = gbm(country_destination ~ ., data=training, n.trees=500, cv.folds = 4 )
best_size = gbm.perf(model_boost,method="cv")
predictions_boost = predict(model_boost, test, best_size, type="response")

errorRate_boost = sum(predictions_boost != test$country_destination)/nrow(test)

# Generate submission
write.csv(predictions_dt, file="submission1.csv")






