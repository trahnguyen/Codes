library(dplyr)
library(tidyr)
library(forcats)
library(readr)
library(nnet)
library(caret)
library(NeuralNetTools)
library(rpart)
library(rpart.plot)
library(gmodels)
library(caretEnsemble)
source('~/Babson/QTM Machine Learning/BabsonAnalytics.R')

# LOAD
df = read.csv('~/Babson/QTM Machine Learning/Group Project/house-prices-advanced-regression-techniques/train.csv')

# MANAGE
# Replace NAs 
sort(colSums(is.na(df)), decreasing = TRUE) # count and sort the number of NAs in the df
df = df %>% mutate_if(is.character, ~replace_na(.,"None")) # replace all NAs for character variables
sort(colSums(is.na(df)), decreasing = TRUE) # check NAs for the non-character variables
df = df %>% mutate_if(is.integer, ~replace_na(.,0)) # replace NAs in integer variables with 0

# Transform variables
df$Id = NULL
df$Utilities = NULL # Utilities has 2 levels but NoSeWa has 1 value -> discard this column
df$SaleType = NULL
df$Condition2 = NULL
df$Heating = NULL
df$RoofMatl = NULL
df$Street = NULL
df = df %>% mutate_if(is.character, as.factor) # convert all character variables to factors
df = df %>% 
  mutate_if(is.factor, ~fct_lump_prop(., 0.05)) 

# Group sets of variables -> compare before and after running models 
df = df %>%
  mutate(TotalBath = FullBath + HalfBath*0.5 + BsmtFullBath + BsmtHalfBath*0.5) %>%
  mutate(AfterRemodel = YrSold - YearRemodAdd) %>%
  mutate(TotalSquareFeet= GrLivArea + TotalBsmtSF) %>%
  mutate(Porch= WoodDeckSF + EnclosedPorch + OpenPorchSF + X3SsnPorch + ScreenPorch)

(l <- sapply(df, function(x) is.factor(x)))
m <- df[, l]
ifelse(n <- sapply(m, function(x) length(levels(x))) == 1, "DROP", "NODROP") # check to see if the df has any 1-level factor variables


# PARTITION
set.seed(1234)
training_size = round(nrow(df)*0.6)
training_cases = sample(nrow(df), training_size)
training = df[training_cases, ]
test = df[-training_cases, ]

summary(test)

# BUILD LINEAR REGRESSION MODEL
model_lm = lm(SalePrice ~ ., data=training, )
model_lm = step(model_lm)
summary(model_lm)
# 1  problem: Coefficients: (1 not defined because of singularities) 
# -> BsmtExposureNone, perfect collinearity (?) not sure how to fix this 

# PREDICT
# Drop unused factor levels to predict from unseen data
predictions_lm = predict(model_lm, test)

observations = test$SalePrice
errors_lm = observations - predictions_lm
rmse_lm = sqrt(mean(errors_lm^2)) 
mape_lm = mean(abs(errors_lm/observations))
# Benchmark
predictions_bench = mean(training$SalePrice)
errors_bench = observations - predictions_bench
rmse_bench = sqrt(mean(errors_bench^2)) 
mape_bench = mean(abs(errors_bench/observations))


# BUILD TREE REGRESSION MODEL
stopping_rules = rpart.control(minsplit = 2, minbucket = 1, cp = 0.01)
model_rt = rpart(SalePrice ~ ., data=training, control = stopping_rules)
model_rt = easyPrune(model_rt)
rpart.plot(model_rt)
predictions_rt = predict(model_rt, test)
errors_rt = observations - predictions_rt
mape_rt = mean(abs(errors_rt/observations))
rmse_rt = sqrt(mean(errors_rt^2))

# BUILD NEURAL NETS MODEL
set.seed(1234)
training_cases_nn = sample(nrow(predict(preProcess(df, method=c("range")), df)),round(0.6*nrow(predict(preProcess(df, method=c("range")), df))))
training_nn = predict(preProcess(df, method=c("range")), df)[training_cases_nn,]
test_nn = predict(preProcess(df, method=c("range")), df)[-training_cases_nn,]

model_nn = nnet(SalePrice ~ ., data=training_nn, size = 4, linout = TRUE)
predictions_nn = predict(model_nn, test_nn)
errors_nn = observations - predictions_nn
rmse_nn = sqrt(mean(errors_nn^2)) 
mape_nn = mean(abs(errors_nn/observations))

# STACKING
pred_lm_full = predict(model_lm, df)
pred_rt_full = predict(model_rt, df)
pred_nn_full = predict(model_nn, df)
df_stack = cbind(df,pred_lm_full,pred_rt_full,pred_nn_full)
train_stack = df_stack[sample(nrow(df_stack), round(nrow(df_stack)*0.6)), ]
test_stack = df_stack[-sample(nrow(df_stack), round(nrow(df_stack)*0.6)), ]

stopping_rules = rpart.control(minbucket=1, minsplit=1, cp=-1)
stack = rpart(SalePrice ~ ., data = train_stack)
stack = easyPrune(stack)
rpart.plot(stack)
