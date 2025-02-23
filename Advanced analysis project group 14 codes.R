
ls()
rm(list = ls())
getwd()
setwd("E:/3rd year/sem2/ST 3082/Data analysis project 1")
# Load the data set
Sleep_data=read.csv(file="sleeptime_prediction_dataset.csv",header = TRUE,sep = ",")

# List of time-related columns
time_columns = c("WorkoutTime", "ReadingTime", "PhoneTime" , "WorkHours", "RelaxationTime","SleepTime")

# Calculate the sum of each row for the selected columns
TotalTime=rowSums(Sleep_data[,time_columns])

# Filter rows where the total time is less than or equal to 24 hours
Data=Sleep_data[TotalTime <= 24, ]

# Load necessary library
library(MASS)
library(stats)
library(car)
library(nortest)
library(randomForest)
library(caret)
library(xgboost)
library(dplyr)
library(e1071)
library(rpart)
library(ggplot2)
library(gridExtra)

# Compute Mahalanobis Distance
data_matrix <- as.matrix(Data)  
center <- colMeans(data_matrix)   
cov_matrix <- cov(data_matrix)    

# Calculate MD for each row
MD <- mahalanobis(data_matrix, center, cov_matrix)

# Set threshold (Chi-square critical value for 95% confidence, df = num of variables)
threshold <- qchisq(0.95, df = ncol(Data))

# Flag outliers
outliers <- which(MD > threshold)

# Print outlier indices
print(outliers)


set.seed(123)

index=sample(1:nrow(Data),0.2*nrow(Data))

# test data set
test_Data=Data[index,]


# train data set
train_Data=Data[-index,]


Xc=train_Data[,1:6]
yc=train_Data[,7]

Xt=test_Data[,1:6]
yt=test_Data[,7]



#____________________________________________________________________________________________________________________________________________
# ************************** MLR model *********************************

# ********************************forward selection************************************
mlr_model <- lm(SleepTime ~ PhoneTime + WorkHours + CaffeineIntake + WorkoutTime + RelaxationTime, data = train_Data)

# Display the final selected model
summary(mlr_model)
mlr_model$coefficients

# Predict on training data
train_predictions <- predict(mlr_model, newdata = train_Data)
# Compute RMSE
train_rmse <- sqrt(mean((train_Data$SleepTime - train_predictions)^2))
# Compute R-squared
train_r2 <- 1 - sum((train_Data$SleepTime - train_predictions)^2) / sum((train_Data$SleepTime - mean(train_Data$SleepTime))^2)


# Predict on test  data
test_predictions <- predict(mlr_model, newdata = test_Data)
# Compute RMSE
test_rmse <- sqrt(mean((test_Data$SleepTime - test_predictions)^2))
# Compute R-squared
test_r2 <- 1 - sum((test_Data$SleepTime - test_predictions)^2) / sum((test_Data$SleepTime - mean(test_Data$SleepTime))^2)



cat("Training RMSE:", train_rmse, "\n")
cat("Training R²:", train_r2, "\n")
cat("Test RMSE:", test_rmse, "\n")
cat("Test R²:", test_r2, "\n")


test_results <- data.frame(Actual = yt, Predicted = test_predictions)

# Scatter plot
p1=ggplot(test_results, aes(x = Actual, y = Predicted)) +
  geom_point(color = "#F8C2DA", alpha = 0.6) +  
  geom_abline(intercept = 0, slope = 1, color = "blue", linetype="dashed") +  
  labs(title = "MLR",
       x = "Actual Sleep Time",
       y = "Predicted Sleep Time") +
  theme_minimal()



# ******************** Assumption checking ************************

# Residual plot - Independence and homoscedasticity 
plot(train_predictions,residuals(mlr_model),col="purple",xlab = "Fitted values",ylab = "Residuals",main = "Residual vs Fitted values")


# Histogram of residuals
hist(residuals(mlr_model), main = "Histogram of Residuals", col = "lightblue")

# Q-Q Plot - normality test 
qqnorm(residuals(mlr_model),col="lightblue")
qqline(residuals(mlr_model), col = "red")

# Shapiro-Wilk test for residuals
shapiro.test(residuals(mlr_model))


ad.test(residuals(mlr_model))  # Anderson-Darling Test
lillie.test(residuals(mlr_model))  # Lilliefors Test


plot(fitted(mlr_model), residuals(mlr_model), 
     main = "Residuals vs Fitted", 
     xlab = "Fitted Values", ylab = "Residuals")
abline(h = 0, col = "red")


# __________________________________________________________________________________________________________________________________________________
# ******************************** Random Forest model ********************************
# cv for RAndom forest



# Define cross-validation method
train_control <- trainControl(method = "cv", number = 10)  # 10-fold CV

# Define hyperparameter grid
tune_grid <- expand.grid(mtry = c(2, 3, 4, 5),   # Number of variables per split
                         splitrule = "variance", # Default for regression
                         min.node.size = c(5, 10, 15))  # Minimum node size

# Train Random Forest model with cross-validation
rf_cv <- train(SleepTime ~ ., 
               data = train_Data,
               method = "ranger",  # Faster implementation of RF
               trControl = train_control,
               tuneGrid = tune_grid,
               num.trees = 500,  # Number of trees
               importance = "impurity")  # Get feature importance

# Print best parameters
print(rf_cv$bestTune)
# Fit Random Forest model
rf_model <- randomForest(SleepTime ~ ., data = train_Data, ntree = 500,mtry=3,nodesize=5,importance=TRUE)

# Predictions on training and test data
train_pred<- predict(rf_model, train_Data)
test_pred <- predict(rf_model, test_Data)

# Function to calculate RMSE
rmse <- function(actual, predicted) {
  sqrt(mean((actual - predicted)^2))
}

# Calculate RMSE for training and test sets
train_rmse <- rmse(train_Data$SleepTime, train_pred)
test_rmse <- rmse(test_Data$SleepTime, test_pred)

# Calculate R² for training and test sets
train_r2 <- cor(train_Data$SleepTime, train_pred)^2
test_r2 <- cor(test_Data$SleepTime, test_pred)^2

# Print results
cat("Train RMSE:", train_rmse, "\n")
cat("Train R²:", train_r2, "\n")
cat("Test RMSE:", test_rmse, "\n")
cat("Test R²:", test_r2, "\n")

# Variable Importance Plot
varImpPlot(rf_model)

barplot(varImp(rf_model)$Overall, names.arg = rownames(varImp(rf_model)),
        main = "Feature Importance in Random forest",
        col = "steelblue", las = 2)



# Fit Random Forest model by removing least important data
Im_rf_model <- randomForest(SleepTime ~ WorkoutTime + PhoneTime + WorkHours , data = train_Data, ntree = 500,mtry=3,nodesize=5,importance=TRUE)

# Predictions on training and test data
train_pred2 <- predict(Im_rf_model, train_Data)
test_pred2 <- predict(Im_rf_model, test_Data)

# Function to calculate RMSE
rmse <- function(actual, predicted) {
  sqrt(mean((actual - predicted)^2))
}

# Calculate RMSE for training and test sets
train_rmse <- rmse(train_Data$SleepTime, train_pred2)
test_rmse <- rmse(test_Data$SleepTime, test_pred2)

# Calculate R² for training and test sets
train_r2 <- cor(train_Data$SleepTime, train_pred2)^2
test_r2 <- cor(test_Data$SleepTime, test_pred2)^2

# Print results
cat("Train RMSE:", train_rmse, "\n")
cat("Train R²:", train_r2, "\n")
cat("Test RMSE:", test_rmse, "\n")
cat("Test R²:", test_r2, "\n")


test_results <- data.frame(Actual = yt, Predicted = test_pred2)

# Scatter plot
p2=ggplot(test_results, aes(x = Actual, y = Predicted)) +
  geom_point(color = "#AFEEEE", alpha = 0.6) +  
  geom_abline(intercept = 0, slope = 1, color = "blue", linetype="dashed") +  
  labs(title = "Random Forest",
       x = "Actual Sleep Time",
       y = "Predicted Sleep Time") +
  theme_minimal()

#__________________________________________________________________________________________________________________________
# ************************************** XG Boost ********************************************************


# Convert data to xgboost's DMatrix format
train_matrix <- xgb.DMatrix(data = as.matrix(Xc), label = yc)

test_matrix <- xgb.DMatrix(data = as.matrix(Xt), label = yt)


# Define parameters for the XGBoost model
params <- list(
  objective = "reg:squarederror",  # For regression tasks
  eta = 0.1,
  max_depth = 3,
  nrounds = 100
)

# Train the XGBoost model
model <- xgb.train(params = params, data = train_matrix,nrounds = 100,importance=TRUE)

# Make predictions on training and testing data
train_pred <- predict(model, train_matrix)
test_pred <- predict(model, test_matrix)

# Calculate RMSE and R-squared for training data
train_rmse <- sqrt(mean((train_pred - yc)^2))
train_r2 <- 1 - sum((train_pred - yc)^2) / sum((yc - mean(yc))^2)

# Calculate RMSE and R-squared for testing data
test_rmse <- sqrt(mean((test_pred - yt)^2))
test_r2 <- 1 - sum((test_pred - yt)^2) / sum((yt - mean(yt))^2)

# Print the results
cat("Train RMSE:", train_rmse, "\n")
cat("Train R squared:", train_r2, "\n")
cat("Test RMSE:", test_rmse, "\n")
cat("Test R squared:", test_r2, "\n")

# ************** Optimizing the model ************************

# Define XGBoost parameters
xgb_grid <- expand.grid(
  nrounds = c(50,100),     # Number of boosting rounds
  eta = c(0.01, 0.1),   # Learning rate
  max_depth = c(3, 6) , # Depth of trees
  gamma = c(0, 1),        # Minimum loss reduction
  colsample_bytree = c(0.6,1),  # Feature selection
  min_child_weight = c( 3, 5),
  subsample = c( 0.8, 1)
  
)
set.seed(123)
# Define training control for cross-validation
train_control <- trainControl(
  method = "cv",            # Cross-validation
  number = 5,               # 5-fold CV
  verboseIter = TRUE        # Show progress
)

# Train XGBoost model with CV
xgb_model <- train(
  x = Xc, 
  y = yc,
  method = "xgbTree",
  trControl = train_control,
  tuneGrid = xgb_grid,
  metric = "RMSE"
)

# Get best tuned parameters
best_params <- xgb_model$bestTune
print(best_params)

# Make predictions on test set
predictions <- predict(xgb_model, Xt)

# Calculate performance metrics
test_RMSE <- sqrt(mean((yt - predictions)^2))
test_R2 <- cor(yt, predictions)^2

# parameters belong to optimum model 
params <- list(
  objective = "reg:squarederror",  # For regression tasks
  eta = 0.1,
  
  max_depth = 3,
  colsample_bytree=0.6,
  min_child_weight=5,
  subsample=0.8,
  nrounds = 100
)

# Train the XGBoost model
model <- xgb.train(params = params, data = train_matrix,nrounds = 100,importance=TRUE)

# Make predictions on training and testing data
train_pred3 <- predict(model, train_matrix)
test_pred3 <- predict(model, test_matrix)

#Calculate RMSE and R-squared for training data
train_rmse <- sqrt(mean((train_pred3 - yc)^2))
train_r2 <- 1 - sum((train_pred3 - yc)^2) / sum((yc - mean(yc))^2)

# Calculate RMSE and R-squared for testing data
test_rmse <- sqrt(mean((test_pred3 - yt)^2))
test_r2 <- 1 - sum((test_pred3 - yt)^2) / sum((yt - mean(yt))^2)

# Print the results
cat("Train RMSE:", train_rmse, "\n")
cat("Train R squared:", train_r2, "\n")
cat("Test RMSE:", test_rmse, "\n")
cat("Test R squared:", test_r2, "\n")

test_results <- data.frame(Actual = yt, Predicted = test_pred3)

# Scatter plot
p3=ggplot(test_results, aes(x = Actual, y = Predicted)) +
  geom_point(color = "#CD853F", alpha = 0.6) +  
  geom_abline(intercept = 0, slope = 1, color = "blue", linetype="dashed") +  
  labs(title = "XGBoost",
       x = "Actual Sleep Time",
       y = "Predicted Sleep Time") +
  theme_minimal()


# ______________________________________________________________________________________________________
# ************************** SVR model *******************************

# Standardize the features (mean = 0, standard deviation = 1)
set.seed(123)
X_train_scaled <- scale(Xc)
X_test_scaled <- scale(Xt, center = attr(X_train_scaled, "scaled:center"), 
                       scale = attr(X_train_scaled, "scaled:scale"))

# Train the SVM model (using radial basis function kernel)
svm_model <- svm(x = X_train_scaled, y = yc, kernel = "radial", cost = 1,gamma = 0.3)

# Make predictions on training and testing data
train_pred4 <- predict(svm_model, X_train_scaled)
test_pred4 <- predict(svm_model, X_test_scaled)

# Calculate RMSE and R-squared for the training set
train_rmse <- sqrt(mean((train_pred4 - yc)^2))
train_r2 <- 1 - sum((train_pred4 - yc)^2) / sum((yc - mean(yc))^2)

# Calculate RMSE and R-squared for the test set

test_rmse <- sqrt(mean((test_pred4 - yt)^2))
test_r2 <- 1 - sum((test_pred4 - yt)^2) / sum((yt - mean(yt))^2)

# Print the results
cat("Train RMSE:", train_rmse, "\n")
cat("Train R squared:", train_r2, "\n")
cat("Test RMSE:", test_rmse, "\n")
cat("Test R squared:", test_r2, "\n")

# Define hyperparameter grid for tuning
tune_grid <- expand.grid(
  cost = c(0.01,0.1,1),
  gamma = c( 0.1,0.2,0.3)
)


# Perform 5-fold cross-validation to find the best parameters
set.seed(123)
tuned_svm <- tune(
  svm,
  train.x = X_train_scaled,
  train.y = yc,
  kernel = "radial",
  ranges = list(cost = tune_grid$cost, gamma = tune_grid$gamma)
)


# Best hyperparameters
best_model <- tuned_svm$best.model
best_params <- tuned_svm$best.parameters
cat("Best Cost (C):", best_params$cost, "\n")
cat("Best Gamma (γ):", best_params$gamma, "\n")


# Make predictions on training and testing data
train_pred <- predict(best_model, X_train_scaled)
test_pred <- predict(best_model, X_test_scaled)


# Calculate RMSE and R-squared for the training set
train_rmse <- sqrt(mean((train_pred - yc)^2))
train_r2 <- 1 - sum((train_pred - yc)^2) / sum((yc - mean(yc))^2)

# Calculate RMSE and R-squared for the test set

test_rmse <- sqrt(mean((test_pred - yt)^2))
test_r2 <- 1 - sum((test_pred - yt)^2) / sum((yt - mean(yt))^2)



# Print the results
cat("Train RMSE:", train_rmse, "\n")
cat("Train R squared:", train_r2, "\n")
cat("Test RMSE:", test_rmse, "\n")
cat("Test R squared:", test_r2, "\n")

# Computing other errors for testing set
adjusted_r2=function(r2,n,p){
  return(1-((1-r2)*(n-1)/(n-p-1)))
}
test_mse=mean((test_pred4-yt)^2)
test_mae=mean(abs(test_pred4-yt))
test_adj_r2=adjusted_r2(test_r2,length(yt),ncol(Xt))
test_rmsle=sqrt(mean((log1p(test_pred4)-log1p(yt))^2))
test_mape=mean(abs((test_pred4-yt)/yt))*100

# Print the results
cat("Test MSE:", test_mse, "\n")
cat("Test MAE:", test_mae, "\n")
cat("Test RMSLE:", test_rmsle, "\n")
cat("Test MAPE:", test_mape, "\n")
cat("Test Adjusted R squared:", test_adj_r2, "\n")


test_results <- data.frame(Actual = yt, Predicted = test_pred4)

# Scatter plot
p4=ggplot(test_results, aes(x = Actual, y = Predicted)) +
  geom_point(color = "#D8BFD8", alpha = 0.6) +  
  geom_abline(intercept = 0, slope = 1, color = "blue", linetype="dashed") +  
  labs(title = "SVR",
       x = "Actual Sleep Time",
       y = "Predicted Sleep Time") +
  theme_minimal()

#______________________________________________________________________________________
# *************************** Regression Trees ************************************

# Assuming you have a dataset with features X and target y
# X is the feature matrix, and y is the target variable vector


# Train the regression tree model using rpart
reg_tree_model <- rpart(yc ~ ., data = cbind(Xc, yc), method = "anova")

# Make predictions on training and testing data
train_pred <- predict(reg_tree_model, Xc)
test_pred <- predict(reg_tree_model, Xt)

# Calculate RMSE and R-squared for training data
train_rmse <- sqrt(mean((train_pred - yc)^2))
train_r2 <- 1 - sum((train_pred - yc)^2) / sum((yc - mean(yc))^2)

# Calculate RMSE and R-squared for testing data
test_rmse <- sqrt(mean((test_pred - yt)^2))
test_r2 <- 1 - sum((test_pred - yt)^2) / sum((yt - mean(yt))^2)

# Print the results
cat("Train RMSE:", train_rmse, "\n")
cat("Train R squared:", train_r2, "\n")
cat("Test RMSE:", test_rmse, "\n")
cat("Test R squared:", test_r2, "\n")

importance=varImp(reg_tree_model)

barplot(importance$Overall, names.arg = rownames(importance),
        main = "Feature Importance in Regression Tree",
        col = "steelblue", las = 2)
           


reg_tree_model2 <- rpart(yc ~ PhoneTime+RelaxationTime+WorkHours+WorkoutTime, data = cbind(Xc, yc), method = "anova")
summary(reg_tree_model2)
# Make predictions on training and testing data
train_pred5 <- predict(reg_tree_model2, Xc)
test_pred5 <- predict(reg_tree_model2, Xt)

# Calculate RMSE and R-squared for training data
train_rmse2 <- sqrt(mean((train_pred5 - yc)^2))
train_r22 <- 1 - sum((train_pred5 - yc)^2) / sum((yc - mean(yc))^2)

# Calculate RMSE and R-squared for testing data
test_rmse2 <- sqrt(mean((test_pred5 - yt)^2))
test_r22 <- 1 - sum((test_pred5 - yt)^2) / sum((yt - mean(yt))^2)

# Print the results
cat("Train RMSE:", train_rmse2, "\n")
cat("Train R squared:", train_r22, "\n")
cat("Test RMSE:", test_rmse2, "\n")
cat("Test R squared:", test_r22, "\n")


# Create a data frame with actual and predicted values
test_results <- data.frame(Actual = yt, Predicted = test_pred5)

# Scatter plot
p5=ggplot(test_results, aes(x = Actual, y = Predicted)) +
  geom_point(color = "lightgreen", alpha = 0.6) +  # Scatter points
  geom_abline(intercept = 0, slope = 1, color = "red", linetype="dashed") +  # Reference line (ideal prediction)
  labs(title = "Regression Tree",
       x = "Actual Sleep Time",
       y = "Predicted Sleep Time") +
  theme_minimal()

grid.arrange(p1,p2,p3,p4, p5 ,nrow = 2)
