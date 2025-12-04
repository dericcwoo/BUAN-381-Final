
# --------------------------------------------
# BUAN-381 Predictive Analysis Final Project
# --------------------------------------------

# LOAD LIBRARIES
library(dplyr)

# LOAD DATASET
qb <- read.csv('/Users/dwoo/Downloads/Fall 2025/BUAN-381/qb_index_no_tier.csv')

# --------------------------------------------
# CLEAN THE DATA — remove QBs with zero NFL attempts
# --------------------------------------------

qb1 <- qb[qb$nfl.att > 0, ]
qb1 <- na.omit(qb1)   # optional but recommended
qb1 <- qb1[qb1$nfl.att >= 10, ]   # keep QBs with at least 10 attempts

# --------------------------------------------
# PARTITIONING OUR DATA (using qb1)
# --------------------------------------------

set.seed(123)

train_idx <- sample(1:nrow(qb1), size = 0.6 * nrow(qb1))
train <- qb1[train_idx, ]

temp <- qb1[-train_idx, ]
val_idx <- sample(1:nrow(temp), size = 0.5 * nrow(temp))

validation <- temp[val_idx, ]
test <- temp[-val_idx, ]

# SAVE PARTITIONS
write.csv(train, "qb_train.csv", row.names = FALSE)
write.csv(validation, "qb_validation.csv", row.names = FALSE)
write.csv(test, "qb_test.csv", row.names = FALSE)

# --------------------------------------------
# OLS MODEL (uses cleaned train set)
# --------------------------------------------

model <- lm(nfl.qbr ~ p.yds + p.td + p.ypa + coach.tenure + conf.str + drafted.team.winpr 
            + drafted_team_ppg_rk, ,
            data = train)

summary(model)

# --------------------------------------------
# RMSE and MAE functions
# --------------------------------------------

rmse <- function(actual, predicted) sqrt(mean((actual - predicted)^2))
mae  <- function(actual, predicted) mean(abs(actual - predicted))

# MAKE PREDICTIONS
train_pred <- predict(model, newdata = train)
valid_pred <- predict(model, newdata = validation)

# METRICS
train_rmse <- rmse(train$nfl.qbr, train_pred)
train_mae  <- mae(train$nfl.qbr, train_pred)

valid_rmse <- rmse(validation$nfl.qbr, valid_pred)
valid_mae  <- mae(validation$nfl.qbr, valid_pred)

train_rmse; train_mae
valid_rmse; valid_mae

# --------------------------------------------
# CORRELATION HEATMAP (using qb1, not qb)
# --------------------------------------------

# Install corrplot if needed
if (!require(corrplot)) {
  install.packages("corrplot")
  library(corrplot)
}

# Select variables from cleaned dataset qb1
num_vars <- qb1 %>%
  select(nfl.qbr, cmp.pct, p.yds, p.td, p.ypa, rate, conf.str, coach.tenure, 
         drafted.team.winpr, drafted_team_ppg_rk) %>%
  na.omit()

cor_matrix <- cor(num_vars)

corrplot(cor_matrix,
         method = "color",
         type = "upper",
         tl.col = "black",
         tl.srt = 45,
         addCoef.col = "black",
         number.cex = 0.7,
         col = colorRampPalette(c("blue","white","red"))(200))

cor(qb1$nfl.qbr, qb1$drafted_team_ppg_rk)
cor(qb1$nfl.qbr, qb1$drafted.team.winpr)
cor(qb1$nfl.qbr, qb1$qb.num.picked)

# --------------------------------------------
# 4A BIVARIATE MODEL
# --------------------------------------------

nfl.qbr ~ qb.num.picked

library(ggplot2)

# Build bivariate linear model
model_4a <- lm(nfl.qbr ~ qb.num.picked, data = train)

# Summary of model
summary(model_4a)

# Predictions
train_pred_4a <- predict(model_4a, newdata = train)
valid_pred_4a <- predict(model_4a, newdata = validation)

# Compute RMSE + MAE
rmse <- function(actual, predicted) sqrt(mean((actual - predicted)^2))
mae  <- function(actual, predicted) mean(abs(actual - predicted))

train_rmse_4a <- rmse(train$nfl.qbr, train_pred_4a)
train_mae_4a  <- mae(train$nfl.qbr, train_pred_4a)

valid_rmse_4a <- rmse(validation$nfl.qbr, valid_pred_4a)
valid_mae_4a  <- mae(validation$nfl.qbr, valid_pred_4a)

train_rmse_4a; train_mae_4a
valid_rmse_4a; valid_mae_4a

# -------------------------------
# SCATTERPLOT + REGRESSION LINE
# -------------------------------
ggplot(train, aes(x = qb.num.picked, y = nfl.qbr)) +
  geom_point(color = "darkblue", alpha = 0.6) +
  geom_smooth(method = "lm", color = "red", se = FALSE, linewidth = 1.2) +
  labs(title = "Bivariate Linear Regression: NFL QBR vs Draft Pick Number",
       x = "QB Draft Pick Number",
       y = "NFL QBR") +
  theme_minimal(base_size = 14)

# -------------------------------------
# TASK 4B: Polynomial (Quadratic) Model
# -------------------------------------

# Build polynomial model of degree 2
model_4b <- lm(nfl.qbr ~ qb.num.picked + I(qb.num.picked^2), data = train)

# Model Summary
summary(model_4b)

# Predictions
train_pred_4b <- predict(model_4b, newdata = train)
valid_pred_4b <- predict(model_4b, newdata = validation)

# RMSE and MAE calculations
rmse <- function(actual, predicted) sqrt(mean((actual - predicted)^2))
mae  <- function(actual, predicted) mean(abs(actual - predicted))

train_rmse_4b <- rmse(train$nfl.qbr, train_pred_4b)
train_mae_4b  <- mae(train$nfl.qbr, train_pred_4b)

valid_rmse_4b <- rmse(validation$nfl.qbr, valid_pred_4b)
valid_mae_4b  <- mae(validation$nfl.qbr, valid_pred_4b)

train_rmse_4b; train_mae_4b
valid_rmse_4b; valid_mae_4b

# Plot: Polynomial Curve Fit
library(ggplot2)

ggplot(train, aes(x = qb.num.picked, y = nfl.qbr)) +
  geom_point(alpha = 0.6, color = "darkblue") +
  stat_smooth(method = "lm",
              formula = y ~ x + I(x^2),
              color = "red",
              se = FALSE,
              linewidth = 1.2) +
  labs(title = "Polynomial Bivariate Regression: NFL QBR ~ Draft Pick Number (Quadratic)",
       x = "QB Draft Pick Number",
       y = "NFL QBR") +
  theme_minimal(base_size = 14)

# --------------------------------------------
# TASK 4C — Manual Ridge & LASSO Regression
# --------------------------------------------

# Extract predictors
x <- train$qb.num.picked
y <- train$nfl.qbr

# Centering (standard practice)
x_center <- x - mean(x)
y_center <- y - mean(y)

# Tuning parameter
lambda <- 1   # choose 1 for demonstration (can try 0.1, 1, 10)

# RIDGE REGRESSION FORMULA
ridge_coef <- sum(x_center * y_center) / (sum(x_center^2) + lambda)
ridge_intercept <- mean(y) - ridge_coef * mean(x)

ridge_coef
ridge_intercept

# Predictions
train_pred_ridge <- ridge_intercept + ridge_coef * x
valid_pred_ridge <- ridge_intercept + ridge_coef * validation$qb.num.picked

# Error functions
rmse <- function(a, p) sqrt(mean((a - p)^2))
mae  <- function(a, p) mean(abs(a - p))

ridge_train_rmse <- rmse(train$nfl.qbr, train_pred_ridge)
ridge_valid_rmse <- rmse(validation$nfl.qbr, valid_pred_ridge)

ridge_train_rmse; ridge_valid_rmse

# LASSO REGRESSION FORMULA

# OLS slope
beta_ols <- sum(x_center * y_center) / sum(x_center^2)

# Soft-thresholding operator
lasso_coef <- sign(beta_ols) * max(0, abs(beta_ols) - lambda / (2 * sum(x_center^2)))
lasso_intercept <- mean(y) - lasso_coef * mean(x)

lasso_coef
lasso_intercept

# Predictions
train_pred_lasso <- lasso_intercept + lasso_coef * x
valid_pred_lasso <- lasso_intercept + lasso_coef * validation$qb.num.picked

lasso_train_rmse <- rmse(train$nfl.qbr, train_pred_lasso)
lasso_valid_rmse <- rmse(validation$nfl.qbr, valid_pred_lasso)

lasso_train_rmse; lasso_valid_rmse

##############################################
# TASK 4D — GLM (Gaussian / Identity Link)
##############################################

# Fit the GLM model
model_4d <- glm(nfl.qbr ~ qb.num.picked,
                data = train,
                family = gaussian(link = "identity"))

summary(model_4d)

##############################################
# Predictions
##############################################

train_pred_4d <- predict(model_4d, newdata = train, type = "response")
valid_pred_4d <- predict(model_4d, newdata = validation, type = "response")

##############################################
# Error Metrics
##############################################

rmse <- function(a, p) sqrt(mean((a - p)^2))
mae  <- function(a, p) mean(abs(a - p))

train_rmse_4d <- rmse(train$nfl.qbr, train_pred_4d)
train_mae_4d  <- mae(train$nfl.qbr, train_pred_4d)

valid_rmse_4d <- rmse(validation$nfl.qbr, valid_pred_4d)
valid_mae_4d  <- mae(validation$nfl.qbr, valid_pred_4d)

train_rmse_4d; train_mae_4d
valid_rmse_4d; valid_mae_4d

##############################################
# Plot (Same Scatterplot with GLM Fit)
##############################################

library(ggplot2)

ggplot(train, aes(x = qb.num.picked, y = nfl.qbr)) +
  geom_point(alpha = 0.6, color = "darkgreen") +
  geom_smooth(method = "glm",
              method.args = list(family = gaussian(link = "identity")),
              color = "orange",
              se = FALSE,
              linewidth = 1.2) +
  labs(
    title = "GLM Regression (Gaussian): NFL QBR ~ Draft Pick Number",
    x = "QB Draft Pick Number",
    y = "NFL QBR"
  ) +
  theme_minimal(base_size = 14)







##### QBR BAR CHART BY COLLEGE CONFERENCE #####

library(dplyr)
library(ggplot2)

# Use your cleaned dataset qb1
# Make sure conference variable is named correctly (common names shown below)
# If your dataset uses a different column name, update "conference"

# 1. Summarize total QBR by conference
qbr_by_conf <- qb1 %>%
  group_by(conf) %>%    # change "conf" if your column name is different
  summarise(total_qbr = sum(nfl.qbr, na.rm = TRUE)) %>%
  arrange(desc(total_qbr))

# 2. Create bar chart
ggplot(qbr_by_conf, aes(x = reorder(conf, total_qbr), y = total_qbr)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  coord_flip() +   # makes it easier to read
  labs(title = "Total NFL QBR by Conference",
       x = "Conference",
       y = "Sum of NFL QBR") +
  theme_minimal(base_size = 14)

##### QB BAR CHART BY COLLEGE TEAM #####

library(dplyr)
library(ggplot2)

# Summarize total NFL QBR by college team
qbr_by_team <- qb1 %>%
  group_by(college) %>%    
  summarise(total_qbr = sum(nfl.qbr, na.rm = TRUE)) %>%
  arrange(desc(total_qbr))

# Keep only the Top 25
qbr_top25 <- qbr_by_team %>% 
  slice_max(total_qbr, n = 25)

# Bar chart of Top 25 colleges by total QBR
ggplot(qbr_top25, aes(x = reorder(college, total_qbr), y = total_qbr)) +
  geom_bar(stat = "identity", fill = "darkorange") +
  coord_flip() +
  labs(title = "Top 25 Colleges by Total NFL QBR",
       x = "College Team",
       y = "Sum of NFL QBR") +
  theme_minimal(base_size = 14)


