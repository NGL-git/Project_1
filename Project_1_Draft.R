# Draft Code for Project 1

library(dplyr)
library (ggplot2)
library(tidyverse)
library(readxl)
library(broom)
library(car)
library(ggfortify)
library(tidymodels)
library(vip)
library(performance)
library(glmnet)
library(ggplot2)
library(plotly)
library(corrr)
library(emo)
library(gghighlight)

trees <- read.csv("Data/Data_1993_Final.csv")
glimpse(trees)
# ______________________________________

trees %>%
  filter(na.rm = TRUE) %>%
  ggplot(aes(SDI_20th, DeadDist)) +
  geom_point() +
  geom_smooth() +
  labs(x = "Stand Density Index (SDI) at 20th percentile", y = "Linear Distance to Nearest Brood Tree")



# * Create Recipe----
## specify variable relationships
## specify (training) data
## feature engineer
## process recipe on data
trees_recipe <- trees %>% 
  recipe(DeadDist ~ TreeDiam + Infest_Serv2 +  Neigh_SDI_1.4th + BA_Infest_1.4th) %>% 
  step_sqrt(all_outcomes()) %>% 
  step_corr(all_predictors()) 


# View feature engineered data
trees_recipe %>% 
  prep() %>% 
  juice()


# * Create Model ----
lm_model <- 
  linear_reg() %>% 
  set_engine("lm")

# * Create Workflow ----
trees_wflow <- 
  workflow() %>% 
  add_model(lm_model) %>% 
  add_recipe(trees_recipe)

trees_wflow


trees_fit <- 
  trees_wflow %>% 
  fit(data = trees)


trees_fit %>% 
  extract_fit_parsnip() %>% 
  tidy()

# visual
trees_fit %>% 
  extract_fit_parsnip() %>% 
  check_model()




# Tidymodels Version ----

# * Create Ridge Regression Model ----
# penalty == lambda
# mixture == alpha
# Note: parsinp allows for a formula method (formula specified in recipe above)
# Remember that glmnet() require a matrix specification


# Create training/testing data
trees_split <- initial_split(trees)
trees_train <- training(trees_split)
trees_test <- testing(trees_split)


# picking Dr. Smirnova's best lambda estimate; can estimate with tune() - see below
ridge_mod <-
  linear_reg(mixture = 0, penalty = 0.1629751) %>%  #validation sample or resampling can estimate this
  set_engine("glmnet")

# verify what we are doing
ridge_mod %>% 
  translate()


# create a new recipe; could use `add_step()` to recipe created above
trees_rec <- trees_train %>% 
  recipe(DeadDist ~ TreeDiam + Infest_Serv2 +  BA_20th + BA_Infest_1.2th) %>% 
  step_sqrt(all_outcomes()) %>% 
  step_corr(all_predictors()) %>% 
  step_normalize(all_numeric(), -all_outcomes()) %>% 
  step_zv(all_numeric(), -all_outcomes()) #%>% 
# prep()


trees_ridge_wflow <- 
  workflow() %>% 
  add_model(ridge_mod) %>% 
  add_recipe(trees_rec)

trees_ridge_wflow


trees_ridge_fit <- 
  trees_ridge_wflow %>% 
  fit(data = trees_train)


trees_ridge_fit %>% 
  extract_fit_parsnip() %>% 
  tidy()

trees_ridge_fit %>% 
  extract_preprocessor()

trees_ridge_fit %>% 
  extract_spec_parsnip()


# refit best model on training and evaluate on testing
last_fit(
  trees_ridge_wflow,
  trees_split
) %>%
  collect_metrics()


# verify Ridge Regression performance with standard linear regression approach
lm(sqrt(DeadDist) ~ TreeDiam + Infest_Serv2 +  BA_20th + BA_Infest_1.2th, data = trees) %>% 
  glance()




# LASSO ----

# create bootstrap samples for resampling and tuning the penalty parameter
set.seed(1234)
trees_boot <- bootstraps(trees_train)

# create a grid of tuning parameters
lambda_grid <- grid_regular(penalty(), levels = 50)


lasso_mod <-
  linear_reg(mixture = 1, penalty = tune()) %>% 
  set_engine("glmnet")

# verify what we are doing
lasso_mod %>% 
  translate()


# create workflow
trees_lasso_wflow <- 
  workflow() %>% 
  add_model(lasso_mod) %>% 
  add_recipe(trees_rec)


set.seed(2020)
lasso_grid <- tune_grid(
  trees_lasso_wflow,
  resamples = trees_boot,
  grid = lambda_grid
)


# let's look at bootstrap results
lasso_grid %>%
  collect_metrics()

# visual
lasso_grid %>%
  collect_metrics() %>%
  ggplot(aes(penalty, mean, color = .metric)) +
  geom_errorbar(aes(
    ymin = mean - std_err,
    ymax = mean + std_err
  ),
  alpha = 0.5
  ) +
  geom_line(linewidth = 1.5) +
  facet_wrap(~.metric, scales = "free", nrow = 2) +
  scale_x_log10() +
  theme(legend.position = "none")




lowest_rmse <- lasso_grid %>%
  select_best("rmse")

# update our final model with lowest rmse
final_lasso <- finalize_workflow(
  trees_lasso_wflow,
  lowest_rmse
)



final_lasso %>% 
  fit(trees_train) %>%
 extract_fit_parsnip() %>% 
  tidy()
# note that penalty (lambda) is close to zero; hence near equivalent to lm() solution  

# visual
# variable importance plot
final_lasso %>%
  fit(trees_train) %>%
  pull_workflow_fit() %>%
  vi(lambda = lowest_rmse$penalty) %>%
  mutate(
    Importance = abs(Importance),
    Variable = fct_reorder(Variable, Importance)
  ) %>%
  ggplot(aes(x = Importance, y = Variable, fill = Sign)) +
  geom_col() +
  scale_x_continuous(expand = c(0, 0)) +
  labs(y = NULL)

# or

final_lasso %>% 
  fit(trees_train) %>% 
  extract_fit_parsnip() %>% 
  vip::vip()


last_fit(
  final_lasso,
  trees_split
) %>%
  collect_metrics()

# predictor graph



# ------------------------------------

# correlation matrix

trees_cor <- trees %>% 
  correlate() 

# table
trees_cor %>% 
  rearrange() %>% 
  fashion() %>% 
  knitr::kable()

# scatterplot

plot_tree <- 
  ggplot(trees, aes(SDI_20th, DeadDist)) +
  geom_point(aes(colour = TreeDiam)) +
  scale_color_gradientn(colours = rainbow(5)) +
  theme_bw() +
  gghighlight(DeadDist > 80)

ggplotly(plot_tree)

# line graph

tree_line <- ggplot(data = trees, aes(SDI_20th, DeadDist)) +
  geom_smooth(color = "purple") +
  theme_bw() +
  labs(x = "Stand Density Index at 1/20th Acre", y = "Minimum Distance to Nearest Brood Tree")

ggplotly(tree_line)


# data table

DT::datatable(trees, options = list(bPaginate = FALSE
))


# -----------------------------

# wrangle data

set.seed(123)
splits <- trees %>%
  initial_split(prop = 0.80)

# create model

feature.plot <- trees %>%
  ggplot(aes(DeadDist, Infest_Serv2, colour == TreeDiam)) +
  geom_boxplot() +
  geom_jitter(alpha = 0.25) +
  theme_minimal(base_size = 18) +
  scale_color_viridis_d(end = 0.4, option = "viridis") 
feature.plot



