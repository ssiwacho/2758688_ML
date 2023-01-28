dat <- read.csv("https://raw.githubusercontent.com/ssiwacho/2758688_ML/main/week%201/TeacherSalaryData.csv")
head(dat)
dat<-dat[,-1]

library(dplyr)
glimpse(dat)
dat <- dat %>% 
  mutate_if(is.character, factor)
## data preprocessing
# step 1 : fit full model
full<-lm(salary ~ . ,data= train)
summary(full)

# step 2: ตัด sex ออก
reduce1 <- lm(salary ~ ., data=train[,-5])
summary(reduce1)


## fitting model


## best subset regression
install.packages("leaps")
library(leaps)
regsubset<-regsubsets(salary~. , data=dat)
sum<-summary(regsubset)


log(dat$salary, base=10)

# z-score = (x-mean)/sd

glimpse(dat)
split<-initial_split(data=dat,
                     prop = 0.8)
train<-split %>% training()
test<-split %>% testing()


train_rec<-recipe(salary ~ ., data = train)%>%
  step_normalize(salary)%>%
  step_() %>%
  step_() %>%
  step() %>%
  prep()

train_preproc <- train_rec %>%
  bake(new_data=NULL)

test_preproc<-train_rec %>% 
  bake(new_data = test)



dat<-read.csv("https://raw.githubusercontent.com/ssiwacho/2758688_ML/79a225047656cb9a22a4e1b78835b8bdd91a1d26/week%201/classification.csv")
glimpse(dat)
dat<-dat[,-1]

# splitting data
split<-initial_split(data=dat,
                     prop = 0.8)
train<-split %>% training()
test<-split %>% testing()
glimpse(train)
# data preprocessing
# helper function
rec<-recipe(Class ~ . ,data= train)%>%
  step_normalize(all_numeric_predictors())%>%
  prep()
train_preproc <- rec %>% bake(new_data = NULL)
test_preproc <- rec%>% bake(new_data = test)
## create 10-folds CV datasets
fold_data<-vfold_cv(data = train_preproc,
         v=10,
         repeats=1,
         strata = Class)
## model specification (with random grid search)
tree0<-decision_tree(mode="classification")

tree<-decision_tree(mode = "classification",
              cost_complexity = tune(),
              tree_depth = tune(),
              min_n = tune())

tree_workflow<- workflow() %>%
  add_recipe(rec) %>% # preprocessing
  add_model(tree)     # model specification

# create random grid

dt_grid2<-grid_random(hardhat::extract_parameter_set_dials(tree_workflow),
            size=20)
set.seed(002)
dt_grid <- grid_random(parameters(tree_workflow),
            size=20)
# create custom evaluation metric
eval_metric <- metric_set(roc_auc, sens, spec)

# fitting model (hyperparameter tuning)
tree_tune_fit <- tree_workflow %>% tune_grid(resamples = fold_data,
                            grid = dt_grid,
                            metric = eval_metric)
tree_tune_fit %>% 
  collect_metrics()


tree_tune_fit %>% 
  collect_metrics(summarize= F)%>%
  group_by(cost_complexity,tree_depth, min_n ,.metric)%>%
  summarise(mean = mean(.estimate),
            sd = sd(.estimate),
            min = min(.estimate),
            med = median(.estimate),
            max = max(.estimate))
show_best(tree_tune_fit, n=5)
mybest <- select_best(tree_tune_fit, metric="roc_auc")
### ---- end of hyperparameter tuning

### my best decision tree
dt_final <- tree_workflow %>%
  finalize_workflow(parameters = mybest)%>%
  last_fit(split = split)
dt_final %>%
  collect_metrics()
pred<-dt_final %>%
  collect_predictions()
pred %>% conf_mat(truth = Class,
                  estimate = .pred_class)%>% summary()
pred%>%roc_curve(truth = Class, .pred_drop)%>%
  autoplot()






?finalize_model

?tune_grid

?grid_random










