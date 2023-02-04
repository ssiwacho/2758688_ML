dat<-read.csv("https://raw.githubusercontent.com/ssiwacho/2758688_ML/main/week%201/TeacherSalaryData.csv")
glimpse(dat)
dat<-dat[,-1]
library(dplyr)
dat$rank <- factor(dat$rank)
dat$discipline<-factor(dat$discipline)
dat %>% 
  select(salary, starts_with("yrs")) %>%
  cor()


p1<-dat %>%
  ggplot(aes(x = yrs.since.phd, y= salary))+
  geom_point()

p2<-dat %>%
  ggplot(aes(x = yrs.service, y= salary))+
  geom_point()
library(gridExtra)
grid.arrange(p1 ,p2, ncol=2)

dat %>%
  pivot_longer(cols=c("yrs.since.phd","yrs.service"),
               names_to="predictor",values_to="value")%>%
  ggplot()+
  geom_point(aes(x=value, y=salary))+
  facet_wrap(.~ factor(predictor),
             scales = "free_x")

### ---- scate
plot(x =dat$yrs.since.phd,  y= dat$salary)
abline(lm(salary ~yrs.since.phd, data=dat))
### ---


plot(x =dat$rank,  y= dat$salary)
plot(x =dat$rank,  y= dat$discipline)




dat %>%
  ggplot(aes(x = yrs.since.phd, y= salary))+
  geom_point(aes(col = factor(sex)))+
  geom_smooth(aes(col = factor(sex)), method="lm")



# data splitting
split<-initial_split(data =dat, prop = 0.8)
train <- training(split)
test <- testing(split)

### data preprocessing
salary_rec <- recipe(salary ~ ., data=train) %>%
  step_string2factor(sex) %>%
  step_normalize(all_numeric_predictors()) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_log(salary)%>%
  step_interact(terms = ~ starts_with("yrs"):sex_Male + starts_with("yrs"):discipline_B)

# resampling
train_resamples <- vfold_cv(data = train, v = 5)

# model specification (parsnip)
regularized_mod <- linear_reg(penalty = tune(), mixture = tune()) %>%
  set_engine("glmnet") %>%
  set_mode("regression")

## create workflow
regularized_mod_wk <- workflow() %>%
  add_recipe(salary_rec) %>%
  add_model(regularized_mod)

# create grid of penalty
p<-parameters(penalty(), mixture())
grid1<-grid_regular(p, levels=5)
grid2<-grid_random(p, size=30)

eval_metric <- metric_set(rsq, rmse, mae)

tune_fit <-regularized_mod_wk %>% 
  tune_grid(
    resamples = train_resamples,
    grid = grid2,
    metrics = eval_metric,
    control = control_grid(verbose = TRUE, save_pred = TRUE,
                           parallel_over = "everything")
  )

tune_fit %>% collect_metrics()
tune_fit %>% autoplot()
show_best(tune_fit, n=10, metric = "rsq")
best <- show_best(tune_fit, n=10, metric = "rsq")[1,]

last_fit_reg<- regularized_mod_wk %>%
  finalize_workflow(best) %>%
  last_fit(split)

last_fit_reg %>% collect_metrics()
last_fit_reg %>% collect_predictions() %>%
  ggplot(aes(x=salary, y=.pred))+
  geom_point()






















































