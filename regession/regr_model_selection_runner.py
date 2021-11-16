import regr_model_selection
import statsmodels.api as sm

petfinder = regr_model_selection.ModelSelector('petfinder data/train.csv', 'petfinder data/train_test_val.json')

# ---- Split data in training & test and find all possible regressor combinations ----
train, train_labels, test, test_labels = petfinder.prepare_train_test_data()
regr_combo = petfinder.find_all_regressor_combos(train)

# ---- Find best possible linear regression model by cross validation ----
cv_best_regressors, cv_model = petfinder.best_model_cv(train, train_labels, regr_combo, 5)
# Calculate RMSE of model
cv_test_data = test[list(cv_best_regressors)]
cv_rmse = petfinder.model_rmse(cv_model, cv_test_data, test_labels)

# ---- Find best possible linear regression model by AIC score ----
aic_best_regressors, aic_model = petfinder.best_model_aic(train, train_labels, regr_combo)
# Calculate RMSE of model
aic_test_data = test[list(aic_best_regressors)]
aic_test_data = sm.add_constant(aic_test_data)
aic_rmse = petfinder.model_rmse(aic_model, aic_test_data, test_labels)

# ---- Baseline RMSE ----
base_rmse = petfinder.baseline_rmse(train_labels, test_labels)
