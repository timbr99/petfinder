import os
import json

import numpy as np
import pandas as pd
import itertools
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn import svm
import statsmodels.api as sm


class ModelSelector:
    """
    A class that manages the linear regression model selection process.
    """

    def __init__(self, dataset_path, train_test_json_path):
        """
        Initialize ModelSelector class.
        
        :param dataset_path: path of dataset
        :param train_test_json_path: path to JSON with IDs of images used for 
                                     testing the model (this is used so we can 
                                     compare our linear models to the 
                                     transformer model)
                                    
        :return: None              
        """

        self.data = pd.read_csv(dataset_path)

        with open(train_test_json_path) as f:
            self.train_test_json = json.load(f)

    def prepare_train_test_data(self):
        """
        Splits up the imported dataset into a training and test 
        set using the train_test_json. 
        
        :return: train dataframe, train labels dataframe, test dataframe,
                 test labels dataframe
        """

        test_id = []
        test_id_list = self.train_test_json['test']
        for each_id in test_id_list:
            clean_id = os.path.basename(each_id).replace('.jpg', '')
            test_id.append(clean_id)

        train = self.data[-self.data['Id'].isin(test_id)]
        train_clean = train.drop(['Id', 'Pawpularity'], axis=1)
        train_labels = train.loc[:, 'Pawpularity']

        test = self.data[self.data['Id'].isin(test_id)]
        test_clean = test.drop(['Id', 'Pawpularity'], axis=1)
        test_labels = test.loc[:, 'Pawpularity']

        return train_clean, train_labels, test_clean, test_labels

    def find_all_regressor_combos(self, train):
        """
        Finds all possible regressor combinations from train data set.
        
        :param train: train dataframe
        
        :return: list containing tuples of regressor combinations
        """
        regressors = train.columns
        regressor_combinations = []
        for i in range(len(regressors) + 1):
            combinations_object = itertools.combinations(regressors, i)
            combinations_list = list(combinations_object)
            regressor_combinations.extend(list(combinations_list))

        return regressor_combinations

    def best_model_cv(self, train, train_labels, regressor_combinations, k_folds):
        """
        This function creates a linear model for each possible combination of 
        regressors in order to determine which set of regressors creates the 
        best model. This is done through cross validation using RMSE as the
        metric for model comparison.
        
        :param train: train dataframe
        :param train_labels: train labels dataframe
        :param regressor_combinations: list of all possible regressor combos
        :param k_folds: number of folds desired during cross validation process
        
        :return: Tuple with best regressor combination and linear model 
                 constructed from these regressors
        """

        regr = LinearRegression()
        scores = []
        for i, combination in enumerate(regressor_combinations):
            if len(combination) == 0:  # Ignore combination with no regressors.
                scores.append(-1337)
                continue
            train_clean_mod = train[list(combination)]
            score = cross_val_score(regr, train_clean_mod, train_labels,
                                    cv=k_folds,
                                    scoring='neg_root_mean_squared_error').mean()
            scores.append(score)

        best_regressor_combo = regressor_combinations[scores.index(max(scores))]

        best_model = LinearRegression().fit(train[list(best_regressor_combo)],
                                            train_labels)

        return best_regressor_combo, best_model

    def best_model_aic(self, train, train_labels, regressor_combinations):
        """
        This function creates a linear model for each possible combination of 
        regressors in order to determine which set of regressors creates the 
        best model. This is done by using AIC score as the metric for model
        comparison.
        
        :param train: train dataframe
        :param train_labels: train labels dataframe
        :param regressor_combinations: list of all possible regressor combos
        
        :return: Tuple with best regressor combination and linear model 
                 constructed from these regressors
        """
        aic_scores = []
        for i, combination in enumerate(regressor_combinations):
            if len(combination) == 0:  # Ignore combination with no regressors.
                aic_scores.append(999999)
                continue
            train_clean_mod = train[list(combination)]
            train_clean_mod = sm.add_constant(train_clean_mod)
            regr = sm.OLS(train_labels, train_clean_mod).fit()
            aic_scores.append(regr.aic)

        best_regressor_combo = regressor_combinations[aic_scores.index(min(aic_scores))]

        aic_train = train[list(best_regressor_combo)]
        aic_train = sm.add_constant(aic_train)
        best_model = sm.OLS(train_labels, aic_train).fit()

        return best_regressor_combo, best_model

    def model_rmse(self, model, test, test_labels):
        """
        This function calculates the RMSE score of your model of choice. The
        input model takes in data from the test set to generate predictions. 
        These prediction alongside the test labels are used to calculate RMSE.
        
        :param model: model that you want to evaluate
        :param test: test dataframe
        :param test_labels: test labels dataframe 
        
        :return: rmse score 
        """
        model_pred = model.predict(test)
        rmse_score = np.sqrt(mean_squared_error(test_labels, model_pred))

        return rmse_score

    def baseline_rmse(self, train_labels, test_labels):
        """
        Calculates RMSE score if you use the mean of the training labels
        as your prediction for the test set. Serves as a baseline RMSE score
        to compare to the RMSE scores generated from the linear models.
        
        :param train_labels: train labels data
        :param test_labels: test labels data
        
        :return: rmse score 
        """

        baseline = mean_squared_error([train_labels.mean()] * len(test_labels),
                                      test_labels,
                                      squared=False)

        return baseline
