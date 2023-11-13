# arrays / DataFrames
import pandas as pd
import numpy as np
# os
import os
# GridSearch
from sklearn.model_selection import GridSearchCV
# Metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
# Decision Tree
from sklearn.tree import DecisionTreeClassifier as dt, plot_tree, export_text
# Regression
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import TweedieRegressor
from sklearn.linear_model import LassoLars

# KNN
from sklearn.neighbors import KNeighborsClassifier
# scaler
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, QuantileTransformer
# splitter
from sklearn.model_selection import train_test_split
# imputer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
# model used
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import Dense, Flatten

metrics_df = pd.DataFrame()

def metrics_dataframe(model,RMSE,r2):
    '''
    Keep track and automatically append data to compare models.
    '''
    metrics_df = pd.DataFrame(data=[
            {
                'model':model,
                'rmse':RMSE,
                'r^2':r2
            }
            ])
    return metrics_df

def save_metrics(df, model, RMSE, r2):
    '''
    Used to automatically save metrics data on to a dataframe, would not allow for
    duplicate model names to be saved
    '''

    df.loc[len(df)] = [model, RMSE, r2]
    df = df[~df.duplicated('model')]
    return df

def model_df(df,scaler='MinMax'):
    '''
    Function to create a model ready dataframe.
    '''
    
    # dropping columns
    model_df = df.drop(columns=['artist','title_medium','dimension_in','date_sold','auction_house'])
    
    # scaling data
            # selecting
    if scaler == 'MinMax':
        scaler = MinMaxScaler()
    elif scaler == 'Standard':
        scaler = StandardScaler()
    elif scaler == 'Robust':
        scaler = RobustScaler()
    
    df_scaled = scaler.fit_transform(model_df)
    df_scaled = pd.DataFrame(df_scaled, columns=model_df.columns)
    print (f'Scaler with params of:\n----\n{scaler.get_params()}')
    return model_df, df_scaled


def linear_regression_compiled(x_train, y_train, x_validate, y_validate, target_mean, target_std,metrics_df=metrics_df, test= True):
    '''
    Generates the linear regression sklearn model.
    Finds the best fit parameters using GridSearchCV by SKLearn
    x_train = features
    y_train = target
    target_mean = mean of the original target variable (before transformations)
    target_std = standard deviation of the original target variable (before transformations)
    '''
    # Parameters defined for GridSearch, train model
    param_grid = {'fit_intercept': [True, False], 'n_jobs': [-1, 1]}
    linreg = LinearRegression()
    linreg_gs = GridSearchCV(linreg, param_grid, cv=5)
    linreg_gs.fit(x_train, y_train)

    # Predictions based on trained model
    model_prediction_train = linreg_gs.predict(x_train)
    model_prediction_test = linreg_gs.predict(x_validate)

    # Inverse transform predictions
    model_prediction_train = (model_prediction_train * target_std) + target_mean
    model_prediction_test = (model_prediction_test * target_std) + target_mean

    # Output best parameters
    print(f'Best fit parameters (Determined by GridSearchCV): {linreg_gs.best_params_}')

    # Generate metrics
    mse_train = mean_squared_error((y_train * target_std) + target_mean, model_prediction_train)
    rmse_train = np.sqrt(mse_train)
    r2_train = r2_score((y_train * target_std) + target_mean, model_prediction_train)
    print(f'Training MSE: {mse_train}, Training RMSE: {rmse_train}, Training R^2: {r2_train}')

    # Test metrics
    mse_test = mean_squared_error((y_validate * target_std) + target_mean, model_prediction_test)
    rmse_test = np.sqrt(mse_test)
    r2_test = r2_score((y_validate * target_std) + target_mean, model_prediction_test)
    print(f'Test MSE: {mse_test}, Test RMSE: {rmse_test}, Test R^2: {r2_test}')

    # Defining if test or val
    if test == True:
        model = 'test-OLS'
    else:
        model = 'val-OLS'
    # Evaluated df
    save_metrics(df = metrics_df, model = model, RMSE = rmse_test, r2= r2_test)

    # Return the prediction array
    return model_prediction_test, metrics_df

def lasso_lars_compiled(x_train, y_train, x_validate, y_validate, target_mean, target_std, metrics_df=metrics_df, test = True):
    '''
    Generates the linear regression sklearn model.
    Finds the best fit parameters using GridSearchCV by SKLearn
    x_train = features
    y_train = target
    target_mean = mean of the original target variable (before transformations)
    target_std = standard deviation of the original target variable (before transformations)
    '''
    # Parameters defined for GridSearch, train model
    param_grid = {'fit_intercept': [True, False], 'alpha': [0.1, 0.01,0.001,0.0001]}
    lasso_lars = LassoLars()
    lasso_lars_gs = GridSearchCV(lasso_lars, param_grid, cv=5)
    lasso_lars_gs.fit(x_train, y_train)

    # Predictions based on trained model
    model_prediction_train = lasso_lars_gs.predict(x_train)
    model_prediction_test = lasso_lars_gs.predict(x_validate)

    # Inverse transform predictions
    model_prediction_train = (model_prediction_train * target_std) + target_mean
    model_prediction_test = (model_prediction_test * target_std) + target_mean

    # Output best parameters
    print(f'Best fit parameters (Determined by GridSearchCV): {lasso_lars_gs.best_params_}')

    # Generate metrics
    mse_train = mean_squared_error((y_train * target_std) + target_mean, model_prediction_train)
    rmse_train = np.sqrt(mse_train)
    r2_train = r2_score((y_train * target_std) + target_mean, model_prediction_train)
    print(f'Training MSE: {mse_train}, Training RMSE: {rmse_train}, Training R^2: {r2_train}')

    # Test metrics
    mse_test = mean_squared_error((y_validate * target_std) + target_mean, model_prediction_test)
    rmse_test = np.sqrt(mse_test)
    r2_test = r2_score((y_validate * target_std) + target_mean, model_prediction_test)
    print(f'Test MSE: {mse_test}, Test RMSE: {rmse_test}, Test R^2: {r2_test}')

    # Defining val or test
    if test == True:
        model = 'test-Lasso_Lars'
    else:
        model = 'val-Lasso_Lars'

    save_metrics(df = metrics_df, model = model, RMSE = rmse_test, r2 = r2_test)

    # Return the prediction array
    return model_prediction_test, metrics_df

def tweedie_regressor_compiled(x_train, y_train, x_validate, y_validate, target_mean, target_std,metrics_df=metrics_df, test=True):
    '''
    Generates the TweedieRegressor sklearn model.
    Finds the best fit parameters using GridSearchCV by SKLearn
    x_train = features
    y_train = target
    target_mean = mean of the original target variable (before transformations)
    target_std = standard deviation of the original target variable (before transformations)
    '''
    # Parameters defined for GridSearch, train model
    param_grid = {'power': [0, 1, 2], 'alpha': [0.1, 0.01, 0.001, 0.0001]}
    tweedie = TweedieRegressor()
    tweedie_gs = GridSearchCV(tweedie, param_grid, cv=5)
    tweedie_gs.fit(x_train, y_train)

    # Predictions based on trained model
    model_prediction_train = tweedie_gs.predict(x_train)
    model_prediction_test = tweedie_gs.predict(x_validate)

    # Inverse transform predictions
    model_prediction_train = (model_prediction_train * target_std) + target_mean
    model_prediction_test = (model_prediction_test * target_std) + target_mean

    # Output best parameters
    print(f'Best fit parameters (Determined by GridSearchCV): {tweedie_gs.best_params_}')

    # Generate metrics
    mse_train = mean_squared_error((y_train * target_std) + target_mean, model_prediction_train)
    rmse_train = np.sqrt(mse_train)
    r2_train = r2_score((y_train * target_std) + target_mean, model_prediction_train)
    print(f'Training MSE: {mse_train}, Training RMSE: {rmse_train}, Training R^2: {r2_train}')

    # Test metrics
    mse_test = mean_squared_error((y_validate * target_std) + target_mean, model_prediction_test)
    rmse_test = np.sqrt(mse_test)
    r2_test = r2_score((y_validate * target_std) + target_mean, model_prediction_test)
    print(f'Test MSE: {mse_test}, Test RMSE: {rmse_test}, Test R^2: {r2_test}')

    # Defining val or test
    if test == True:
        model = 'test-Tweedie'
    else:
        model = 'val-Tweedie'
    
    # Eval DF
    save_metrics(df = metrics_df, model = model, RMSE = rmse_test, r2= r2_test)

    # Return the prediction array
    return model_prediction_test, metrics_df