# arrays
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
# Logistic Regression
from sklearn.linear_model import LinearRegression
# KNN
from sklearn.neighbors import KNeighborsClassifier
# scaler
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, QuantileTransformer
# splitter
from sklearn.model_selection import train_test_split
# imputer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score


def model_df(df):
    '''
    Function to create a model ready dataframe.
    '''
    
    # dropping columns
    model_df = df.drop(columns=['artist','title_medium','dimension_in','date_sold','auction_house'])
    
    # scaling data
            # selecting 
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(model_df)
    df_scaled = pd.DataFrame(df_scaled, columns=model_df.columns)
    print (f'Scaler with params of:\n----\n{scaler.get_params()}')
    return model_df, df_scaled

def get_classification_report(x_test, y_pred):
    '''
    Returns classification report as a dataframe.
    '''
    report = classification_report(x_test, y_pred, output_dict=True)
    df_classification_report = pd.DataFrame(report).transpose()
    df_classification_report = df_classification_report.sort_values(by=['f1-score'], ascending=False)
    return df_classification_report

def metrics(TN,FP,FN,TP):
    '''
    True positive(TP),
        True negative (TN),
        False positive (FP),
        False negative (FN)

        Reminder:
        false pos true neg
        false neg true pos
    '''
    combined = (TP + TN + FP + FN)

    accuracy = (TP + TN) / combined

    TPR = recall = TP / (TP + FN)
    FPR = FP / (FP + TN)

    TNR = TN / (FP + TN)
    FNR = FN / (FN + TP)


    precision =  TP / (TP + FP)
    f1 =  2 * ((precision * recall) / ( precision + recall))

    support_pos = TP + FN
    support_neg = FP + TN

    print(f"Accuracy: {accuracy}\n")
    print(f"True Positive Rate/Sensitivity/Recall/Power: {TPR}")
    print(f"False Positive Rate/False Alarm Ratio/Fall-out: {FPR}")
    print(f"True Negative Rate/Specificity/Selectivity: {TNR}")
    print(f"False Negative Rate/Miss Rate: {FNR}\n")
    print(f"Precision/PPV: {precision}")
    print(f"F1 Score: {f1}\n")
    print(f"Support (0): {support_pos}")
    print(f"Support (1): {support_neg}")

def log_regression_compiled(x_train, y_train, x_validate, y_validate):
    '''
    Generates the logistic regression sklearn model.
    Finds the best fit C parameter using GridSearchCV by SKLearn
    x_train = features
    y_train = target
    '''
    # Parameters defined for GridSearch, train model
    param_grid = [
    {'penalty': ['l1'], 'solver': ['liblinear', 'saga'], 'C': np.arange(.1,5,.1)},
    {'penalty': ['l2'], 'solver': ['liblinear', 'saga', 'lbfgs', 'newton-cg'], 'C': np.arange(.1,5,.1)},
    {'penalty': ['none'], 'solver': ['lbfgs', 'newton-cg'], 'C': np.arange(.1,5,.1)}
]
    logreg_tuned = lr(max_iter=500)
    logreg_tuned_gs = GridSearchCV(logreg_tuned, param_grid, cv=5)
    logreg_tuned_gs.fit(x_train,y_train)

    # Predictions based on trained model
    y_predictions_lr_tuned = logreg_tuned_gs.predict(x_validate)
    y_predictions_lr_prob_tuned = logreg_tuned_gs.predict_proba(x_validate)

    # Output best C parameter
    print(f'Best fit parameter (Determined by GridSearchCV): {logreg_tuned_gs.best_params_}')

    # model object
    logit = lr(**logreg_tuned_gs.best_params_, random_state=4343)
    # fit
    logit.fit(x_train,y_train)
    # predict
    model_prediction = logit.predict(x_train)
    model_prediction_test = logit.predict(x_validate)

    # generate metrics
    TN, FP, FN, TP = confusion_matrix(y_train, model_prediction).ravel()
    get_classification_report(y_train,model_prediction)
    metrics(TN, FP, FN, TP)
    # test metrics
    TN, FP, FN, TP = confusion_matrix(y_validate, model_prediction_test).ravel()
    get_classification_report(y_validate,model_prediction_test)

    return



def linear_regression_compiled(x_train, y_train, x_validate, y_validate, target_mean, target_std):
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
    model_prediction_train = np.exp((model_prediction_train * target_std) + target_mean)
    model_prediction_test = np.exp((model_prediction_test * target_std) + target_mean)

    # Output best parameters
    print(f'Best fit parameters (Determined by GridSearchCV): {linreg_gs.best_params_}')

    # Generate metrics
    mse_train = mean_squared_error(np.exp((y_train * target_std) + target_mean), model_prediction_train)
    rmse_train = np.sqrt(mse_train)
    r2_train = r2_score(np.exp((y_train * target_std) + target_mean), model_prediction_train)
    print(f'Training MSE: {mse_train}, Training RMSE: {rmse_train}, Training R^2: {r2_train}')

    # Test metrics
    mse_test = mean_squared_error(np.exp((y_validate * target_std) + target_mean), model_prediction_test)
    rmse_test = np.sqrt(mse_test)
    r2_test = r2_score(np.exp((y_validate * target_std) + target_mean), model_prediction_test)
    print(f'Test MSE: {mse_test}, Test RMSE: {rmse_test}, Test R^2: {r2_test}')

    # Return the prediction array
    return model_prediction_test