from sklearn.model_selection import KFold
from numpy import asarray
import numpy as np
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from math import sqrt
from xgboost import XGBRegressor


# calculate MAPE
def mean_absolute_percentage_error(y_true, y_pred):
    y_true = asarray(y_true)
    y_pred = asarray(y_pred)
    try:
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    except:
        return 0


# Evaluate model
def evaluate_model_reg(model, X, y, verbose):
    predictions = model.predict(X)

    # Printing metrics
    r2 = r2_score(y, predictions)
    mae = mean_absolute_error(y, predictions)
    mse = mean_squared_error(y, predictions)
    rmse = sqrt(mse)
    mape = mean_absolute_percentage_error(y, predictions)
    if verbose:
        print("MAPE = %0.3f%%, MAE = %0.3f, MSE = %0.3f, RMSE = %0.3f, R Squared = %0.3f" % (mape, mae, mse, rmse, r2))
    return mape, mae, mse, rmse, r2


# Find mean and std of all metrics
def total_metrics_reg(mape_list, mae_list, mse_list, rmse_list, r2_list):#ADD VERBOSE HERE!
    result = {}
    print("Total")
    print('______________________________')
    mape_list = asarray(mape_list)
    result['mape'] = {}
    result['mape']['mean'] = mape_list.mean()
    result['mape']['std'] = mape_list.std()
    print("MAPE mean = %f%%, std = %f%%" % (mape_list.mean(), mape_list.std()))
    mae_list = asarray(mae_list)
    result['mae'] = {}
    result['mae']['mean'] = mae_list.mean()
    result['mae']['std'] = mae_list.std()
    print("MAE mean = %f, std = %f" % (mae_list.mean(), mae_list.std()))
    mse_list = asarray(mse_list)
    result['mse'] = {}
    result['mse']['mean'] = mse_list.mean()
    result['mse']['std'] = mse_list.std()
    print("MSE mean = %f, std = %f" % (mse_list.mean(), mse_list.std()))
    rmse_list = asarray(rmse_list)
    result['rmse'] = {}
    result['rmse']['mean'] = rmse_list.mean()
    result['rmse']['std'] = rmse_list.std()
    print("RMSE mean = %f, std = %f" % (rmse_list.mean(), rmse_list.std()))
    r2_list = asarray(r2_list)
    result['r2'] = {}
    result['r2']['mean'] = r2_list.mean()
    result['r2']['std'] = r2_list.std()
    print("R2 mean = %f, std = %f" % (r2_list.mean(), r2_list.std()))


# for regression
def reg(X, y, k, verbose, model_params={}, fit_params={}):
    mape_list = list()
    mae_list = list()
    mse_list = list()
    rmse_list = list()
    r2_list = list()

    # k-fold cross validation
    pass_num = 0
    for train_index, test_index in KFold(n_splits=k, shuffle=True).split(X):
        X_train, X_test = X.loc[train_index], X.loc[test_index]
        y_train, y_test = y.loc[train_index], y.loc[test_index]

        # evaluation set
        eval_set = [(X_test, y_test)]

        # create the model
        model = XGBRegressor(**model_params)

        # train the model
        model.fit(X_train, y_train, eval_set=eval_set, **fit_params)

        # verbosity
        if verbose:
            pass_num = pass_num + 1
            print('Pass number = %d' % pass_num)

        # get accuracy metrics
        mape, mae, mse, rmse, r2 = evaluate_model_reg(model=model, X=X_test, y=y_test, verbose=verbose)

        # Store individual results in a list
        mape_list.append(mape)
        mae_list.append(mae)
        mse_list.append(mse)
        rmse_list.append(rmse)
        r2_list.append(r2)

    # Total metrics
    return total_metrics_reg(mape_list, mae_list, mse_list, rmse_list, r2_list)


# cross validation
def cv(X, y, k, verbose=False, mode='r', model_params={}, fit_params={}):
    if mode == 'r':
        return reg(X=X, y=y, k=k, verbose=verbose, model_params=model_params, fit_params=fit_params)
