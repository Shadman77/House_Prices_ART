from data import get_data
from xgb_cv import cv
from xgboost import XGBRegressor
import pandas as pd
import numpy as np
import sys


def main():
    # get data
    X_train, y_train, X_test = get_data()

    # hyper - parameters
    params = dict(n_estimators=400, max_depth=4, eta=0.09, gamma=0, min_child_weight=0, subsample=0.8,
                  colsample_bytree=0.8, colsample_bylevel=0.6, colsample_bynode=0.2)

    # fit parameters
    #fit_params = {'early_stopping_rounds': 10, 'verbose': True}
    fit_params = {'verbose': True}

    # cross validation
    print('=============================================')

    cv(X=X_train, y=y_train, k=10, verbose=True, mode='r', model_params=params, fit_params=fit_params)

    # find result
    def res():
        df_test = pd.read_csv('data/test.csv', skipinitialspace=True, verbose=True)

        model = XGBRegressor(**params)
        model.fit(X_train, y_train, **fit_params)
        predictions = model.predict(X_test)
        print(predictions)
        df_test['SalePrice'] = np.exp(predictions)
        result = df_test[['Id', 'SalePrice']]
        result.to_csv('data/new/xgb.csv', index=False)
        print('Done!')

    res()


if __name__ == '__main__':
    main()
