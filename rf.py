from data import get_data
from cv import cv
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np


def main():
    X_train, y_train, X_test = get_data()

    # hyper-parameter
    params = {'n_estimators': 500, 'max_samples': None, 'max_features': 0.4, 'n_jobs':-1}

    # create the model
    model = RandomForestRegressor(**params)

    # cross-validation
    cv(X=X_train, y=y_train, k=10, verbose=True, mode='r', model=model)

    # find result
    def res():
        df_test = pd.read_csv('data/test.csv', skipinitialspace=True, verbose=True)

        model = RandomForestRegressor(**params)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        print(predictions)
        df_test['SalePrice'] = np.exp(predictions)
        result = df_test[['Id', 'SalePrice']]
        result.to_csv('data/new/rf.csv', index=False)
        print('Done!')

    res()


if __name__ == '__main__':
    main()
