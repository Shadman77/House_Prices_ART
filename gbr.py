from data import get_data
from cv import cv
from sklearn.ensemble import GradientBoostingRegressor
import pandas as pd
import numpy as np


def main():
    X_train, y_train, X_test = get_data()

    # hyper-parameter
    params = dict(verbose=False, learning_rate=0.09, n_estimators=300, subsample=0.8, max_depth=3)

    # create the model
    model = GradientBoostingRegressor(**params)

    # cross-validation
    cv(X=X_train, y=y_train, k=10, verbose=True, mode='r', model=model)

    # find result
    def res():
        df_test = pd.read_csv('data/test.csv', skipinitialspace=True, verbose=True)

        model = GradientBoostingRegressor(**params)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        print(predictions)
        df_test['SalePrice'] = np.exp(predictions)
        result = df_test[['Id', 'SalePrice']]
        result.to_csv('data/new/grb.csv', index=False)
        print('Done!')

    res()


if __name__ == '__main__':
    main()
