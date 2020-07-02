import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import sys


def one_hot(df):
    col_list = ['MSSubClass', 'MSZoning', 'Alley', 'LotShape', 'LandContour',
                'Utilities', 'LotConfig', 'LandSlope', 'Condition1',
                'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st',
                'Exterior2nd', 'MasVnrType', 'Foundation', 'Heating', 'GarageType',
                'MiscFeature', 'SaleType', 'SaleCondition', 'Street', 'Neighborhood']

    # fill missing values
    for col in col_list:
        df[col].fillna('not_specified', inplace=True)

    # encode
    encoder = OneHotEncoder(sparse=False)
    df_encoded = pd.DataFrame(encoder.fit_transform(df[col_list]))
    df_encoded.columns = encoder.get_feature_names(col_list)
    df.drop(col_list, axis=1, inplace=True)
    df = pd.concat([df, df_encoded], axis=1)

    # df.to_csv('data/new/one-hot.csv', index=False)

    return df


# integer encoding
def int_encode(df):
    col_list = ['ExterQual', 'ExterCond',
                'BsmtQual', 'BsmtCond', 'BsmtExposure',
                'BsmtFinType1', 'BsmtFinType2', 'HeatingQC',
                'CentralAir', 'KitchenQual', 'Functional',
                'FireplaceQu', 'GarageFinish', 'GarageQual',
                'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'Electrical']

    # fill null values if any - there are some
    for col in col_list:
        df[col].fillna(0, inplace=True)

    def int_code1(x):
        vals = [0, 'Po', 'Fa', 'TA', 'Gd', 'Ex']
        return vals.index(x)

    df['ExterQual'] = df['ExterQual'].apply(int_code1)
    df['ExterCond'] = df['ExterCond'].apply(int_code1)
    df['HeatingQC'] = df['HeatingQC'].apply(int_code1)
    df['KitchenQual'] = df['KitchenQual'].apply(int_code1)

    def int_code2(x):
        vals = [0, 'NA', 'Po', 'Fa', 'TA', 'Gd', 'Ex']
        return vals.index(x)

    df['BsmtQual'] = df['BsmtQual'].apply(int_code2)
    df['BsmtCond'] = df['BsmtCond'].apply(int_code2)
    df['FireplaceQu'] = df['FireplaceQu'].apply(int_code2)
    df['GarageQual'] = df['GarageQual'].apply(int_code2)
    df['GarageCond'] = df['GarageCond'].apply(int_code2)

    def int_code3(x):
        vals = [0, 'NA', 'No', 'Mn', 'Av', 'Gd']
        return vals.index(x)

    df['BsmtExposure'] = df['BsmtExposure'].apply(int_code3)

    def int_code4(x):
        vals = [0, 'Na', 'Unf',
                'LwQ', 'Rec', 'BLQ', 'ALQ', 'GLQ']
        return vals.index(x)

    df['BsmtFinType1'] = df['BsmtFinType1'].apply(int_code4)
    df['BsmtFinType2'] = df['BsmtFinType2'].apply(int_code4)

    def int_code5(x):
        vals = [0, 'N', 'Y']
        return vals.index(x)

    df['CentralAir'] = df['CentralAir'].apply(int_code5)

    def int_code6(x):
        vals = [0, 'Sal', 'Sev', 'Maj2', 'Maj1', 'Mod',
                'Min2', 'Min1', 'Typ']
        return vals.index(x)

    df['Functional'] = df['Functional'].apply(int_code6)

    def int_code7(x):
        vals = [0, 'NA', 'Unf', 'RFn', 'Fin']
        return vals.index(x)

    df['GarageFinish'] = df['GarageFinish'].apply(int_code7)

    def int_code8(x):
        vals = [0, 'N', 'P', 'Y']
        return vals.index(x)

    df['PavedDrive'] = df['PavedDrive'].apply(int_code8)

    def int_code9(x):
        vals = [0, 'NA', 'Fa', 'TA', 'Gd', 'Ex']
        return vals.index(x)

    df['PoolQC'] = df['PoolQC'].apply(int_code9)

    def int_code10(x):
        vals = [0, 'NA', 'MnWw', 'GdWo', 'MnPrv', 'GdPrv']
        return vals.index(x)

    df['Fence'] = df['Fence'].apply(int_code10)

    def int_code11(x):
        vals = [0, 'Mix', 'FuseP', 'FuseF', 'FuseA', 'SBrkr']
        return vals.index(x)

    df['Electrical'] = df['Electrical'].apply(int_code11)

    return df


# separate labels
def separate_labels(data_set):
    labels = data_set.pop('SalePrice')
    return data_set, labels


def get_stats(data_set):
    stats = data_set.describe()
    stats = stats.transpose()
    return stats


def normalize(x, train_stats):
    return (x - train_stats['mean']) / train_stats['std']


def get_data():
    # read data
    df_train = pd.read_csv('data/train.csv', skipinitialspace=True, verbose=True)
    df_test = pd.read_csv('data/test.csv', skipinitialspace=True, verbose=True)
    print(df_train.info())
    print(df_test.info())

    # get labels
    X, y_train = separate_labels(df_train)

    # log transformation of price
    y_train = np.log(y_train)

    # combine data
    X = pd.concat([X, df_test])
    X.reset_index(drop=True, inplace=True)
    # X.to_csv('data/cleaned.csv')
    # sys.exit()

    # ont-hot encode
    X = one_hot(X)
    print(X.info())

    # integer encode
    X = int_encode(X)

    # make sure there are no nan values
    X.fillna(0, inplace=True)
    print('Null values = %d' % (X.isnull().sum().sum()))

    # drop unwanted cols
    X = X.drop(['Id'], axis=1)

    # normalize
    '''
    X_stats = get_stats(X)
    X = normalize(X, X_stats)
    print(X.info())
    '''

    # seperate data
    X_train = X.iloc[:len(df_train.index)]
    X_train.to_csv('data/new/train_x_final.csv', index=False)
    X_test = X.iloc[len(df_train.index):]

    return X_train, y_train, X_test
