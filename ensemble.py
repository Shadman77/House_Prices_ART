import pandas as pd
import numpy as np

def main():
    xgb_df = pd.read_csv('data/new/xgb.csv', skipinitialspace=True, verbose=True)
    rf_df = pd.read_csv('data/new/rf.csv', skipinitialspace=True, verbose=True)
    lin_df = pd.read_csv('data/new/lin.csv', skipinitialspace=True, verbose=True)
    gbr_df = pd.read_csv('data/new/grb.csv', skipinitialspace=True, verbose=True)

    xgb = xgb_df['SalePrice'] * 0.4
    gbr = gbr_df['SalePrice'] * 0.2
    rf = rf_df['SalePrice'] * 0.2
    lin = lin_df['SalePrice'] * 0.2
    ensem_df = xgb_df
    ensem_df['SalePrice'] = np.round(xgb + gbr + rf + lin)
    ensem_df.to_csv('data/new/ensem.csv', index=False)


if __name__ == '__main__':
    main()