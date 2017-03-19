import numpy as np
import pandas as pd

#import statsmodel
from sklearn.linear_model import LogisticRegression

from habitat import filenames


def main():
    baseline_data = pd.read_csv(filenames.kevins_baseline_dfile, encoding='utf-16le', index_col='ID')
    baseline_data.replace('*', np.nan, inplace=True)
    baseline_data['Age'] = baseline_data['Age'].astype(float)
    baseline_data['EmailDlvrd'] = baseline_data['EmailDlvrd'].astype(float)
    baseline_data['EmailDlvrd^2'] = baseline_data['EmailDlvrd^2'].astype(float)
    baseline_data['Exp(-EmailDlvrd/248)'] = baseline_data['Exp(-EmailDlvrd/248)'].astype(float)
    baseline_data['1stGft$'] = baseline_data['1stGft$'].str.strip('$').astype(float)
    baseline_data['Median$'] = baseline_data['Median$'].astype(float)
#    print(baseline_data.dtypes)
    categorical_vars = baseline_data.select_dtypes(include=['object'])
    categorical_vars.drop('2ndGift$', inplace=True, axis=1)
    categoricals = pd.get_dummies(categorical_vars, dummy_na=True)
    continuous_vars = baseline_data.select_dtypes(include=['float', 'int64'])
    continuous_cols_to_drop = ['RESI1', 'RESI2', 'RESI3', 'RESI4', 'RESI6', 'RESI7', 'RESI8', 'RESI10', 'FITS12', 'DEVRES12', 'SDEVRES12', 'COEF12', 'DCHI12']
    continuous_vars.drop(continuous_cols_to_drop, axis=1, inplace=True)
    print(categoricals.columns.tolist())
    input('')
    print(continuous_vars.columns.tolist())
    input('')
    predictor_df = pd.merge(categoricals, continuous_vars, left_index=True, right_index=True)
    print(predictor_df.columns.tolist())
#    print(baseline_data.dtypes)
    input('')
    X = predictor_df.copy()
    X.drop('2ndGft(y/n)', inplace=True, axis=1)
    y = predictor_df.loc[:, '2ndGft(y/n)']

    model = LogisticRegression(fit_intercept=True)
    mdl = model.fit(X, y)
    print(mdl)

if __name__ == '__main__':
    main()