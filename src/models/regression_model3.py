import re
import os
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import f_regression
from sklearn.preprocessing import Imputer
from sklearn.metrics import r2_score, mean_absolute_error, median_absolute_error
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler, LabelBinarizer, PolynomialFeatures
from sklearn.utils.validation import check_is_fitted
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import TheilSenRegressor,  HuberRegressor, Lasso, LinearRegression

from .. import filenames as filenames

class ColumnSelector(TransformerMixin):
    '''
    For data held in a column in a pandas dataframe, select a subset of the data by columns and
    return it as a matrix
    '''
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self  # Not relevant, but necessary

    def transform(self, X, y=None):
        types = X[self.columns].dtypes
        if types == 'object':
            Xt = X[self.columns].astype(str).values.reshape(-1,1)
        elif types == 'float64':
            Xt = X[self.columns].astype(float).values.reshape(-1,1)
        elif types == 'int64':
            Xt = X[self.columns].astype(int).values.reshape(-1,1)
        else:
            print("Don't know this type in {} at ColumnSelector Class...".format(self.columns))
            print('Halt and check this out...')
            return None

        if self.columns == 'age':
            Xt = X[self.columns].fillna(X[self.columns].mean()).values.reshape(-1,1)
        return Xt

class HistBinner(BaseEstimator, TransformerMixin):
    '''
    Bins continuous data into groups based on defined number of bins and a histogram
    '''
    def __init__(self, num_bins=4):
        self.num_bins = num_bins

    def fit(self, X, y=None):
        return self     # Not relevant, but necessary

    def transform(self, X, y=None, **fit_params):
        check_is_fitted(self, 'num_bins')
#        print(X)
        X = X.astype(np.float64)
        _, self.bin_edges = np.histogram(X, bins=self.num_bins, density=True)
        Xt = np.zeros(shape=(X.shape[0], self.num_bins))
        for i in range(X.shape[0]):
            temp = X[i,0]
            if temp < self.bin_edges[1]:
                Xt[i,0] = 1
            elif self.bin_edges[1] >= temp < self.bin_edges[2]:
                Xt[i,1] = 1
            elif self.bin_edges[2] >= temp < self.bin_edges[3]:
                Xt[i,2] = 1
            else:
                Xt[i,3] = 1
        return Xt

class CountTrimmer(BaseEstimator, TransformerMixin):
    """Counts the occurrence of unique text in a column, returns everything greater than a defined minimum,
    and converts everything that occurs less frequently to a default value.  Usually "blank".
    """

    def __init__(self, min_cutoff=10, replacement_string='blank'):
        '''
        Initializes the transformer
        :param min_cutoff: Minimum # of times a value must occur to be included after the transformation
        :param replacement_value: Value used to replace the trimmed values
        '''
        if type(replacement_string) != str:
            raise ValueError('Replacement string must be a string')
        self.min_cutoff=min_cutoff
        self.replacement_string=replacement_string

    def fit(self, X, column):
        '''
        :param X: Incoming dataframe
        :param column: Name of the column to transform
        :return: returns an instance of self
        '''
        self.column = column

        counted_vars = X[self.column].value_counts()
#        vars_to_keep = counted_vars[counted_vars > self.min_cutoff].index
#        vars_to_keep_mask = X[column].isin(vars_to_keep)
#        X[column].ix[vars_to_keep_mask == False] = self.replacement_value
#        X[column].fillna(self.replacement_value, inplace=True)
        self.vars_to_keep_ = counted_vars[counted_vars > self.min_cutoff].index
        return self

    def transform(self, X):
        '''
        Transforms the class labels to b
        :param X:
        :param column:
        :return: Dataframe
        '''

        check_is_fitted(self, 'vars_to_keep_')
        check_is_fitted(self, 'column')

        vars_to_keep_mask = X[self.column].isin(self.vars_to_keep_)
        X[self.column].ix[vars_to_keep_mask == False] = self.replacement_string
        X[self.column].fillna(self.replacement_string, inplace=True)
        return X

    from sklearn.base import BaseEstimator,TransformerMixin

class LogTransformer(BaseEstimator,TransformerMixin):
    def __init__(self, constant=1, base='e'):
        from numpy import log,log10
        if base == 'e' or base == np.e:
            self.log = log
        elif base == '10' or base == 10:
            self.log = log10
        else:
            base_log = np.log(base)
            self.log = lambda x: np.log(x)/base_log
        self.constant = constant

    def fit(self, X, y=None):
        return self

    def transform(self, features):
        return self.log(features+self.constant)

def _convert_dates(df):
    '''
    Convert the date columns to datetimes and then to ordinal values for gift dates & indicator value for appeal Y/N
    :param df:
    :return: dataframe with date values cleaned up and ready for making classifier predictions.
    '''
    date_cols = ['first appeal date', 'first gift date', 'second gift date']
    for d_ in date_cols:
        df[d_] = pd.to_datetime(df[d_], yearfirst=True)
    df.sort_values(by=['first gift date'], inplace=True)
    df['first gift date'] = df['first gift date'].apply(lambda x: x.toordinal())
    df['first appeal date'] = np.where(df['first appeal date'].isnull(), 0, 1)
    df.drop('second gift date', inplace=True, axis=1)
    return df

def _split_email(df, cutoff=10, predict=False):
    '''
    Takes in the merged dataframe and cleans up the email addresses, breaking up the tld and root domains
    :param df: Merged dataframe
    :param cutoff:  The minumum cutoff for a tld or secondary domain name to be included as unique in the dataset
    :return:
    '''
    df['email domain'] = df['email domain'].str.lower()
    errors = []
    valid_tlds = ['com', 'edu', 'net', 'org', 'fm', np.nan, 'mn', 'fm', 'us']
    for idx in df.index:
        try:
            if len(df.loc[idx, 'email domain'].split('.')) == 2:
                df.loc[idx, 'secondary domain'] = df.loc[idx, 'email domain'].split('.')[0]
                df.loc[idx, 'tld'] = df.loc[idx, 'email domain'].split('.')[1]
            elif (len(df.loc[idx, 'secondary domain'].split('.'))) > 2:
                df.loc[idx, 'secondary domain'] = ' '.join([x for x in df.loc[idx, 'email domain'].split('.')[0:-1]])
                df.loc[idx, 'tld'] = df.loc[idx, 'email domain'].split('.')[-1]
            elif df.loc[idx, 'email domain'].endswith('com'):
                df.loc[idx, 'secondary domain'] = df.loc[idx, 'email domain'][:-3]
                df.loc[idx, 'tld'] = df.loc[idx, 'email domain'][-3:]
            else:
                pass
        except AttributeError:
            df.loc[idx, 'secondary domain'] = np.nan
            df.loc[idx, 'tld'] = np.nan
        except IndexError:
            errors.append(df.loc[idx])
    if errors:
        print('Check the errors in the email addresses...')
        print('Press <enter> to continue...')
    else:
        pass
    # Find the invalid tld's and process them
    invalid_tlds = df[df['tld'].isin(valid_tlds) == False]
    for idx in invalid_tlds.index:
        temp_ = invalid_tlds.loc[idx, 'tld']
        temp_ = str(temp_)
        if bool(re.search(r'\w[com]\w', temp_)) == True:
            df.loc[idx, 'tld'] = re.sub(r'\w[com]\w', 'com', temp_)
        elif bool(re.search(r'\w[net]+>', temp_)) == True:
            df.loc[idx, 'tld'] = re.sub(r'\w[net]+>', 'net', temp_)
        elif bool(re.search(r'\w[org]+>', temp_)) == True:
            df.loc[idx, 'tld'] = re.sub(r'\w[org]+>', 'org', temp_)
        else:
            print('No luck fixing email domain for {}'.format(idx))
            errors.append(df.loc[idx])
    if not errors:
        print('No errors were found in the cleaning of email addresses.')
    else:
        print('Errors in the email addresses...')
    # Get the set of top level domains that occur more than 3 times in the dataset.  Convert the others to NaN,
    # and then to "Blank"
#    df = _reduce_value_counts(train=True, df=df, column='tld', min_cutoff=cutoff)
    if predict is False:
        tld_trimmer = CountTrimmer()
        df = tld_trimmer.fit_transform(df, column='tld')
        joblib.dump(tld_trimmer, filenames.tld_trimmer)
        secondary_domain_trimmer = CountTrimmer()
        df = secondary_domain_trimmer.fit_transform(df, column='secondary domain')
        joblib.dump(secondary_domain_trimmer, filenames.secondary_domain_trimmer)

    else:
        print('Loading email transformers...')
        tld_trimmer = joblib.load(filenames.tld_trimmer)
        secondary_domain_trimmer = joblib.load(filenames.secondary_domain_trimmer)
        df = tld_trimmer.transform(df)
        df = secondary_domain_trimmer.transform(df)
    # Get the set of secondary email domains that occur more than 3 times in the dataset.  Convert the others to NaN,
    # and then to "Blank"
    df.drop('email domain', axis=1, inplace=True)
    return df


def _create_categorical_variables(df, quentiles,  include_age_quentiles):
    '''
    Create categorical variables where specified and return training dataframe for classifier.
    :param df: incoming dataframe
    :param q: size of the quentiles to split ages into categorical variables
    :return: Dataframe of categorical variables.
    '''
    lifecycle_dummies = pd.get_dummies(data=df['lifecycle stage'], dummy_na=True)
    email_categorical = pd.get_dummies(data=df['email'], prefix='email', dummy_na=True)
    email_tld = pd.get_dummies(data=df['tld'], prefix='tld', dummy_na=True)
    gender_categorical = pd.get_dummies(data=df['gender'], dummy_na=False)
    org_contact = pd.get_dummies(data=df['is org contact?'], prefix='org_contact', dummy_na=False)
    dfs = [lifecycle_dummies, email_categorical, email_tld, gender_categorical, org_contact]
    if include_age_quentiles:
        # If include_age_quentiles is True, create categories of ages based on the dataset.
        age_quentiles = pd.qcut(df['age'], q=quentiles)
        age_quentile_df = pd.get_dummies(data=age_quentiles, dummy_na=True, prefix='age_bin').astype(float)
        dfs.append(age_quentile_df)
    else:
        # Else, treat ages as numerical variables, and fill in the mean ages
        ages = df['age'].to_frame()
        ages.fillna(ages.mean(), inplace=True)
        dfs.append(ages)
    merged_df = pd.DataFrame()
    for df_ in dfs:
        if merged_df.empty:
            merged_df = df_
        else:
            merged_df = pd.merge(merged_df, df_, left_index=True, right_index=True)
    merged_df['made second donation'] = df['made second donation'].to_frame()
    return merged_df


def _clean_categorical(df, column):
    df[column] = df[column].str.lower()
    df[column].fillna('unknown', inplace=True)
    return df


def main(df, zipcodes=filenames.zipcode_dfile, predict=False):
    print('Loading dataframe...')
    if predict is False:
        combined_df = pd.read_csv(df, dtype={'ID': str}, low_memory=False)
        combined_df.set_index('ID', inplace=True)
        cols = combined_df.columns.tolist()
        cols = [col.lower() for col in cols]
        combined_df.columns = cols
        combined_df['2ndgft$v2'].fillna(inplace=True, value=0)
        y_temp = combined_df['2ndgft(y/n)v2']
        X_train, X_test, _, _ = train_test_split(combined_df, y_temp, test_size=0.1, random_state=69, stratify=y_temp)
        combined_df = X_train
        validation_df = X_test
        validation_df.to_csv(filenames.validation_df, index_label='id')
    else:
        combined_df = pd.read_csv(df, dtype={'id': str}, low_memory=False)
        combined_df.set_index('id', inplace=True)

        # Load the trained pipeline and classifier models
        pipeline = joblib.load(filenames.regression_pipeline)           # Load the trained pipeline
        grid = joblib.load(filenames.regression_model)                # assign the saved classifier

    combined_df.to_csv(os.path.join(filenames.interim_data, 'test_file.csv'))
    print('Combined DF columns are:  ')
    print(combined_df.columns.tolist())
    print('Shape of the combined DF is:  {}'.format(combined_df.shape))
    print('Sort the combined DF by dates, and pass it for predictions...')
    combined_df.sort_values(by=['first gift date'], inplace=True)

#    combined_df.dropna(subset=['age','totevnts','totmtngfone','population', 'wealthy','landarea'], inplace=True, axis=0)

    if predict is False:
        print('Dropping rows without second gift amounts for regression work...')
        input('Press <enter> to continue...')
        combined_df = combined_df.dropna(subset=['2ndgft$v2'], axis=0)
        X_train = combined_df
        print(X_train)
        input('')
#        y_train = X_train['2ndgft$v2'].apply(lambda x: np.log10(x+1))
        y_train = X_train['2ndgft$v2']
        print(y_train)
        input('')

        pipeline = Pipeline([
            # Use FeatureUnion to grab columns and binarize them using labelbinarizer
            ('features', FeatureUnion([
                ('categorical_features', FeatureUnion([
                    # Select, transform, and join labels as shown below
                     ('transform_gender', Pipeline([
                        ('gender_selector', ColumnSelector(columns='gender')),
                        ('gender_binarizer', LabelBinarizer()),
                        ])),
                    ('transform apl', Pipeline([
                        ('apl_selector', ColumnSelector(columns='aplcat1st')),
                        ('binarizer', LabelBinarizer()),
                    ])),
                    ('transform unsolicit', Pipeline([
                        ('selector', ColumnSelector(columns='1stgftunsolicit')),
                        ('binarize', LabelBinarizer()),
                    ])),
                    ('transform', Pipeline([
                        ('select pmnt type', ColumnSelector(columns='1stgftpmt')),
                        ('binarize', LabelBinarizer()),
                    ])),
                ])),
                # Add selected continuous data
                ('get and scale continuous_features', Pipeline([
                    ('continuous_features', FeatureUnion([
#                        ('first gift date', ColumnSelector(columns='first gift date')),
                        ('take log of first gift amt',Pipeline([
                            ('add_first_gift_amount', ColumnSelector(columns='1stgft$v2')),
#                            ('take log', LogTransformer(base=10)),
                            ])),
                        ('add_age_values', Pipeline([
                            ('get ages', ColumnSelector(columns='age')),
                            ('impute mean ages', Imputer()),
                            ])),
#                        ('add wages per household', ColumnSelector(columns='wages/household')),
                        ('add event participations', Pipeline([
                            ('select events', ColumnSelector(columns='totevnts')),
                            ('impute missing', Imputer()),
                            ('transform events', LogTransformer(base=10)),
                            ])),
                        ('add meetings', Pipeline([
                            ('select meetings', ColumnSelector(columns='totmtngfone')),
                            ('impute missing', Imputer()),
                            ('take log of meetings', LogTransformer(base=10)),
                            ])),
#                        ('add population', ColumnSelector(columns='population')),
                        ('add wealthy', Pipeline([
                            ('get wealthy column', ColumnSelector(columns='wealthy')),
                            ('impute missing', Imputer()),
                            ])),
                        ('add land area', Pipeline([
                            ('select area', ColumnSelector(columns='landarea')),
                            ('impute missing', Imputer()),
                        ])),
                        ])),
                    ('scale_continuous', StandardScaler()),
                    ])),
                ])),
#            ('add interactions', PolynomialFeatures())
            ])
        # Fit and transform the training data, then train the classifier
        Xt_train = pipeline.fit_transform(X_train)
        params = {}
        grid = GridSearchCV(estimator=HuberRegressor(), scoring='neg_median_absolute_error', cv=10, param_grid=params)
#        grid = GridSearchCV(estimator=RandomForestRegressor(), scoring='r2', cv=10, param_grid=params)
        grid.fit(Xt_train, y_train)
#        score = grid.score(Xt_train, y_train)

        selected, pvals = f_regression(Xt_train, y_train.values)  #
        np.savetxt(os.path.join(filenames.interim_data, 'Xt.csv'), Xt_train, delimiter=',')
        np.savetxt(os.path.join(filenames.interim_data, 'pvals.csv'), pvals, delimiter=',')

        yhat = grid.predict(Xt_train)
        yhat = pd.DataFrame(yhat, index=y_train.index)
        print('Overall R^2 for model is:  {}'.format(r2_score(y_train, yhat)))
        print('The Mean Absolute Error for the model is:  {}'.format(mean_absolute_error(y_train, yhat)))
        print('The Median Absolute Error for the model is:  {}'.format(median_absolute_error(y_train, yhat)))
        y_predicted = pd.concat([y_train, yhat], axis=1)
        y_predicted.columns=['y', 'yhat']
#        y_predicted = y_predicted.applymap(lambda x: 10**(x-1))
        y_predicted.to_csv(os.path.join(filenames.interim_data, 'training_predictions.csv'))
        joblib.dump(pipeline, filenames.regression_pipeline, compress=True)
        joblib.dump(grid, filenames.regression_model, compress=True)
        return y_predicted, combined_df, pipeline

    elif predict is True:
        #Transform the entire dataset and predict the results
        y = combined_df['2ndgft$v2']
        y = y.fillna(0)
        Xt = pipeline.transform(combined_df)
        np.savetxt(os.path.join(filenames.interim_data, 'Xt_prediction.csv'), Xt, delimiter=',')
        yhat = grid.predict(Xt)
#        yhat = 10**yhat
        print(yhat)
        print('R^2 score is:  {}'.format(r2_score(y, yhat)))
        print('The Mean Absolute Error is:  {}'.format(mean_absolute_error(y, yhat)))
        print('The Median Absolute Error is:  {}'.format(median_absolute_error(y, yhat)))
        yhat_df = pd.DataFrame(yhat, index=y.index)
        yhat_df = pd.concat([y, yhat_df], ignore_index=True, axis=1)
        yhat_df.to_csv(filenames.predicted_donation_amts, index_label=0, header=['y', 'yhat'])

    else:
        print('Something went wrong.')

    print('Done.')

if __name__ == '__main__':
    main(df=filenames.combined_df, zipcodes=filenames.zipcode_dfile, predict=False)