import re
import os
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegressionCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, recall_score, precision_score
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer
from sklearn.utils.validation import check_is_fitted
from sklearn.externals import joblib
from sklearn_pandas import DataFrameMapper

from .. import filenames as filenames


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
        X = X.astype(np.float64)
        mn = np.nanmean(X, axis=0).reshape(-1,1)
        inds = np.where(np.isnan(X))
#        print(X.shape, mn.shape)
#        print(inds)
        X[inds] = np.take(mn, inds[1])
        try:
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

        except ValueError:
            print('Caught a problem binning data...')
            print(X)
            return None

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


def _convert_dates(df):
    '''
    Convert the date columns to datetimes and then to ordinal values for gift dates & indicator value for appeal Y/N
    :param df:
    :return: dataframe with date values cleaned up and ready for making classifier predictions.
    '''
    print('Converting dates to datetime...')
    date_cols = ['unemployment_date']
    for d_ in date_cols:
        df[d_] = pd.to_datetime(df[d_], yearfirst=True)
    df.sort_values(by=['unemployment_date'], inplace=True)
    df['unemployment_date'] = df['unemployment_date'].apply(lambda x: x.toordinal())
    return df


def _split_email(df, cutoff=10, predict=False):
    '''
    Takes in the merged dataframe and cleans up the email addresses, breaking up the tld and root domains
    :param df: Merged dataframe
    :param cutoff:  The minumum cutoff for a tld or secondary domain name to be included as unique in the dataset
    :return:
    '''
    print('Cleaning up email...')
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
    print('Done with email...')
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
        combined_df = pd.read_csv(df, dtype={'id': str}, low_memory=False)
        combined_df.set_index('id', inplace=True)
        cols = combined_df.columns.tolist()
        cols = [col.lower() for col in cols]
        combined_df.columns = cols
        y_temp = combined_df['2ndgft']
        X_train, X_test, _, _ = train_test_split(combined_df, y_temp, test_size=0.2, random_state=69, stratify=y_temp)
        combined_df = X_train
        validation_df = X_test
        validation_df.to_csv(filenames.validation_df, index_label='id')
    else:
        combined_df = pd.read_csv(df, dtype={'id': str}, low_memory=False)
        combined_df.set_index('id', inplace=True)

#    combined_df = _convert_dates(combined_df)
    combined_df['zipcode'] = combined_df['zipcode'].astype(str)
    combined_df['zipcode'] = combined_df['zipcode'].str.rstrip('.0')
    combined_df['zipcode'].fillna('00000', inplace=True)
    combined_df.reset_index(inplace=True)
    combined_df.set_index('id', inplace=True)

    if predict is True:
        # Load the trained pipeline and classifier models
        pipe = joblib.load(filenames.l1_classifier_pipeline)           # Load the trained pipeline
        clf = joblib.load(filenames.l1_trained_classifier)                # assign the saved classifier

    print('Combined DF columns are:  ')
    print(combined_df.index)
    print('Shape of the combined DF is:  {}'.format(combined_df.shape))
    print('Sort the combined DF by dates, and pass it for predictions...')
#    combined_df.sort_values(by=['unemployment_date'], inplace=True)
    combined_df.to_csv(os.path.join(filenames.interim_data, 'test_file.csv'))
    scores = []

    if predict is False:
        X = combined_df.drop(['2ndgft'], axis=1)
        y = combined_df['2ndgft']

        mapper = DataFrameMapper([
            ('gender_f', None),
            ('gender_m', None),
            ('gender_unkn', None),
            ('isorgcntct_no', None),
            ('isorgcntct_yes', None),
            ('age_29', None),
            ('age_34', None),
            ('age_39', None),
            ('age_44', None),
            ('age_49', None),
            ('age_54', None),
            ('age_59', None),
            ('age_64', None),
            ('age_69', None),
            ('age_74', None),
            ('age_79', None),
            ('age_80+', None),
            ('age_na', None),
            ('zip_550', None),
            ('zip_551', None),
            ('zip_553', None),
            ('zip_554', None),
            ('zip_555-559', None),
            ('zip_blank', None),
            ('zip_mid-atl', None),
            ('zip_mid-ohio', None),
            ('zip_midwest', None),
            ('zip_ne', None),
            ('zip_se', None),
            ('zip_south', None),
            ('zip_w-mn', None),
            ('zip_west', None),
            ('zip_wisc', None),
            ('noemail', None),
            ('gmail', None),
            ('yahoo', None),
            ('comcast', None),
            ('hotmail', None),
            ('aol', None),
            ('msn', None),
            ('wellsfargo', None),
            ('umn', None),
            ('valspar', None),
            ('charter', None),
            ('earthlink', None),
            ('tchabitat', None),
            ('mac', None),
            ('securian', None),
            ('q', None),
            ('edu', None),
            ('org', None),
            ('1stgftpmt_buschk', None),
            ('1stgftpmt_cash', None),
            ('1stgftpmt_crcrd', None),
            ('1stgftpmt_othr', None),
            ('1stgftpmt_perschk', None),
            ('aplcat1st_bike', None),
            ('aplcat1st_bwithk', None),
            ('aplcat1st_corpee', None),
           ('aplcat1st_golf', None),
           ('aplcat1st_gv2tmax', None),
           ('aplcat1st_hhnews', None),
           ('aplcat1st_lochs', None),
           ('aplcat1st_other', None),
           ('aplcat1st_routine', None),
           ('aplcat1st_special', None),
           ('aplcat1st_sprgala', None),
           ('aplcat1st_tribute', None),
           ('aplcat1st_unsolicit', None),
           ('aplcat1st_volntr', None),
           ('aplcat1st_web', None),
           ('aplcat1st_wofh', None),
           ('log10(1stgft$v2)', StandardScaler()),
           ('1stgft$v2', StandardScaler()),
            ('hbsptscrbin_0-5', None),
            ('hbsptscrbin_10', None),
            ('hbsptscrbin_15', None),
            ('hbsptscrbin_20-25', None),
            ('hbsptscrbin_30-40+', None),
           (['wealthy'], [Imputer(), StandardScaler()]),
           (['wages/household'], Imputer()),
           (['totalwages'], [Imputer(), StandardScaler()]),
           (['landarea'], [Imputer(), StandardScaler()]),
           (['housingunits'], [Imputer(), StandardScaler()]),
            ('unemploymentatfirstdonation', StandardScaler()),
            ('unemployment 6 mos before end donation', StandardScaler()),
            ('slopeofunemployment', StandardScaler()),
            ])
        pipe = Pipeline([('featurize', mapper),
#                             ('add_interactions', PolynomialFeatures(interaction_only=True))
                         ])
        # Fit and transform the training data, then train the classifier
        Xt = pipe.fit_transform(combined_df)
        Xt_ = pd.DataFrame(Xt)
        Xt_.to_csv(os.path.join(filenames.interim_data, 'Xt.csv'))
        clf = LogisticRegressionCV(class_weight='balanced', n_jobs=-1, tol=0.01, max_iter=500, penalty='l1',
                                   solver='liblinear', verbose=1, cv=5)
        clf.fit(Xt, y)
#        Xt2 = pipe.transform(X_test)
#        scores.append(clf.score(Xt2, y_test))
        print('Fit.')
        joblib.dump(pipe, filenames.l1_classifier_pipeline)            # Save the trained pipeline transformer
        joblib.dump(clf, filenames.l1_trained_classifier)                  # Save the trained classifier

        # Transform the entire dataset and predict the results
        target_names = ['class 0', 'class 1']
        Xt = pipe.transform(combined_df)
        print('The shape of Xt is:  '.format(Xt.shape))
        yhat = clf.predict(Xt)
        y = combined_df['2ndgft']
        print('Overall accuracy score is:  {}'.format(accuracy_score(y, yhat)))
        print('Precision score is:  {}'.format(precision_score(y, yhat, labels=target_names)))
        print('Recall score is:  {}'.format(recall_score(y, yhat, labels=target_names)))
        yhat_df = pd.DataFrame(yhat, columns=['yhat'], index=combined_df.index.tolist())
        yhat_df.index.name='id'
        if hasattr(clf, 'predict_proba'):
            yhat_proba = clf.predict_proba(Xt)
        elif hasattr(clf, 'decision_function'):
            yhat_proba = None
        else:
            yhat_proba = None
        print(combined_df.index)
        yhat_proba_df = pd.DataFrame(yhat_proba, index=combined_df.index, columns=['yhat_proba_0', 'yhat_proba_1'])
        print(yhat_proba_df.head())
        y_pred = pd.concat( [yhat_df, yhat_proba_df], axis=1, ignore_index=True )
        y_pred = pd.concat([y_pred, combined_df['2ndgft'].to_frame()], axis=1, ignore_index=True)

        print('Saving training set prediction results to filenames.ts_predicted_classes')
        print(y_pred.head())
        y_pred.columns = ['yhat', 'yhat_proba_0', 'yhat_proba_1', '2ndgft']
        y_pred.to_csv(filenames.ts_predicted_classes, index='id')
        print('Score is:  {}'.format(scores))
        print('Classification report ')
        print(classification_report(y, yhat, target_names=target_names))
        if yhat_proba is not None:
            print('ROC AUC Score is:  {}'.format(roc_auc_score(y, yhat)))
        else:
            print(yhat_proba)
            print('Can not calculate ROC AUC Score.')
        return clf, y, yhat

    elif predict is True:
        # Transform the entire dataset and predict the results
        Xt = pipe.transform(combined_df)
        print(Xt)
        yhat = clf.predict(Xt)
        yhat_df = pd.DataFrame(yhat, columns=['yhat'], index=combined_df.index.tolist())
        yhat_df.index.name='id'
        if hasattr(clf, 'predict_proba'):
            yhat_proba = clf.predict_proba(Xt)

        elif hasattr(clf, 'decision_function'):
            yhat_proba = None
        else:
            yhat_proba = None

        yhat_proba_df = pd.DataFrame(yhat_proba, index=combined_df.index, columns=['yhat_proba_0', 'yhat_proba_1'])
        y_pred = pd.concat([yhat_df, yhat_proba_df], axis=1, ignore_index=True)
        y_pred = pd.concat([y_pred, combined_df['2ndgft'].to_frame()], axis=1, ignore_index=True)
        y_pred.columns = ['yhat', 'yhat_proba_0', 'yhat_proba_1', '2ndgft']
        print('Saving validation prediction results to filenames.predicted_classes')
        y_pred.to_csv(filenames.predicted_classes, index='id')

    else:
        print('Something went wrong.')

    print('Done.')

if __name__ == '__main__':
    main(df=filenames.combined_df, zipcodes=filenames.zipcode_dfile, predict=False)