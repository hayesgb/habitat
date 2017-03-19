import re

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler, LabelBinarizer
from sklearn.utils.validation import check_is_fitted
from sklearn.externals import joblib

import filenames as filenames

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
        else:
            Xt = X[self.columns].astype(float).values.reshape(-1,1)

        if self.columns == 'age':
            Xt = X[self.columns].fillna(X[self.columns].median()).values.reshape(-1,1)
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
        # Else, treat ages as numerical variables, and fill in the median ages
        ages = df['age'].to_frame()
        ages.fillna(ages.median(), inplace=True)
        dfs.append(ages)
    merged_df = pd.DataFrame()
    for df_ in dfs:
        if merged_df.empty:
            merged_df = df_
        else:
            merged_df = pd.merge(merged_df, df_, left_index=True, right_index=True)
    merged_df['made second donation'] = df['made second donation'].to_frame()
#    print(merged_df.head())
#    input('')
    return merged_df


def _clean_categorical(df, column):
    df[column] = df[column].str.lower()
    df[column].fillna('unknown', inplace=True)
#    print('Cleaned up {}'.format(column))
#    print(df[column])
#    input('')
    return df


def main(df=filenames.combined_df, zipcodes=filenames.zipcode_dfile, predict=False):
    if predict is False:
        combined_df = pd.read_csv(df, index_col='id', low_memory=False)
        combined_df = _convert_dates(combined_df)
        combined_df = _clean_categorical(combined_df, column='marital status')
        combined_df = _split_email(combined_df, cutoff=25)
        zipcode_df = pd.read_csv(zipcodes, index_col='id')
        zipcode_df.index = zipcode_df.index.astype(str)
        zipcode_df[['main zipcode', 'secondary zipcode']] = zipcode_df[['main zipcode', 'secondary zipcode']].astype(str)
        zipcode_trimmer = CountTrimmer(replacement_string='00000')
        zipcode_df = zipcode_trimmer.fit_transform(zipcode_df, column='main zipcode')
        joblib.dump(zipcode_trimmer, filenames.zipcode_trimmer)
    if predict is True:
        print('Loading datafile & preparing to make predictions...')
        combined_df = pd.read_csv(df, index_col='id', low_memory=False)
        combined_df = _convert_dates(combined_df)
        combined_df = _clean_categorical(combined_df, column='marital status')
        combined_df = _split_email(combined_df, cutoff=25, predict=True)
        print('Loading zipcode datafile to make predictions...')
        zipcode_df = pd.read_csv(zipcodes, index_col='id')
        zipcode_df.index = zipcode_df.index.astype(str)
        zipcode_df[['main zipcode', 'secondary zipcode']] = zipcode_df[['main zipcode', 'secondary zipcode']].astype(str)
        zipcode_trimmer = joblib.load(filenames.zipcode_trimmer)
        zipcode_df = zipcode_trimmer.transform(zipcode_df)
        # Load the trained pipeline and classifier models
        pipeline = joblib.load(filenames.classifier_pipeline)
        clf_ = joblib.load(filenames.trained_classifier)                # assign the saved classifier
    combined_df = pd.merge(combined_df, zipcode_df, left_index=True, right_index=True)
    print(combined_df.head())
    print('Combined DF columns are:  ')
    print(combined_df.columns.tolist())
    print('Shape of the combined DF is:  {}'.format(combined_df.shape))
    input('Press <enter> to continue...')

    if predict is False:
    #    lr = LogisticRegression()
    #    gnb = GaussianNB()
    #    svc = LinearSVC()
        rfc = RandomForestClassifier()
    #    knn_clf = KNeighborsClassifier()
    #    centroid_clf = NearestCentroid()
    #    xgb = xgboost.XGBClassifier()
        for clf, name in [#(lr, 'Logistic'),
                          #(gnb, 'Naive Bayes'),
                          #(svc, 'Support Vector Classifier'),
                          (rfc, 'Random Forest Classifier'),
                          #(knn_clf, 'KNN Classifier'),
                          #(centroid_clf, 'Centroid Classifier'),
                          #(xgb, 'XGBoost Classifier')
                          ]:
            scores = []
            tscv = TimeSeriesSplit(n_splits=10)
            for train_idx, test_idx in tscv.split(combined_df):
                X_train, X_test = combined_df.iloc[train_idx], combined_df.iloc[test_idx]
                y_train, y_test = combined_df['made second donation'].iloc[train_idx], combined_df['made second donation'].iloc[test_idx]
                pipeline = Pipeline([
                    # Use FeatureUnion to grab columns and binarize them using labelbinarizer
                    ('features', FeatureUnion([
                        ('categorical_features', FeatureUnion([
                            # Select, transform, and join labels as shown below
                             ('transform_gender', Pipeline([
                                ('gender_selector', ColumnSelector(columns='gender')),
                                ('gender_binarizer', LabelBinarizer()),
                                ])),
                             ('transform_org', Pipeline([
                                ('org_selector', ColumnSelector(columns ='is org contact?')),
                                ('org_binarizer', LabelBinarizer()),
                                ])),
                             ('transform_persona', Pipeline([
                                ('persona_selector', ColumnSelector(columns='persona')),
                                ('persona_binarizer', LabelBinarizer()),
                                ])),
                             ('transform_lifecycle', Pipeline([
                                ('lifecycle_selector', ColumnSelector(columns='lifecycle stage')),
                                ('lifecycle_binarizer', LabelBinarizer()),
                                ])),
                             ('transform_email', Pipeline([
                                ('email_selector', ColumnSelector(columns='secondary domain')),
                                ('email_binarizer', LabelBinarizer()),
                                ])),
                             ('transform_tld', Pipeline([
                                ('tld_selector', ColumnSelector(columns='tld')),
                                ('tld_binarizer', LabelBinarizer()),
                                ])),
                            ('transform_zipcodes', Pipeline([
                                ('zipcode_selector', ColumnSelector(columns='main zipcode')),
                                ('zipcode_binarizer', LabelBinarizer()),
                            ])),
                            ('transform_maritals', Pipeline([
                                ('marital_status_selector', ColumnSelector(columns='marital status')),
                                ('marital_status_binarizer', LabelBinarizer()),
                            ])),
                            ])),
                        # Add selected continuous data
                        ('get and scale continuous_features', Pipeline([
                            ('continuous_features', FeatureUnion([
                                ('add_first_gift_amount', ColumnSelector(columns='first gift amount')),
                                ('add_first_gift_date', ColumnSelector(columns='first gift date')),
                                ('add_age_values', ColumnSelector(columns='age')),
                            ])),
                            ('scale_continuous', StandardScaler())
                            ]))
                        ]))
                    ])
                # Fit and transform the training data, then train the classifier
                Xt_train = pipeline.fit_transform(X_train)
                clf_ = clf.fit(Xt_train, y_train)
                joblib.dump(pipeline, filenames.classifier_pipeline)
                joblib.dump(clf, filenames.trained_classifier)
                # Transform the test data and predict the results
                Xt_test = pipeline.transform(X_test)
                yhat = clf_.predict(Xt_test)
                scores.append(accuracy_score(y_test, yhat))
            print('Training accuracy score for {} is:  {}, +/- {}'.format(name, np.mean(scores), np.std(scores)*2))
            print('Scores are:  {}'.format(scores))

        # Transform the entire dataset and predict the results
        Xt = pipeline.transform(combined_df)
        yhat = clf_.predict(Xt)

        y = combined_df['made second donation']
        scores.append(accuracy_score(y, yhat))
        print('Overall accuracy score for {} is:  {}'.format(name, np.mean(scores)))

        target_names = ['class 0', 'class 1']
        if hasattr(clf_, 'predict_proba'):
            Y_proba = clf_.predict_proba(Xt)
            Y_proba = Y_proba[:,1]
        elif hasattr(clf_, 'decision_function'):
            prob_pos = clf_.decision_function(Xt)
            Y_proba = prob_pos
        else:
            Y_proba = None
        print('Classification report for {}'.format(name))
        print(classification_report(y, yhat, target_names=target_names))
        if Y_proba is not None:
            print('ROC AUC Score is:  {}'.format(roc_auc_score(y, yhat)))
        else:
            print(Y_proba)
            print('Can not calculate ROC AUC Score.')

        return clf_, y, yhat


    elif predict is True:
        # Transform the entire dataset and predict the results
        Xt = pipeline.transform(combined_df)
        yhat = clf_.predict(Xt)
        print(yhat)
    else:
        print('Something went wrong.')


if __name__ == '__main__':
    main()