# -*- coding: utf-8 -*-
#import os
#import click
#import logging
#from dotenv import find_dotenv, load_dotenv
import sys
print(sys.path)
import lat_long_calc
import pandas as pd
import utilities as utilities


import filenames

def _create_joined_df(file1, file2):
    print('Reading file 1')
    df1 = pd.read_excel(file1)#, dtype={'re constituent id':'str'})#, encoding='utf-8')
    cols = df1.columns.tolist()
    cols = [col.lower() for col in cols]
    df1.columns = cols
    df1['re constituent id'] = df1['re constituent id'].astype(str)
    df1.dropna(subset=['re constituent id'], axis=0, inplace=True)
    print('Are any of the id values NaNs?  {}'.format(df1['re constituent id'].isnull().values.any()))
    df1 = df1.set_index('re constituent id')
    l_ = df1.index.tolist()
    df1.index = utilities.remove_trailing_zeros(l_)

    print('Reading file 2')
    df2 = pd.read_csv(file2, dtype={'ID':'str'})
    cols = df2.columns.tolist()
    cols = [col.lower() for col in cols]
    df2.columns = cols
    df2['id'] = df2['id'].str.strip()
    print('Are any of the id values in df2 NaNs?  {}'.format(df2['id'].isnull().values.any()))
    print('Length of df2 is:  {}'.format(len(df2)))
    df2 = df2.set_index('id')
    l_ = df2.index.tolist()
    df2.index = utilities.remove_trailing_zeros(l_)
    print('Shape of df1 = {}'.format(df1.shape))
    print('Shape of df2 = {}'.format(df2.shape))
    print('Merging df1 and df2 on the ids in Raisers Edge...')

    df3 = pd.merge(df2, df1, left_index=True, right_index=True, how='outer')
    print('The shape of the merged dframe is:  {}'.format(df3.shape))
#    df3.to_csv(filenames.joined_dfile, index='id')
    return df3


def main(file1, file2, output_filepath, get_latlngs = False):
    '''
    Data preparation starts here.  We will read the two datafiles shown below, merge them, clean them, and break them
    into chunks for cleaning.  We will end by creating a composite datafile for analysis and prediction
    :param file1: online_datafile.csv
    :param file2: raisers_edge_datafile.csv
    :return:
    '''
    df = _create_joined_df(file1=file1, file2=file2)
    print('All of the available columns are:')
    print(df.columns.tolist())
    input('Press <enter> to continue.')
    print('Creating a giving dataframe...')
    giving_cols = ['first appeal date', 'first gift date', 'first gift amount', 'second gift date', 'second gift amount',
               'total given', 'total number of gifts', 'average gift amount', 'largest gift amount',
               'smallest gift amount', 'total unsolicited gifts']
    giving_df = df.loc[:,giving_cols]
    print('Creating a location dataframe...')
    location_cols = ['city_x', 'state', 'zip', 'city_y', 'state/region', 'zip code', 'ip country code']
    location_df = df.loc[:,location_cols]
#    print(location_df.shape)
#    input('')
    print('Creating a persona dataframe...')
    persona_cols = ['gender', 'age', 'email domain', 'is org contact?', 'persona', 'lifecycle stage', 'marital status']
    persona_df = df.loc[:, persona_cols]
    # Specify date columns, and columns containing donation amounts, and clean them up.
    tseries = ['first appeal date', 'first gift date', 'second gift date']
    donation_amounts = ['first gift amount', 'second gift amount', 'total given', 'average gift amount',
                        'largest gift amount', 'smallest gift amount']
    cleaned_giving_df = utilities.clean_giving_profile(giving_df, time_cols=tseries, donation_cols=donation_amounts)
    # Clean up the zipcode dataframe by splitting the zipcodes into main and secondary, then attempt to reconcile the
    # zipcodes.
    zipcode_df = utilities.clean_zipcodes(df=location_df)
    zipcode_df.to_csv(filenames.zipcode_dfile, index_label='id')
    if get_latlngs is True:
        lat_long_calc.create_distance_matrix(zipcode_df)
    else:
        print('Already have the pairwise distance matrix for users...')
    print('Merging the giving, location, and donation dataframes and saving to a combined file...')
    combined_df = pd.merge(cleaned_giving_df, persona_df, left_index=True, right_index=True)
    print(combined_df.columns)
    combined_df.to_csv(output_filepath, index_label='id')


if __name__=='__main__':
    main(file1=filenames.online_datafile, file2=filenames.raisers_edge_datafile, output_filepath=filenames.combined_df)

