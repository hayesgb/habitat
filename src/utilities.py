import os
import time

import numpy as np
import pandas as pd
from uszipcode import ZipcodeSearchEngine
from . import filenames as filenames


def drop_ignores(dframe, reference):
    '''
    Take in a dataframe and a data dictionary.  Drop any columns from the dframe that the dictionary
    states it should be ignored.
    :param dframe:  Incoming datasource
    :param reference:  Data Dictionary
    :return:  A dataframe that has had columns slated to be ignored from the dataframe
    '''
    print('Dropping ignores using data dictionary from data')
    cols = dframe.columns.tolist()              # Get a list of dframe columns
    cols = [col.lower().strip() for col in cols] # Convert dframe columns to lowercase & strip
    dframe.columns = cols                        # Replace dframe colnames with lowercase
    for col in cols:
        if dframe[col].dtype == 'object':
#            print('Changing...{}'.format(col))
            try:
                dframe[col] = dframe[col].str.lower()
                dframe[col] = dframe[col].str.strip()
            except:
                print('Could not change case of {}'.format(col))
        else:
#            print('Skipping...{}'.format(col))
            pass
    ignores = reference[reference.iloc[:,1] == 'ignore']
    dict_rows = ignores.ID.tolist()
    xref = [col for col in cols if col in dict_rows]  # Get the rows that occur as column headers
    print(xref)
    dframe.drop(xref, axis=1, inplace=True)    # Drop "ignores" from the dataset
    return dframe

def remove_trailing_zeros(idx):
    idx_ = [idx1.split('.')[0] if idx1.endswith('.0') else idx1 for idx1 in idx]
    print('Length of the returned list is:  {}'.format(len(idx_)))
    return idx_

def strip_whitespaces_from_lists(series):
    for idx in series.index:
        try:
            list_ = series[idx].tolist()
            new_list_ = [l.strip() for l in list_]
            series[idx] = new_list_
        except AttributeError:
            pass
    return series

def _convert_zip( city,state,idx, errors):
    try:
        search = ZipcodeSearchEngine()
        result = search.by_city_and_state(city, state)
        zipcode = result[0]['Zipcode']
#        print('Zipcode for {} , {} is: {}'.format(city, state, zipcode))
        return zipcode, errors
    except:
        print('Error!  Could not find city {} in state {} for id#:  {}.'.format(city, state, idx))
        zipcode = None
        errors[idx] = (city, state)
        return zipcode, errors

def _get_city(zipcodes, df, idx, fullzip):
    '''
    Where possible, match up a city with the zipcode
    :param zipcodes:
    :param df:
    :param idx:
    :param fullzip:
    :return:
    '''
    print('Getting cities for {}..'.format(idx))
    if type(df.loc[idx, 'city_x']) == str:
        zipcodes.loc[idx, fullzip[2]] = df.loc[idx, 'city_x']
    elif type(df.loc[idx, 'city_y']) == str:
        zipcodes.loc[idx, [fullzip[2]]] = (df.loc[idx, 'city_y'].split(',')[0])
    else:
        print('No apparent city match...')

    return zipcodes

def get_zipcodes(df, col1='zip', col2='zip code', impute_zipcode=True):
    '''
    Takes in the target dataframe, cleans up the zip codes and returns a finalized zip code for each id
    :param dataframe of messy zipcode data.
    :param col1 - column label of zip codes
    :param col2 - column label of zipcodes
    :param impute_zipcode - Boolean for whether to impute most frequent zipcodes on unknowns
    :return: zipcodes
    '''
    print('Parsing zip codes...')
    errors = {}
    fullzip = ['main zipcode', 'secondary zipcode', 'city']
    city_state_cols = ['city_x', 'state', 'city_y', 'state/region']
    zipcodes = pd.DataFrame(index = df.index, columns=fullzip)
    for idx in df.index:
        try:
            if (df.loc[idx, col1] == df.loc[idx, col2]) & (df.loc[idx, col1][0] != '0'):
                temp = df.loc[idx, col1][0].split('-')
                if len(temp) == 2:
                    zipcodes.loc[idx, fullzip[0]] = temp[0]
                    zipcodes.loc[idx, fullzip[1]] = temp[1]
                elif len(temp) == 1:
                    zipcodes.loc[idx, fullzip[0]] = temp[0]
                    if type(df.loc[idx, 'city_x']) == str:
                        zipcodes.loc[idx, fullzip[2]] = df.loc[idx, 'city_x']
                else:
#                    print('Could not solve identical column problem...')
                    pass
            elif len(df.loc[idx, col1]) == 2:
                zipcodes.loc[idx, fullzip[0]] = df.loc[idx, col1][0]
                zipcodes.loc[idx, fullzip[1]] = df.loc[idx, col1][1]
                print(zipcodes.loc[idx, fullzip[0]])
            #elif type(df.loc[idx, col2]) != np.float:
            #    if (len(df.loc[idx, col2]) == 2):
            #        zipcodes.loc[idx, fullzip[0]] = df.loc[idx, col2][0]
            #        zipcodes.loc[idx, fullzip[1]] = df.loc[idx, col2][1]
            elif len(df.loc[idx, col1][0]) == 5:
                zipcodes.loc[idx, fullzip[0]] = df.loc[idx, col1][0]
            if type(df.loc[idx, col2]) != np.float:
                if len(df.loc[idx, col2][0]) == 5:
                    zipcodes.loc[idx, fullzip[0]] = df.loc[idx, col1][0]
            elif len(df.loc[idx, col1][0]) == 9:
                zipcodes.loc[idx, fullzip[0]] = df.loc[idx, col1][0][0:5]
                zipcodes.loc[idx, fullzip[1]] = df.loc[idx, col1][0][5:]
            elif (type(df.loc[idx, 'city_x']) == str) & (type(df.loc[idx, 'state']) == str):
#                print('Parsing city_x and state for {}...'.format(idx))
                city1 = df.loc[idx, city_state_cols[0]]
                if len(city1.split()) > 1:
                    city1 = [word.strip('.') for word in city1]
                    city1 = ''.join(city1)
                state1 = df.loc[idx, 'state'].split(',')[0].strip()
                zipcode_, errors = _convert_zip(city1, state1, idx, errors)
                if zipcode_ is None:
#                    print('Need to write code to find zipcode if it occurs in the dataframe.')
                    time.sleep(.5)
                else:
                    zipcodes.loc[idx, fullzip[0]] = zipcode_
            elif (type(df.loc[idx, 'city_y']) == str):
#                print('Parsing city_y column ...')
                city2 = None
                state2 = None
                city_y_col = df.loc[idx, 'city_y'].split(',')
                if (len(city_y_col)) == 1:  # if 'city_y' only contains one item after converting to a list and splitting by commas
#                    print('The length of city_y_col is:  {}'.format(len(city_y_col)))
                    time.sleep(.5)
                    city2 = city_y_col[0]
                    if (type(df.loc[idx, 'state/region']) == str):
                        region_col = df.loc[idx, 'state/region'].split(',')
                        if len(region_col[0]) == 2:
                            state2 = region_col[0].strip()
#                            print('The state for {} is {}'.format(city2, state2))
                    if (city2 is not None) & (state2 is not None):
                        zipcode_, errors =  _convert_zip(city2, state2, idx, errors)
                        if zipcode_ is None:
                            pass
                        else:
                            pass
                elif (len(city_y_col)) == 2:
                    city2 = city_y_col[0]
                    state2 = city_y_col[1]
                    print('Double check col {}'.format(idx))
                    input('')
                    zipcodes.loc[idx, fullzip[0]], errors = _convert_zip(city2, state2, idx, errors)  # Changed this to city2, state2
                else:
                    # Need to search dataframe to find matching zipcode in case of inability to locate in database
#                    print('Could not resolve {}'.format(df.loc[idx]))
                    pass
            elif (type(df.loc[idx, 'state/region']) == str):
                print('Parsing state/region column...')
                region_col = df.loc[idx, 'state/region'].split(',')
                if (len(region_col)) == 1:  # and if state/region only contains one item
                    if len(region_col[0]) == 2:  # and if the state/region column is only two letters long
                        state2 = region_col[0].strip()
                elif len(region_col[0]) == 2:       # elif the state region column is more than two letters long
                    if ',' in region_col[0]:
                        city2 = region_col[0]
                        state2 = region_col[1]
                        zipcodes.loc[idx, fullzip[0]] = city2
                        zipcodes.loc[idx, fullzip[1]] = state2
                else:
#                    print('Could no solve state/region column...')
                    pass
            else:
                print('Skipped {}'.format(df.loc[idx]))
                input('')
            zipcodes = _get_city(zipcodes=zipcodes, df=df, idx=idx, fullzip=fullzip)
        except TypeError:
            print('Error created by {}'.format(idx))
            print(df[idx])
            input('')
    zipcodes.fillna('0', inplace=True)
    for idx in zipcodes.index:
        print('Reverse fit the primary zip codes...')
        if zipcodes.loc[idx, fullzip[0]] == '0':
            if zipcodes.loc[idx, fullzip[2]] != '0':
                city = zipcodes.loc[idx, fullzip[2]]
                temps_ = zipcodes[zipcodes[fullzip[2]] == city]
                best_zipcode_guess = temps_.iloc[:,0].value_counts()#.tolist()[1]
                best_zipcode_guess.dropna(inplace=True, axis=0)
                best_guess = best_zipcode_guess.idxmax()
#                print('Best guess for {} is {}'.format(city, best_guess))
                zipcodes.loc[idx, fullzip[0]] = best_guess
    # Strip any letters out of zipcodes in both column 1 and columns 2
    zipcodes[fullzip[0]] = [zcode.lstrip('a-zA-Z') for zcode in zipcodes[fullzip[0]]]
    zipcodes[fullzip[1]] = [zcode.lstrip('a-zA-Z') for zcode in zipcodes[fullzip[1]]]
    zipcodes[fullzip[0]] = [zcode.rjust(5, '0') for zcode in zipcodes[fullzip[0]]]
    zipcodes[fullzip[1]] = [zcode.rjust(4, '0') for zcode in zipcodes[fullzip[1]]]
    print('The number of undefined zipcodes is:  {}'.format(len(zipcodes[zipcodes[fullzip[0]] == '00000'])))
    print('The number of undefined secondary zipcodes is {}'.format(len(zipcodes[zipcodes[fullzip[1]] == '0000'])))
    print('They\'re save in the file named undefined.csv')
    undefined = zipcodes[zipcodes[fullzip[0]] == '0000']
    undefined.to_csv(os.path.join(filenames.data_folder, 'undefined.csv'))

    # If impute zipcodes is True, drop any zipcodes that are zero and get the mode
    # as the zipcode to impute to unknown zipcodes
    if impute_zipcode is True:
        imputed_zipcode = zipcodes[zipcodes[fullzip[0]] != '00000']
        imputed_zipcode = imputed_zipcode[fullzip[0]].value_counts().idxmax()
        print('We will impute the most frequently occuring zip code on those with no defined data.')
        print('The most common zipcode in the main dataset is:  {}'.format(imputed_zipcode))
        zipcodes[fullzip[0]].replace('00000', imputed_zipcode, inplace=True)
    else:
        print('Will need to deal with unknown zipcodes as NaNs...')
    print('Saving zipcode errors to a csv file titled:  zip_errors.csv')
    zip_errors = pd.DataFrame(errors)
    zip_errors.to_csv(os.path.join(filenames.data_folder, 'zipcode_errors.csv'))

    # Get demographic data
    zipcode_demographic_dict={}
    with ZipcodeSearchEngine() as search:
        for idx in zipcodes.index:
            if idx in zipcode_demographic_dict:
                pass
            else:
                zipcode_demographic_dict[idx] = search.by_zipcode(zipcodes.loc[idx, 'main zipcode']).to_dict()
    zipcode_demographic_df = pd.DataFrame.from_dict(zipcode_demographic_dict, orient='index')
    print(zipcode_demographic_df.columns.tolist())
    cols = zipcode_demographic_df.columns.tolist()
    cols = [col.lower() for col in cols]
    zipcode_demographic_df.columns = cols
    zipcode_demographic_df.index.names=['id']
#    zipcode_demographic_df['wealth/household'] = zipcode_demographic_df['wealthy'].divide(zipcode_demographic_df['houseofunits'])
#    zipcode_demographic_df['wages/household'] = zipcode_demographic_df['totalwages'].divide(zipcode_demographic_df['houseofunits'])
#    zipcode_demographic_df.to_csv(filenames.zipcode_demographic_data)

    return zipcodes, zipcode_demographic_df

def clean_giving_profile(df, time_cols, donation_cols):
    '''
    Take in time and donation amounts.  Drop rows where there is no indication of a first time donation,
    and strip the $ and commas from the donation info, converting to floats.
    Also create a binary indiator column showing that a second gift was given.
    :param df: incoming giving dataframe
    :param time_cols: a list of columns containing time series data in the giving dataframe to be operated on
    :param donation_cols: a list of columns containg data about amounts gifted.
    :return: a dataframe of dates and floats, with an addtional binary indicator column showing whether a second gift
    was given.
    '''
    print('Cleaning up the giving profile data.')
#    df = pd.read_csv(dfile, index_col='id')
#    tseries = ['first appeal date', 'first gift date', 'second gift date']
#    donation_amounts = ['first gift amount', 'second gift amount', 'total given', 'average gift amount',
#                        'largest gift amount', 'smallest gift amount']
    try:
        print('Drop the rows that contain zero first time donation dates.')
        for ts in time_cols:
            df.loc[:, ts] = pd.to_datetime(df.loc[:, ts])
            df = df[df['first gift date'].notnull()]
    except ValueError:
        print('The dates in time series failed to convert properly')

    try:
        for donation in donation_cols:
            df.loc[:, donation] = df.loc[:, donation].replace('[$,]', '', regex=True).astype(float)
    except ValueError:
        print('The donation columns are not present in the dataframe columns.')
    df['made second donation'] = np.where(df['second gift date'].notnull(), 1, 0)
    print(df)
    return df

def clean_zipcodes(df):
    '''
    Take in a location dataframe and clean it up.  If no zipcode is given, impute the most commonly
    frequently zipcode in the dataframe.
    :param df: a dataframe that contain zipcodes, cities, states, countries, from multiple sources
    :return: a cleaned list of zipcodes, where on column is the main zipcode, there's a column
    '''
#    df = pd.read_csv(fname, index_col='id')
    print('Cleaning up the zipcodes...')
    print(df['zip'])
    print(df['zip'].dtypes)
    input('')
    df['zip'].fillna('0', inplace=True)
    df['zip'] = df['zip'].str.split('-')
#    print(df['zip'].shape)
#    input('')
    df['zip'] = strip_whitespaces_from_lists(df['zip'])
    df['zip code'].fillna('0', inplace=True)
    df['zip code'] = df['zip code'].str.split('-')
#    df.drop('nan', inplace=True, axis=0)          # Drop any rows that contain nans in the index

    df['zip code'] = strip_whitespaces_from_lists(df['zip code'])
    df, zipcode_demographic_df = get_zipcodes(df, impute_zipcode=False)
    print('Saving zipcode datafile...')
    df.to_csv(filenames.zipcode_dfile, index_label='id')
    print('Shape of the zipcode datafile is: {}'.format(df.shape))
    return df, zipcode_demographic_df



if __name__ == '__main__':
#    clean_giving_profile(filenames.giving_dfile)
    clean_zipcodes(filenames.location_dfile)

