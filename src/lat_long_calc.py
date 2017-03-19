import os
import socket
import time

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from geopy import exc
from geopy.distance import VincentyDistance
from tqdm import tqdm

import filenames


def _extract_lat_lngs(zip_, xml_output, lat_lng_dict, errors):
    try:
        if xml_output is None:
            lag_lng_dict[zip_] = np.nan
        else:
            soup = BeautifulSoup(xml_output.text, 'xml')
            lat = soup.find_all('lat')[0].get_text()
            lng = soup.find_all('lng')[0].get_text()
            lat_lng_dict[zip_] = (lat, lng)
    except IndexError:
        print('Zipcode {} generated an error.'.format(zip_))
        errors.append(zip_)
        lat_lng_dict[zip_] = None
        time.sleep(120)
    return lat_lng_dict, errors

def _get_location_from_geonames(zip_x, lat_lng_dict, errors, username):
    params = {'username': username, 'countryBias': 'USA', 'postalcode': zip_x, 'maxRows': 1}
    location_xml = requests.get('http://api.geonames.org/postalCodeSearch?', params=params)
    lat_lng_dict, errors = _extract_lat_lngs(zip_=zip_x, xml_output=location_xml, lat_lng_dict=lat_lng_dict, errors=errors)
    city = lat_lng_dict.get(zip_x)
    return lat_lng_dict, errors, city


def create_distance_matrix(df, username):
    skipped = []
    errors = []
    lat_lng_dict = {}   # Create a dictionary to store lat long values to we can avoid repeatedly hitting the server
#    df = pd.read_csv(dfile, index_col='id', dtype=str)
    df['main zipcode'] = df['main zipcode'].str.rjust(5, '0')
    df['secondary zipcode'] = df['secondary zipcode'].str.rjust(4, '0')
    X = np.empty((df.shape[0], df.shape[0]))
    for i, m in tqdm(enumerate(df.index.tolist())):
        for j, n in tqdm(enumerate(df.index.tolist())):
            try:
                zip_m = df.loc[m, 'main zipcode']
                if zip_m in lat_lng_dict:
                    if lat_lng_dict.get(zip_m) != np.nan:
                        city1 = lat_lng_dict.get(zip_m)
                    else:
                        lat_lng_dict, error, city1 = _get_location_from_geonames(username=username, zip_x=zip_m,
                                                                                 lat_lng_dict=lat_lng_dict, errors=errors)
                else:
                    lat_lng_dict, error, city1 = _get_location_from_geonames(username=username, zip_x=zip_m,
                                                                             lat_lng_dict=lat_lng_dict, errors=errors)

                zip_n = df.loc[n, 'main zipcode']
                if zip_n in lat_lng_dict:
                    if lat_lng_dict.get(zip_n) != np.nan:
                        city2 = lat_lng_dict.get(zip_n)
                    else:
                        lat_lng_dict, error, city2 = _get_location_from_geonames(username=username, zip_x=zip_n,
                                                                                 lat_lng_dict=lat_lng_dict, errors=errors)
                else:
                    lat_lng_dict, error, city2 = _get_location_from_geonames(username=username, zip_x=zip_n,
                                                                             lat_lng_dict=lat_lng_dict, errors=errors)

                if (city1 is not None) & (city2 is not None):
                    X[i,j] = VincentyDistance(city1, city2).miles
                else:
                    X[i,j] = np.nan
            except (error.URLError, error.HTTPError, exc.GeocoderTimedOut, exc.GeocoderServiceError, socket.timeout) as e:
                X[i,j] = np.nan
    print('Saving pairwise distance matrix for ids to file titled "distances.csv"')
    df2 = pd.DataFrame(X, index=df.index, columns=df.index.tolist())
    skipped_df = pd.DataFrame(skipped)
    df2.to_csv(os.path.join(filenames.data_folder, 'distances.csv'))
    print('Saving skipped data to "skipped_distances.csv"')
    skipped_df.to_csv(os.path.join(filenames.data_folder, 'skipped_distances.csv'))
    print('Saving a dictionary of zipcode lat/longs to "zipcode_dict.csv"')
    zipcodes_df = pd.DataFrame(lat_lng_dict)
    zipcodes_df.to_csv(os.path.join(filenames.data_folder, 'zipcode_dict.csv'))
    print('Errors saved were:  ')
    print(errors)
    print('Done.')

if __name__=='__main__':
    create_distance_matrix(filenames.zipcode_dfile, username='hayesgb01')