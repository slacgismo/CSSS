#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script queries the Pecan Street PostgreSQL database and creates the data file required to run the
"SolarDisagg Individual Home Tutroial" Jupyter notebook.

- 8 homes
- 4 home have solar, 4 do not
- 2 solar proxies
- 10 days of training data
- 5 days of test data
- 2 tuning systems

We are querying the university.electricity_egauge_15min table for individual home electrical load ("use") and solar
generation ("gen").

NOTE: if you are not using a university license, you must change the "university" to the appropriate database name.

The solar disaggregation algorithm expects net load measurements, which we construct from the use and gen columns.
"""

import sqlalchemy as sq
import numpy as np
import pandas as pd
import pytz
from time import time
import datetime as dt

HOME_IDS_SOLAR      = [3367, 7793, 7875, 9932]
HOME_IDS_NO_SOLAR   = [4998, 5949, 7641, 8292]
SOLAR_PROXY_IDS     = [4297, 8084]
START_DATE          = '04-01-2015'
END_DATE            = '04-16-2015'

def main():
    # Read key for PecanSt, Stored in text file
    print('loading Pecan Street key file...')
    with open('../keys/pecanstkey.txt', 'r') as f:
        key = f.read().strip()
        f.close()

    # Create engine
    print('creating sql engine...')
    engine = sq.create_engine("postgresql+psycopg2://{}@dataport.pecanstreet.org:5434/postgres".format(key))

    # Query Pecan Street database for home energy use and generation
    print('querying home energy database...')
    ti = time()
    duse = HOME_IDS_SOLAR + HOME_IDS_NO_SOLAR + SOLAR_PROXY_IDS
    query = """
        SELECT dataid, local_15min, use, gen 
        FROM university.electricity_egauge_15min
        WHERE local_15min
        BETWEEN '{}' AND '{}'
        AND electricity_egauge_15min.dataid in (
    """.format(START_DATE, END_DATE) + ','.join([str(d) for d in duse]) + """)
        ORDER BY local_15min
    ;"""
    load_data = pd.read_sql_query(query, engine)
    tf = time()
    deltat = (tf - ti) / 60.
    print('query of {} values took {:.2f} minutes'.format(load_data.size, deltat))

    # Post-processing steps: set time index, will NaNs with zeros, net load calculation, aggregate net load calculation
    print('post-processing queried data...')
    load_data.rename(columns={'local_15min': 'time'}, inplace=True)
    load_data['time'] = pd.DatetimeIndex(load_data['time'])
    load_data.set_index('time', inplace=True)
    load_data.fillna(value=0, inplace=True)

    # Construct net load for each house
    load_data['netload'] = load_data['use'] - load_data['gen']

    # Construct aggregate net load. To do this, we must align the timeseries axis for all house IDs.
    grouped_data = load_data.groupby('dataid')
    ids = grouped_data.groups.keys()                                # should be the same as 'duse'
    load_data = pd.concat([grouped_data.get_group(k) for k in ids], axis=1, keys=ids).swaplevel(axis=1)
    del load_data['dataid']                                         # no longer need as IDs are in the column header
    load_data['netload', 'agg'] = load_data['netload'].sum(axis=1)  # aggregate net load

    # Rename column headers from site ID numbers to something more descriptive (saves looking what what IDs are in
    # what set later)
    d1 = {'solar_home': HOME_IDS_SOLAR, 'home': HOME_IDS_NO_SOLAR, 'proxy': SOLAR_PROXY_IDS}
    d2 = {v: k + str(i + 1) for k, l in d1.items() for i, v in enumerate(l)}
    d2['agg'] = 'agg'
    mi = load_data.columns
    new_labels = [d2[k] for k in mi.levels[1]]
    load_data.columns = pd.MultiIndex(levels=[mi.levels[0], new_labels], labels=mi.labels)

    # Query Pecan Street database for weather data
    print('querying ambient temperature data from weather table...')
    locs = pd.read_sql_query(
        """
        SELECT distinct(latitude,longitude), latitude
        FROM university.weather
        ORDER BY latitude
        LIMIT 10;
        """,
        engine
    )
    locs['location'] = ['Austin', 'San Diego', 'Boulder']           # Ascending order by latitude
    locs.set_index('location', inplace=True)
    weather = pd.read_sql_query(
        """
        SELECT localhour, temperature
        FROM university.weather
        WHERE localhour
        BETWEEN '{}' and '{}'
        AND latitude = {}
        ORDER BY localhour;
        """.format(START_DATE, END_DATE, locs.loc['Austin']['latitude']),
        engine
    )
    weather.rename(columns={'localhour': 'time'}, inplace=True)
    # timestamps come in local hour
    weather['time'] = weather['time'].map(lambda x: x.replace(tzinfo=None))
    weather['time'] = pd.to_datetime(weather['time'])
    weather.set_index('time', inplace=True)
    weather = weather.asfreq('15Min').interpolate('linear')         # Upsample from 1hr to 15min to match load data


    combined_data = load_data.copy()
    combined_data[weather.columns] = weather                        # Table merging using column assignment
    combined_data.drop(combined_data.index[-1], inplace=True)       # Drop 12AM measurement on the 16, leaving 15 full
                                                                    # days of data
    # Construct input data structures for SolarDisagg_IndvHome class
    # netloads = load_data['netload'].drop('agg', axis=1).values
    # solarproxy = load_data['gen'][SOLAR_PROXY_IDS].values
    # if np.average(solarproxy) >= 0:
    #     solarproxy *= -1                                            # generation should be negative
    # names = ['solar_{}'.format(d) for d in load_data['netload'].drop('agg', axis=1).columns]

    return combined_data


if __name__ == "__main__":
    pd.set_option('display.max_rows', 20)
    pd.set_option('display.max_columns', 10)
    k = main()
    # print(len(k))
    print(k.iloc[50:60])
    # print(k['temperature'].values)
    # print(k.head())
    print(k.tail())
    k.to_csv('../data/tutorial_data.csv')