from importlib import reload
import datetime as dt
import numpy as np
import pandas as pd
from time import time
from os import path
import os
from pathlib import Path
import pickle as pk
import matplotlib.pyplot as plt
import sqlalchemy as sq
from copy import deepcopy



class Setup_load(object):
    """
    This class queries the load and weather data from Pecan Street and save them in a csv file.
    In case the file already exists, it just load it.
    There are two different methods to load the data in the favourite format, depending on the application.
    
    """
    
#    def __init__(self):
#        print('No init required')

    def QueryOrLoad(self,start_date = '01-01-2015', end_date = '01-01-2017'):

        fp = 'data/netloadsolaridentify_{}_{}.csv'.format(start_date, end_date)
        fw = 'data/weather_netloadsolaridentify_{}_{}.csv'.format(start_date, end_date)
        
        # Read the keys
        with open('keys/pecanstkey.txt', 'r') as f:
            key = f.read().strip()
            f.close()
        engine = sq.create_engine("postgresql+psycopg2://{}@dataport.pecanstreet.org:5434/postgres".format(key))
        
        
        if not path.exists(fp):
            ti = time()
            # Find sites with complete data for the requested time period and join
            print('determining sites with full data...')
            query = """
                SELECT e.dataid
                FROM university.electricity_egauge_15min e
                WHERE local_15min
                BETWEEN '{}' AND '{}'
                AND e.dataid IN (
                    SELECT m.dataid
                    FROM university.metadata m
                    WHERE m.city = 'Austin'
                )
        
                GROUP BY dataid
                HAVING count(e.use) = (
                    SELECT MAX(A.CNT)
                    FROM (
                        SELECT dataid, COUNT(use) as CNT
                        FROM university.electricity_egauge_15min
                        WHERE local_15min
                        BETWEEN '{}' AND '{}'
                        GROUP BY dataid
                    ) AS A
                );
            """.format(start_date, end_date, start_date, end_date)
            metadata = pd.read_sql_query(query, engine)
            duse = metadata.values.squeeze()
            print('querying load and generation data...')
            query = """
                SELECT dataid, local_15min, use, gen 
                FROM university.electricity_egauge_15min
                WHERE local_15min
                BETWEEN '{}' AND '{}'
                AND electricity_egauge_15min.dataid in (
            """.format(start_date, end_date) + ','.join([str(d) for d in duse]) + """)
                ORDER BY local_15min;
            """
            load_data = pd.read_sql_query(query, engine)
            tf = time()
            deltat = (tf - ti) / 60.
            print('query of {} values took {:.2f} minutes'.format(load_data.size, deltat))
            load_data.to_csv(fp)
            
            # Weather data
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
                """.format(start_date, end_date, locs.loc['Austin']['latitude']),
                engine
            )
            weather.rename(columns={'localhour': 'time'}, inplace=True) # Rename 
            weather['time'] = weather['time'].map(lambda x: x.replace(tzinfo=None))
            weather['time'] = pd.to_datetime(weather['time'])
            weather.set_index('time', inplace=True)
            weather = weather[~weather.index.duplicated(keep='first')]
            weather = weather.asfreq('15Min').interpolate('linear')         # Upsample from 1hr to 15min to match load data
            weather.to_csv(fw)
        else:
            ti = time()
            load_data = pd.read_csv(fp)
            weather = pd.read_csv(fw)
            tf = time()
            deltat = (tf - ti)
            print('reading {} values from csv took {:.2f} seconds'.format(load_data.size, deltat))
        
        #Load Setup - set index and fill na    
        load_data.rename(columns={'local_15min': 'time'}, inplace=True)
        load_data['time'] = pd.DatetimeIndex(load_data['time'])
        load_data.set_index('time', inplace=True)
        load_data.fillna(value=0, inplace=True)
        if 'Unnamed: 0' in load_data.columns:
            del load_data['Unnamed: 0'] # useless column
        
        # Weather Setup
        weather['time'] = pd.DatetimeIndex(weather['time'])
        weather.set_index('time', inplace=True)
        
        # Redefine start_date and end_date so that the weather and load_data dataset match in time stamps and you take the dates common to both.
        start_date = max(weather.index[0],load_data.index[0])
        end_date = min(weather.index[-1],load_data.index[-1])
        weather = weather[(weather.index >= pd.to_datetime(start_date)) & (weather.index <= pd.to_datetime(end_date))]
        lst = list(set(weather.index)-set(load_data['use'].index)) # when you interpolate hourly data to 15m resolution it also interpolates in the changing time hours. This code inidividuates those times and then I drop them
        weather = weather.drop(lst)
        load_data = load_data[(load_data.index >= pd.to_datetime(start_date)) & (load_data.index <= pd.to_datetime(end_date))]
        
        # NetLoad
        load_data['netload'] = load_data['use'] - load_data['gen']
        load_data.head()
        
        self.load_data = load_data
        self.weather = weather
        
        
    def load_setup(self):
        
        # Group data by ids and reshape the load_data DataFrame
        self.load_data = self.load_data[~self.load_data['dataid'].isin([484, 871, 9609, 9938, 8282, 2365, 5949])] #I removed those Ids that showed  negative netload at some point but zero generation
        grouped_data = self.load_data.groupby('dataid')
        ids = grouped_data.groups.keys()                                # should be the same as 'duse'
        homeids = np.unique(self.load_data['dataid'])
        load_data_2 = deepcopy(pd.concat([grouped_data.get_group(k) for k in ids], axis=1, keys=ids).swaplevel(axis=1))
        del load_data_2['dataid'] # no longer need as IDs are in the column header
        self.load_data_2 = load_data_2
        
        # Print to see it makes sense. 
#        self.load_data['netload'][list(ids_all)[0:2]]  # example on how to slice this MutiIndex DataFrame
        
        # Output a brief summary of the dataset and create the solar_ids dictionary        
        sums = grouped_data.sum()
        solar_ids = {
            'solar': list(sums.index[sums['gen'] > 0]),
            'nosolar': list(sums.index[sums['gen'] <= 0])
        }
        n = len(solar_ids['solar'])
        
        print('There are %d homes with complete data' % len(sums))
        print('%d homes solar' % len(solar_ids['solar']))
        print('%d homes with no solar' % len(solar_ids['nosolar']))
        
        
        return self.load_data, self.load_data_2, self.weather, grouped_data, ids, solar_ids, homeids