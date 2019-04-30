# Solar Disaggregation extended validation
'''
The aim of this script is to perform again the validation of the streaming problem presented in MT paper BuildSys 2018
'''
import sys
sys.path.append("..") # Adds higher directory to python modules path.
from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csss
import matplotlib.pyplot as plt
import datetime as dt
import pickle as pk
from imp import reload
import copy
from time import time as t_clock
from copy import deepcopy
import seaborn as sns
from csss.SolarDisagg import createTempInput
from csss.utilities import Setup_load as SetUp_load, unique_list
import math

get_ipython().run_line_magic('matplotlib', 'inline')

#%% Loading the data

Setup_load = deepcopy(SetUp_load())
Setup_load.QueryOrLoad()
load_data, load_data_2, weather, grouped_data, ids, solar_ids, homeids  = Setup_load.load_setup()
#%% Remove those solar houses that have generation = 0 for continous N_obs.
N_obs_day = 96 # number of observations in one day.
N_obs = 3*N_obs_day
remove = []
for i in solar_ids['solar']:
    df = grouped_data.get_group(i)
    roll_mean = df.gen.rolling(N_obs).mean()
    if np.nansum(roll_mean < 0.001) > 0:
        remove.append(i)
len(remove)

## Update the data accordingly
ids = list(set(ids) - set(remove))
load_data = load_data[~load_data['dataid'].isin(remove)]
solar_ids['solar'] = list(set(solar_ids['solar']) - set(remove))
solar_ids['nosolar'] = list(set(solar_ids['nosolar']) - set(remove))
grouped_data = load_data.groupby("dataid")
#%% Do a quick plot of load data
fig = plt.figure()
ax = plt.gca()
np.random.seed(6)
dataid = np.random.choice(grouped_data.mean().index, size=1)[0]
data = load_data.groupby('dataid').get_group(dataid)
start = dt.datetime(2015,7,6)
dur   = dt.timedelta(days = 1)
ind = (data.index > start) & (data.index < start + dur)
data = data.loc[ind,:]

ax.plot(data.index,-data['gen'], label = 'solar generation')
ax.plot(data.index,data['use'], label = 'consumption')
ax.plot(data.index,data['netload'], label = 'netload')
ax.set_title(dataid)
ax.legend()
plt.xticks(rotation = 90)
plt.show()

#%% Real Time Disaggregation
Nhomes, Nproxies, Ndays, Ntunesys, Nsolar = 8,2,10,2,4

def run_scenario_stream(Nhomes, Nproxies, Ndays, Ntunesys, Nsolar, sdmod_tune = None):
    
#    solarids = proxy_ids + IDS_solar
#    proxyids = proxy_ids   # Homes to be used as solar proxy
#    solids  = IDS_solar    # Homes with solar
#    homeids = homes_ids # np.concatenate([solids,nosolids])  # Homes used for solar disagg
#    nosolids   = list(set(homeids)-set(solids))    # Homes without solar. 
#    tuneids  = tune_ids #solids[:Ntunesys]  # Home to be used for tuning - those are chosen different from the houses and from the proxies
#    dataids = solarids+nosolids # +tuneids # Ids for all the systems ####### 
    
    solarids = np.random.choice(solar_ids['solar'], Nsolar+Nproxies, replace=False)
    proxyids = solarids[:Nproxies]   # Homes to be used as solar proxy
    solids  = solarids[Nproxies:]    # Homes with solar
    nosolids   = np.random.choice(solar_ids['nosolar'], Nhomes-Nsolar, replace=False)    # Homes without solar. 
    homeids = np.concatenate([solids,nosolids]) # Homes used for solar disagg
    tuneids  = homeids[:Ntunesys]  # Home to be used for tuning
    dataids = np.concatenate([solarids,nosolids]) # Ids for all the systems
    
    
    ## Set up data 
    first = True
    for did in dataids:
        dat = grouped_data.get_group(did)[['use','gen']]
        if did in solarids:
            dat['gen'] = -dat['gen']
        else:
            dat['gen'] = 0
        dat['netload_%s' % did] = dat['use'] + dat['gen']
        dat.columns = ['demand_%s' % did, 'solar_%s' % did, 'netload_%s' % did]

        if first:
            fulldata = dat
            first = False
        else:
            fulldata = pd.concat([fulldata,dat], axis = 1)

    ## Create aggregate load and aggregate net load columns. 
    netloadcols = ['netload_%s' % d for d in homeids]
    loadcols    = ['demand_%s' %  d for d in homeids]
    proxycol    = ['solar_%s' %   d for d in proxyids]

    fulldata['AggregateNetLoad'] = np.sum(fulldata[netloadcols], axis = 1)
    fulldata['AggregateLoad']    = np.sum(fulldata[loadcols], axis = 1)
    fulldata['Date'] = [dt.datetime(t.year, t.month, t.day, 0, 0, 0) for t in fulldata.index]
        
    ## Time align weather data with load data using linear interpolation 
    xp  = [t.value for t in weather.index]
    x = [t.value for t in fulldata.index]
    fulldata['AustinTemp'] = np.interp(x = x ,xp=xp, fp =  weather['temperature'])

    ## Get list of dates with complete readings
    dates = fulldata[['Date','AggregateLoad']].groupby('Date').count()   ### Find number of readings for each date
    numel = dates['AggregateLoad']                                      
    dates = dates.index[numel == np.max(numel)]                          ### Choose only dates with complete readings

    ## Chose 2N sets of dates, half for training and half for validation. 
    dates = dates[ np.random.choice(np.arange(len(dates)), replace=False, size=Ndays*2) ] ### Randomly select Ndays dates
    induse      = fulldata['Date'].isin(dates[:Ndays])  ### Subset data
    induse_test = fulldata['Date'].isin(dates[Ndays:])  ### Subset data
    data = fulldata.loc[induse,:]       # Training Set
    data_test = fulldata.loc[induse_test,:]  # Validation Set

    ## Get HOD regressors
    hod = pd.Series([t.hour for t in data.index])
    hod = pd.get_dummies(hod)

    ## Get temperature regressors
    Tmin, Tmax, tempregress = regressor=createTempInput(data['AustinTemp'], 10)

    ## Prepare data for ingestion into SolarDisagg method. 
    loadregressors = deepcopy(np.hstack([hod,tempregress]))
    tuningregressors = np.array(hod)
    netload = np.array(data[netloadcols])
    solarproxy = np.array(data[proxycol])
    names = ['solar_%s' % d for d in homeids]        


    if sdmod_tune is None:
        ## Construct solar disaggregation problem
        reload(csss.SolarDisagg)
        sdmod0 = copy.deepcopy(csss.SolarDisagg.SolarDisagg_IndvHome(netloads=netload, solarregressors=solarproxy, loadregressors=loadregressors, tuningregressors = tuningregressors, names = names))
    
        ## add true vectors
        for d in homeids:
            source_name = 'solar_%s' % d
            sdmod0.addTrueValue(name=source_name, trueValue=data[source_name])
    
        ## Add true aggregate load
        sdmod0.addTrueValue(name = 'AggregateLoad', trueValue=data['AggregateLoad'])
    
        ## Construct and solve the problem. 
        sdmod0.constructSolve(solver = 'GUROBI')
        sdmod_tune = copy.deepcopy(sdmod0)
        sdmod_tune.fitTuneModels(['solar_%s' % d for d in tuneids])
        sdmod_tune.tuneAlphas()
        sdmod_tune.constructSolve(solver = 'GUROBI')
    else:
        sdmod_tune = deepcopy(sdmod_tune)

 ### Create streaming problem on training data
    ### ******************************************
    aggregateNetLoad = np.sum(netload, axis = 1)

    sdmod_st_train = csss.SolarDisagg.SolarDisagg_IndvHome_Realtime(sdmod_tune, 
                                                                   aggregateNetLoad=aggregateNetLoad,
                                                              solarregressors=solarproxy, 
                                                              loadregressors=loadregressors, 
                                                              tuningregressors= tuningregressors)

    sdmod_st_train.tuneAlphas()
    sdmod_st_train.constructSolve()

    ## add true vectors
    for d in homeids:
        source_name = 'solar_%s' % d
        sdmod_st_train.addTrueValue(name=source_name, trueValue=data[source_name])

    ## Add true aggregate load
    sdmod_st_train.addTrueValue(name = 'AggregateLoad', trueValue=data['AggregateLoad'])


    ### Create streaming problem on historical data
    ### ******************************************
    ## Get HOD regressors
    hod = pd.Series([t.hour for t in data_test.index])
    hod = pd.get_dummies(hod)

    ## Get temperature regressors
    Tmin, Tmax, tempregress = regressor=createTempInput(data_test['AustinTemp'], 10, maxTemp=Tmax, minTemp=Tmin)
    tempregress = np.array(np.array(tempregress))
    

    ## Prepare data for ingestion into SolarDisagg streaming method. 
    loadregressors = np.hstack([hod,tempregress])
    netload = np.array(data_test[netloadcols])
    solarproxy = np.array(data_test[proxycol])
    tuningregressors = np.array(hod)
    names = ['solar_%s' % d for d in homeids]


    aggregateNetLoad = np.sum(netload, axis = 1)

    sdmod_st_test = csss.SolarDisagg.SolarDisagg_IndvHome_Realtime(sdmod_tune, 
                                                                   aggregateNetLoad=aggregateNetLoad,
                                                              solarregressors=solarproxy, 
                                                              loadregressors=loadregressors, 
                                                              tuningregressors= tuningregressors)

    sdmod_st_test.tuneAlphas()
    sdmod_st_test.constructSolve()

    ## add true vectors
    for d in homeids:
        source_name = 'solar_%s' % d
        sdmod_st_test.addTrueValue(name=source_name, trueValue=data_test[source_name])

    ## Add true aggregate load
    sdmod_st_test.addTrueValue(name = 'AggregateLoad', trueValue=data_test['AggregateLoad'])

    outdict = {}
    outdict['tuned_model'] = sdmod_tune
    outdict['initial_model'] = sdmod0
    outdict['stream_model_train'] = sdmod_st_train
    outdict['stream_model_test'] = sdmod_st_test
    outdict['dates']    = dates[:Ndays]
    outdict['test_dates'] = dates[Ndays:]
    outdict['times']    = data.index
    outdict['proxyids'] = proxyids
    outdict['homeids']  = homeids
    outdict['tuneids']  = tuneids
    outdict['solids']  = solids
    outdict['nosolids']  = nosolids
    outdict['data']  = data
    return(outdict)
    
#%% Create Output Data Frame
    
def create_outputdf_stream(scen_out, name, scen_n = 1):
    mod = scen_out['tuned_model']
    
    scen_out['initial_model'].calcPerformanceMetrics()
    scen_out['tuned_model'].calcPerformanceMetrics()
    scen_out['stream_model_train'].calcPerformanceMetrics()
    scen_out['stream_model_test'].calcPerformanceMetrics()

    df0  = scen_out['initial_model'].performanceMetrics
    dft  = scen_out['tuned_model'].performanceMetrics
    dfs  = scen_out['stream_model_train'].performanceMetrics
    dfs2 = scen_out['stream_model_test'].performanceMetrics
    
    ## Is the model tuned
    df0['tuned']  = False
    dft['tuned']  = True
    dfs['tuned']  = True
    dfs2['tuned'] = True
    
    ## Is the model streaming
    df0['streaming']  = False
    dft['streaming']  = False
    dfs['streaming']  = True
    dfs2['streaming'] = True

    ## Is the model on test data
    df0['testdata']  = False
    dft['testdata']  = False
    dfs['testdata']  = False
    dfs2['testdata'] = True


    dfout = pd.concat([df0,dft,dfs,dfs2])
    
    dfout['scen_name'] = name
    dfout['dataid']    = 0
    dfout['N_sys']     = len(mod.models)-1
    dfout['N_read']    = len(mod.aggregateSignal) 
    dfout['N_prox']    = 0
    dfout['N_tunesys'] = len(scen_out['tuneids'])
    dfout['istunesys'] = False
    dfout['frac_reverse'] = np.ones(dfout.shape[0]) * np.nan
    dfout['isSolar']   = False
    dfout['scen']    = scen_n

    dfout = dfout.reset_index().set_index(['models'])
    for did in scen_out['homeids']:
        name = 'solar_' + str(did)
        dfout.loc[(name),'frac_reverse'] = np.mean( mod.netloads[name] < 0 )
        dfout.loc[(name),'N_prox'] = mod.models[name]['regressor'].shape[1]
        dfout.loc[(name),'dataid'] = np.int(did)
        dfout.loc[(name),'istunesys'] = (did in scen_out['tuneids'])
    return(dfout)

scen_out = run_scenario_stream(8,2,10,2,4)
df = create_outputdf_stream(scen_out, 'test', scen_n = 2)
#%% Run sensitivity on the streaming problem
csvname = 'Optimal_Sens_streaming'
N_each = 50
first = True
for i in range(N_each):
    scen_name = 'scenario_days_%d_%d' % (Ndays, i)
    print('\r' + scen_name, end = '')
    try:
        scen_out = run_scenario_stream(Nhomes=8,Nproxies=2,Ndays=30,Ntunesys=2,Nsolar=4)
        ## Create output dfs
        df0 = create_outputdf_stream(scen_out, scen_name, scen_n = i)
    except:
        df0 = pd.DataFrame()

    if first:
        df_all = df0
        first = False
    else:
        df_all = pd.concat([df_all,df0])
    print(i)
    fp = 'Validation/sensitivity/' + csvname
    df_all.to_csv(fp+'.csv')
    save_pkl(df_all,fp+'.pkl')
#%%
df_all.reset_index(inplace=True)  
df_all.set_index(['dataid'], inplace = True)
df_all = df_all[df_all.index != 0]
df_all['isSolar'] = [df_all.index[i] in solar_ids['solar'] for i in range(len(df_all))]
N_each = 50
models = ['train', 'stream_train','stream_test']
metrics = ['cv_pos']#['cv_pos']
isSolar = True
house_type = ['solar','no_solar']
box_vectors = pd.DataFrame(index = house_type,columns = models)
for i in house_type:
    if i == 'solar':
        metrics = ['cv_pos']
        isSolar = True
    else:
        metrics = ['rmse']
        isSolar = False
    for j in models:
        if j == 'train':
            models_param = [False, False]
        elif j == 'stream_train':
            models_param = [True, False]
        elif j == 'stream_test':
            models_param = [True, True]
        index = [(df_all['tuned'] == True) & (df_all['isSolar'] == isSolar) & (df_all['streaming'] == models_param[0]) & (df_all['testdata'] == models_param[1])] #& (df_all['istunesys'] == False)
        box_vectors.loc[i,j] = df_all.loc[index[0]][metrics].values.squeeze()
#%% Plot results
#f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
y_max = 1.95
models_title = ['Train', 'Stream Train','Stream Test']
bp = []
plt.close()
f, (ax1, ax2) = plt.subplots(1, 2, figsize = (12,4))#, sharex=True)
bp.append(ax1.boxplot([np.abs(box_vectors.loc['solar',i]) for i in models],labels=[str(a) for a in models_title],showmeans=True))
#f.suptitle('Solar houses')
ax1.set_ylabel('CV')
#ax1.set_ylim([0,y_max])
#ax1.set_xlabel('Number of days')
ax1.set_title('Solar houses')
bp.append(ax2.boxplot([np.abs(box_vectors.loc['no_solar',i]) for i in models],labels=[str(a) for a in models_title],showmeans=True))
ax2.set_ylabel('RMSE')
#ax2.set_ylim([0,y_max])
ax2.set_title('Non solar houses')
#plt.show()
plt.savefig('Validation/figures/Optimal_streaming_sensitivity_sd')
#%%
box_vectors_ = pd.DataFrame(index = house_type,columns = ['means','medians'])
for count,i in enumerate(box_vectors_.index):
    for j in box_vectors_.columns:
        box_vectors_.loc[i,j] = [medline.get_ydata()[0] for medline in bp[count][j]]

