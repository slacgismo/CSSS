# Solar Disaggregation extended validation
'''
The aim of this python file is to reperform the validation that has been done in MT paper Buildsys 2018,
Extending the number of houses and days. 
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
from csss.utilities import Setup_load as SetUp_load, save_pkl, load_pkl

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


# ## Set up an example Solar Disaggregation on multiple homes

# ## Create the individual homes model
# This code creates the CSSS model for disaggregating solar at individual homes. 
# This is the training problem, not the real time problem. 
# See note below on issues with values of alpha. 
# 
# ### Alpha and solver errors
# The solver is sensitive to large values of alpha, particularly with more data.  There seems to be an inverse relationship between the maximum size of alpha and the amount of data. The errors scale with t he magnitude of alpha, not the difference between alphas, so by scaling the maximum alpha to 1, we may solve this problem. 

# ### Steady case: 
# 8 homes (4 solar, 4 no solar), 2 proxies, 10 days of readings, randomly sampled, 2 tuning systems.

# ## Sensitivity Cases
# - Pre and post tuning
# - fraction feed-in (defined by system)
# - Ndays (randomly chosen days)
# - Nproxies (randomly chosen systems)
# - Ntunesys (randomly chosen systems)

#%% Scenario N_homes

#Nhomes,Nproxies, Ndays, Ntunesys, Nsolar, proxy_tuning = 8,2,10,2,4,False
def run_scenario(Nhomes,Nproxies, Ndays, Ntunesys, Nsolar, proxy_tuning = False, hourly = False):
    '''
    param: Nhomes total number of houses
    param: Nproxies number of proxies used
    param: Ntunesys number of tuning systems
    param: Ndays number of days
    param: Nsolar number of solar houses
    param: proxy tuning. If True the tuning sys are disregarded and the proxies are used instead. If 'One' use only the first proxy
    param: hourly use hourly resolution
    '''
    
    ## Identify dataids to use
    solarids = np.random.choice(solar_ids['solar'], Nsolar+Nproxies, replace=False)
    proxyids = solarids[:Nproxies]   # Homes to be used as solar proxy
    solids  = solarids[Nproxies:]    # Homes with solar
    nosolids   = np.random.choice(solar_ids['nosolar'], Nhomes-Nsolar, replace=False)    # Homes without solar. 
    homeids = np.concatenate([solids,nosolids]) # Homes used for solar disagg
    tuneids  = homeids[:Ntunesys]  # Home to be used for tuning
    
    if proxy_tuning == True:
        tuneids = deepcopy(proxyids)
        homeids = np.concatenate([tuneids,homeids])
    elif proxy_tuning == 'One':
        proxyids = [proxyids[0]]
        tuneids = deepcopy(proxyids)
        homeids = np.concatenate([tuneids,homeids])
        
    dataids = np.concatenate([solarids,nosolids]) # Ids for all the systems
    
    ## Set up data 
    first = True
    for did in dataids:
        dat = load_data.groupby("dataid").get_group(did)[['use','gen']]
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

    ## Get indices for Ndays random dates
    dates = fulldata[['Date','AggregateLoad']].groupby('Date').count()   ### Find number of readings for each date
    numel = dates['AggregateLoad']                                      
    dates = dates.index[numel == np.max(numel)]                          ### Choose only dates with complete readings
    dates = dates[ np.random.choice(np.arange(len(dates)), replace=False, size=Ndays) ] ### Randomly select Ndays dates
#    start_sequence = int(np.random.choice(np.arange(len(dates)-Ndays), replace=False, size=1))
#    dates = dates[start_sequence:start_sequence+Ndays] ### select continous Ndays dates
    induse = fulldata['Date'].isin(dates)  ### Subset data
    data = fulldata.loc[induse,:]

    if hourly == True:    
        data = data.resample('H').mean().dropna()

    ## Get HOD regressors
    hod = pd.Series([t.hour for t in data.index])
    hod = pd.get_dummies(hod)

    ## Get temperature regressors
    Tmin, Tmax, tempregress = regressor=createTempInput(data['AustinTemp'], 10)

    ## Prepare data for ingestion into SolarDisagg method. 
    loadregressors = np.hstack([hod,tempregress])
    netload = np.array(data[netloadcols])
    solarproxy = np.array(data[proxycol])
    names = ['solar_%s' % d for d in homeids]

    ## Construct solar disaggregation problem
#    reload(csss.SolarDisagg)
    sdmod0 = csss.SolarDisagg.SolarDisagg_IndvHome(netloads=netload, solarregressors=solarproxy, loadregressors=loadregressors,tuningregressors = hod, names = names)


    ## add true vectors
    for d in homeids:
        source_name = 'solar_%s' % d
        sdmod0.addTrueValue(name=source_name, trueValue=data[source_name])

    ## Add true aggregate load
    sdmod0.addTrueValue(name = 'AggregateLoad', trueValue=data['AggregateLoad'])

    ## Construct and solve the problem. 
    solver = None
    sdmod0.constructSolve(solver = solver)
    sdmod_tune = copy.deepcopy(sdmod0)
    sdmod_tune.fitTuneModels(['solar_%s' % d for d in tuneids])
    sdmod_tune.tuneAlphas()
    sdmod_tune.constructSolve(solver = solver)
    
    outdict = {}
    outdict['tuned_model'] = sdmod_tune
    outdict['initial_model'] = sdmod0
    outdict['dates'] = dates
    outdict['times'] = data.index
    outdict['proxyids'] = proxyids
    outdict['homeids']  = homeids
    outdict['tuneids']  = tuneids
    outdict['solids']  = solids
    outdict['nosolids']  = nosolids
    outdict['data']  = data
    return(outdict)


#%% scen_out = run_scenario(20,2,10,2,10)
#import csss
Nhomes = 10
Nproxies = 2
Ndays = 10
Ntunesys = 2
Nsolar = 5
t1 = t_clock()
scen_out = run_scenario(Nhomes,Nproxies, Ndays, Ntunesys, Nsolar,proxy_tuning = False, hourly = False)
t2 = t_clock()
print('It took {} minutes to solve {} days.'.format((t2-t1)/60,Ndays))

# In[39]:
def create_outputdf(scen_out, name,scen_n = 1,Sens_Type='houses'):
    mod = scen_out['tuned_model']
    
    scen_out['initial_model'].calcPerformanceMetrics()
    scen_out['tuned_model'].calcPerformanceMetrics()
    df0 = scen_out['initial_model'].performanceMetrics
    dft = scen_out['tuned_model'].performanceMetrics
    
    df0['tuned'] = 'ini'
    dft['tuned'] = 'tune'
    df0['tuned_temp'] = 'ini' # I need it for the index
    dft['tuned_temp'] = 'tune' # I need it for the index
    dfout = pd.concat([df0,dft])
    
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
    
    dfout = dfout.reset_index().set_index(['models','tuned'])
    for did in scen_out['homeids']:
        for tuned in list(['ini','tune','solar']):
            name = 'solar_' + str(did)
            dfout.loc[(name,tuned),'frac_reverse'] = np.mean( mod.netloads[name] < 0 )
            dfout.loc[(name,tuned),'N_prox'] = mod.models[name]['regressor'].shape[1]
            dfout.loc[(name,tuned),'dataid'] = np.int(did)
            dfout.loc[(name,tuned),'istunesys'] = (did in scen_out['tuneids'])
            dfout.loc[(name,tuned),'isSolar'] = (did in scen_out['solids'])
    
    if Sens_Type == 'houses':
        dfout = dfout.reset_index().set_index(['dataid','N_sys','tuned_temp','scen'])        
        index = pd.MultiIndex.from_tuples(dfout.index.values.tolist(), names=['dataid', 'N_houses','tuning','scen'])
        dfout.set_index(index,inplace = True)
    else:
        dfout['N_days']    = int(len(mod.aggregateSignal)/96)
        dfout = dfout.reset_index().set_index(['dataid','N_days','tuned_temp','scen'])        
        index = pd.MultiIndex.from_tuples(dfout.index.values.tolist(), names=['dataid', 'N_days','tuning','scen'])
        df
    return(dfout)

df = create_outputdf(scen_out, 'test',Sens_Type = 'N_days')


#%% ## Run Sansitivity Cases!

#csvname = 'ScenarioAttmempt9.csv'
first = True
N_each = 50

## Sensitivity of number of proxies
csvname = 'Nproxies'
N_each = 50
first = True
errors = 0
for Nproxies in np.arange(1,6):
    for i in range(N_each):
        scen_name = 'scenario_prox_%d_%d' % (Nproxies, i)
        print('\r' + scen_name, end = '')
        try:
            scen_out = run_scenario(Nhomes=8,Nproxies=Nproxies,Ndays=10,Ntunesys=2,Nsolar=4)
            ## Create output dfs
            df0 = create_outputdf(scen_out, scen_name)
        except:
            print('Exception Triggered')
            df0 = pd.DataFrame()
            errors+=1
            
        if first:
            df_all = df0
            first = False
        else:
            df_all = pd.concat([df_all,df0])
            df_all.to_csv('data/' + csvname)
        print(Nproxies,i)
        fp = 'Validation/sensitivity/' + csvname
        df_all.to_csv(fp+'.csv')
        save_pkl(df_all,fp+'.pkl')
   

# Sensitivity of number of homes
csvname = 'Nhouses_large_'
N_each = 50
first = True
errors = pd.Series(0,index = np.array([8,20,50,80,120]))#, columns = ['error'])
Ntunesys = 2
#for Nhomes in np.array([8,20,50,80,120]):
for Nhomes in np.array([50,80,120]):
    for i in range(N_each):
        scen_name = 'scenario_home_%d_%d' % (Nhomes, i)
        print('\r' + scen_name, end = '')
        Nsolar = int((Nhomes-Ntunesys)/2)+Ntunesys#(np.ceil(1 + (Nhomes-1)*0.4))
        try:
            scen_out = run_scenario(Nhomes=Nhomes,Nproxies=2,Ndays=30,Ntunesys=2,Nsolar=Nsolar)
            ## Create output dfs
            df0 = create_outputdf(scen_out, scen_name)
        except:
            print('Exception Triggered')    
            df0 = pd.DataFrame()
            errors.loc[Nhomes]+=1
            
        if first:
            df_all = df0
            first = False
        else:
            df_all = pd.concat([df_all,df0])
            df_all.to_csv('data/' + csvname)
        print(Nproxies,i)
        fp = 'Validation/sensitivity/' + csvname
        df_all.to_csv(fp+'.csv')
        save_pkl([df_all, errors],fp+'.pkl')
#aa = load_pkl(fp+'.pkl')

         
## Sensitivity of number of readings
csvname = 'Ndays'
N_each = 50
first = True
for Ndays in np.array([5,10,20,30]):
    for i in range(N_each):
        scen_name = 'scenario_days_%d_%d' % (Ndays, i)
        print('\r' + scen_name, end = '')
        try:
            scen_out = run_scenario(Nhomes=8,Nproxies=3,Ndays=Ndays,Ntunesys=2,Nsolar=4)
            ## Create output dfs
            df0 = create_outputdf(scen_out, scen_name)
        except:
            df0 = pd.DataFrame()

        if first:
            df_all = df0
            first = False
        else:
            df_all = pd.concat([df_all,df0])
        print(Ndays,i)
        fp = 'Validation/sensitivity/' + csvname
        df_all.to_csv(fp+'.csv')
        save_pkl(df_all,fp+'.pkl')

## Sensitivity on optimal paramters
csvname = ['OptSens_pt1','OptSens_pt2','OptSens_st2']
mode = ['One',True, False]
Nproxies_vec = [1,2,2]
csvname = ['OptSens_nohourly_','OptSens_hourly_']
mode = [False,False]
hh_set = [False,True]
Nproxies_vec = [2,2]
N_each = 50
first = True
df_all = {}
for i in range(N_each):
    for j in range(len(csvname)):
        np.random.seed(i)
        scen_name = 'scenario_days_%d_%d' % (j, i)
        print('\r' + scen_name, end = '')
        try:
            scen_out = run_scenario(Nhomes=8,Nproxies=Nproxies_vec[j],Ndays=30,Ntunesys=2,Nsolar=4, proxy_tuning = mode[j], hourly = hh_set[j])
            ## Create output dfs
            df0 = create_outputdf(scen_out, scen_name)
        except:
            df0 = pd.DataFrame()
    
        if i == 0:
            df_all[csvname[j]] = df0
        else:
            df_all[csvname[j]] = pd.concat([df_all[csvname[j]],df0])
        print(j,i)
        fp = 'Validation/sensitivity/' + csvname[j]
#        df_all.to_csv(fp+'.csv')
        save_pkl(df_all,fp+'.pkl')
        
#%% Analysis of the results and plots
name,sens_var,xlabel = 'Nhouses', 'N_houses', 'N houses'
name,sens_var,xlabel = 'Nproxies', 'N_prox', 'N proxies'
name,sens_var,xlabel = 'Ndays', 'N_read', 'N days'
fp = 'Validation/sensitivity/'+name+'.pkl'
df_all = load_pkl(fp)

#%%
#plt.rcParams.update(plt.rcParamsDefault)
df_all.reset_index(inplace=True)  
df_all.set_index(['dataid'], inplace = True)
df_all = df_all[df_all.index != 0]
N_each = 50
N_houses = np.sort(df_all[sens_var].unique())[:-1]
models = ['ini', 'tune']
metrics = ['cv_pos','rmse']
isSolar = [True,False]
box_vectors = pd.DataFrame(index = pd.MultiIndex.from_tuples([(int(i),j) for j in isSolar for i in N_houses]), columns = models)
for l,k in enumerate(isSolar):
    for j in models:
        for i in N_houses:
            index = [(df_all['tuning'] == j) & (df_all['isSolar'] == k) & (df_all[sens_var] == i)] #& (df_all['istunesys'] == False)
            box_vectors.loc[(i,k),j] = df_all.loc[index[0]][metrics[l]].values.squeeze()

#%% Plot results
#f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
y_max = 2
models_title = ['Initial model','Tuned model']
figure_title = ['Solar houses','Non solar houses']
y_label = ['CV','RMSE']
bp = []
plt.close('all')
f, ax = plt.subplots(2, 2, sharex = True, sharey = 'row', figsize = (10,6))
for j in range(2):
    for k in range(2):
        bp.append(ax[k,j].boxplot([np.abs(box_vectors.loc[(i,isSolar[k]),models[j]]) for i in N_houses],labels=[str(a) for a in N_houses],showmeans=True))
        ax[k,j].set_ylabel(y_label[k])
        if k ==0:
            ax[k,j].set_ylim([0,y_max])
        ax[k,j].set_xlabel(xlabel)
        ax[k,j].set_title(figure_title[k]+' - '+models_title[j])
        ax[k,j].grid(True)
#plt.show()
plt.savefig('Validation/figures/'+name)#Optimal_sensitivity_sd_std_tuning')

#%% N proxies
ii = 0
csvname = ['OptSens_pt1','OptSens_pt2','OptSens_st2']
csvname = ['OptSens_st2','OptSens_hourly']
fp = 'Validation/sensitivity/'+csvname[ii]+'.pkl'
df_all_all = load_pkl(fp)
#%
#plt.rcParams.update(plt.rcParamsDefault)
df_all = deepcopy(df_all_all[csvname[ii]])
df_all.reset_index(inplace=True)  
df_all.set_index(['dataid'], inplace = True)
df_all = df_all[df_all.index != 0]
N_each = 50
N_houses = (np.sort(df_all.N_read.unique())/96)[:-1] # sensitivity variable. I did not change name just to save time
models = ['ini', 'tune']
metrics = ['cv_pos']#['cv_pos']
isSolar = True
house_type = ['solar','no_solar']
box_vectors = pd.DataFrame(index = house_type,columns = models)
for i in house_type:
    if i == 'solar':
        metrics = ['cv_pos']
        isSolar = True
    else:
        metrics = ['rmse_pos']
        isSolar = False
    for j in models:
        index = [(df_all['tuning'] == j) & (df_all['isSolar'] == isSolar)] #& (df_all['istunesys'] == False)
        box_vectors.loc[i,j] = df_all.loc[index[0]][metrics].values.squeeze()
#% Plot results
#f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
y_max = 0.9
models_title = ['Initial model','Tuned model']
bp = []
#plt.close()
f, (ax1, ax2) = plt.subplots(2, 1, sharex = True, figsize = (6,6))#, sharex=True)
bp.append(ax1.boxplot([np.abs(box_vectors.loc['solar',i]) for i in models],labels=[str(a) for a in models_title],showmeans=True))
#f.suptitle('Solar houses')
ax1.set_ylabel('CV')
#ax1.set_ylim([0,y_max])
#ax1.set_xlabel('Number of days')
ax1.set_title('Solar houses')
bp.append(ax2.boxplot([np.abs(box_vectors.loc['no_solar',i]) for i in models],labels=[str(a) for a in models_title],showmeans=True))
ax2.set_ylabel('RMSE')
#ax2.set_xlabel('Number of days')
#ax2.set_ylim([0,y_max])
ax2.set_title('Non solar houses')
#plt.show()
#plt.savefig('Validation/figures/'+csvname[ii])#Optimal_sensitivity_sd_std_tuning')
#%% Calculate mean and median
box_vectors_ = pd.DataFrame(index = house_type,columns = ['means','medians'])
for count,i in enumerate(box_vectors_.index):
    for j in box_vectors_.columns:
        box_vectors_.loc[i,j] = [medline.get_ydata()[0] for medline in bp[count][j]]
box_vectors_


#%% N proxies
csvname = ['OptSens_nohourly_','OptSens_hourly_']
resolution_str = ['15 minutes data','hourly data']
for ii in range(2):
    
    fp = 'Validation/sensitivity/'+csvname[ii]+'.pkl'
    df_all_all = load_pkl(fp)
    #%
    #plt.rcParams.update(plt.rcParamsDefault)
    df_all = deepcopy(df_all_all[csvname[ii]])
    df_all.reset_index(inplace=True)  
    df_all.set_index(['dataid'], inplace = True)
    df_all = df_all[df_all.index != 0]
    N_each = 50
    N_houses = (np.sort(df_all.N_read.unique())/96)[:-1] # sensitivity variable. I did not change name just to save time
    models = ['ini', 'tune']
    metrics = ['cv_pos']#['cv_pos']
    isSolar = True
    house_type = ['solar','no_solar']
    box_vectors = pd.DataFrame(index = house_type,columns = models)
    for i in house_type:
        if i == 'solar':
            metrics = ['cv_pos_max']
#            metrics = ['max_sol_pred']
            isSolar = True
        else:
            metrics = ['rmse_pos']
#            metrics = ['max_sol_pred']
            isSolar = False
        for j in models:
            index = [(df_all['tuning'] == j) & (df_all['isSolar'] == isSolar)] #& (df_all['istunesys'] == False)
            box_vectors.loc[i,j] = df_all.loc[index[0]][metrics].values.squeeze()
    #% Plot results
    #f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    y_max = 0.9
    models_title = ['Initial model','Tuned model']
    bp = []
    #plt.close()
    if ii == 0:
        f, ax = plt.subplots(2, 2, sharex = True, sharey = 'row', figsize = (10,10))#, sharex=True)
    bp.append(ax[0,ii].boxplot([np.abs(box_vectors.loc['solar',i]) for i in models],labels=[str(a) for a in models_title],showmeans=True))
    #f.suptitle('Solar houses')
    ax[0,ii].set_ylabel('CV')
#    ax[0,ii].set_ylabel('Max solar predicted')
    ax[0,ii].grid(True)
    #ax1.set_ylim([0,y_max])
    #ax1.set_xlabel('Number of days')
    ax[0,ii].set_title('Solar houses - '+resolution_str[ii])
    bp.append(ax[1,ii].boxplot([np.abs(box_vectors.loc['no_solar',i]) for i in models],labels=[str(a) for a in models_title],showmeans=True))
    ax[1,ii].set_ylabel('RMSE')
#    ax[1,ii].set_ylabel('Max solar predicted')
    ax[1,ii].grid(True)
    #ax2.set_xlabel('Number of days')
#    ax[1,ii].set_ylim([0,3])
    ax[1,ii].set_title('Non solar houses')
#plt.show()
#plt.savefig('Validation/figures/'+csvname[ii])#Optimal_sensitivity_sd_std_tuning')
#%%
box_vectors_ = pd.DataFrame(index = house_type,columns = ['means','medians'])
for count,i in enumerate(box_vectors_.index):
    for j in box_vectors_.columns:
        box_vectors_.loc[i,j] = [medline.get_ydata()[0] for medline in bp[count][j]]
box_vectors_