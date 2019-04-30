# Solar Disaggregation extended validation
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
from csss.utilities import Setup_load as SetUp_load

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

def run_scenario_perfect_load(homes_ids,proxy_ids, DATES, tune_ids, IDS_solar):
#def run_scenario_perfect_load(Nhomes,Nproxies, Ndays, Ntunesys, Nsolar):
    ## Identify dataids to use
#    if Ntunesys > Nproxies:
#        print('Error: it must be Ntunesys <= Nproxies ')
#        return
#    solarids = np.random.choice(solar_ids['solar'], Nsolar+Nproxies, replace=False)
#    proxyids = solarids[:Nproxies]   # Homes to be used as solar proxy
#    solids  = solarids[Nproxies:]    # Homes with solar 
#    nosolids   = np.random.choice(solar_ids['nosolar'], Nhomes-Nsolar, replace=False)    # Homes without solar. 
#    homeids = np.concatenate([solids,nosolids]) # Homes used for solar disagg
#    tuneids  = homeids[:Ntunesys]  # Home to be used for tuning
#    dataids = np.concatenate([solarids,nosolids]) # Ids for all the systems
#    
    
    solarids = proxy_ids + IDS_solar
    proxyids = proxy_ids   # Homes to be used as solar proxy
    solids  = IDS_solar    # Homes with solar
    homeids = homes_ids # np.concatenate([solids,nosolids])  # Homes used for solar disagg
    nosolids   = list(set(homeids)-set(solids))    # Homes without solar. 
    
    tuneids  = tune_ids #solids[:Ntunesys]  # Home to be used for tuning - those are chosen different from the houses and from the proxies
    dataids = solarids+nosolids # +tuneids # Ids for all the systems #######
    
    
    
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

    ## Get indices for Ndays random dates
#    dates = fulldata[['Date','AggregateLoad']].groupby('Date').count()   ### Find number of readings for each date
#    numel = dates['AggregateLoad']                                      
#    dates = dates.index[numel == np.max(numel)]                          ### Choose only dates with complete readings
##     dates = dates[ np.random.choice(np.arange(len(dates)), replace=False, size=Ndays) ] ### Randomly select Ndays dates
#    start_sequence = int(np.random.choice(np.arange(len(dates)-Ndays), replace=False, size=1))
#    dates = dates[start_sequence:start_sequence+Ndays] ### select continous Ndays dates
    dates = DATES
#    induse = fulldata['Date'].isin(dates)  ### Subset data
    induse = fulldata.index.isin(dates)
    data = fulldata.loc[induse,:]

    ## Get HOD regressors
    hod = pd.Series([t.hour for t in data.index])
    hod = pd.get_dummies(hod)

    ## Get temperature regressors
    Tmin, Tmax, tempregress = regressor=createTempInput(data['AustinTemp'], 10)

    ## Prepare data for ingestion into SolarDisagg method. 
#    loadregressors = deepcopy(np.hstack([hod,tempregress]))
    netload = np.array(data[netloadcols])
    solarproxy = np.array(data[proxycol])
    names = ['solar_%s' % d for d in homeids]        

    ## Construct solar disaggregation problem
###### Test if the model improves when the load itself is used as a regressor. Perfect Info. I want to see which value of Theta is attributed and the alpha parameter too.
    loadregressors = deepcopy(np.expand_dims(data['AggregateLoad'].values, axis = 1)) # I do not include an offset too becuase then I want to see the Theta Value.     
#    loadregressors = deepcopy(np.hstack([hod,tempregress]))
    reload(csss.SolarDisagg)
    sdmod0 = copy.deepcopy(csss.SolarDisagg.SolarDisagg_IndvHome(netloads=netload, solarregressors=solarproxy, loadregressors=loadregressors, tuningregressors = hod, names = names))

    ## add true vectors
    for d in homeids:
        source_name = 'solar_%s' % d
        sdmod0.addTrueValue(name=source_name, trueValue=data[source_name])

    ## Add true aggregate load
    sdmod0.addTrueValue(name = 'AggregateLoad', trueValue=data['AggregateLoad'])

    ## Construct and solve the problem. 
    time_t = t_clock()
    print(t_clock())
    sdmod0.constructSolve(solver = 'GUROBI')
    print(t_clock()-time_t)
    time_t = t_clock()
    sdmod_tune = copy.deepcopy(sdmod0)
    print(t_clock()-time_t)
    time_t = t_clock()
    sdmod_tune.fitTuneModels(['solar_%s' % d for d in tuneids])
    print(t_clock()-time_t)
    time_t = t_clock()
    sdmod_tune.tuneAlphas()
    print(t_clock()-time_t)
    time_t = t_clock()
    sdmod_tune.constructSolve(solver = 'GUROBI')
    print(t_clock()-time_t)
    time_t = t_clock()

##### Comparison with a tuned model using the standard loadregressor
    loadregressors = deepcopy(np.hstack([hod,tempregress]))
    reload(csss.SolarDisagg)
    sdmod_tune_std = copy.deepcopy(csss.SolarDisagg.SolarDisagg_IndvHome(netloads=netload, solarregressors=solarproxy, loadregressors=loadregressors,tuningregressors = hod, names = names))
    ## add true vectors
    for d in homeids:
        source_name = 'solar_%s' % d
        sdmod_tune_std.addTrueValue(name=source_name, trueValue=data[source_name])

    ## Add true aggregate load
    sdmod_tune_std.addTrueValue(name = 'AggregateLoad', trueValue=data['AggregateLoad'])

    ## Construct and solve the problem. 
    sdmod_tune_std.constructSolve(solver = 'GUROBI')
    print(t_clock()-time_t)
    time_t = t_clock()
    sdmod_tune_std.fitTuneModels(['solar_%s' % d for d in tuneids])
    print(t_clock()-time_t)
    time_t = t_clock()
    sdmod_tune_std.tuneAlphas()
    print(t_clock()-time_t)
    time_t = t_clock()
    sdmod_tune_std.constructSolve(solver = 'GUROBI')
    print(t_clock()-time_t)
    time_t = t_clock()


    outdict = {}
    outdict['tuned_model'] = sdmod_tune
    outdict['initial_model'] = sdmod0
    outdict['tuned_model_std'] = sdmod_tune_std
    outdict['dates'] = dates
    outdict['times'] = data.index
    outdict['proxyids'] = proxyids
    outdict['homeids']  = homeids
    outdict['tuneids']  = tuneids
    outdict['solids']  = solids
    outdict['nosolids']  = nosolids
    outdict['data']  = data
    return(outdict)

#%% Create Output Data Frame
def create_outputdf(scen_out, name = 'output' ,scen_n = 1):
    mod = scen_out['tuned_model']
    
    scen_out['initial_model'].calcPerformanceMetrics()
    scen_out['tuned_model'].calcPerformanceMetrics()
    scen_out['tuned_model_std'].calcPerformanceMetrics()
    df0 = scen_out['initial_model'].performanceMetrics
    dft = scen_out['tuned_model'].performanceMetrics
    dfs = scen_out['tuned_model_std'].performanceMetrics
    
    
    df0['tuned'] = 'ini'
    dft['tuned'] = 'tune'
    dfs['tuned'] = 'tune_std'
    df0['tuned_temp'] = 'ini' # I need it for the index
    dft['tuned_temp'] = 'tune' # I need it for the index
    dfs['tuned_temp'] = 'tune_std' # I need it for the index
    dfout = pd.concat([df0,dft,dfs])
    
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
        for tuned in list(['ini','tune','tune_std']):
            name = 'solar_' + str(did)
            dfout.loc[(name,tuned),'frac_reverse'] = np.mean( mod.netloads[name] < 0 )
            dfout.loc[(name,tuned),'N_prox'] = mod.models[name]['regressor'].shape[1]
            dfout.loc[(name,tuned),'dataid'] = np.int(did)
            dfout.loc[(name,tuned),'istunesys'] = (did in scen_out['tuneids'])
            dfout.loc[(name,tuned),'isSolar'] = (did in scen_out['solids'])
    
    
    dfout = dfout.reset_index().set_index(['dataid','N_sys','tuned_temp','scen'])        
    index = pd.MultiIndex.from_tuples(dfout.index.values.tolist(), names=['dataid', 'N_houses','tuning','scen'])
    dfout.set_index(index,inplace = True)
    return(dfout)   
    
#%% Run example
t1 = t_clock()
rs_vec = [67,68,69]
N_proxy_vec = [1,2,3]
#rs_vec = [67]
#N_proxy_vec = [1]
scen_out_all = pd.DataFrame(columns = rs_vec, index = N_proxy_vec)
#df_out_all = pd.DataFrame(columns = rs_vec, index = N_proxy_vec)
df_out_all = {}
k = 0
fp = 'results_sd_1year.pkl'
for i in range(len(rs_vec)):
    for j in range(len(N_proxy_vec)):
        rs = rs_vec[i]
        N_proxy = N_proxy_vec[j]
#        rs = 67
        np.random.seed(rs)
#        N_proxy = 2
        N_tunes = 2
        proxy_tune_ids = list(np.random.choice(solar_ids['solar'],N_proxy+N_tunes, replace = False))
        proxy_ids = deepcopy(proxy_tune_ids[:N_proxy])
        tune_ids = deepcopy(proxy_tune_ids[N_proxy:N_proxy+N_tunes])
        IDS_solar = deepcopy((list(set(solar_ids['solar']) - set(proxy_ids))))#[0:6] + tune_ids)
        homes_ids = deepcopy(IDS_solar)#+list(np.random.choice(solar_ids['nosolar'],1,replace = False)))
        DATES = load_data_2.index[(load_data_2.index >= load_data_2.index[0])&(load_data_2.index < '2016-01-01')]#[0:1*96]
        ## Quick plot check
        for i in proxy_tune_ids:
            plt.figure()
            load_data_2['gen'][i].plot()
        try:
            scen_out_all.loc[N_proxy,rs] = [run_scenario_perfect_load(homes_ids,proxy_ids, DATES, tune_ids, IDS_solar)]
        except:
            df_out_all[(N_proxy,rs)] = np.NaN
        else:
            df_out_all[(N_proxy,rs)] = create_outputdf(scen_out_all.loc[N_proxy,rs][0], name = 'output' ,scen_n = k)
            with open(fp,"wb") as f:
                pk.dump([df_out_all,N_proxy_vec,rs_vec],f)
        k+=1
t2 = t_clock()
print('Performing {} simulations took {} minutes'.format(k,(t2-t1)/60))

#%% Analysis of the results

df_out = deepcopy(df_out_all[(N_proxy,rs)])
scen_out = deepcopy(scen_out_all.loc[(N_proxy,rs)][0])
#%%

#%%
#ids = ['solar_'+ str(idd) for idd in unique_list(scen_out['homeids'] + scen_out['tuneids'])] + ['AggregateLoad']
fp = 'results_sd_1year_scen_out.pkl'
def save_results(fp,scen_out,df_out,N_proxy,rs):
    ids = list(scen_out['tuned_model'].models.keys())
    Output_dict = {}
    models = ['initial_model','tuned_model','tuned_model_std' ]
    for model in models:
        Output_dict[model] = {}
    for model in models:
        for idd in ids:
            if idd[0] == 's':
                Output_dict[model][int(idd[6:])] = deepcopy(scen_out[model].models[idd])
            else:
                Output_dict[model][idd] = deepcopy(scen_out[model].models[idd])
    other_outputs = list(set(list(scen_out.keys())) - set(models))
    #for item in models:
    #    other_outputs.remove(item)
    #['dates','times','proxyids','homeids','tuneids','solids','nosolids','data']
    for name in other_outputs:
        Output_dict[name] = deepcopy(scen_out[name])
    with open(fp,"wb") as f:
        pk.dump([Output_dict,df_out,N_proxy,rs],f)
    print('Results correctly saved in '+fp)
save_results(fp,scen_out)
def load_results(fp):
    with open(fp,"rb") as f:
        Output_dict,df_out,N_proxy,rs = pk.load(f)
    return Output_dict,df_out,N_proxy,rs
scen_out,df_out,N_proxy,rs = load_results(fp)
#%%
def unique_list(x):
#    x is a list
    u_list = []
    for el in x:
        if el not in u_list:
            u_list.append(el)
    return u_list
#%%
ids = df_out.index.levels[0][1:]
N_houses = df_out.index.levels[1][0]
n_scen = df_out.index.levels[3][0]
models = df_out.index.levels[2]
model = 'tune_std'
metric = 'cv_pos'
index = unique_list(df_out[df_out['istunesys'] == False].index.get_level_values('dataid'))
index.remove(0)
df_res = pd.DataFrame(index = index, columns = models)
for i in range(len(models)):
    model = models[i]    
    df_res.loc[:,model] = df_out.loc[(ids,N_houses,model,n_scen)][df_out['istunesys'] == False][metric].values
df_res = np.abs(df_res)
df_res
plt.scatter(range(len(df_res)),(df_res['tune_std'] - df_res['tune']))
plt.xlabel('house ID')
plt.ylabel('cv_pos')
plt.savefig('figures/solar_estimation_1year_cv_comp.png',bbox_inches = 'tight')
#plt.show()
#%% #Best 8236 #Worst 9935 for both models
plt.close()
for idd in [8236, 9935]:
    df_actual = grouped_data.get_group(idd)
    df_actual = df_actual[df_actual.index < '2016-01-01']
    mask = df_actual['gen'] > 0.05*df_actual['gen'].mean()
    s_l_ratio = df_actual.gen[mask].mean()/df_actual.use[mask].mean() 
    print(idd,s_l_ratio)
#    start = np.random.choice(range(len(df_actual)),1)[0]
    start = 15762
    n_pts = 5* 96
    fig, ax = plt.subplots()
    plt.plot(df_actual['gen'].index[start:start+n_pts],df_actual['gen'].values[start:start+n_pts], label = 'actual', linewidth=1)
    plt.plot(df_actual['gen'].index[start:start+n_pts],-scen_out['tuned_model'][idd]['source'].value.A1[start:start+n_pts] , label = 'tuned_perfect', linewidth=1)
    plt.plot(df_actual['gen'].index[start:start+n_pts],-scen_out['tuned_model_std'][idd]['source'].value.A1[start:start+n_pts] , label = 'tuned_std', linewidth=1)
    if idd == 9935:
        plt.plot(df_actual['gen'].index[start:start+n_pts],-scen_out['initial_model'][idd]['source'].value.A1[start:start+n_pts] , label = 'initial', linewidth=1)
    plt.setp(ax.lines, linewidth=1.5)
    plt.legend(loc= 'upper right')
    plt.title('Solar estimation for ID: '+str(idd) + ' Solar/Load = ' +str(round(s_l_ratio,2)))
#    plt.savefig('figures/solar_estimation_1year' + str(idd)+'.png',bbox_inches = 'tight')
    plt.show()