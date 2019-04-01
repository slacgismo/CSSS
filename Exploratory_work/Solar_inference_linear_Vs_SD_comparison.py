'''
The aim of this python file is to run simulations to compare the solar disaggregation optimsation problem with the linear model approach.
'''

import sys
sys.path.append("..") # Adds higher directory to python modules path.
# import ipy_autoreload
from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csss
import datetime as dt
import pickle as pk
from imp import reload
import copy
from time import time as t_clock
from copy import deepcopy
import seaborn as sns
from csss.SolarDisagg import createTempInput, createSolarDisaggIndvInputs, Solar_Disagg_linear_model
from csss.utilities import Setup_load as SetUp_load
import math
import pprint
from Custom_Functions.error_functions import cv_pos,rmse_pos
from csss.utilities import save_pkl, load_pkl


# ### Load the data
Setup_load = deepcopy(SetUp_load())
Setup_load.QueryOrLoad()#(start_date = '01-01-2015', end_date = '01-10-2015')
load_data, load_data_2, weather, grouped_data, ids, solar_ids, homeids  = Setup_load.load_setup()
load_data_2['temperature'] = weather['temperature']

## Remove those solar houses that have generation = 0 for continous N_obs.
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
load_data_2.swaplevel(axis = 1).drop(remove,axis = 1, inplace = True)
solar_ids['solar'] = list(set(solar_ids['solar']) - set(remove))
solar_ids['nosolar'] = list(set(solar_ids['nosolar']) - set(remove))
grouped_data = load_data.groupby("dataid")

## Do a quick plot of load data
df = deepcopy(grouped_data.get_group(solar_ids['solar'][0]))
np.random.seed(9)
start = np.random.choice(range(len(df)))
df.iloc[start:start+2*96,1:].plot()
plt.show()
#%% ## Parameters setup for the performance evaluation
df = deepcopy(load_data_2)
N_each = 50
Nproxies = 2
Ntunesys = 2
Ndays = 30
df_all_ = pd.DataFrame()
df0 = pd.DataFrame()
Nhomes_vec = np.array([8])#,45,80,120])
errors = pd.Series(0,index = Nhomes_vec)
for Nhomes in Nhomes_vec:
    for j in range(N_each):
        np.random.seed(j)
        for hourly in [False,True]:
            df = deepcopy(load_data_2)
            t1 = t_clock()
         ## Choose homes and proxies
            proxy_ids = list(np.random.choice(solar_ids['solar'],Nproxies, replace = False))
            tune_ids = list(np.random.choice(list(set(solar_ids['solar'])-set(proxy_ids)),Ntunesys,replace = False))
            if not tune_ids:
                tune_ids = proxy_ids
            Nsolar = int(Nhomes/2)
            solarIDS =  list(np.random.choice(list(set(solar_ids['solar'])-set(tune_ids)-set(proxy_ids)),Nsolar,replace = False))
            nosolarIDS =  list(np.random.choice(solar_ids['nosolar'],int(Nhomes-Nsolar),replace = False))
            home_ids = solarIDS+nosolarIDS+tune_ids
        #    home_ids = list(np.random.choice(ids,Nhomes,replace = False)) + tune_ids
            
             ## Choose the time index - when perform it 
            n_days = Ndays # number of days to analyse
            resolution_minutes = (df.index[1]-df.index[0]).seconds/60 # determine the resolution of the data
            start_date = np.random.choice(np.arange(0,(len(df)-int(24*60/resolution_minutes*(n_days+1)))))
            time_index = [df.index[start_date],df.index[start_date+int(24*60/resolution_minutes*n_days)]] # beginiing and end datetime of the window to analyse
            index_time = df.index[(df.index >= time_index[0]) & (df.index <= time_index[1])] # index useful to plot later
            df = df.loc[index_time]
            if hourly == True:    
                df = df.resample('H').mean().dropna()
            index_time = df.index
             ## Creating the data for ingestion in Solar Disagg
            data = createSolarDisaggIndvInputs(df, home_ids, solar_proxy_ids= proxy_ids)
            # pprint.pprint(data, indent=1)
                    
            
         ## Solar Disaggregation and Linear model comparison for the same houses ids
            
             ## Solar Disagg method
            ## Initial model
            try:
                reload(csss.SolarDisagg)
                sdmod0 = deepcopy(csss.SolarDisagg.SolarDisagg_IndvHome(**data))
                sdmod0.constructSolve(solver = None) # if solver = None, it uses the default one    
            ## Initialising the tuned model and adding tuning system generation
                sdmod_tune = deepcopy(sdmod0)
                for key in tune_ids:
                   sdmod_tune.addTrueValue(-df['gen'][key].loc[index_time], str(key))
            ## update the alpha weights and perform the disaggregation
                sdmod_tune.fitTuneModels(list(map(str,tune_ids)))
                sdmod_tune.tuneAlphas()
                sdmod_tune.constructSolve(solver = None)
            ## add true value and calculate the performances
                def addTrue_CalcPerf(sdmod, tuning = 'sd_tuned'):
                    for key in home_ids:
                        sdmod.addTrueValue(-df['gen'][key].loc[index_time], str(key))
                    sdmod.models['AggregateLoad']['source'] = np.sum([sdmod.models[str(idd)]['source'] for idd in data['names']], axis = 0)
                
            ## Swap_AggLoad_FeederGen(sdmod_tune)
                    ## I use the model of the AggregateLoad to fill it with the estimated aggregated generation at the feeder level. So now the aggregate load is actually the feeder generation to calculate the performances..
                    sdmod.addTrueValue(name = 'AggregateLoad', trueValue=-df.loc[index_time,'gen'][list(map(int,data['names']))].sum(axis = 1))
                    sdmod.calcPerformanceMetrics()
                    sdmod.snr(df_true = df.loc[index_time])
                    df_sd = sdmod.performanceMetrics
                    df_sd['isSolar'] = [int(i) in solar_ids['solar'] for i in df_sd.index if i != 'AggregateLoad'] + [False]
                    df_sd['model']   = tuning
                    df_sd['Nhouses']   = Nhomes
                    df_sd['hourly_resolution']  = hourly
                    df_sd = pd.concat([df_sd, sdmod.SNR], axis=1)
                    return df_sd
                
                df_sd0 = addTrue_CalcPerf(sdmod0, tuning = 'sd_initial')
                df_sd  = addTrue_CalcPerf(sdmod_tune, tuning = 'sd_tuned')
            except:
                print('Exception Triggered')    
                errors.loc[Nhomes]+=1
                df_sd0 = pd.DataFrame()
                df_sd = pd.DataFrame()
            
            ## Linear model
            reload(csss.SolarDisagg)
            ## Initialise the solar disagg linear class
            sdlinear = deepcopy(Solar_Disagg_linear_model(netloads = data['netloads'],solarregressors = data['solarregressors'], loadregressors = data['loadregressors'], names = data['names']))
            sdlinear.fit()
            solar_pred = (sdlinear.predict())
            sdlinear.calcPerformanceMetrics(df_true = -df.loc[index_time,'gen'][list(map(int,data['names']))])
            df_lm = sdlinear.performanceMetrics
            sdlinear.snr(df_true = df.loc[index_time])
            df_lm['isSolar'] = [int(i) in solar_ids['solar'] for i in df_lm.index if i != 'AggregateLoad'] + [False]
            df_lm['model'] = 'linear'
            df_lm['Nhouses']   = Nhomes
            df_lm['hourly_resolution']  = hourly
            df_lm = pd.concat([df_lm, sdlinear.SNR], axis=1)
            df0 = pd.concat([df_sd0,df_sd,df_lm])
            df_all_ = pd.concat([df_all_,df0])
            fp = 'Exploratory_work/data/results_sd_VS_linear_data_resolution'
            save_pkl([df_all_,errors],fp+'.pkl')
            if j == 9:
               df_all_.to_csv(fp+'.csv')
            print(Nhomes,j)
            t2 = t_clock()
            print('1 iteration took {}'.format((t2-t1)/60))
        
#for i in solar_pred.columns:
#    plt.figure()
#    plt.plot(solar_pred[i]
#%% Plots - linear model
n_homes = len(home_ids)
n_pts = len(index_time) # number of points to plot
fig, ax = plt.subplots(nrows= n_homes+1, ncols=1, sharex=True, figsize=(16,1.5*n_homes)) # n_pts/96*1.5
for i in range(n_homes):
    idd = str(home_ids[i])
    ax[i].plot(index_time,-solar_pred[idd].values[0:n_pts], label = 'gen_'+idd)
    ax[i].plot(index_time,df['gen'][int(idd)].loc[index_time].values[:n_pts], label = 'true_'+idd)
    ax[i].legend()
## add Feeder Level generation plot
ax[i+1].plot(index_time,-solar_pred[data['names']].sum(axis = 1).values[0:n_pts], label = 'gen_'+'feeder')
ax[i+1].plot(index_time,np.sum([df['gen'][int(idd)].loc[index_time].values[:n_pts] for idd in home_ids], axis = 0), label = 'true_'+'feeder')
ax[i+1].legend()
plt.tight_layout()
plt.show()

## Plots solar disagg
n_homes = len(home_ids)
n_pts = len(index_time) # number of points to plot
fig, ax = plt.subplots(nrows= n_homes+1, ncols=1, sharex=True, figsize=(16,1.5*n_homes)) # n_pts/96*1.5
for i in range(n_homes):
    idd = str(home_ids[i])
    ax[i].plot(index_time,-sdmod_tune.models[idd]['source'].value[0:n_pts], label = 'gen_'+idd)
    ax[i].plot(index_time,df['gen'][int(idd)].loc[index_time].values[:n_pts], label = 'true_'+idd)
    ax[i].legend()
## add Feeder Level generation plot
ax[i+1].plot(index_time,np.sum([-sdmod_tune.models[str(idd)]['source'].value[0:n_pts] for idd in home_ids], axis = 0), label = 'gen_'+'feeder')
ax[i+1].plot(index_time,np.sum([df['gen'][int(idd)].loc[index_time].values[:n_pts] for idd in home_ids], axis = 0), label = 'true_'+'feeder')
ax[i+1].legend()
plt.tight_layout()
plt.show()
# plt.savefig('figures/example_solar_disag_3months_px.png')
# Calculate the estimated SNR for the non solar houses. 
#%% Linear Vs Tuned
df_all = deepcopy(df_all_)
df_all = df_all[df_all.index != 'AggregateLoad']
#N_each = 50
models = ['sd_initial','sd_tuned','linear']
metrics = ['cv_pos','rmse']
isSolar = [True,False]
box_vectors = pd.DataFrame(index = isSolar, columns = models)
for l,k in enumerate(isSolar):
    for j in models:
        index = [(df_all['model'] == j) & (df_all['isSolar'] == k)]# & (df_all[sens_var] == i)] #& (df_all['istunesys'] == False)
        box_vectors.loc[k,j] = df_all.loc[index[0]][metrics[l]].values.squeeze()

#%% Plot results
#f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
y_max = 2
models_title = ['Initial model','Tuned model']
figure_title = ['Solar houses','Non solar houses']
y_label = ['CV','RMSE']
bp = []
plt.close('all')
f, ax = plt.subplots(1, 2, sharex = True, figsize = (10,4))
for j in range(2):
    bp.append(ax[j].boxplot([np.abs(box_vectors.loc[isSolar[j],k]) for k in models ],labels=[k for k in models],showmeans=True))
    ax[j].set_ylabel(y_label[j])
    if j ==0:
        ax[j].set_ylim([0,y_max])
    ax[j].set_title(figure_title[j])
    ax[j].grid(True)
plt.show()
#plt.savefig('Validation/figures/'+'sd_linear')#Optimal_sensitivity_sd_std_tuning')

box_vectors_ = pd.DataFrame(index = isSolar,columns = ['means','medians'])
for count,i in enumerate(box_vectors_.index):
    for j in box_vectors_.columns:
        box_vectors_.loc[i,j] = [medline.get_ydata()[0] for medline in bp[count][j]]
box_vectors_

#%% Feeder Level disaggregation
df_all = deepcopy(df_all_)
df_all = df_all[df_all.index == 'AggregateLoad']
#N_each = 50
models = ['linear', 'sd_tuned']
metrics = ['cv_pos','rmse']
isSolar = [False,True]
box_vectors = pd.DataFrame(index = isSolar, columns = models)
for l,k in enumerate(isSolar):
    for j in models:
        index = [(df_all['model'] == j) & (df_all['isSolar'] == k)]# & (df_all[sens_var] == i)] #& (df_all['istunesys'] == False)
        box_vectors.loc[k,j] = df_all.loc[index[0]][metrics[l]].values.squeeze()

#%% Plot results
#f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
y_max = 2
models_title = ['Initial model','Tuned model']
figure_title = ['Feeder level disaggregation']# houses','Non solar houses']
y_label = ['CV','RMSE']
bp = []
plt.close('all')
f, ax = plt.subplots(1, 1, sharex = True, sharey = 'row', figsize = (10,6))
for j in range(1):
    bp.append(ax.boxplot([np.abs(box_vectors.loc[isSolar[j],k]) for k in models ],labels=[k for k in models],showmeans=True))
    ax.set_ylabel(y_label[j])
#    if k ==0:
#        ax[k,j].set_ylim([0,y_max])
    ax.set_title(figure_title[j])
    ax.grid(True)
#plt.show()
plt.savefig('Validation/figures/'+'feeder_disagg_cv')#Optimal_sensitivity_sd_std_tuning')

box_vectors_ = pd.DataFrame(index = isSolar,columns = ['means','medians'])
for count,i in enumerate(box_vectors_.index):
    for j in box_vectors_.columns:
        box_vectors_.loc[i,j] = [medline.get_ydata()[0] for medline in bp[count][j]]
box_vectors_

#%% houses sensitivity Tuned Vs Linear
name,sens_var,xlabel = 'results_sd_VS_linear_Nhouses_snr', 'Nhouses', 'N houses'
name,sens_var,xlabel = 'results_sd_VS_linear_data_resolution', 'hourly_resolution', 'N houses'
fp = Explorato+name+'.pkl'
df_all = load_pkl(fp)
df_all = df_all[df_all.index != 'AggregateLoad']
N_each = 50
N_houses = np.sort(df_all[sens_var].unique())
models = ['linear', 'sd_tuned']
metrics = ['cv_pos','rmse']
isSolar = [True,False]
box_vectors = pd.DataFrame(index = pd.MultiIndex.from_tuples([(int(i),j) for j in isSolar for i in N_houses]), columns = models)
for l,k in enumerate(isSolar):
    for j in models:
        for i in N_houses:
            index = [(df_all['model'] == j) & (df_all['isSolar'] == k) & (df_all[sens_var] == i)] #& (df_all['istunesys'] == False)
            box_vectors.loc[(i,k),j] = df_all.loc[index[0]][metrics[l]].values.squeeze()

#%% Plot results
#f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
y_max = 2
models_title = ['Linear model','Tuned model']
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
plt.show()
#plt.savefig('Inference Approach/figures/'+name)#Optimal_sensitivity_sd_std_tuning')
#%% houses sensitivity Tuned Vs Linear - Feeder Level
name,sens_var,xlabel = 'results_sd_VS_linear_Nhouses_snr', 'Nhouses', 'N houses'
fp = 'Inference Approach/'+name+'.pkl'
df_all = load_pkl(fp)
df_all = df_all[df_all.index == 'AggregateLoad']
N_each = 50
N_houses = np.sort(df_all[sens_var].unique())
models = ['linear', 'sd_tuned']
metrics = ['cv_pos','cv_pos_max']
isSolar = [False,False]
box_vectors = pd.DataFrame(index = pd.MultiIndex.from_tuples([(int(i),j) for j in metrics for i in N_houses]), columns = models)
for l,k in enumerate(isSolar):
    for j in models:
        for i in N_houses:
            index = [(df_all['model'] == j) & (df_all['isSolar'] == k) & (df_all[sens_var] == i)] #& (df_all['istunesys'] == False)
            box_vectors.loc[(i,metrics[l]),j] = df_all.loc[index[0]][metrics[l]].values.squeeze()

#%% Plot results
#f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
y_max = 2
models_title = ['Linear model','Tuned model']
figure_title = ['Feeder level disaggregation','Non solar houses']
y_label = ['CV','CV_max']
bp = []
plt.close('all')
f, ax = plt.subplots(2, 2, sharex = True, sharey = 'row', figsize = (10,6))
for j in range(2):
    for k in range(2):
        bp.append(ax[k,j].boxplot([np.abs(box_vectors.loc[(i,metrics[k]),models[j]]) for i in N_houses],labels=[str(a) for a in N_houses],showmeans=True))
        ax[k,j].set_ylabel(y_label[k])
#        if k ==0:
#            ax[k,j].set_ylim([0,y_max])
        ax[k,j].set_xlabel(xlabel)
        ax[k,j].set_title(figure_title[0]+' - '+models_title[j])
        ax[k,j].grid(True)
#plt.show()
plt.savefig('Inference Approach/figures/'+name+'_feeder')#Optimal_sensitivity_sd_std_tuning')
#%% SNR
name,sens_var,xlabel = 'results_sd_VS_linear_Nhouses_snr', 'Nhouses', 'N houses'
fp = 'Inference Approach/'+name+'.pkl'
df_all = load_pkl(fp)
df_all = df_all[df_all.index != 'AggregateLoad']

plt.figure()
plt.close('all')
plt.scatter(df_all['snr_true'][df_all['isSolar']==True],-df_all['cv_pos'][df_all['isSolar']==True])
plt.ylim([0,3])
plt.xlim([0,5])


    plt.close()
    fig = plt.figure()
    class_names = ['solar','non solar']
    model = 'sd_tuned'
    plt.scatter(df_all['snr_est'][(df_all['isSolar']==True) & (df_all['model']==model)],np.ones(int(sum((df_all['isSolar']==True)& (df_all['model']==model)))),c='red')
    plt.scatter(df_all['snr_est'][(df_all['isSolar']==False) & (df_all['model']==model)],0.1+np.ones(int(sum((df_all['isSolar']==False)& (df_all['model']==model)))),c='green')
    #plt.scatter(alpha.p1,np.ones(len(alpha)),c=alpha.solar_bin)
    plt.legend(class_names)
    #plt.xlim([-0.05,0.05])
    plt.ylim([0.9,1.2])
    plt.show()
    
#%% Data Resolution Initial - Tuned - Linear
name,sens_var,xlabel = 'results_sd_VS_linear_data_resolution', 'hourly_resolution', 'N houses'
fp = 'Exploratory_work/data/'+name+'.pkl'
df_all = load_pkl(fp)[0]
df_all = df_all[df_all.index != 'AggregateLoad']
N_each = 50
#N_houses = np.sort(df_all[sens_var].unique())
N_houses = [False,True]
models = ['sd_initial', 'sd_tuned']#, 'linear']
metrics = ['cv_pos','rmse']
isSolar = [True,False]
box_vectors = pd.DataFrame(index = pd.MultiIndex.from_tuples([(i,j) for j in isSolar for i in N_houses]), columns = models)
for l,k in enumerate(isSolar):
    for j in models:
        for i in N_houses:
            index = [(df_all['model'] == j) & (df_all['isSolar'] == k) & (df_all[sens_var] == i)] #& (df_all['istunesys'] == False)
            box_vectors.loc[(i,k),j] = df_all.loc[index[0]][metrics[l]].values.squeeze()

#%% Plot results
#f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
y_max = 2
models_title = ['Initial model','Tuned model']#,'Linear model']
figure_title = ['Solar houses','Non solar houses']
resolution_title = ['15m','hourly']
y_label = ['CV','RMSE']
bp = []
plt.close('all')
f, ax = plt.subplots(2, 2, sharex = True, sharey = 'row', figsize = (10,6))
for j in range(2):
    for k in range(2):
        bp.append(ax[k,j].boxplot([np.abs(box_vectors.loc[(N_houses[j],isSolar[k]),i]) for i in models],labels=[a for a in models_title],showmeans=True))
        ax[k,j].set_ylabel(y_label[k])
        if k ==0:
            ax[k,j].set_ylim([0,y_max])
#        ax[k,j].set_xlabel(xlabel)
#        ax[k,j].set_title(figure_title[k]+' - '+resolution_title[j])
        ax[k,j].grid(True)
#plt.show()
plt.savefig('Exploratory_work/figures/'+'data_resolution_it')#Optimal_sensitivity_sd_std_tuning')
#%% Data Resolution Initial - Tuned - Linear - Feeder Level
name,sens_var,xlabel = 'results_sd_VS_linear_data_resolution', 'hourly_resolution', 'N houses'
fp = 'Exploratory_work/data/'+name+'.pkl'
df_all = load_pkl(fp)[0]
df_all = df_all[df_all.index == 'AggregateLoad']
N_each = 50
#N_houses = np.sort(df_all[sens_var].unique())
N_houses = [False,True]
models = ['sd_initial', 'sd_tuned']#, 'linear']
metrics = ['cv_pos','cv_pos_max']
isSolar = [False,False]
box_vectors = pd.DataFrame(index = pd.MultiIndex.from_tuples([(i,j) for j in metrics for i in N_houses]), columns = models)
for l,k in enumerate(isSolar):
    for j in models:
        for i in N_houses:
            index = [(df_all['model'] == j) & (df_all['isSolar'] == k) & (df_all[sens_var] == i)] #& (df_all['istunesys'] == False)
            box_vectors.loc[(i,metrics[l]),j] = df_all.loc[index[0]][metrics[l]].values.squeeze()

#%% Plot results
#f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
y_max = 2
models_title = ['Initial model','Tuned model']#,'Linear model']
figure_title = ['Feeder level disaggregation - 15m','Feeder level disaggregation - hourly']
y_label = ['CV','CV_max']
bp = []
plt.close('all')
f, ax = plt.subplots(2, 2, sharex = True, sharey = 'row', figsize = (10,6))
for j in range(2):
    for k in range(2):
        bp.append(ax[k,j].boxplot([np.abs(box_vectors.loc[(N_houses[j],metrics[k]),i]) for i in models],labels=[str(a) for a in models_title],showmeans=True))
        ax[k,j].set_ylabel(y_label[k])
#        if k ==0:
#            ax[k,j].set_ylim([0,y_max])
#        ax[k,j].set_xlabel(xlabel)
#        ax[k,j].set_title(figure_title[j])#+' - '+models_title[j])
        ax[k,j].grid(True)
#plt.show()
plt.savefig('Exploratory_work/figures/'+'_feeder'+'data_resolution')#Optimal_sensitivity_sd_std_tuning')