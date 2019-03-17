import csss as CSSS
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression as LR
import copy
from Custom_Functions.error_functions import rmse_pos
##### Davide Modifications: 
#    I enriched the calcPerformanceMetrics routine with new performance metrics
#    I added the possibility to feed which IDS do not have solar and contrain their generation to 0
#    I added a function Mape_mod
#    I added a class: Solar_Disagg_linear_model to perform solar disagg using a n inference aproach with a linear model
#    I added a method to calculate the signal2noise ratio after the disaggregation has been performed
class SolarDisagg_IndvHome(CSSS.CSSS):
    def __init__(self, netloads, solarregressors, loadregressors, tuningregressors=None, names=None, nosolar_ids = None):
        """

        :param netloads:        np.array of net loads at each home, with columns corresponding to entries of "names" if
                                available.
        :param solarregressors: np.array of solar regressors (N_s X T)
        :param loadregressors:  np.array of load regressors (N_l x T)
        :param tuningregressors:
        :param names:
        """

        ## Find aggregate net load, and initialize problem.
        agg_net_load = np.sum(netloads, axis = 1)
        CSSS.CSSS.__init__(self, agg_net_load)

        ## If no names are input, create names based on id in vector.
        self.N, self.M = netloads.shape
        if names is None:
            self.names = [str(i) for i in np.arange(self.M)]
        else:
            self.names = names

        ## Store net loads as a dictionary
        self.netloads = {}
        for i in range(self.M):
            name = self.names[i]
            self.netloads[name] = netloads[:,i]

        ## If no tuning regressors are input, use an intercept only
        if tuningregressors is None:
            tuningregressors = np.ones((self.N,1))

        ## Store solar and load regressors, solar regressors, and begin true solar dict
        self.solarRegressors = solarregressors
        self.loadRegressors  = loadregressors
        self.tuningRegressors  = tuningregressors
        self.trueValues      = {}
        self.nosolar_ids     = nosolar_ids

        ## Cycle through each net load, and create sources.
        for source_name in self.names:
            self.addSource(regressor=solarregressors, name = source_name, alpha = 1)

            ## Add constraints that solar generation cannot exceed zero or net load.
            self.addConstraint( self.models[source_name]['source'] <= np.array(self.netloads[source_name]) )
            self.addConstraint( self.models[source_name]['source'] <= 0 )
            self.addConstraint( self.models[source_name]['theta'] >= 0 ) ####################
            
        
##################
        if self.nosolar_ids is not None:
            for source_name in self.nosolar_ids:
                self.addConstraint( self.models[source_name]['source'] == 0 )
##################

        ## Add the aggregate load source
        self.addSource(regressor=loadregressors, name = 'AggregateLoad', alpha = 1)
        self.addConstraint( self.models['AggregateLoad']['source'] > 0 )

    def Solar_var_norm(self):
        return(None)
        ## Placeholder for variance prediction for tuning

    def Total_NL_var(self):
        return(None)
        ## Placeholder for variance prediction for tuning

    def addTrueValue(self, trueValue, name):
        ## Function to add true solar for a given model

        ## Check that true value is correct number of dimensions
        trueValue = trueValue.squeeze()
        if not (trueValue.shape == (self.N,)):
            raise Exception('True value of a solar or load signal must be one dimensional and length N = %d' % self.N)

        if name not in (self.names + ['AggregateLoad']):
            raise Exception('Must input a valid household identifier or \"AggregateLoad\"')

        ## Add True Value
        self.trueValues[name] = trueValue
        return(None)


    def calcPerformanceMetrics(self, dropzeros = False):
        ## Function to calculate performance metrics
        # Dropping zeros is intended to remove nightime solar.

        df = pd.DataFrame()
        df['models'] = self.models.keys()
        df['rmse']   = np.zeros(df.shape[0]) * np.nan
        df['cv']     = np.zeros(df.shape[0]) * np.nan
        df['mae']    = np.zeros(df.shape[0]) * np.nan
        df['pmae']   = np.zeros(df.shape[0]) * np.nan
        df['mbe']    = np.zeros(df.shape[0]) * np.nan
        df['mean']   = np.zeros(df.shape[0]) * np.nan
        df['MAPE']   = np.zeros(df.shape[0]) * np.nan
        df['mae_max']= np.zeros(df.shape[0]) * np.nan
        df['cv_max'] = np.zeros(df.shape[0]) * np.nan
        df['max_sol_pred'] = np.zeros(df.shape[0]) * np.nan
        
        df['cv_pos']     = np.zeros(df.shape[0]) * np.nan
        df['rmse_pos']   = np.zeros(df.shape[0]) * np.nan
        df['mae_pos']    = np.zeros(df.shape[0]) * np.nan
        df['pmae_pos']   = np.zeros(df.shape[0]) * np.nan
        df['mbe_pos']    = np.zeros(df.shape[0]) * np.nan
        df['mean_pos']   = np.zeros(df.shape[0]) * np.nan
        df['MAPE_pos']   = np.zeros(df.shape[0]) * np.nan
        df['mae_pos_max']= np.zeros(df.shape[0]) * np.nan
        df['cv_pos_max'] = np.zeros(df.shape[0]) * np.nan
        df               = df.set_index('models')

        for name in self.trueValues.keys():
            truth   = self.trueValues[name]
            est     = np.array(self.models[name]['source'].value).squeeze()

            ## Calculate metrics.
            df.loc[name,'mbe']  = np.mean((truth-est))
            df.loc[name,'mean'] = np.mean((truth))
            df.loc[name,'rmse'] = np.sqrt(np.mean((truth-est)**2))
            df.loc[name,'mae']  = np.mean(np.abs((truth-est)))
            df.loc[name,'MAPE'] = MAPE_mod(est,truth,thrs=0.001)
            df.loc[name,'max_sol_pred'] = np.max(np.abs(est))
            try:
                df.loc[name,'rmse_pos'] = rmse_pos(est,truth)
            except:
                df.loc[name,'rmse_pos'] = np.nan
        
            if not (df.loc[name,'mean'] == 0):
                df.loc[name,'cv']   = df.loc[name,'rmse'] / np.mean(truth)
                df.loc[name,'pmae'] = df.loc[name,'mae']  / np.mean(truth)
                df.loc[name,'mae_max']  =  df.loc[name,'mae']/ np.max(np.abs(truth))
                df.loc[name,'cv_max'] = df.loc[name,'rmse'] / np.max(np.abs(truth))
                
                ## Find metrics for  positive indices only
                posinds = np.abs(truth) > (0.05 * np.abs(np.mean(truth)))
                truth = truth[posinds]
                est   = est[posinds]
#                df.loc[name,'rmse_pos'] = np.sqrt(np.mean((truth-est)**2))
                df.loc[name,'mae_pos']  = np.mean(np.abs((truth-est)))
                df.loc[name,'cv_pos']   = df.loc[name,'rmse_pos'] / np.mean(truth)
                df.loc[name,'pmae_pos'] = df.loc[name,'mae_pos']  / np.mean(truth)
                df.loc[name,'mbe_pos']  = np.mean((truth-est))
                df.loc[name,'mean_pos'] = np.mean((truth))
                df.loc[name,'MAPE_pos'] = MAPE_mod(est,truth)
                df.loc[name,'mae_pos_max']  = df.loc[name,'mae_pos'] / np.max(np.abs(truth))
                df.loc[name,'cv_pos_max']   = df.loc[name,'rmse_pos'] / np.max(np.abs(truth))
                
                
        self.performanceMetrics = df

        return(None)


    def tuneAlphas_v1(self, tuneSys = None, filter_vec = np.ones(12)/12.0, var_lb_fraction = 0.01):
        ## Function to autotune alphas given some true solar information.
        if tuneSys is None:
            ## If no name for a tuning system is input, use all systems for which
            # a truth is known.
            tuneSys = self.trueValues.keys()
            if 'AggregateLoad' in tuneSys: tuneSys.remove('AggregateLoad')


        ## For each system used for tuning, filter the square residuals.
        filt_sq_resid_norm = np.ones((self.N,len(tuneSys)))
        i=0
        for name in tuneSys:
            truth      = self.trueValues[name].squeeze()
            #modelest   = self.models[name]['regressor'] * self.models[name]['theta']
            #modelest   = np.array(modelest.value).squeeze()

            ## Run a quick regression to collect expected value of
            # the lienar model given truth
            model       = LR()
            model.fit(X = self.models[name]['regressor'], y = truth)
            modelest    = model.predict(X = self.models[name]['regressor'])

            ## Create a rough capacity estimate from the theta values
            capest     = np.sum(self.models[name]['theta'].value)
            resid_norm      = (truth - modelest) / capest
            #filt_resid = convolve_cyc( resid, filter_vec )
            #sq_resid_demean   = ( resid - filt_resid ) ** 2
            sq_resid = resid_norm ** 2

            filt_sq_resid_norm[:,i] = convolve_cyc( sq_resid , filter_vec )
            i=i+1

        ## Average the filtered squared residuals
        ave_filt_sq_resid_norm = np.mean(filt_sq_resid_norm, axis = 1)

        ## Create alphas for each other PV system
        total_sol_var   = np.zeros(self.N) ## Instantiate vector for total variance of PV signals,
        total_model_est = np.zeros(self.N) ## Instantiate vector for linear model prediction of net load.

        ## Cycle through each solar model and tune alphas
        for name in self.models.keys():
            ## Model estimated value
            model_est = self.models[name]['regressor'] * self.models[name]['theta'] ## model estimate
            model_est = np.array(model_est.value).squeeze()
            total_model_est = total_model_est + model_est             ## total model estimate

            ## Don't solve for aggregate load yet
            if name.lower() == 'aggregateload':
                continue

            capest      = np.sum(self.models[name]['theta'].value)  ### Rough capacity estimate
            mean_abs_nl = np.mean(np.abs( self.netloads[name] ))    ### Mean absolute net load
            lb_var      = (mean_abs_nl * var_lb_fraction) ** 2                  ### Lower bound on variance
            sol_var     = ave_filt_sq_resid_norm * (capest ** 2)    ### Solar variance (unconstrained)
            sol_var[sol_var < lb_var ] = lb_var                     ### Constrain the solar variance
            total_sol_var = total_sol_var + sol_var                 ### Track the total variance of solar

            alpha = sol_var ** -1         ## alpha
            self.models[name]['alpha'] = alpha

        ## Tune load alphas.
        lb_var            = (np.mean(np.abs(self.aggregateSignal)) * var_lb_fraction)    ** 2 ## LOWER BOUND OF VARIANCE, 1%
        total_residual_sq = (self.aggregateSignal.squeeze() - total_model_est.squeeze()) ** 2 ## Square residuals of aggregate signal prediction
        total_var_filt    = convolve_cyc(total_residual_sq, filter_vec)   ## Filter square residuals as variance estimate
        load_var_est      = total_var_filt - total_sol_var                ## Estimate of load variance
        load_var_est[load_var_est < lb_var] = lb_var                      ## Enforce lower bound on variance
        alpha = load_var_est ** -1
        self.models['AggregateLoad']['alpha'] = alpha

        ## Scale all alphas
        self.scaleAlphas()
        self.updateSourceObj('all')
        return(None)

    def fitTuneModels(self, tuneSys = None, var_lb_fraction = 0.05, tuningRegressors = None):
        firsttune = True

        if tuneSys is None:
            ## If no name for a tuning system is input, use all systems for which
            # a truth is known.
            tuneSys = self.trueValues.keys()
            if 'AggregateLoad' in tuneSys: tuneSys.remove('AggregateLoad')

        ## Allow user to place new tuning regressors here.
        if tuningRegressors is not None:
            self.tuningRegressors = tuningRegressors


        ## Cycle through each tuning system and collect data for a model.
        for name in tuneSys:
            truth      = self.trueValues[name].squeeze()

            ## Run a quick regression to collect expected value of
            # the lienar model given the true solar signal.
            model       = LR()
            model.fit(X = self.models[name]['regressor'], y = truth)
            modelest    = model.predict(X = self.models[name]['regressor'])

            ## Create a rough capacity estimate from the theta values
            capest_tune     = np.sum(model.coef_)
            resid_norm = (truth - modelest) / capest_tune

            if firsttune:
                firsttune = False
                sq_resid_norm = resid_norm ** 2
                X = self.tuningRegressors
            else:
                sq_resid_norm  = np.concatenate([ sq_resid_norm , resid_norm ** 2] )
                X = np.concatenate([ X, self.tuningRegressors])


        # Build model to predict normalized variances
        self.Solar_var_norm = LR()
        self.Solar_var_norm.fit(y = (sq_resid_norm), X = X)

        # Set lower bound for each PV system now
        for name, m in self.models.items():
            ## use the aggregate signal for aggregate load
            if name.lower() == 'aggregateload':
                mean_abs_nl = np.mean(np.abs(self.aggregateSignal))
            else:
                mean_abs_nl = np.mean(np.abs(self.netloads[name]))

            m['var_lb'] = (mean_abs_nl * var_lb_fraction) ** 2      ### Lower bound on variance

        ## Build model to predict aggregate net load
        model = LR()
        X = np.hstack([self.loadRegressors, self.solarRegressors])
        model.fit(y = self.aggregateSignal, X = X)
        lin_est = model.predict(X = X)

        ## Collect square residuals and predict them
        total_sq_resid = (self.aggregateSignal - lin_est)**2
        self.Total_NL_var   = LR()
        self.Total_NL_var.fit(y = total_sq_resid, X = self.tuningRegressors)

    def tuneAlphas(self):
        # Instantiate vectors for the total solar variance and total estiamted net load by the model.
        total_sol_var   = np.zeros(self.N) ## Instantiate vector for total variance of PV signals,

        ## Cycle through each solar model and tune alphas
        pred_sq_resid = self.Solar_var_norm.predict(X = self.tuningRegressors)

        for name,m in self.models.items():
            ## Don't solve for aggregate load yet
            if name.lower() == 'aggregateload':
                continue

            capest      = np.sum(m['theta'].value)  ### Rough capacity estimate
            lb_var      = m['var_lb']                               ### Lower bound on variance
            sol_var     = pred_sq_resid * (capest ** 2)             ### Solar variance (unconstrained)
            sol_var[sol_var < lb_var ] = lb_var                     ### Constrain the solar variance
            total_sol_var = total_sol_var + sol_var                 ### Track the total variance of solar

            alpha = sol_var ** -1         ## alpha
            self.models[name]['alpha'] = alpha

        ## Tune load alphas.
        lb_var            = self.models['AggregateLoad']['var_lb'] ## LOWER BOUND OF VARIANCE, 1%
        total_var_filt    = self.Total_NL_var.predict(X = self.tuningRegressors) ## Use linear model to predict total variace
        load_var_est      = total_var_filt - total_sol_var                     ## Estimate of load variance
        load_var_est[load_var_est < lb_var] = lb_var                           ## Enforce lower bound on variance
        alpha = load_var_est ** -1
        self.models['AggregateLoad']['alpha'] = alpha

        ## Scale all alphas
        self.scaleAlphas()
        self.updateSourceObj('all')
        #return(None)

    def scaleAlphas(self, scale_to = 1.0):
        ## Find the maximum value of alpha
        alpha_max = 0
        for name, m in self.models.items():
            if np.max(m['alpha']) > alpha_max:
                alpha_max = np.max(m['alpha'])


        ## Scale other values of alpha
        for name, m in self.models.items():
            m['alpha'] = np.array( m['alpha'] / alpha_max * scale_to ).squeeze()
            self.updateSourceObj(name)



        return(None)
        
    def snr(self,df_true = None):
        '''
        calculate the SNR(signal to noise ratio)
        param df_true is the data frame with true solar, use and netload
        return: dataframe with index ids and column estimated and true snr
        '''
        SNR = pd.DataFrame(index = self.names+['AggregateLoad'], columns = ['snr_est','snr_true'])
        for idd in self.names:
            idd = int(idd)
            if df_true is not None:
                mask = df_true['gen'][idd] > 0.05*df_true['gen'][idd].mean()
                SNR.loc[str(idd),'snr_true'] = np.square(np.linalg.norm(df_true.gen[idd][mask],2)/np.linalg.norm(df_true.use[idd][mask],2))       
            solar_signal_est = -np.squeeze(np.array(self.models[str(idd)]['source'].value))
            mask = solar_signal_est > 0.05*solar_signal_est.mean()
            solar_signal_est = solar_signal_est[mask] 
            use_signal_est   = df_true['netload'][idd].values[mask] + solar_signal_est
            SNR.loc[str(idd),'snr_est'] = np.square(np.linalg.norm(solar_signal_est,2)/np.linalg.norm(use_signal_est,2))
        self.SNR = SNR
        SNR.fillna(0, inplace = True)
        return

class SolarDisagg_IndvHome_Realtime(CSSS.CSSS):
    def __init__(self, sdmod, aggregateNetLoad, solarregressors, loadregressors, tuningregressors = None):
        ## Inputs
        # netloads:         np.array of net loads at each home, with columns corresponding to entries of "names" if available.
        # solarregressors:  np.array of solar regressors (N_s X T)
        # loadregressors:   np.array of load regressors (N_l x T)
        CSSS.CSSS.__init__(self, aggregateNetLoad)

        self.N = len(aggregateNetLoad)
        self.M = sdmod.M

        ## If no tuning regressors are input, use an intercept only
        if tuningregressors is None:
            tuningregressors = np.ones((self.N,1))

        ## Store solar and load regressors, solar regressors, and begin true solar dict
        self.solarRegressors   = solarregressors
        self.loadRegressors    = loadregressors
        self.tuningRegressors  = tuningregressors
        self.trueValues        = {}

        ## Can I inherit methods?
        self.Solar_var_norm = sdmod.Solar_var_norm
        self.Total_NL_var   = sdmod.Total_NL_var

        ## Inherit properties from the fitted class
        self.names = sdmod.names

        ## Cycle through each net load, and create sources.
        for source_name in sdmod.names:
            self.addSource(regressor=solarregressors, name = source_name, alpha = 1)
            self.models[source_name]['theta'].value = sdmod.models[source_name]['theta'].value

            ## Assign Capacity estimates and cutoffs for tuning
            self.models[source_name]['var_lb']  = sdmod.models[source_name]['var_lb']

            ## Add constraints that solar generation cannot exceed zero or net load.
            self.addConstraint( self.models[source_name]['source'] <= 0 )

        ## Add the aggregate load source
        self.addSource(regressor=loadregressors, name = 'AggregateLoad', alpha = 1)
        self.addConstraint( self.models['AggregateLoad']['source'] > 0 )
        self.models['AggregateLoad']['theta'].value = sdmod.models['AggregateLoad']['theta'].value
        self.models['AggregateLoad']['var_lb']  = sdmod.models['AggregateLoad']['var_lb']
        ## FixThetas
        self.fixThetas()
        self.updateSourceObj('all')

        ## Copy all true trueValues

    def tuneAlphas(self):
        # Instantiate vectors for the total solar variance and total estiamted net load by the model.
        total_sol_var   = np.zeros(self.N) ## Instantiate vector for total variance of PV signals,

        ## Cycle through each solar model and tune alphas
        pred_sq_resid = self.Solar_var_norm.predict(X = self.tuningRegressors)

        for name,m in self.models.items():
            ## Don't solve for aggregate load yet
            if name.lower() == 'aggregateload':
                continue


            capest      = np.sum(m['theta']      )                  ### Rough capacity estimate
            lb_var      = m['var_lb']                               ### Lower bound on variance
            sol_var     = pred_sq_resid * (capest ** 2)             ### Solar variance (unconstrained)
            sol_var[sol_var < lb_var ] = lb_var                     ### Constrain the solar variance
            total_sol_var = total_sol_var + sol_var                 ### Track the total variance of solar

            alpha = sol_var ** -1         ## alpha
            self.models[name]['alpha'] = alpha

        ## Tune load alphas.
        lb_var            = self.models['AggregateLoad']['var_lb'] ## LOWER BOUND OF VARIANCE, 1%
        total_var_filt    = self.Total_NL_var.predict(X = self.tuningRegressors) ## Use linear model to predict total variace
        load_var_est      = total_var_filt - total_sol_var                     ## Estimate of load variance
        load_var_est[load_var_est < lb_var] = lb_var                           ## Enforce lower bound on variance
        alpha = load_var_est ** -1
        self.models['AggregateLoad']['alpha'] = alpha

        ## Scale all alphas
        self.scaleAlphas()
        self.updateSourceObj('all')
        #return(None)

    def scaleAlphas(self, scale_to = 1.0):
        ## Find the maximum value of alpha
        alpha_max = 0
        for name, m in self.models.items():
            if np.max(m['alpha']) > alpha_max:
                alpha_max = np.max(m['alpha'])


        ## Scale other values of alpha
        for name, m in self.models.items():
            m['alpha'] = np.array( m['alpha'] / alpha_max * scale_to ).squeeze()
            self.updateSourceObj(name)



        return(None)

    def addTrueValue(self, trueValue, name):
        ## Function to add true solar for a given model

        ## Check that true value is correct number of dimensions
        trueValue = trueValue.squeeze()
        if not (trueValue.shape == (self.N,)):
            raise Exception('True value of a solar or load signal must be one dimensional and length N = %d' % self.N)

        if name not in (self.names + ['AggregateLoad']):
            raise Exception('Must input a valid household identifier or \"AggregateLoad\"')

        ## Add True Value
        self.trueValues[name] = trueValue
        return(None)

    def calcPerformanceMetrics(self, dropzeros = False, MAPE = False):
        ## Function to calculate performance metrics
        # Dropping zeros is intended to remove nightime solar.

        df = pd.DataFrame()
        df['models'] = self.models.keys()
        df['rmse']   = np.zeros(df.shape[0]) * np.nan
        df['cv']     = np.zeros(df.shape[0]) * np.nan
        df['mae']    = np.zeros(df.shape[0]) * np.nan
        df['pmae']   = np.zeros(df.shape[0]) * np.nan
        df['mbe']    = np.zeros(df.shape[0]) * np.nan
        df['mean']   = np.zeros(df.shape[0]) * np.nan
        
        df['cv_pos']     = np.zeros(df.shape[0]) * np.nan
        df['rmse_pos']   = np.zeros(df.shape[0]) * np.nan
        df['mae_pos']    = np.zeros(df.shape[0]) * np.nan
        df['pmae_pos']   = np.zeros(df.shape[0]) * np.nan
        df['mbe_pos']    = np.zeros(df.shape[0]) * np.nan
        df['mean_pos']   = np.zeros(df.shape[0]) * np.nan
        
        df               = df.set_index('models')

        for name in self.trueValues.keys():
            truth   = self.trueValues[name]
            est     = np.array(self.models[name]['source'].value).squeeze()

            ## Calculate metrics.
            df.loc[name,'mbe']  = np.mean((truth-est))
            df.loc[name,'mean'] = np.mean((truth))
            df.loc[name,'rmse'] = np.sqrt(np.mean((truth-est)**2))
            df.loc[name,'mae']  = np.mean(np.abs((truth-est)))
            if not (df.loc[name,'mean'] == 0):
                df.loc[name,'cv']   = df.loc[name,'rmse'] / np.mean(truth)
                df.loc[name,'pmae'] = df.loc[name,'mae']  / np.mean(truth)

                ## Find metrics for  positive indices only
                posinds = np.abs(truth) > (0.05 * np.abs(np.mean(truth)))
                truth = truth[posinds]
                est   = est[posinds]
                df.loc[name,'rmse_pos'] = np.sqrt(np.mean((truth-est)**2))
                df.loc[name,'mae_pos']  = np.mean(np.abs((truth-est)))
                df.loc[name,'cv_pos']   = df.loc[name,'rmse_pos'] / np.mean(truth)
                df.loc[name,'pmae_pos'] = df.loc[name,'mae_pos']  / np.mean(truth)
                df.loc[name,'mbe_pos']  = np.mean((truth-est))
                df.loc[name,'mean_pos'] = np.mean((truth))

        self.performanceMetrics = df

        return(None)


def createTempInput(temp, size, minTemp=None, maxTemp=None, intercept=False):
    if (minTemp is None):
        minTemp = min(temp)
    if maxTemp is None:
        maxTemp = max(temp)
    minBound = int(np.floor(minTemp / size)) * size
    maxBound = int(np.floor(maxTemp / size)) * size + size

    rangeCount = int((maxBound - minBound) / size)
    result = np.zeros((len(temp), rangeCount + intercept))
    t = 0
    for elem in temp:
        fullRanges = min(int(np.floor((elem - minBound) / size)), rangeCount - 1)
        fullRanges = max(0, fullRanges)
        bound = (minBound + fullRanges * size)
        lastRange = elem - bound
        res = [size for elem in range(fullRanges)]
        res.append(lastRange)
        for var in range(rangeCount - fullRanges - 1):
            res.append(0)
        if intercept:
            res.append(1)  ## Include an intercept

        result[t, :] = np.array(res)
        t += 1
    return minTemp, maxTemp, result


def createSolarDisaggIndvInputs(data, home_ids, solar_proxy_ids, time_index = None):
    """
    Helper function to create inputs to SolarDisagg_IndvHome from a time-series csv file. This function works with the
    data file generation script, tutorial_data_setup.py.

    :param data:                A path to the csv file (probably the one created by tutorial_data_setpu.py)
    :param solar_proxy_ids:     The site ID numbers associated with the solar proxy signals
    :param includes_agg_col:    If true, the csv contains an aggregate net load column
    :param time_index:          List with start and end or index type
    :return:                    A dictionary mapping to the constructor for SolarDisagg_IndvHome
    """
    try:
        df = pd.read_csv(data, index_col=0, header=[0, 1], parse_dates=[0])
    except ValueError:
        if isinstance(data, pd.DataFrame):
            df = copy.deepcopy(data)
        else:
            print('Please input a path to a csv file or a valid Pandas DataFrame')
            return

    if time_index:
        df = df[(df.index >= time_index[0]) & (df.index <= time_index[1])]

    if isinstance(solar_proxy_ids, str):
        solar_proxy_ids = np.load(solar_proxy_ids)

    netloads = df['netload'][home_ids].values
    try:
        solarproxy = df['gen'][solar_proxy_ids].values
    except KeyError:
        solarproxy = df['gen'][list(map(str, solar_proxy_ids))].values
    if np.average(solarproxy) >= 0:
        solarproxy *= -1                                            # generation should be negative

    hod = df.index.hour.values
    hod = np.array(pd.get_dummies(hod))
    Tmin, Tmax, temp_regress = createTempInput(df['temperature'].values.squeeze(), 10)

    data_out = {
        'netloads':         netloads,
        'solarregressors':  solarproxy,
        'loadregressors':   np.hstack((np.ones((len(hod),1)),hod, temp_regress)),
        'tuningregressors': hod,
        'names':            home_ids
    }
    
#   changes the names of the models to strings becuase the other functions do not like numbers as names
#    if not isinstance(home_ids[0], str):
    data_out['names'] = list(map(str,home_ids))

    return data_out


## Function for a cyclic convolution filter, because apparantely there isn't one already in python
def convolve_cyc(x, filt, left = True):
    if (len(filt) % 2) == 1:
        pad_l = np.int((len(filt)-1)/2)
        pad_r = pad_l
    elif left:
        pad_l = np.int(len(filt)/2)
        pad_r = pad_l-1
    else:
        pad_r = np.int(len(filt)/2)
        pad_l = pad_r-1


    x = np.concatenate([x[-pad_l:], x, x[:pad_r]])
    x = np.convolve(x, filt, mode = 'valid')
    return(x)

def MAPE_mod(predicted,actual,thrs = None):
    # 0<thrs<1 : thrs consider there is no error if the absolute difference predicted Vs actual <= than thrs*abs(max(actual))
    
    if thrs:
        posinds = np.abs(actual-predicted) >= thrs*np.max(np.abs(actual))
        predicted_m = predicted[posinds] 
        actual_m = actual[posinds]
        error = np.sum(np.abs(actual_m-predicted_m)/np.abs(actual_m))/len(actual)
    else:
        error = np.mean(np.abs(actual-predicted)/np.abs(actual))
    
    return error

class Solar_Disagg_linear_model(object):
    """
    This class performs the solar disaggregation using the coefficients attributed to the solar model
    Fit a linear model to predict the netload at the individual house and then use only the coefficient associated with the solarregressor.
    """
    
    def __init__(self,netloads,loadregressors, solarregressors, names):
        
        self.netloads        = netloads
        self.loadregressors  = loadregressors
        self.solarregressors = solarregressors
        self.names           = names
        self.Nproxies        = solarregressors.shape[1]
        
    def fit(self):
        """
          X:regression matrix
          y:true values
        """
        
        X = copy.deepcopy(np.hstack((self.loadregressors,self.solarregressors)))
        y = copy.deepcopy(self.netloads)
        
        LSE = copy.deepcopy(LR())
        LSE.fit(X,y)
        self.model = LSE
        
    def predict(self):
        """
          X:regression matrix
        """
        yp = np.matmul(self.solarregressors,(self.model.coef_[:,-self.Nproxies:]).T)
        yp = pd.DataFrame(yp,columns = self.names)
        yp[yp.columns[yp.mean()>0]] = 0 ## Set to zero the generation at those houses where it is attributed positive. Here the input proxy has negative generation. If the generation is predicted positive, the model is finding correlations with proxy data at non solar houses, therefore I set it to zero.
        yp['AggregateLoad'] = copy.deepcopy(yp.sum(axis = 1))
        self.yp = yp
        return self.yp
    
    def calcPerformanceMetrics(self,df_true):
        ## Function to calculate performance metrics
        # Dropping zeros is intended to remove nightime solar.

        df = pd.DataFrame()
        df['models'] = self.names + ['AggregateLoad']
        df['rmse']   = np.zeros(df.shape[0]) * np.nan
        df['cv']     = np.zeros(df.shape[0]) * np.nan
        df['mae']    = np.zeros(df.shape[0]) * np.nan
        df['pmae']   = np.zeros(df.shape[0]) * np.nan
        df['mbe']    = np.zeros(df.shape[0]) * np.nan
        df['mean']   = np.zeros(df.shape[0]) * np.nan
        df['MAPE']   = np.zeros(df.shape[0]) * np.nan
        df['mae_max']= np.zeros(df.shape[0]) * np.nan
        df['cv_max'] = np.zeros(df.shape[0]) * np.nan
        df['max_sol_pred'] = np.zeros(df.shape[0]) * np.nan
        
        df['cv_pos']     = np.zeros(df.shape[0]) * np.nan
        df['rmse_pos']   = np.zeros(df.shape[0]) * np.nan
        df['mae_pos']    = np.zeros(df.shape[0]) * np.nan
        df['pmae_pos']   = np.zeros(df.shape[0]) * np.nan
        df['mbe_pos']    = np.zeros(df.shape[0]) * np.nan
        df['mean_pos']   = np.zeros(df.shape[0]) * np.nan
        df['MAPE_pos']   = np.zeros(df.shape[0]) * np.nan
        df['mae_pos_max']= np.zeros(df.shape[0]) * np.nan
        df['cv_pos_max'] = np.zeros(df.shape[0]) * np.nan
        df               = df.set_index('models')

        for name in self.names+['AggregateLoad']:
            if name == 'AggregateLoad':
                truth = df_true.loc[:,list(map(int,self.names))].sum(axis = 1).values
                est   = self.yp[name].values
            else:
                truth   = df_true.loc[:,int(name)].values
                est     = self.yp[name].values

            ## Calculate metrics.
            df.loc[name,'mbe']  = np.mean((truth-est))
            df.loc[name,'mean'] = np.mean((truth))
            df.loc[name,'rmse'] = np.sqrt(np.mean((truth-est)**2))
            df.loc[name,'mae']  = np.mean(np.abs((truth-est)))
            df.loc[name,'MAPE'] = MAPE_mod(est,truth,thrs=0.001)
            df.loc[name,'max_sol_pred'] = np.max(np.abs(est))
            try:
                df.loc[name,'rmse_pos'] = rmse_pos(est,truth)
            except:
                df.loc[name,'rmse_pos'] = np.nan
        
            if not (df.loc[name,'mean'] == 0):
                df.loc[name,'cv']   = df.loc[name,'rmse'] / np.mean(truth)
                df.loc[name,'pmae'] = df.loc[name,'mae']  / np.mean(truth)
                df.loc[name,'mae_max']  =  df.loc[name,'mae']/ np.max(np.abs(truth))
                df.loc[name,'cv_max'] = df.loc[name,'rmse'] / np.max(np.abs(truth))
                
                ## Find metrics for  positive indices only
                posinds = np.abs(truth) > (0.05 * np.abs(np.mean(truth)))
                truth = truth[posinds]
                est   = est[posinds]
#                df.loc[name,'rmse_pos'] = np.sqrt(np.mean((truth-est)**2))
                df.loc[name,'mae_pos']  = np.mean(np.abs((truth-est)))
                df.loc[name,'cv_pos']   = df.loc[name,'rmse_pos'] / np.mean(truth)
                df.loc[name,'pmae_pos'] = df.loc[name,'mae_pos']  / np.mean(truth)
                df.loc[name,'mbe_pos']  = np.mean((truth-est))
                df.loc[name,'mean_pos'] = np.mean((truth))
                df.loc[name,'MAPE_pos'] = MAPE_mod(est,truth)
                df.loc[name,'mae_pos_max']  = df.loc[name,'mae_pos'] / np.max(np.abs(truth))
                df.loc[name,'cv_pos_max']   = df.loc[name,'rmse_pos'] / np.max(np.abs(truth))
                
                
        self.performanceMetrics = df

        return(None)
        
    def snr(self,df_true = None):
        '''
        calculate the SNR(signal to noise ratio)
        param df_true is the data frame with true solar, use and netload
        return: dataframe with index ids and column estimated and true snr
        '''
        SNR = pd.DataFrame(index = self.names+['AggregateLoad'], columns = ['snr_est','snr_true'])
        for idd in self.names:
            idd = int(idd)
            if df_true is not None:
                mask = df_true['gen'][idd] > 0.05*df_true['gen'][idd].mean()
                SNR.loc[str(idd),'snr_true'] = np.square(np.linalg.norm(df_true.gen[idd][mask],2)/np.linalg.norm(df_true.use[idd][mask],2))       
            solar_signal_est = -self.yp[str(idd)].values.squeeze()
            mask = solar_signal_est > 0.05*solar_signal_est.mean()
            solar_signal_est = solar_signal_est[mask] 
            use_signal_est   = df_true['netload'][idd].values[mask] + solar_signal_est
            SNR.loc[str(idd),'snr_est'] = np.square(np.linalg.norm(solar_signal_est,2)/np.linalg.norm(use_signal_est,2))
            SNR.fillna(0, inplace = True)
        self.SNR = SNR
        return
    


