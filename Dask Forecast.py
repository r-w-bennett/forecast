import pandas as pd
from prophet import Prophet
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, STATUS_FAIL
from hyperopt.exceptions import AllTrialsFailed
import numpy as np
from multiprocessing import Value, Lock
from dask import delayed, compute
import logging
import warnings
import os
import sys
import time
import traceback

start_time = time.time()

##Logging##

# Disable all logging messages
logging.disable(logging.CRITICAL)

# Disable all warning messages
warnings.filterwarnings("ignore")

##End Logging##


## Load Data

# Define a function to safely read Excel sheets
def safe_read_excel(file_path, sheet_name):
    try:
        return pd.read_excel(file_path, sheet_name=sheet_name)
    except FileNotFoundError:
        logging.warning(f"File {file_path} not found.")
        sys.exit(1)
    except Exception as e:
        logging.warning(f"An error occurred while reading {sheet_name}: {e}")
        sys.exit(1)

# Define the base directory and file name
base_directory = 'C:\\PythonLocal'
file_name = 'Train and Test Data.xlsx'

# Create full file path
full_file_path = os.path.join(base_directory, file_name)

# Load data using the safe_read_excel function
train_data = safe_read_excel(full_file_path, 'Train Data')
test_data = safe_read_excel(full_file_path, 'Test Data')
forecast_template = safe_read_excel(full_file_path, 'Forecast Template Table')

## End Load Data


## Preprocess Data

# Generate a list of unique item codes
history_all = pd.concat([train_data, test_data], ignore_index=True)
test_unique_item_codes = history_all['Item Code'].unique().tolist()

# Create lists to store excluded and included item codes
excluded_item_codes = []
unique_item_codes = []

for code in test_unique_item_codes:
    if len(train_data[train_data['Item Code'] == code]) < 4:
        excluded_item_codes.append(code)
    else:
        unique_item_codes.append(code)
    
# SARIMAX Preprocessing

# Train
train_sarimax = train_data.copy()
train_sarimax.set_index('Date', inplace=True)
train_sarimax['Item Code'] = train_sarimax['Item Code'].astype(np.int64)
train_sarimax['WeekOfYear'] = train_sarimax.index.isocalendar().week
train_sarimax['MonthOfYear'] = train_sarimax.index.month
train_sarimax['WeekOfYear'] = train_sarimax['WeekOfYear'].astype(np.int64)
train_sarimax['MonthOfYear'] = train_sarimax['MonthOfYear'].astype(np.int64)
train_sarimax['Weight'] = train_sarimax['Weight'].astype('float64')

# Test
test_sarimax = test_data.copy()
test_sarimax.set_index('Date', inplace=True)
test_sarimax['Item Code'] = test_sarimax['Item Code'].astype(np.int64)
test_sarimax['WeekOfYear'] = test_sarimax.index.isocalendar().week
test_sarimax['MonthOfYear'] = test_sarimax.index.month
test_sarimax['WeekOfYear'] = test_sarimax['WeekOfYear'].astype(np.int64)
test_sarimax['MonthOfYear'] = test_sarimax['MonthOfYear'].astype(np.int64)
test_sarimax['Weight'] = test_sarimax['Weight'].astype('float64')

# Historical Data
history_sarimax = pd.concat([train_sarimax, test_sarimax], ignore_index=True)

# Template
forecast_template_sarimax = forecast_template.copy()
forecast_template_sarimax.set_index('Date', inplace=True)
forecast_template_sarimax['Item Code'] = forecast_template_sarimax['Item Code'].astype(np.int64)
forecast_template_sarimax['WeekOfYear'] = forecast_template_sarimax.index.isocalendar().week
forecast_template_sarimax['MonthOfYear'] = forecast_template_sarimax.index.month
forecast_template_sarimax['WeekOfYear'] = forecast_template_sarimax['WeekOfYear'].astype(np.int64)
forecast_template_sarimax['MonthOfYear'] = forecast_template_sarimax['MonthOfYear'].astype(np.int64)

#Prophet Preprocessing

# Train
train_prophet = train_data.copy()
train_prophet['Item Code'] = train_prophet['Item Code'].astype(np.int64)
train_prophet.rename(columns={'Date': 'ds', 'Weight': 'y'}, inplace=True)
train_prophet['ds'] = pd.to_datetime(train_prophet['ds'])

# Test
test_prophet = test_data.copy()
test_prophet['Item Code'] = test_prophet['Item Code'].astype(np.int64)
test_prophet.rename(columns={'Date': 'ds', 'Weight': 'y'}, inplace=True)
test_prophet['ds'] = pd.to_datetime(test_prophet['ds'])

# Template
forecast_template_prophet = forecast_template.copy()
forecast_template_prophet['Item Code'] = forecast_template_prophet['Item Code'].astype(np.int64)
forecast_template_prophet.rename(columns={'Date': 'ds'}, inplace=True)
forecast_template_prophet['ds'] = pd.to_datetime(forecast_template_prophet['ds'])

# Historical Data
history_prophet = pd.concat([train_prophet, test_prophet], ignore_index=True)

# ARIMA Preprocessing

# Train
train_arima = train_data.copy()
train_arima.set_index('Date', inplace=True)
train_arima['Item Code'] = train_arima['Item Code'].astype(np.int64)
train_arima['Weight'] = train_arima['Weight'].astype('float64')

# Test
test_arima = test_data.copy()
test_arima.set_index('Date', inplace=True)
test_arima['Item Code'] = test_arima['Item Code'].astype(np.int64)
test_arima['Weight'] = test_arima['Weight'].astype('float64')

# Historical Data
history_arima = pd.concat([train_arima, test_arima], ignore_index=True)

# Template
forecast_template_arima = forecast_template.copy()
forecast_template_arima.set_index('Date', inplace=True)
forecast_template_arima['Item Code'] = forecast_template_arima['Item Code'].astype(np.int64)

##End Preprocess Data


## Define model search space

# SARIMAX
# Define individual hyperparameter choices
p_values = [0, 1, 2, 3, 4, 5]
d_values = [0, 1, 2]
q_values = [0, 1, 2, 3, 4, 5]
P_values = [0, 1]
D_values = [0, 1]
Q_values = [0, 1]
s_values = [0, 52]
trend_values = ['n', 'c', 't', 'ct']

# Generate all combinations
all_combinations = []
for p in p_values:
    for d in d_values:
        for q in q_values:
            for P in P_values:
                for D in D_values:
                    for Q in Q_values:
                        for s in s_values:
                            for trend in trend_values:
                                # Exclude incompatible combinations where s=0 and P, D, Q are not zero
                                if s == 0 and (P != 0 or D != 0 or Q != 0):
                                    continue
                                all_combinations.append({
                                    'p': p,
                                    'd': d,
                                    'q': q,
                                    'P': P,
                                    'D': D,
                                    'Q': Q,
                                    's': s,
                                    'trend': trend
                                })

# Now `all_combinations` contains only compatible sets of hyperparameters
# Convert it to a hyperopt-compatible format
sarimax_space = hp.choice('params', all_combinations)

# Prophet
prophet_space = {
    'seasonality_mode': hp.choice('seasonality_mode', ['additive', 'multiplicative']),
    'changepoint_prior_scale': hp.uniform('changepoint_prior_scale', 0.001, 0.5),
    'seasonality_prior_scale': hp.uniform('seasonality_prior_scale', 0.01, 10)
}

# ARIMA
# Define individual hyperparameter choices
p_values = [0, 1, 2, 3, 4]
d_values = [0, 1, 2]
q_values = [0, 1, 2, 3, 4]
trend_values = ['n', 'c', 't', 'ct']

# Generate all combinations
arima_combinations = []
for p in p_values:
    for d in d_values:
        for q in q_values:
            for trend in trend_values:
                # Exclude incompatible combinations
                if (d > 0 and trend == 'c') or (d > 1 and (trend == 'c' or trend == 't')):
                    continue
                arima_combinations.append({
                    'p': p,
                    'd': d,
                    'q': q,
                    'trend': trend
                })
                
# Now `arima_combinations` contains only compatible sets of hyperparameters
# Convert it to a hyperopt-compatible format
arima_space = hp.choice('params', arima_combinations)

## End Define model search space


##Dataframe Preparations

# Create empty DataFrames to store results
results_df = pd.DataFrame(columns=['Item Code', 'BestModel', 'BestParams', 'BestLoss'])
excluded_df = pd.DataFrame(excluded_item_codes, columns=['Item Code'])
model_params_df = pd.DataFrame()
final_forecast_df = pd.DataFrame()

# Placeholder values for excluded codes
excluded_df['BestModel'] = 'no model'
excluded_df['BestParams'] = None
excluded_df['BestLoss'] = float('inf')

##End Dataframe Preparations


##Model Objective Functions

# Define WMAPE calculation function (Used for establishing baseline values)
def calculate_wmape(true_values, forecast_values):
    sum_actuals = sum(abs(x) for x in true_values)
    if sum_actuals == 0:
        if sum(abs(x) for x in forecast_values) == 0:
            return 0.0
        else:
            return float('inf')
    return 100 * sum(abs(true_value - forecast_value) for true_value, forecast_value in zip(true_values, forecast_values)) / sum_actuals

# Define SARIMAX objective function
def objective_sarimax(params, train_sarimax_filtered, test_sarimax_filtered):
    try:
        exog_train = train_sarimax_filtered[['WeekOfYear', 'MonthOfYear']]
        exog_test = test_sarimax_filtered[['WeekOfYear', 'MonthOfYear']]
        
        model = SARIMAX(train_sarimax_filtered['Weight'], 
                        exog=exog_train,
                        order=(params['p'], params['d'], params['q']),
                        seasonal_order=(params['P'], params['D'], params['Q'], params['s']),
                        trend=params['trend'],
                        enforce_stationarity=False,
                        enforce_invertibility=False)
        
        model_fit = model.fit(disp=False)      
        pred = model_fit.forecast(steps=len(test_sarimax_filtered), exog=exog_test)
        actual = test_sarimax_filtered['Weight']
        wmape = (sum(abs(actual - pred)) / sum(actual)) * 100
        
        return {'loss': wmape, 'status': 'ok'}
    except Exception as e:
        return {'loss': float('inf'), 'status': 'fail'}

# Define Prophet objective function
def objective_prophet(params, train_prophet_filtered, test_prophet_filtered):
    try:
        non_nan_rows = train_prophet_filtered['y'].dropna().shape[0]
        if non_nan_rows < 2:
            return {'loss': float('inf'), 'status': 'fail'}

        model = Prophet(daily_seasonality=False,
                        weekly_seasonality=True,
                        seasonality_mode=params['seasonality_mode'], 
                        changepoint_prior_scale=params['changepoint_prior_scale'], 
                        seasonality_prior_scale=params['seasonality_prior_scale'])
        model.fit(train_prophet_filtered)
        last_date_train = train_prophet_filtered['ds'].max()
        future_dates = pd.date_range(start=last_date_train + pd.Timedelta(days=7), periods=len(test_prophet_filtered), freq='W-MON')
        future = pd.DataFrame({'ds': future_dates})
        forecast = model.predict(future)
        actual = test_prophet_filtered['y']
        pred = forecast['yhat'][-len(test_prophet_filtered):]
        wmape = (sum(abs(actual - pred)) / sum(actual)) * 100

        return {'loss': wmape, 'status': 'ok'}
    except Exception as e:
        return {'loss': float('inf'), 'status': 'fail'}

# Define ARIMA objective function
def objective_arima(params, train_arima_filtered_weight, test_arima_filtered_weight):
    try:
        model = ARIMA(train_arima_filtered_weight, 
                      order=(params['p'], params['d'], params['q']),
                      trend=params['trend'])
        model_fit = model.fit()
        pred = model_fit.forecast(steps=len(test_arima_filtered_weight))
        actual = test_arima_filtered_weight
        wmape = (sum(abs(actual - pred)) / sum(actual)) * 100

        return {'loss': wmape, 'status': 'ok'}
    except Exception as e:
        return {'loss': float('inf'), 'status': 'fail'}

# Define variable to store results from worker processes
results_list = []

# Define a counter for Loop
counter = Value('i', 0)
counter_lock = Lock()

# This should be defined before the main block
def update_counter(new_row):
    global results_list
    results_list.append(new_row)
    with counter_lock:
        try:
            counter.value += 1
            elapsed_time = time.time() - start_time
            time_per_item_code = elapsed_time / counter.value
            remaining_item_codes = len(random_item_codes) - counter.value
            estimated_time_remaining = time_per_item_code * remaining_item_codes

            # Convert estimated time from seconds to a more readable format
            m, s = divmod(estimated_time_remaining, 60)
            h, m = divmod(m, 60)

            print(f"Optimization completed for {counter.value} of {len(random_item_codes)} item codes. Estimated time remaining: {int(h)}:{int(m):02d}:{int(s):02d}")
        except Exception as e:
            print(f"An exception occurred while updating the counter: {type(e).__name__}, {str(e)}")
    pass

# Define function for loop multiprocessing
def optimize_item_code(item_code):
        
    print(f"Starting optimization for Item Code: {item_code}")

    best_loss = float('inf')
    best_model = 'no model'
    best_params = None
    all_failed = True

    # Filter data by item code
    train_sarimax_filtered = train_sarimax[train_sarimax['Item Code'] == item_code].asfreq('W-MON')
    test_sarimax_filtered = test_sarimax[test_sarimax['Item Code'] == item_code].asfreq('W-MON')
    train_prophet_filtered = train_prophet[train_prophet['Item Code'] == item_code].drop(columns=['Item Code']).reset_index(drop=True)
    test_prophet_filtered = test_prophet[test_prophet['Item Code'] == item_code].drop(columns=['Item Code']).reset_index(drop=True)
    train_prophet_filtered['ds'] = pd.to_datetime(train_prophet_filtered['ds'])
    train_prophet_filtered['y'] = train_prophet_filtered['y'].astype('float64')
    test_prophet_filtered['ds'] = pd.to_datetime(test_prophet_filtered['ds'])
    test_prophet_filtered['y'] = test_prophet_filtered['y'].astype('float64')
    train_arima_filtered = train_arima[train_arima['Item Code'] == item_code].asfreq('W-MON')
    test_arima_filtered = test_arima[test_arima['Item Code'] == item_code].asfreq('W-MON')
    train_arima_filtered_weight = train_arima_filtered['Weight'].asfreq('W-MON')
    test_arima_filtered_weight = test_arima_filtered['Weight'].asfreq('W-MON')
    train_baseline_filtered = train_data[train_data['Item Code'] == item_code]
    test_baseline_filtered = test_data[test_data['Item Code'] == item_code]

    # Calculate Naive Forecast WMAPE
    naive_forecast = [train_baseline_filtered['Weight'].iloc[-1]] * len(test_baseline_filtered)
    naive_wmape = calculate_wmape(test_baseline_filtered['Weight'], naive_forecast)

    if naive_wmape < best_loss:
        best_loss = naive_wmape
        best_model = 'Naive'
        best_params = 'Standard'
        all_failed = False

    # Calculate Average of Last 4 Weeks Forecast WMAPE
    avg_last_4_weeks = sum(train_baseline_filtered['Weight'].iloc[-4:]) / 4
    average_forecast = [avg_last_4_weeks] * len(test_baseline_filtered)
    average_wmape = calculate_wmape(test_baseline_filtered['Weight'], average_forecast)

    if average_wmape < best_loss:
        best_loss = average_wmape
        best_model = 'Average'
        best_params = 'Standard'
        all_failed = False
    
    try:
        # SARIMAX optimization
        recent_data = train_sarimax_filtered.tail(13)
        non_zero_count = (recent_data['Weight'] > 0).sum()
        check_sarimax = len(train_sarimax_filtered) >= 104 and non_zero_count >= 9

        successful_evals = 0
        total_evals = 0
        early_stop_triggered = False
        trials = Trials()
        setattr(trials, 'best_loss_so_far', float('inf'))
        setattr(trials, 'no_improvement_count', 0)

        def early_stopping_fn(trials, *args):
            if len(trials.trials) == 0:
                return False, args

            successful_evals = sum([1 for trial in trials.trials if trial['result']['status'] == 'ok'])

            print(f"Number of successful evaluations: {successful_evals}")

            if successful_evals >= 15:
                sorted_trials = sorted(trials.trials, key=lambda x: x['result'].get('loss', float('inf')))
                current_best_loss = sorted_trials[0]['result'].get('loss', float('inf'))

                if current_best_loss < trials.best_loss_so_far:
                    trials.best_loss_so_far = current_best_loss
                    trials.no_improvement_count = 0
                else:
                    trials.no_improvement_count += 1

                print(f"Checking early stopping: best_loss_so_far = {trials.best_loss_so_far}, no_improvement_count = {trials.no_improvement_count}")

                if trials.no_improvement_count >= 10:
                    print(f"Early stopping activated.")
                    return True, args

            return False, args
                        
        if not check_sarimax:
            print(f"Skipping SARIMAX for Item Code: {item_code} due to insufficient data.")
        
        else:
            while successful_evals < 30 and total_evals < 300 and not early_stop_triggered:
                try:
                    best_sarimax = fmin(
                        fn=lambda params: objective_sarimax(params, train_sarimax_filtered, test_sarimax_filtered),
                        space=sarimax_space, algo=tpe.suggest, trials=trials, max_evals=total_evals + 1,
                        early_stop_fn=early_stopping_fn
                    )
                    
                    best_params_sarimax = all_combinations[best_sarimax['params']]
                    
                    # Check for early stopping
                    if early_stopping_fn(trials)[0]:
                        early_stop_triggered = True
                    
                except AllTrialsFailed:
                    print(f"All trials for SARIMAX failed for Item Code: {item_code}")
                    break

                total_evals += 1
                successful_evals = sum(1 for trial in trials.trials if trial['result']['status'] == STATUS_OK)
            
            if trials.best_trial['result']['status'] == STATUS_OK:
                all_failed = False
                if trials.best_trial['result']['loss'] < best_loss:
                    best_loss = trials.best_trial['result']['loss']
                    best_model = 'SARIMAX'
                    best_params = best_params_sarimax
                    
    except Exception as e:
        print(f"Exception in SARIMAX: {e}")
        traceback.print_exc()

    try:
        # Prophet optimization

        # Initialize counts and trials
        successful_evals = 0
        total_evals = 0
        trials = Trials()

        while successful_evals < 10 and total_evals < 100:
            try:
                best_prophet = fmin(fn=lambda params: objective_prophet(params, train_prophet_filtered, test_prophet_filtered),
                                    space=prophet_space, algo=tpe.suggest, trials=trials, max_evals=total_evals+1)
            except AllTrialsFailed:
                print(f"All trials for Prophet failed for Item Code: {item_code}")
                break
            total_evals +=1

            # Count successful trials
            successful_evals = sum(1 for trial in trials.trials if trial['result']['status'] == STATUS_OK)
            seasonality_conversion = ['additive', 'multiplicative']
            best_prophet['seasonality_mode'] = seasonality_conversion[best_prophet['seasonality_mode']]

        if trials.best_trial['result']['status'] == STATUS_OK:
            all_failed = False
            if trials.best_trial['result']['loss'] < best_loss:
                best_loss = trials.best_trial['result']['loss']
                best_model = 'Prophet'
                best_params = best_prophet
    except Exception as e:
        print(f"Exception in Prophet: {e}")

    try:
        # ARIMA optimization

        # Initialize counts and trials
        successful_evals = 0
        total_evals = 0
        
        trials = Trials()

        while successful_evals < 20 and total_evals < 200:
            try:
                best_arima = fmin(fn=lambda params: objective_arima(params, train_arima_filtered_weight, test_arima_filtered_weight),
                                space=arima_space, algo=tpe.suggest, trials=trials, max_evals=total_evals+1)
                
                best_params_arima = arima_combinations[best_arima['params']]
            except AllTrialsFailed:
                print(f"All trials for ARIMA failed for Item Code: {item_code}")
                break
            total_evals +=1

            # Count successful trials
            successful_evals = sum(1 for trial in trials.trials if trial['result']['status'] == STATUS_OK)

        if trials.best_trial['result']['status'] == STATUS_OK:
            all_failed = False
            if trials.best_trial['result']['loss'] < best_loss:
                best_loss = trials.best_trial['result']['loss']
                best_model = 'ARIMA'
                best_params = best_params_arima
    except Exception as e:
        print(f"Exception in ARIMA: {e}")

    if all_failed:
        new_row = pd.DataFrame({
            'Item Code': [item_code],
            'BestModel': ['no model'],
            'BestParams': [None],
            'BestLoss': ['undefined']
        })
    else:
        new_row = pd.DataFrame({
            'Item Code': [item_code],
            'BestModel': [best_model],
            'BestParams': [best_params],
            'BestLoss': [best_loss]
        })
    
    # Update the shared counter
    with counter.get_lock():
        counter.value += 1

    print(f"Completed optimization for Item Code: {item_code}")
    return new_row
pass

##End Model Objective Functions##


##Multiprocessing##

# Get the number of available CPU cores
num_cores = os.cpu_count()

if __name__ == '__main__':
    random_item_codes = unique_item_codes
    print(f"Random item codes: {random_item_codes}")

    delayed_results = []

    for item_code in random_item_codes:
        result = delayed(optimize_item_code)(item_code)
        delayed_results.append(result)

    results_list = compute(*delayed_results)

    # Apply the callback function to each result
    for result in results_list:
        update_counter(result)

    if results_list:
        results_df = pd.concat(results_list, ignore_index=True)
        # Concatenate results_df with excluded_df
        final_results_df = pd.concat([results_df, excluded_df], ignore_index=True)
        print(final_results_df)
        model_params_df = final_results_df.copy()
    else:
        print("No DataFrames to concatenate.")
        
##End Multiprocessing##


##Forecasting Objective Functions##

# Function for SARIMAX model
def run_sarimax(item_code, best_params, history_sarimax, forecast_template_sarimax):
    # Filtering by item code for both history and forecast template
    history_filtered_sarimax = history_sarimax[history_sarimax['Item Code'] == item_code]
    forecast_template_filtered_sarimax = forecast_template_sarimax[forecast_template_sarimax['Item Code'] == item_code]
    
    # Extracting the relevant columns and parameters
    y_train_sarimax = history_filtered_sarimax['Weight']
    exog_train_sarimax = history_filtered_sarimax[['WeekOfYear', 'MonthOfYear']]
    exog_forecast_sarimax = forecast_template_filtered_sarimax[['WeekOfYear', 'MonthOfYear']]
    
    # Fit the SARIMAX model using the best parameters and exogenous factors
    model_sarimax = SARIMAX(y_train_sarimax, 
                        exog=exog_train_sarimax,
                        order=(best_params['p'], best_params['d'], best_params['q']),
                        seasonal_order=(best_params['P'], best_params['D'], best_params['Q'], best_params['s']),
                        trend=best_params['trend'],
                        enforce_stationarity=False,
                        enforce_invertibility=False)
    
    model_fit_sarimax = model_sarimax.fit(disp=False)
    
    # Generate forecast using the fitted model and exogenous factors for future dates
    forecast_sarimax = model_fit_sarimax.get_forecast(steps=len(forecast_template_filtered_sarimax), exog=exog_forecast_sarimax)
    
    # Extracting only the forecasted values to return
    forecast_values_sarimax = forecast_sarimax.predicted_mean

    # Assuming forecast_template_filtered_sarimax has a Date column for the forecast dates
    forecast_dates_sarimax = forecast_template_filtered_sarimax.index
    
    output_df_sarimax = pd.DataFrame({
        'Item Code': [item_code]*len(forecast_values_sarimax),
        'Date': forecast_dates_sarimax,
        'Weight': forecast_values_sarimax
    })

    return output_df_sarimax

# Function for Prophet model
def run_prophet(item_code, best_params, history_prophet, forecast_template_prophet):
    # Filtering by item code for both history and forecast template
    history_filtered_prophet = history_prophet[history_prophet['Item Code'] == item_code]
    forecast_template_filtered_prophet = forecast_template_prophet[forecast_template_prophet['Item Code'] == item_code]
    
    # Fit the Prophet model with the additional parameters
    model_prophet = Prophet(
        changepoint_prior_scale=best_params['changepoint_prior_scale'],
        seasonality_prior_scale=best_params['seasonality_prior_scale'],
        seasonality_mode=best_params['seasonality_mode']
    )
    model_prophet.fit(history_filtered_prophet)
    
    # Generate forecast
    forecast_prophet = model_prophet.predict(forecast_template_filtered_prophet)
    
    # Extract only the forecasted values and create output in common format
    forecast_values_prophet = forecast_prophet['yhat']
    forecast_dates_prophet = forecast_prophet['ds']
    output_df_prophet = pd.DataFrame({
        'Item Code': [item_code]*len(forecast_values_prophet),
        'Date': forecast_dates_prophet,
        'Weight': forecast_values_prophet
    })
    
    return output_df_prophet

# Function for ARIMA model
def run_arima(item_code, best_params, history_arima, forecast_template_arima):
    # Filtering by item code for both history and forecast template
    history_filtered_arima = history_arima[history_arima['Item Code'] == item_code]
    forecast_template_filtered_arima = forecast_template_arima[forecast_template_arima['Item Code'] == item_code]
    
    # Extracting the relevant columns
    y_train_arima = history_filtered_arima['Weight']
    
    # Fit the ARIMA model using the best parameters
    model_arima = ARIMA(y_train_arima, 
                        order=(best_params['p'], best_params['d'], best_params['q']),
                        trend=best_params['trend'])
    
    model_fit_arima = model_arima.fit()
    
    # Generate forecast using the fitted model
    forecast_arima = model_fit_arima.forecast(steps=len(forecast_template_filtered_arima))
    
    # Extracting only the forecasted values to return
    forecast_values_arima = forecast_arima
    
    # Assuming forecast_template_filtered_arima has a Date column for the forecast dates
    forecast_dates_arima = forecast_template_filtered_arima.index
    
    # Create output in common format
    output_df_arima = pd.DataFrame({
        'Item Code': [item_code]*len(forecast_values_arima),
        'Date': forecast_dates_arima,
        'Weight': forecast_values_arima
    })
    
    return output_df_arima

# Function for other methods like Average, Naive, or no model
def run_other(item_code, best_model, history_all, forecast_template):
    # Filtering by item code for both history and forecast template
    history_filtered_other = history_all[history_all['Item Code'] == item_code]
    forecast_template_filtered_other = forecast_template[forecast_template['Item Code'] == item_code]
    
    # Initialize an empty list to store the forecast values
    forecast_values_other = []
    
    # Generate forecast based on the best_model
    if best_model == 'Average' or best_model == 'no model':
        # Compute the average of historical data
        average_value = history_filtered_other['Weight'].iloc[-4:].mean()
        forecast_values_other = [average_value] * len(forecast_template_filtered_other)
        
    elif best_model == 'Naive':
        # Use the last observed value as the naive forecast
        naive_value = history_filtered_other['Weight'].iloc[-1]
        forecast_values_other = [naive_value] * len(forecast_template_filtered_other)
    
    # Assuming forecast_template_filtered_other has a Date column for the forecast dates
    forecast_dates_other = forecast_template_filtered_other['Date']
    
    # Create output in common format
    output_df_other = pd.DataFrame({
        'Item Code': [item_code] * len(forecast_values_other),
        'Date': forecast_dates_other,
        'Weight': forecast_values_other
    })
    
    return output_df_other

##End Forecasting Objective Functions##


##Forecasting##

# Assuming final_results_df holds the best_params and best_model for each item code
for index, row in model_params_df.iterrows():
    item_code = row['Item Code']
    best_params = row['BestParams']
    best_model = row['BestModel']
    
    # Initialize an empty DataFrame to hold the output of a single iteration
    output_df_iteration = pd.DataFrame()
    
    if best_model == 'SARIMAX':
        output_sarimax = run_sarimax(item_code, best_params, history_sarimax, forecast_template_sarimax)
        output_df_iteration = output_sarimax
    
    elif best_model == 'Prophet':
        output_prophet = run_prophet(item_code, best_params, history_prophet, forecast_template_prophet)
        output_df_iteration = output_prophet
        
    elif best_model == 'ARIMA':
        output_arima = run_arima(item_code, best_params, history_arima, forecast_template_arima)
        output_df_iteration = output_arima
        
    else:  # For 'Average', 'Naive', or 'no model'
        output_other = run_other(item_code, best_model, history_all, forecast_template)
        output_df_iteration = output_other

    # Concatenate output_df_iteration to final_forecast_df
    final_forecast_df = pd.concat([final_forecast_df, output_df_iteration], ignore_index=True)

# Save the forecast to a CSV file
final_forecast_df.to_csv('forecasts.csv', index=False)