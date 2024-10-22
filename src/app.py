import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns # Visualization
import matplotlib.pyplot as plt # Visualization
from colorama import Fore

from sklearn.model_selection import train_test_split
from statsmodels.tsa.api import VAR
import matplotlib.dates as mdates
from datetime import date

from sklearn.metrics import mean_absolute_error, mean_squared_error
import math

import warnings # Supress warnings 
warnings.filterwarnings('ignore')

np.random.seed(7)

Petrignano_df = pd.read_csv('../data/ACEA/Aquifer_Petrignano.csv')

Petrignano_df.info()

print(Petrignano_df.columns)

Petrignano_df

Petrignano_df['Date'] = pd.to_datetime(Petrignano_df['Date'], format='%d/%m/%Y')

Petrignano_df = Petrignano_df.dropna()

Petrignano_df = Petrignano_df.set_index('Date')

Petrignano_df = Petrignano_df.sort_index(ascending=True)

# Create subplots for each feature
fig, axes = plt.subplots(nrows=7, ncols=1, figsize=(20, 50))
plt.subplots_adjust(hspace=0.5)

# Loop through each feature in the Petrignano_df DataFrame
for idx, feature in enumerate(Petrignano_df.columns):
    # Plot the actual data
    sns.lineplot(x=Petrignano_df.index, y=Petrignano_df[feature], ax=axes[idx], color='dodgerblue', label='Actual')

    # Set titles and labels
    axes[idx].set_title(f'Feature: {feature}', fontsize=16, weight='bold')
    axes[idx].set_ylabel(feature, fontsize=14)
    
    # Set x-axis limits and format
    axes[idx].set_xlim([Petrignano_df.index.min(), Petrignano_df.index.max()])  # Dynamic limits based on index
    axes[idx].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    axes[idx].xaxis.set_major_locator(mdates.YearLocator())
    
    # Add grid and legend
    axes[idx].grid(True)
    axes[idx].legend()

# Set the x-label for the last subplot
axes[-1].set_xlabel('Date', fontsize=14)

# Show the plot
plt.show()

Petrignano_differenced_series = Petrignano_df.diff().dropna()

Petrignano_differenced_series

test_size = 0.10  # 10% for testing

# Resetting the index to allow for splitting
df_reset = Petrignano_differenced_series.reset_index()

# Split the data into training and testing sets
train_df, test_df = train_test_split(df_reset, test_size=test_size, shuffle=False)

# Set 'Date' back as the index for both sets
train_df.set_index('Date', inplace=True)
test_df.set_index('Date', inplace=True)

# Display the sizes of the splits
print(f"Training set size: {len(train_df)}")
print(f"Testing set size: {len(test_df)}")

num_forecast = len(test_df)

# Automatically calculate max_lag based on the number of observations
def calculate_max_lag(data, fraction=0.1):
    """
    Calculate the maximum lag as a fraction of the number of observations in the dataset.
    Default fraction is 10%.
    """
    return max(1, int(len(data) * fraction))  # Ensuring at least 1 lag is selected

# Calculate dynamic max_lag as 10% of the total number of rows in the training data
max_lag = calculate_max_lag(train_df, fraction=0.1)  # Adjust the fraction as needed
print(f"Automatically calculated max_lag: {max_lag}")

# Fit the VAR model on training data
model = VAR(train_df)

# Select the optimal lag length based on multiple criteria (e.g., AIC, BIC, HQIC)
lag_order = model.select_order(maxlags=max_lag)

# Print the summary of optimal lags for various criteria
print("Optimal Lag Orders (AIC, BIC, HQIC, FPE):\n", lag_order.summary())

# Access the optimal lag according to AIC (or change to other criteria, e.g., BIC)
optimal_lag = lag_order.aic
print(f"Optimal Lag based on AIC: {optimal_lag}")

# Ensure the optimal lag is within reasonable bounds (e.g., less than max_lag)
if optimal_lag > max_lag:
    print(f"Warning: Optimal lag {optimal_lag} exceeds the maximum allowed lag ({max_lag}). Fitting with max_lag instead.")
    optimal_lag = max_lag

# Fit the VAR model using the selected optimal lag
print(f"Fitting VAR model with lag: {optimal_lag}")
model_fitted = model.fit(optimal_lag)

# Print the summary of the fitted VAR model
print("\nFitted VAR Model Summary:\n", model_fitted.summary())

num_lag = max_lag

forecast = model_fitted.forecast(train_df.values[-num_lag:], steps=num_forecast)
# Adjust the number of column names
forecast_df = pd.DataFrame(forecast, columns=[f'Forecast_{i}' for i in range(forecast.shape[1])])

print(forecast_df)

forecast_df.columns = [
    'Rainfall_Bastia_Umbra',
    'Depth_to_Groundwater_P24',
    'Depth_to_Groundwater_P25',
    'Temperature_Bastia_Umbra',
    'Temperature_Petrignano',
    'Volume_C10_Petrignano',
    'Hydrometry_Fiume_Chiascio_Petrignano'
]

print(forecast_df)

# Step 1: Determine the last date in Petrignano_df
last_date = train_df.index[-1]

# Step 2: Generate a date range for the next 12 days
forecast_index = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=num_forecast, freq='D')

# Step 3: Reassign the new index to forecast_df
forecast_df.index = forecast_index

# Print the new forecast_df to check the index
print("Forecast Data with New Index:")
print(forecast_df)

# Step 1: Get the last observation from the original data (before differencing)
last_observation = Petrignano_df.iloc[-1]

# Step 2: Cumulatively sum the differenced forecast to revert differencing
forecast_reverted = forecast_df.cumsum()

# Step 3: Add the last observation to each feature in the forecast to return to the original scale
forecast_reverted = forecast_reverted.add(last_observation, axis=1)

# Print the reverted forecast data
print("Reverted Forecast Data:")
print(forecast_reverted)

# Create subplots for the two features
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(20, 20))
plt.subplots_adjust(hspace=0.5)

# Features to plot
features_to_plot = ['Depth_to_Groundwater_P24', 'Depth_to_Groundwater_P25']

# Loop through each feature in features_to_plot
for idx, feature in enumerate(features_to_plot):
    # Plot actual data from test_df
    sns.lineplot(x=test_df.index, y=test_df[feature], ax=axes[idx], color='dodgerblue', label='Actual')

    # Plot forecast data with bright orange color
    if feature in forecast_df.columns:
        sns.lineplot(x=forecast_df.index, y=forecast_df[feature], ax=axes[idx], color='orange', linestyle='-', label='Forecast')

    # Set titles and labels
    axes[idx].set_title(f'Feature: {feature}', fontsize=16, weight='bold')
    axes[idx].set_ylabel(feature, fontsize=14)

    # Set x-axis limits and format
    axes[idx].set_xlim([test_df.index.min(), forecast_df.index.max()])  # Extend limits to cover both datasets
    axes[idx].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    axes[idx].xaxis.set_major_locator(mdates.YearLocator())

    # Add grid and legend
    axes[idx].grid(True)
    axes[idx].legend()

# Set x-label for the last subplot
axes[-1].set_xlabel('Date', fontsize=14)

# Show the plot for actual vs. forecast
plt.show()

# Create a single plot for both actual and forecast data
fig, axes = plt.subplots(nrows=7, ncols=1, figsize=(20, 50))
plt.subplots_adjust(hspace=0.5)

# Loop through each feature in Petrignano_df
for idx, feature in enumerate(Petrignano_df.columns):
    # Plot actual data
    sns.lineplot(x=Petrignano_df.index, y=Petrignano_df[feature], ax=axes[idx], color='dodgerblue', label='Actual')

    # Plot forecast data with bright orange color and same style as actual data
    if idx < forecast_df.shape[1]:  # Ensure forecast_reverted has enough columns
        sns.lineplot(x=forecast_reverted.index, y=forecast_reverted.iloc[:, idx], ax=axes[idx], color='orange', linestyle='-', label='Forecast')

    # Set titles and labels
    axes[idx].set_title(f'Feature: {feature}', fontsize=16, weight='bold')
    axes[idx].set_ylabel(feature, fontsize=14)
    
    # Set x-axis limits and format
    axes[idx].set_xlim([Petrignano_df.index.min(), forecast_reverted.index.max()])  # Extend limits to cover both datasets
    axes[idx].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    axes[idx].xaxis.set_major_locator(mdates.YearLocator())
    
    # Add grid and legend
    axes[idx].grid(True)
    axes[idx].legend()

# Set x-label for the last subplot
axes[-1].set_xlabel('Date', fontsize=14)

# Show the plot for actual vs. forecast
plt.show()

# Calculate MAE
mae = mean_absolute_error(test_df[['Depth_to_Groundwater_P24','Depth_to_Groundwater_P25']], forecast_df[['Depth_to_Groundwater_P24','Depth_to_Groundwater_P25']])
print(f"Mean Absolute Error (MAE): {mae}")

# Calculate MSE
mse = mean_squared_error(test_df[['Depth_to_Groundwater_P24','Depth_to_Groundwater_P25']], forecast_df[['Depth_to_Groundwater_P24','Depth_to_Groundwater_P25']])
print(f"Mean Squared Error (MSE): {mse}")

# Calculate RMSE
rmse = np.sqrt(mse)
print(f"Root Mean Squared Error (RMSE): {rmse}")

Auser_df = pd.read_csv('../data/ACEA/Aquifer_Auser.csv')

Auser_df['Date'] = pd.to_datetime(Auser_df['Date'], format='%d/%m/%Y')

Auser_df = Auser_df.dropna().set_index('Date').sort_index()

def plot_features(df, features, title, forecast_df=None):
    """
    Plot the features of the dataset with optional forecast overlay.
    """
    fig, axes = plt.subplots(nrows=len(features), ncols=1, figsize=(20, len(features) * 7))
    plt.subplots_adjust(hspace=0.5)

    for idx, feature in enumerate(features):
        sns.lineplot(x=df.index, y=df[feature], ax=axes[idx], color='dodgerblue', label='Actual')
        
        if forecast_df is not None and feature in forecast_df.columns:
            sns.lineplot(x=forecast_df.index, y=forecast_df[feature], ax=axes[idx], color='orange', linestyle='-', label='Forecast')

        # Set titles, labels, and formatting
        axes[idx].set_title(f'Feature: {feature}', fontsize=16, weight='bold')
        axes[idx].set_ylabel(feature, fontsize=14)
        axes[idx].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        axes[idx].xaxis.set_major_locator(mdates.YearLocator())
        axes[idx].grid(True)
        axes[idx].legend()

    axes[-1].set_xlabel('Date', fontsize=14)
    plt.show()

    plot_features(Auser_df, Auser_df.columns, 'Actual Data')

    Auser_differenced_series = Auser_df.diff().dropna()

    test_size = 0.20  # 20% for testing

# Resetting the index to allow for splitting
df_reset = Auser_differenced_series.reset_index()

# Split the data into training and testing sets
train_df, test_df = train_test_split(df_reset, test_size=test_size, shuffle=False)

# Set 'Date' back as the index for both sets
train_df.set_index('Date', inplace=True)
test_df.set_index('Date', inplace=True)

# Display the sizes of the splits
print(f"Training set size: {len(train_df)}")
print(f"Testing set size: {len(test_df)}")

# Split data into training and testing sets
#train_df, test_df = train_test_split(Auser_differenced_series, test_size=0.10, shuffle=False)

# Adjust the calculate_max_lag function to limit max_lag
def calculate_max_lag(data, fraction=0.1, max_limit=30):
    """
    Calculate maximum lag as a fraction of the dataset size, but limit it to max_limit.
    """
    calculated_lag = max(1, int(len(data) * fraction))
    return min(calculated_lag, max_limit)  # Add a max_limit to avoid large lags

# Recalculate the optimal lag with an upper limit on maxlags
max_lag_Auser = calculate_max_lag(train_df, fraction=0.1, max_limit=30)
print(f"Max Lag Auser: {max_lag_Auser}")

# Fit VAR model and select optimal lag
model = VAR(train_df)
lag_order = model.select_order(maxlags=max_lag_Auser)
optimal_lag = lag_order.aic
print(f"Optimal Lag based on AIC: {optimal_lag}")

# Fit the model using the optimal lag
model_fitted = model.fit(optimal_lag)
print(model_fitted.summary())

# Forecast the next steps
num_forecast = len(test_df)
forecast = model_fitted.forecast(train_df.values[-optimal_lag:], steps=num_forecast)
forecast_df = pd.DataFrame(forecast, columns=Auser_df.columns)

# Assign proper index to forecast_df
forecast_index = pd.date_range(start=train_df.index[-1] + pd.Timedelta(days=1), periods=num_forecast, freq='D')
forecast_df.index = forecast_index

# Revert differencing to get actual values
last_observation = Auser_df.iloc[-1]
forecast_reverted = forecast_df.cumsum().add(last_observation, axis=1)

# Plot actual vs forecast for specific features
plot_features(test_df, ['Depth_to_Groundwater_SAL', 'Depth_to_Groundwater_LT2'], 'Actual vs Forecast', forecast_reverted)

# Plot actual vs forecast for all features
plot_features(Auser_df, Auser_df.columns, 'Actual vs Forecast for All Features', forecast_reverted)

# Calculate errors (MAE, MSE, RMSE)
def calculate_errors(test_data, forecast_data, features):
    """
    Calculate MAE, MSE, and RMSE for given features.
    """
    mae = mean_absolute_error(test_data[features], forecast_data[features])
    mse = mean_squared_error(test_data[features], forecast_data[features])
    rmse = np.sqrt(mse)
    
    return mae, mse, rmse

# Update feature names based on the Auser dataset
mae, mse, rmse = calculate_errors(test_df, forecast_reverted, ['Depth_to_Groundwater_SAL', 'Depth_to_Groundwater_LT2'])
print(f"MAE: {mae}, MSE: {mse}, RMSE: {rmse}")

