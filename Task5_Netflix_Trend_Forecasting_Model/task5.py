# Task 5 Netflix Trend Forecasting Model

import Loader
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import warnings
warnings.filterwarnings('ignore')  # hides unnecessary math warnings

# Loading the data
data = Loader.load_data('Dataset/Dataset.csv')

# Time Series Aggregation
# Setting the date as the index, to count how many titles were added each month as ("ME")

data.set_index('date_added', inplace=True)
monthly_adds = data.resample('ME').size()

# Starting from 2016
monthly_adds = monthly_adds[monthly_adds.index >= '2016-01-01']

# Train / Test Split
# Training on the past data and testing the model on last 12 months (OCT 2020 - Sept 2021)
train = monthly_adds.iloc[:-12]
test = monthly_adds.iloc[-12:]

# Building the Forecasting model
# Using Exponential Smoothing finds the trend by heavily weighting recent past averages
model = ExponentialSmoothing(train,
                             trend='add',
                             seasonal=None,
                             initialization_method='estimated')

# Predictions for last 12 months
fit_model = model.fit()
predictions = fit_model.forecast(len(test))

# Accuracy evaluation
mean_error = mean_absolute_error(test, predictions)
print(f"Mean Absolute Error: {mean_error:.2f}")
print(
    f"Interpretation: On average, our AI forecast was off by {int(mean_error)} titles per month")

# Visuals
plt.figure(figsize=(12, 6))
plt.plot(train.index,
         train,
         label='Training Data (2016 - 2020)')
plt.plot(test.index,
         test,
         label='Actual Final Year Data (2020-2021)',
         color='green')
plt.plot(test.index,
         predictions,
         label='AI Trend Forecast Vs Reality')

plt.title('Netflix Content Addition: AI Forecast vs Reality')
plt.xlabel('Date')
plt.ylabel('Number of Titles Added per Month')
plt.legend()
plt.grid(True)
plt.show()
