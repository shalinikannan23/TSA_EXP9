## DEVELOPED BY: SHALINI K
## REGISTER NO: 212222240095
## DATE: 
# EX.NO.09        A project on Time series analysis on forecasting using ARIMA model 

### AIM:
To Create a project on Time series analysis on forecasting using ARIMA model in  Python and compare with other models.
### ALGORITHM:
1. Explore the dataset of Raw_sales. 
2. Check for stationarity of time series time series plot
   ACF plot and PACF plot
   ADF test
   Transform to stationary: differencing
3. Determine ARIMA models parameters p, q
4. Fit the ARIMA model
5. Make time series predictions
6. Auto-fit the ARIMA model
7. Evaluate model predictions
### PROGRAM:
```py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

def arima_model(data, target_variable, order):
    # Split data into training and testing sets (80% train, 20% test)
    train_size = int(len(data) * 0.8)
    train_data, test_data = data[:train_size], data[train_size:]

    # Fit the ARIMA model on the training data
    model = ARIMA(train_data[target_variable], order=order)
    fitted_model = model.fit()

    # Forecast for the length of the test data
    forecast = fitted_model.forecast(steps=len(test_data))

    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(test_data[target_variable], forecast))
    print(f"RMSE: {rmse}")

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(train_data.index, train_data[target_variable], label='Training Data', color='blue')
    plt.plot(test_data.index, test_data[target_variable], label='Testing Data', color='red')
    plt.plot(test_data.index, forecast, label='Forecasted Data', color='green')
    plt.xlabel('Date')
    plt.ylabel(target_variable)
    plt.title('ARIMA Forecasting for ' + target_variable)
    plt.legend()
    plt.show()

# Load the Bitcoin data
data = pd.read_csv('coin_Bitcoin.csv')  # Assuming your CSV file is named 'bitcoin_data.csv'

# Convert the 'Date' column to datetime format and set it as the index
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Run the ARIMA model for Bitcoin's 'Close' price
arima_model(data, 'Close', order=(5,1,0))


```
### OUTPUT:

![image](https://github.com/user-attachments/assets/3b9afa0d-6130-48df-bbd9-f898d459fdbc)


### RESULT:
Thus, the program based on the ARIMA model using python is executed successfully.
