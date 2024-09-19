import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam

# Load the data
file_path = '/content/SBIN.NS.csv'
data = pd.read_csv(file_path)

# Convert 'Date' to datetime format and set it as index
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Check for NaN values and handle them
print(data.isna().sum())

# Drop rows with NaN values
data = data.dropna()

# Alternatively, you could fill NaN values with a method like forward fill
# data = data.fillna(method='ffill')

# Step 1: Preprocess the Data
# Select the 'Close' price column (you can add other features as well)
prices = data['Close'].values
prices = prices.reshape(-1, 1)

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_prices = scaler.fit_transform(prices)

# Create sequences for multi-step forecasting
def create_sequences(data, seq_length, forecast_horizon):
    X, y = [], []
    for i in range(len(data) - seq_length - forecast_horizon):
        X.append(data[i:i + seq_length, 0])
        y.append(data[i + seq_length:i + seq_length + forecast_horizon, 0])
    return np.array(X), np.array(y)

seq_length = 60
forecast_horizon = 7  # Predict the next 7 days
X, y = create_sequences(scaled_prices, seq_length, forecast_horizon)

# Split the data
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Reshape for LSTM input
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1)).astype(np.float32)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1)).astype(np.float32)
y_train = y_train.astype(np.float32)
y_test = y_test.astype(np.float32)

# Step 2: Train the Model
# Define the LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(seq_length, 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(forecast_horizon))

# Compile the model with a lower learning rate and gradient clipping
optimizer = Adam(learning_rate=0.0001, clipvalue=1.0)
model.compile(optimizer=optimizer, loss='mean_squared_error')

# Train the model
history = model.fit(X_train, y_train, batch_size=1, epochs=3, validation_data=(X_test, y_test))

# Step 3: Test the Model
# Generate predictions
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)

# Inverse transform the actual values as well
y_test = scaler.inverse_transform(y_test)

# Step 4: Visualize Predictions
# Prepare the data for plotting
train = data[:split + seq_length]
valid = data[split + seq_length:split + seq_length + len(predictions)].copy()
valid['Predictions'] = np.nan

for i in range(len(predictions)):
    valid.iloc[i:i+forecast_horizon, valid.columns.get_loc('Predictions')] = predictions[i]

valid['Actual'] = np.nan
for i in range(len(y_test)):
    valid.iloc[i:i+forecast_horizon, valid.columns.get_loc('Actual')] = y_test[i]

# Plot the results
plt.figure(figsize=(16, 8))
plt.title('Model')
plt.xlabel('Date')
plt.ylabel('Close Price USD ($)')
plt.plot(train['Close'], label='Train')
plt.plot(valid['Close'], label='Actual')
plt.plot(valid['Predictions'], label='Predictions')
plt.legend(['Train', 'Actual', 'Predictions'], loc='lower right')
plt.show()

# Plot actual vs predicted
plt.figure(figsize=(16, 8))
plt.plot(valid.index, valid['Actual'], color='blue', label='Actual')
plt.plot(valid.index, valid['Predictions'], color='red', label='Predicted')
plt.title('Actual vs Predicted Stock Prices')
plt.xlabel('Date')
plt.ylabel('Close Price USD ($)')
plt.legend()
plt.show()