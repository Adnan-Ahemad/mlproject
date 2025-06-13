# mlproject

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

air_quality_data = pd.read_csv("/content/AirQualityUCI.csv")
air_quality_data = air_quality_data.dropna(how='all')
air_quality_data = air_quality_data.dropna(axis=1, how='all')
air_quality_data = air_quality_data.replace(-200, np.nan)
air_quality_data['Date'] = pd.to_datetime(air_quality_data['Date'], format='%d-%m-%Y')
air_quality_data['ds'] = air_quality_data['Date'].astype(str) + ' ' + air_quality_data['Time'].astype(str)
air_quality_data['ds'] = pd.to_datetime(air_quality_data['ds'], format='%Y-%m-%d %H.%M.%S')
numerical_data = air_quality_data.drop(['Date', 'Time', 'ds'], axis=1, errors='ignore')

for col in numerical_data.columns:
    numerical_data[col] = pd.to_numeric(numerical_data[col], errors='coerce')

numerical_data.fillna(numerical_data.mean(), inplace=True)
numerical_data['ds'] = air_quality_data['ds']
data_for_prophet = numerical_data[['ds', 'PT08.S1(CO)']].copy()
data_for_prophet.rename(columns={'PT08.S1(CO)': 'y'}, inplace=True)
data_for_prophet.dropna(subset=['ds', 'y'], inplace=True)

print("Generating Correlation Heatmap...")
correlation_matrix = numerical_data.drop('ds', axis=1, errors='ignore').corr()
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap of Air Quality Data')
plt.show()

print("\nTraining Prophet Model for Accuracy Evaluation...")
train_size = int(len(data_for_prophet) * 0.8)
train_df = data_for_prophet.iloc[:train_size]
test_df = data_for_prophet.iloc[train_size:]
model_accuracy = Prophet()
model_accuracy.fit(train_df)
future_test_dates = pd.DataFrame({'ds': test_df['ds']})
forecast_accuracy = model_accuracy.predict(future_test_dates)
results_df = test_df.set_index('ds').join(forecast_accuracy.set_index('ds')[['yhat']])
results_df.dropna(inplace=True)

mse = mean_squared_error(results_df['y'], results_df['yhat'])
rmse = np.sqrt(mse)
r2 = r2_score(results_df['y'], results_df['yhat'])

print(f"Prophet Model Accuracy on Test Data:")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R-squared (R2): {r2:.2f}")

print("\nMaking specific prediction...")
new_data_point = pd.DataFrame({'ds': [pd.to_datetime('2024-03-15 10:00:00')]})
model_full = Prophet()
model_full.fit(data_for_prophet)
forecast = model_full.predict(new_data_point)
predicted_air_quality = forecast['yhat'][0]
print("Predicted Air Quality:", predicted_air_quality)

def classify_aqi(aqi_value):
    if aqi_value <= 50:
        return "Good"
    elif aqi_value <= 100:
        return "Moderate"
    elif aqi_value <= 150:
        return "Unhealthy for Sensitive Groups"
    elif aqi_value <= 200:
        return "Unhealthy"
    elif aqi_value <= 300:
        return "Very Unhealthy"
    else:
        return "Hazardous"

predicted_category = classify_aqi(predicted_air_quality)
print("Predicted Air Quality Category:", predicted_category)
