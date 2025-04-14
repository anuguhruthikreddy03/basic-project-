import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
import streamlit as st

# Load the dataset
file_path = "climate_change_dataset1.csv"

if not os.path.exists(file_path):
    st.error(f"❌ Error: File '{file_path}' not found. Please upload the correct dataset.")
    st.stop()

df = pd.read_csv(file_path)
st.success("✅ Data loaded successfully!")

# Exploratory Data Analysis (EDA)
st.title("🌱 Climate Change Forecasting")
st.write("### Data Overview")
st.dataframe(df.head())

# Correlation Heatmap
st.write("### Correlation Heatmap")
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', ax=ax)
st.pyplot(fig)

# Temperature Trend Visualization
st.write("### Temperature Trend Over the Years")
fig, ax = plt.subplots(figsize=(10, 6))
sns.lineplot(x='Year', y='Temperature', data=df, ax=ax)
st.pyplot(fig)

# Feature Scaling
numerical_cols = ['Temperature', 'CO2 Emissions (Tons/Capita)', 'Sea Level Rise (mm)',
                  'Rainfall (mm)', 'Population', 'Renewable Energy (%)',
                  'Extreme Weather Events', 'Forest Area (%)']

scaler = MinMaxScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# Train ARIMA Model for Temperature Prediction
st.write("### Model Training (ARIMA)")
train_data = df['Temperature'][:-12]
test_data = df['Temperature'][-12:]

try:
    model = ARIMA(train_data, order=(5, 1, 2))
    model_fit = model.fit()
    st.write(model_fit.summary())
    
    # Model Forecasting
    forecast = model_fit.forecast(steps=12)
    st.write("### Forecasted Temperature for Next 12 Months")
    forecast_df = pd.DataFrame({'Actual': test_data.values, 'Forecast': forecast.values}, index=test_data.index)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(test_data.index, test_data.values, label='Actual', marker='o')
    ax.plot(test_data.index, forecast.values, label='Forecast', linestyle='dashed', marker='x')
    ax.set_title("Actual vs Forecasted Temperature")
    ax.set_xlabel("Time")
    ax.set_ylabel("Temperature")
    ax.legend()
    st.pyplot(fig)
    
    # Model Evaluation
    mse = mean_squared_error(test_data, forecast)
    mae = mean_absolute_error(test_data, forecast)
    st.write(f"📊 Mean Squared Error (MSE): {mse:.4f}")
    st.write(f"📉 Mean Absolute Error (MAE): {mae:.4f}")
    
    # Climate Change Impact Assessment
    st.write("### Climate Change Impact Assessment")
    impact_df = forecast_df.copy()
    impact_df['Difference'] = impact_df['Actual'] - impact_df['Forecast']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=impact_df.index, y=impact_df['Difference'], ax=ax, palette='coolwarm')
    ax.set_title("Impact Difference (Actual - Forecast)")
    ax.set_xlabel("Time")
    ax.set_ylabel("Difference")
    st.pyplot(fig)
    
    st.success("🌎 Climate Change Forecasting Project Execution Completed Successfully!")
except Exception as e:
    st.error(f"❌ ARIMA model training failed: {e}")
    st.stop()