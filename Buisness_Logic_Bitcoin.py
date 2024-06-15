"""
This module provides classes for processing Bitcoin's historical
dataset, analyzing the data and providing a Bitcoin price forcast
using prophet.

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

from datetime import datetime
from prophet import Prophet
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.seasonal import seasonal_decompose


class BitcoinDataProcessor:

    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.preprocess_data()

    def preprocess_data(self):
        """
        Converts csv data from string to float.
        """
        self.data["Date"] = pd.to_datetime(self.data["Date"])
        self.data.set_index("Date", inplace=True)
        self.data["Price"] = self.data["Price"].str.replace(
            ",", "").astype(float)
        self.data["Open"] = self.data["Open"].str.replace(
            ",", "").astype(float)
        self.data["High"] = self.data["High"].str.replace(
            ",", "").astype(float)
        self.data["Low"] = self.data["Low"].str.replace(
            ",", "").astype(float)
        self.data["Vol."] = self.data["Vol."].replace(
            "-", np.nan)
        self.data["Change %"] = self.data["Change %"].str.replace(
            "%", "").astype(float)
        self.data["Vol."] = (self.data["Vol."].str.replace("K", "e3")
                                              .str.replace("M", "e6")
                                              .str.replace("B", "e9")
                                              .astype(float))
        self.data = self.data.sort_index()
        self.data = self.data.asfreq('D')
        self.data["Price"] = self.data["Price"].interpolate(
            method='linear')


class BitcoinPriceForecast(BitcoinDataProcessor):
    def __init__(self, csv_file):
        super().__init__(csv_file)
        self.model = None
        self.train = None
        self.test = None
        self.forecast = None

    def prepare_data_for_prophet(self):
        """
        Selects "Date" and "Price" cols and splits the data
        80%(Training) - 20%(Testing).
        """
        bitcoin_data_prophet = self.data.reset_index()[['Date', 'Price']]
        bitcoin_data_prophet.columns = ['ds', 'y']
        train_size = int(len(bitcoin_data_prophet) * 0.8)
        self.train, self.test = bitcoin_data_prophet[
            :train_size], bitcoin_data_prophet[train_size:]

    def train_model(self):
        """
        Initializes the Prophet model with daily seasonality enabled.
        """
        self.model = Prophet(daily_seasonality=True)
        self.model.fit(self.train)

    def make_forecast(self):
        """
        Calculates the Mean Squared Error (MSE) between the actual test
        data and the forecasted values.
        """
        future = self.model.make_future_dataframe(periods=len(self.test))
        self.forecast = self.model.predict(future)
        forecast_test = self.forecast.set_index(
            'ds').loc[self.test['ds']]['yhat']
        mse = mean_squared_error(self.test['y'], forecast_test)
        print(f"Mean Squared Error: {mse}")

    def plot_forecast(self):
        """
        Plots the forecasted Bitcoin prices against the actual prices.
        """
        forecast_test = self.forecast.set_index(
            'ds').loc[self.test['ds']]['yhat']
        plt.figure(figsize=(12, 6))
        plt.plot(self.test['ds'], self.test['y'], label='Actual')
        plt.plot(self.test['ds'], forecast_test, color='red', label='Forecast')
        plt.title('Bitcoin Price Forecast')
        plt.xlabel('Date')
        plt.ylabel('Price (USD)')
        plt.legend()
        plt.show()


class BitcoinDataAnalyzer(BitcoinDataProcessor):

    def plot_price_over_time(self):
        """
        Plots Bitcoin price over time.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(self.data.index, self.data["Price"])
        plt.title("Bitcoin Price Over Time")
        plt.xlabel("Date")
        plt.ylabel("Price (USD)")
        plt.show()

    def plot_candlestick(self):
        """
        Plots Bitcoins price in candlestick chart.
        """
        data = self.data.dropna()
        fig = go.Figure(data=[go.Candlestick(x=data.index,
                                             open=data["Open"],
                                             high=data["High"],
                                             low=data["Low"],
                                             close=data["Price"]
                                             )])
        fig.update_layout(title="Bitcoin Candlestick Chart",
                          xaxis_title="Date",
                          yaxis_title="Price (USD)")
        fig.show()

    def plot_moving_averages(self):
        """
        Plots 50-day and 200-day moving averages against the price of
        Bitcoin.
        """
        self.data["MA50"] = self.data["Price"].rolling(window=50).mean()
        self.data["MA200"] = self.data["Price"].rolling(window=200).mean()
        plt.figure(figsize=(10, 6))
        plt.plot(self.data.index, self.data["Price"], label="Price")
        plt.plot(self.data.index, self.data[
            "MA50"], label="50 Day Moving Average")
        plt.plot(self.data.index, self.data[
            "MA200"], label="200 Day Moving Average")
        plt.title("Bitcoin Price with Moving Averages")
        plt.xlabel("Date")
        plt.ylabel("Price (USD)")
        plt.legend()
        plt.show()

    def plot_volume_over_time(self):
        """
        Plots Bitcoins trading volume over dataset.
        """
        avg_volume_per_month = self.data.resample('M').mean()
        plt.figure(figsize=(10, 6))
        plt.plot(avg_volume_per_month.index, avg_volume_per_month[
            "Vol."], marker='o')
        plt.yscale('log')
        plt.title("Average Bitcoin Trading Volume Per Month")
        plt.xlabel("Date")
        plt.ylabel("Average Volume (log scale)")
        plt.grid(True)
        plt.show()

    def plot_correlation_heatmap(self):
        """
        Plots correlation heatmap map against all colums of dataset along
        with 50-day and 200-day moving average.
        """
        corr = self.data.corr()
        plt.figure(figsize=(10, 6))
        sns.heatmap(corr, annot=True, cmap="coolwarm")
        plt.title('Correlation Heatmap')
        plt.show()

    def plot_seasonality(self):
        """
        Plot the seasonal decomposition of Bitcoin's price time series
        into trend, seasonal, and residual components using an additive
        model and a period of 30 days.
        """
        decomposition = seasonal_decompose(
            self.data["Price"], model="additive", period=30)
        trend = decomposition.trend
        seasonal = decomposition.seasonal
        residual = decomposition.resid
        plt.figure(figsize=(12, 8))

        plt.subplot(411)
        plt.plot(self.data["Price"], label="Original")
        plt.legend(loc="upper left")
        plt.title("Original Time Series")
        plt.xlabel("Date")
        plt.ylabel("Price (USD)")

        plt.subplot(412)
        plt.plot(trend, label="Trend")
        plt.legend(loc="upper left")
        plt.title("Trend Component")
        plt.xlabel("Date")
        plt.ylabel("Price (USD)")

        plt.subplot(413)
        plt.plot(seasonal, label="Seasonal")
        plt.legend(loc="upper left")
        plt.title("Seasonal Component")
        plt.xlabel("Date")
        plt.ylabel("Seasonal Deviation")

        plt.subplot(414)
        plt.plot(residual, label="Residual")
        plt.legend(loc="upper left")
        plt.title("Residual Component")
        plt.xlabel("Date")
        plt.ylabel("Residual Error")

        plt.tight_layout()
        plt.show()

    def get_bitcoin_prices_on_same_date(self, date):
        """
        Provides the price of Bitcoin on this day for the past 13 years.
        """
        results = {}
        target_date = datetime.strptime(date, '%Y-%m-%d').date()
        for year in range(target_date.year - 1, target_date.year - 14, -1):
            if year < 2024:
                try:
                    same_date = target_date.replace(year=year)
                    price_on_date = self.data.loc[
                        self.data.index == pd.Timestamp(
                            same_date), "Price"].values[0]
                    results[year] = price_on_date
                except IndexError:
                    results[year] = "Missing Data"
        return results

    def plot_returns_distribution(self):
        """
        Plots Bitcoin's returns distribution.
        """
        self.data['Returns'] = self.data['Price'].pct_change()
        plt.figure(figsize=(12, 6))
        sns.histplot(self.data['Returns'].dropna(), bins=50, kde=True)
        plt.title('Distribution of Daily Returns')
        plt.xlabel('Daily Returns')
        plt.ylabel('Frequency')
        plt.show()
