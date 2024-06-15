# Bitcoin Data Analysis and Forecasting Tool

*Disclaimer: This project was designed for the purpose of exploring a large dataset and is not to be considered as financial advice. This tool was not designed to predict the price of Bitcoin and should not be used as such. The dataset was taken from Investing.com.*

## Overview

This module provides classes for processing Bitcoin's historical dataset, analyzing the data, and providing a Bitcoin price forecast using Prophet.

## Features

- Preprocess Bitcoin historical data.
- Analyze Bitcoin data with various visualizations.
- Forecast Bitcoin prices using the Prophet library.
- Interactive user interface for exploring data and forecasts.

## Installation

To get started with this project, clone the repository and install the required dependencies.

```bash
git clone https://github.com/yourusername/bitcoin_analysis_tool.git
cd bitcoin-analysis-tool
pip install -r requirements.txt
Usage
Preprocess Data: Convert and clean the historical data for analysis.
Analyze Data: Generate various plots to understand the historical behavior of Bitcoin prices.
Forecast Prices: Use the Prophet library to forecast future Bitcoin prices.
Example
python
Copy code
from Buisness_Logic_Bitcoin import BitcoinDataProcessor, BitcoinPriceForecast, BitcoinDataAnalyzer

# Preprocess the data
processor = BitcoinDataProcessor('Bitcoin History.csv')

# Analyze the data
analyzer = BitcoinDataAnalyzer('Bitcoin History.csv')
analyzer.plot_price_over_time()
analyzer.plot_candlestick()
analyzer.plot_moving_averages()
analyzer.plot_volume_over_time()
analyzer.plot_correlation_heatmap()
analyzer.plot_seasonality()
analyzer.plot_returns_distribution()

# Forecast prices
forecast = BitcoinPriceForecast('Bitcoin History.csv')
forecast.prepare_data_for_prophet()
forecast.train_model()
forecast.make_forecast()
forecast.plot_forecast()
User Interface
The module also includes a user-friendly interface to interact with the data and forecasts.

python
Copy code
from User_interface_Bitcoin import UserInterface

def main():
    ui = UserInterface()
    ui.display_menu()

if __name__ == "__main__":
    main()
Classes and Methods
BitcoinDataProcessor
__init__(self, csv_file)
Initializes the data processor with the given CSV file and preprocesses the data.

preprocess_data(self)
Converts CSV data from string to float and performs necessary cleaning and formatting.

BitcoinPriceForecast
Inherits from BitcoinDataProcessor.

prepare_data_for_prophet(self)
Prepares data for Prophet by selecting "Date" and "Price" columns and splitting into training and testing sets.

train_model(self)
Initializes and trains the Prophet model with daily seasonality enabled.

make_forecast(self)
Calculates the Mean Squared Error (MSE) between the actual test data and the forecasted values.

plot_forecast(self)
Plots the forecasted Bitcoin prices against the actual prices.

BitcoinDataAnalyzer
Inherits from BitcoinDataProcessor.

plot_price_over_time(self)
Plots Bitcoin price over time.

plot_candlestick(self)
Plots Bitcoin price in a candlestick chart.

plot_moving_averages(self)
Plots 50-day and 200-day moving averages against the price of Bitcoin.

plot_volume_over_time(self)
Plots Bitcoin's trading volume over time.

plot_correlation_heatmap(self)
Plots a correlation heatmap of all columns in the dataset along with 50-day and 200-day moving averages.

plot_seasonality(self)
Plots the seasonal decomposition of Bitcoin's price time series into trend, seasonal, and residual components using an additive model and a period of 30 days.

get_bitcoin_prices_on_same_date(self, date)
Provides the price of Bitcoin on this day for the past 13 years.

plot_returns_distribution(self)
Plots Bitcoin's returns distribution.

UserInterface
__init__(self)
Initializes the user interface with data processor, price forecast, and data analyzer.

display_menu(self)
Displays an interactive menu for the user to choose various analysis and forecasting options.

Disclaimer
This project was designed for the purpose of exploring a large dataset and is not to be considered as financial advice. This tool was not designed to predict the price of Bitcoin and should not be used as such. The dataset was taken from Investing.com.

License
This project is licensed under the MIT License.