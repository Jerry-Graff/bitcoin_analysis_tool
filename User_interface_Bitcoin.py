"""
This module contains classes which deal with the user interface of the
programme. The user interface is designed so the user can explore the
plt charts individually.
"""

from datetime import datetime

from Buisness_Logic_Bitcoin import BitcoinDataProcessor
from Buisness_Logic_Bitcoin import BitcoinPriceForecast
from Buisness_Logic_Bitcoin import BitcoinDataAnalyzer
from Utilities import draw_line, financial_warning


class UserInterface:
    def __init__(self):
        self.data_processor = BitcoinDataProcessor("Bitcoin History.csv")
        self.price_forecast = BitcoinPriceForecast("Bitcoin History.csv")
        self.data_analyzer = BitcoinDataAnalyzer("Bitcoin History.csv")

    def display_menu(self):
        while True:
            print(financial_warning())
            print(f'''\n{draw_line()}
            Bitcoin Data Analysis and Forecasting Tool\n{draw_line()}
                1. Plot Bitcoin Price Over Time
                2. Plot Candlestick Chart
                3. Plot Moving Averages
                4. Plot Volume Over Time
                5. Plot Correlation Heatmap
                6. Plot Seasonality Components
                7. Plot Bitcoin Return Distributions
                8. Get Bitcoin Prices on Same Date in Past Years
                9. Perform Bitcoin Price Forecasting
                0. Exit
                  ''')

            choice = input("What would you like to do: ")
            if choice == "1":
                self.data_analyzer.plot_price_over_time()
            elif choice == "2":
                self.data_analyzer.plot_candlestick()
            elif choice == "3":
                self.data_analyzer.plot_moving_averages()
            elif choice == "4":
                self.data_analyzer.plot_volume_over_time()
            elif choice == "5":
                self.data_analyzer.plot_correlation_heatmap()
            elif choice == "6":
                self.data_analyzer.plot_seasonality()
            elif choice == "7":
                self.data_analyzer.plot_returns_distribution()
            elif choice == "8":
                todays_date = datetime.today().strftime("%Y-%m-%d")
                prices = self.data_analyzer.get_bitcoin_prices_on_same_date(
                    todays_date)
                for year, price in sorted(prices.items(), reverse=True):
                    print(f"On this day in {year}, the price of Bitcoin was: "
                          f"${price}")
            elif choice == "9":
                self.price_forecast.prepare_data_for_prophet()
                self.price_forecast.train_model()
                self.price_forecast.make_forecast()
                self.price_forecast.plot_forecast()
            elif choice == "0":
                print("Exiting Application")
                break
            else:
                print("Invalid choice, please choose again: ")
