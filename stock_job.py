#!/usr/bin/env python3
"""
Stock Market Analysis Job
-------------------------
This script fetches stock data for a list of tickers using yfinance,
calculates various technical indicators, performs Monte Carlo simulations
for price prediction, and optimizes a mock portfolio allocation.

It is designed to simulate a compute-intensive task that takes a few minutes to run.
"""

import os
import sys
import time
import logging
import datetime
import random
import math
import getpass

try:
    import yfinance as yf
    import pandas as pd
    import numpy as np
except ImportError:
    print("Please install required packages: pip install yfinance pandas numpy")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("StockAnalyzer")

# Constants
TICKERS = [
    "AAPL",
    "MSFT",
    "GOOGL",
    "AMZN",
    "TSLA",
    "META",
    "NVDA",
    "AMD",
    "NFLX",
    "INTC",
    "JPM",
    "BAC",
    "WFC",
    "C",
    "GS",
    "MS",
    "V",
    "MA",
    "AXP",
    "PYPL",
]
START_DATE = (datetime.datetime.now() - datetime.timedelta(days=365 * 2)).strftime(
    "%Y-%m-%d"
)
END_DATE = datetime.datetime.now().strftime("%Y-%m-%d")
SIMULATION_RUNS = 5000  # Number of Monte Carlo simulations
FORECAST_DAYS = 252  # Predict next year (trading days)


class TechnicalIndicators:
    """Class to calculate various technical indicators."""

    @staticmethod
    def calculate_sma(data, window=20):
        """Simple Moving Average."""
        return data.rolling(window=window).mean()

    @staticmethod
    def calculate_ema(data, span=20):
        """Exponential Moving Average."""
        return data.ewm(span=span, adjust=False).mean()

    @staticmethod
    def calculate_bollinger_bands(data, window=20, num_std=2):
        """Bollinger Bands."""
        sma = data.rolling(window=window).mean()
        std = data.rolling(window=window).std()
        upper_band = sma + (std * num_std)
        lower_band = sma - (std * num_std)
        return upper_band, lower_band

    @staticmethod
    def calculate_rsi(data, window=14):
        """Relative Strength Index."""
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def calculate_macd(data, slow=26, fast=12, signal=9):
        """Moving Average Convergence Divergence."""
        exp1 = data.ewm(span=fast, adjust=False).mean()
        exp2 = data.ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        return macd, signal_line


class MarketSimulator:
    """Class to perform Monte Carlo simulations."""

    def __init__(self, ticker, data):
        self.ticker = ticker
        # Calculate daily returns
        self.returns = data.pct_change().dropna()
        self.mean_return = self.returns.mean()
        self.std_dev = self.returns.std()
        self.last_price = data.iloc[-1]

    def run_simulation(self, num_simulations, num_days):
        """Runs Monte Carlo simulation to forecast future prices."""
        logger.info(
            f"Running {num_simulations} Monte Carlo simulations for {self.ticker}..."
        )

        simulation_results = np.zeros((num_days, num_simulations))

        for i in range(num_simulations):
            # Generate random daily returns based on normal distribution
            daily_returns = np.random.normal(self.mean_return, self.std_dev, num_days)

            # Calculate price path
            price_path = np.zeros(num_days)
            price_path[0] = self.last_price * (1 + daily_returns[0])

            for t in range(1, num_days):
                price_path[t] = price_path[t - 1] * (1 + daily_returns[t])

            simulation_results[:, i] = price_path

            # Add some artificial processing delay to simulate heavy computation if needed
            # But calculating 5000 paths for 252 days is already decent work
            if i % 1000 == 0:
                self._intensive_calculation_stub()

        return simulation_results

    def _intensive_calculation_stub(self):
        """A stub function to burn some CPU cycles without sleeping."""
        _ = sum(math.sqrt(x) for x in range(10000))


class PortfolioOptimizer:
    """Class to optimize portfolio allocation (mock implementation)."""

    def __init__(self, tickers, returns_df):
        self.tickers = tickers
        self.returns_df = returns_df

    def optimize(self, num_portfolios=10000):
        """Finds the optimal portfolio using Sharpe Ratio via random sampling."""
        logger.info(f"Optimizing portfolio via {num_portfolios} random allocations...")

        results = np.zeros((3, num_portfolios))
        weights_record = []

        for i in range(num_portfolios):
            weights = np.random.random(len(self.tickers))
            weights /= np.sum(weights)
            weights_record.append(weights)

            portfolio_return = np.sum(self.returns_df.mean() * weights) * 252
            portfolio_std_dev = np.sqrt(
                np.dot(weights.T, np.dot(self.returns_df.cov() * 252, weights))
            )

            results[0, i] = portfolio_return
            results[1, i] = portfolio_std_dev
            results[2, i] = results[0, i] / results[1, i]  # Sharpe Ratio

        max_sharpe_idx = np.argmax(results[2])
        optimal_weights = weights_record[max_sharpe_idx]

        return optimal_weights, results


def fetch_data(tickers):
    """Fetches historical data for given tickers."""
    logger.info(
        f"Fetching data for {len(tickers)} tickers from {START_DATE} to {END_DATE}..."
    )
    data = {}
    for ticker in tickers:
        try:
            # Add a small delay/retry logic if needed, but yfinance handles most
            df = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False)
            if not df.empty:
                # Use 'Adj Close' if available, else 'Close'
                if "Adj Close" in df.columns:
                    data[ticker] = df["Adj Close"]
                elif "Close" in df.columns:
                    data[ticker] = df["Close"]
            else:
                logger.warning(f"No data found for {ticker}")
        except Exception as e:
            logger.error(f"Error fetching {ticker}: {e}")
    return pd.DataFrame(data)


def main():
    start_time = time.time()
    logger.info("Starting Stock Market Analysis Job")
    logger.info(f"Running as user: {getpass.getuser()}")
    logger.info(f"Python version: {sys.version}")

    # 1. Fetch Data
    stock_data = fetch_data(TICKERS)
    if stock_data.empty:
        logger.error("No stock data fetched. Exiting.")
        sys.exit(1)

    logger.info(f"Successfully fetched data for {stock_data.shape[1]} tickers.")

    # 2. Technical Analysis per Ticker
    logger.info("Calculating technical indicators...")
    for ticker in stock_data.columns:
        series = stock_data[ticker]

        # Calculate indicators
        sma_50 = TechnicalIndicators.calculate_sma(series, 50)
        ema_20 = TechnicalIndicators.calculate_ema(series, 20)
        upper, lower = TechnicalIndicators.calculate_bollinger_bands(series)
        rsi = TechnicalIndicators.calculate_rsi(series)
        macd, signal = TechnicalIndicators.calculate_macd(series)

        # Log a summary for the latest day
        latest_price = series.iloc[-1]
        latest_rsi = rsi.iloc[-1]
        logger.info(f"{ticker}: Price=${latest_price:.2f}, RSI={latest_rsi:.2f}")

    # 3. Monte Carlo Simulations
    logger.info("Starting Monte Carlo Simulations...")
    forecasts = {}
    for ticker in stock_data.columns[
        :5
    ]:  # Run for first 5 to save some time but still be heavy
        simulator = MarketSimulator(ticker, stock_data[ticker])
        results = simulator.run_simulation(SIMULATION_RUNS, FORECAST_DAYS)

        mean_end_price = np.mean(results[-1, :])
        forecasts[ticker] = mean_end_price
        logger.info(
            f"{ticker}: Forecasted Mean Price in {FORECAST_DAYS} days = ${mean_end_price:.2f}"
        )

    # 4. Portfolio Optimization
    logger.info("Starting Portfolio Optimization...")
    # Calculate daily returns for all stocks
    returns_df = stock_data.pct_change().dropna()

    optimizer = PortfolioOptimizer(stock_data.columns, returns_df)
    optimal_weights, results = optimizer.optimize(
        num_portfolios=20000
    )  # Increased iterations

    logger.info("Optimal Portfolio Allocation:")
    for ticker, weight in zip(stock_data.columns, optimal_weights):
        if weight > 0.01:  # Only show significant weights
            logger.info(f"  {ticker}: {weight * 100:.2f}%")

    end_time = time.time()
    duration = end_time - start_time
    logger.info(f"Analysis completed successfully in {duration:.2f} seconds.")


if __name__ == "__main__":
    main()
