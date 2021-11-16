*Work in progress...*

# Using Reinforcement Learning for Portfolio Optimization

## Introduction

The goal of this project is to use reinforcement learning to optimize the portfolio which consists of stocks and cryptocurrencies. The portfolio is optimized by using a neural network to predict the future value of the portfolio.

## Data

### Stock Data

Stock data is obtained from Yahoo Finance. We have collected stock data for S&P 500 companies. We have created a script to download the data from Yahoo Finance. The data for each company is stored in a separate CSV file and we have merged all the CSV files into one CSV file. The data is stored in the `utils/datasets/all_stocks_5yr.csv` file. we have used the data from the `utils/datasets/all_stocks_5yr.csv` file.

### Cryptocurrency Data

Cryptocurrency data is obtained from Yahoo Finance. We have collected top 9 cryptocurrencies on the basis of market capitalization. The data is obtained from the Yahoo Finance API. We have used the same script to download the data from Yahoo Finance. The data for each cryptocurrency is stored in a separate CSV file and we have merged all the CSV files into one CSV file. The data is stored in the `utils/datasets/all_crypto_5yr.csv` file. We have used the data from the `utils/datasets/all_crypto_5yr.csv` file.

*You can use the script to download last n years of data for a list of stocks or cryptocurrencies from Yahoo Finance. Follow the steps mentioned here[https://github.com/PacificG/PortfolioOptimizationStocksCrypto]*

### Stock & Cryptocurrency Price Prediction

We considering two model architectures for stock & cryptocurrencies price prediction. The first model is a LSTM based RNN model. The second model is a CNN.

### Steps to reproduce the results

Clone this repository using the following command:

```bash
git clone  https://github.com/PacificG/Portfolio-Optimization-using-stocks-and-cryptocurrencies.git
```

In the `Portfolio-Optimization-using-stocks-and-cryptocurrencies` directory, run the following command to set up the environment:

```bash
virtualenv -p python3 venv
```

Activate the virtual environment using the following command:

```bash
source venv/bin/activate # for linux
venv\Scripts\activate # for windows
```

Install the required packages using the following command:

```bash
pip3 install -r requirements.txt
```




