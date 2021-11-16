*Work in progress...*

# Using Reinforcement Learning for Portfolio Optimization

## Introduction

The goal of this project is to use reinforcement learning to optimize the portfolio which consists of stocks and cryptocurrencies. The portfolio is optimized by using a neural network to predict the future value of the portfolio.

## Data

### Stock Data

Stock data is obtained from Yahoo Finance. The data is obtained from the Yahoo Finance API. We have created a 
script to download the data from Yahoo Finance. The data for each company is stored in a separate CSV file and 
we have merged all the CSV files into one CSV file. The data is stored in the `utils/datasets/all_stocks_5yr.csv` file. we have used the data from the `utils/datasets/all_stocks_5yr.csv` file.

### Cryptocurrency Data

Cryptocurrency data is obtained from Yahoo Finance. The data is obtained from the Yahoo Finance API. We have
used the same script to download the data from Yahoo Finance. The data for each cryptocurrency is stored in a separate CSV file and we have merged all the CSV files into one CSV file. The data is stored in the `utils/datasets/all_crypto_5yr.csv` file. We have used the data from the `utils/datasets/all_crypto_5yr.csv` file.

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




* Long short term memory
* Deep Deterministic Policy Gradient

## Dataset
* S&P500 dataset from kaggle found [here](https://www.kaggle.com/camnugent/sandp500)

## References
* [A Deep Reinforcement Learning Framework for the Financial Portfolio Management Problem](https://arxiv.org/abs/1706.10059)
* [Continuous control with deep reinforcement learning](https://arxiv.org/abs/1509.02971)
* The code is inspired by [CSCI 599 deep learning and its applications final project](https://github.com/vermouth1992/drl-portfolio-management) 
* The environment is inspired by [wassname/rl-portfolio-management](https://github.com/wassname/rl-portfolio-management)
* DDPG implementation is inspired by [Deep Deterministic Policy Gradients in TensorFlow](http://pemami4911.github.io/blog/2016/08/21/ddpg-rl.html)


## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

To install the required python packges, browse to the code folder then run ```pip install --user --requirement requirements.txt```

### Running the tests

ddpg_tests.ipynb is a step by step jupyter notebook showing the performance of the trained agent on unseen stocks. You can run this jupyter notebook directly without having to run the training since the training weights are saved in the weigths folder.


### Running the training 

To train the model from scratch and overwrite the saved weights, run stock_trading.py. This could take several hours.

## License

This project is licensed under the MIT License.
