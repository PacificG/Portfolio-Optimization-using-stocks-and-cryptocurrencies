import streamlit as st
import pandas as pd
import pypfopt
import matplotlib.pyplot as plt

# set title as Portfolio Optimization using stocks and cryptocurrencies
st.title('Portfolio Optimization using stocks and cryptocurrencies')

# set header as Portfolio Optimization using stocks and cryptocurrencies
st.header('Portfolio Optimization using stocks and cryptocurrencies')

df = pd.read_csv('all_stocks_5yr.csv')

#show a list of stocks and cryptocurrencies as a multiple dropdown
st.sidebar.header('Select stocks and cryptocurrencies')
stocks = st.sidebar.multiselect('Select stocks and cryptocurrencies', df[" Name"].unique())

#write a funtion to get a list of stocks from yahoo finance
def get_data(stocks, start_date='1/1/2019', end_date='1/1/2020'):
    #get the data from yahoo finance
    df = pd.DataFrame(index=pd.date_range(start_date, end_date))
    for stock in stocks:
        df[stock] = pd.DataFrame(pd.read_csv('https://query1.finance.yahoo.com/v7/finance/download/' + stock + '?period1=1546300800&period2=1577836800&interval=1d&events=history&crumb=QYX2KZL5Q8'))['Close']
    return df

# #show a list of cryptocurrencies as a multiple dropdown
# st.sidebar.header('Select cryptocurrencies')
# cryptos = st.sidebar.multiselect('Select cryptocurrencies', ['ADA-USD', 'ETH-USD', 'BNB-USD', 'BTC-USD', 'DOT1-USD', 'HEX-USD', 'USDT-USD', 'SOL1-USD', 'XRP-USD'])

# #show a slider to select a range of time periods
# st.sidebar.header('Select time period')
# time_period = st.sidebar.slider('Select time period', 1, 10, 5)

#show a text input to get number of years we want to do
st.sidebar.header('Select number of years')
years = st.sidebar.text_input('Select number of years', '5')

#show a text input to get input balance
st.sidebar.header('Enter initial balance')
init_balance = st.sidebar.text_input('Enter initial balance', '$100,000')

#show a text input to get input trading cost
st.sidebar.header('Enter trading cost')
trading_cost = st.sidebar.text_input('Enter trading cost', '0.0025')

#show options for optimization method in PyPortfolioOpt package
st.sidebar.header('Select optimization method')
optimization_method = st.sidebar.radio('Select optimization method', ['Max Sharpe Ratio', 'Min Volatility'])

#show a text input to get input risk free rate
st.sidebar.header('Enter risk free rate')
risk_free_rate = st.sidebar.text_input('Enter risk free rate', '0.00')

#read dasboard.csv data


#filter the dataframe based on the selected stocks
df = df[df[' Name'].isin(stocks)]

df = df[['date', 'close', ' Name']]

df = df.pivot_table(index='date', columns=' Name', values='close')

#plot the data
# st.write(df.plot(figsize=(15, 5)))


from pypfopt.expected_returns import mean_historical_return
from pypfopt.risk_models import CovarianceShrinkage

mu = mean_historical_return(df)
S = CovarianceShrinkage(df).ledoit_wolf()



if optimization_method == 'Max Sharpe Ratio':
    #get the max sharpe ratio portfolio
    weights = pf.max_sharpe()
    st.write('Max Sharpe Ratio')
    from pypfopt.efficient_frontier import EfficientFrontier
    #get the mean variance portfolio
    pf = EfficientFrontier(mu, S)
    # st.write(p.plotting.plot_efficient_frontier(pf, show_covariance=True))
    # st.write(pf.plotting.plot_weights(weights))
    st.write(weights = pf.max_sharpe())
if optimization_method == 'Min Volatility':
    from pypfopt.efficient_frontier import EfficientFrontier
    #get the mean variance portfolio
    pf = EfficientFrontier(mu, S)
    #get the min volatility portfolio
    weights = pf.min_volatility()
    st.write(weights)
    # st.write(pypfopt.plotting.plot_weights(weights))



# st.write(df)




