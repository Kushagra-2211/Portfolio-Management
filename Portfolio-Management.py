#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 23 19:42:12 2026

@author: kush
"""

import pandas as pd
import numpy as np
import datetime as dt
import random
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import plotly.io as pio
import os
pio.renderers.default = 'browser'
#%%
print(os.getcwd())
#%%
close_price_df = pd.read_csv("data/stock_prices.csv")
#%% Converting to datetime 
close_price_df["Date"] = pd.to_datetime(close_price_df["Date"])
close_price_df.set_index("Date", drop=True, inplace= True)
close_price_df.head()
#%%
# calculating percentage daily returns
daily_returns_df = close_price_df.pct_change(1) * 100
daily_returns_df.head()
#%%
daily_returns_df.fillna(0, inplace= True)
daily_returns_df.head()
#%%
#Defining a function to plot financial data
def financial_data_plot(df,title):
    fig = px.line(title=title)
    for col in df.columns:
        fig.add_scatter(x = df.index, y = df[col], name = col )
        fig.update_traces(line_width = 5)
        fig.update_layout({'plot_bgcolor': "white"})
    fig.show()
#%% Plotting stock prices
financial_data_plot(close_price_df, "Stock Prices")
#%% Plotting daily returns
financial_data_plot(daily_returns_df, "Percentage Daily Returns (%)")
#%%
# Heatmap plot of correlation coefficients of stocks under consideration
plt.figure(figsize = (10, 8))
sns.heatmap(daily_returns_df.corr(), annot = True);
#%%
# Defining a function to scale prices of stocks for apt comparision
def price_scaling(raw_prices_df):
    return raw_prices_df / raw_prices_df.iloc[0]
#%%
portfolio_df = close_price_df.copy()
scaled_prices_df = price_scaling(portfolio_df)
scaled_prices_df.head()
#%%
# defining a function to generate random weights
# Method 1 
def random_weights_generator_1(n):
    weights = np.random.rand(n)
    return weights/weights.sum()  # Scaling so that sum of weights = 1
#%%
# Method 2 - Better for portfolio management
def portfolio_weight_generation(n):
    return np.random.dirichlet(np.ones(n))
#%%
# w = portfolio_weight_generation(9)
# w
# w.sum()
#%%
# generating portfolio weights
weights = portfolio_weight_generation(len(close_price_df.columns))
#%%
scaled_prices_df.head()
#%%
# Defining a function that receives the following arguments: 
      # (1) Stocks closing prices
      # (2) Random weights 
      # (3) Initial investment amount
# The function will return a DataFrame that contains the following:
      # (1) Daily value (position) of each individual stock over the specified time period
      # (2) Total daily value of the portfolio 
      # (3) Percentage daily return 

def assest_allocation(df, weights, initial_investment):
    portfolio_df = df.copy()
    scaled_df = price_scaling(portfolio_df)
    
    investment_df = scaled_df * weights * initial_investment
    
    investment_df["Portfolio Value"] = investment_df.sum(axis = 1)
    investment_df["% Change in Portfolio Value"] = investment_df["Portfolio Value"].pct_change(1) * 100
    
    return investment_df
#%%
initial_investment = 1000000
investment_df = assest_allocation(close_price_df, weights, initial_investment)
#%% 
# plotting an interactive graph of total portfolio value over time
financial_data_plot(investment_df[["Portfolio Value"]], 'Total Portfolio Value [$]')
#%%
# plotting an interactive graph of portfolio postions over time
financial_data_plot(investment_df.drop(columns = ["Portfolio Value","% Change in Portfolio Value"]),  'Total Portfolio Value [$]')
#%%
# Defining a simulation engine function 
# The function receives: 
    # (1) portfolio weights
    # (2) initial investment amount
# The function performs asset allocation and calculates portfolio statistical metrics including Sharpe ratio
# The function returns: 
    # (1) Expected portfolio return 
    # (2) Expected volatility 
    # (3) Sharpe ratio 
    # (4) Return on investment 
    # (5) Final portfolio value in dollars
def simulation_engine(df, weights, initial_investment, rf):
    investment_df = assest_allocation(df, weights, initial_investment)
    
    portfolio_start = investment_df["Portfolio Value"].iloc[0]
    portfolio_end = investment_df["Portfolio Value"].iloc[-1]
    
    return_on_investment = (portfolio_end - portfolio_start) / portfolio_start * 100
    
    daily_returns_df = df.pct_change().dropna()
    expected_portfolio_return = weights @ daily_returns_df.mean()
    expected_portfolio_return = expected_portfolio_return * 252

    covariance = daily_returns_df.cov() * 252
    expected_portfolio_variance = weights @ covariance @ weights
    expected_portfolio_volatility = np.sqrt(expected_portfolio_variance)
    
    sharpe_ratio = (expected_portfolio_return - rf) / expected_portfolio_volatility
    
    return expected_portfolio_return, expected_portfolio_volatility, sharpe_ratio, return_on_investment, portfolio_end
#%%
initial_investment = 10000000
rf = 0.03

portfolio_metrics = simulation_engine(close_price_df, weights, initial_investment, rf)
print('Expected Portfolio Annual Return = {:.2f}%'.format(portfolio_metrics[0] * 100))
print('Portfolio Standard Deviation (Volatility) = {:.2f}%'.format(portfolio_metrics[1] * 100))
print('Sharpe Ratio = {:.2f}'.format(portfolio_metrics[2]))
print('Return on Investment = {:.2f}%'.format(portfolio_metrics[3]))
print('Portfolio Final Value = ${:.2f}'.format(portfolio_metrics[4]))

### Output 
# Expected Portfolio Annual Return = 16.58%
# Portfolio Standard Deviation (Volatility) = 21.95%
# Sharpe Ratio = 0.62
# Return on Investment = 213.00%
# Portfolio Final Value = $31300499.52

#%%
# Defining a setup for Morte Carlo Simulations
# Initializing placeholders to store weights, sharpe ratios, expected return and vol, and ROI
# Calling simulation_engine function to fill the placeholders


# # Placeholder to store weights
# weights_mc = np.zeros((sim_runs, n))

# # Placeholder to store sharpe ratios
# sharpe_ratio_mc = np.zeros(sim_runs)

# # Placeholder to store expected returns
# expected_return_mc = np.zeros(sim_runs)

# # Placeholder to store expected volatility
# expected_vol_mc = np.zeros(sim_runs)

# # Placeholder to store return on investment
# roi_mc = np.zeros(sim_runs)

# # Placeholder to store final portfolio value
# portfolio_end_value = np.zeros(sim_runs)

def monte_carlo_runs(sim_runs, df, initial_investment, rf):
    
    n = len(df.columns)
    
    weights_mc = np.zeros((sim_runs, n))
    sharpe_ratio_mc = np.zeros(sim_runs)
    expected_return_mc = np.zeros(sim_runs)
    expected_vol_mc = np.zeros(sim_runs)
    roi_mc = np.zeros(sim_runs)
    portfolio_end_value = np.zeros(sim_runs)
    
    for i in range(sim_runs):
        weights = portfolio_weight_generation(n)
        
        weights_mc[i, :] = weights
        
        expected_return_mc[i], expected_vol_mc[i], sharpe_ratio_mc[i], roi_mc[i], portfolio_end_value[i] = simulation_engine(df, weights, initial_investment, rf)
    
    sharpe_ratio_max = sharpe_ratio_mc.max()
    sim_run_number = sharpe_ratio_mc.argmax()
    weights_max = weights_mc[sim_run_number, :]
    
    sim_out_df = pd.DataFrame({
        "Volatility": expected_vol_mc,
        "Portfolio_Return": expected_return_mc,
        "Sharpe_Ratio": sharpe_ratio_mc,
        "ROI": roi_mc,
        "Final_Value": portfolio_end_value
    })
    
    return weights_max, sharpe_ratio_max, sim_out_df

#%% 

# no.of simulation runs
sim_runs = 10000
initial_investment = 10000000
rf = 0.03
df = close_price_df

weights_max, sr_max, sim_out_df = monte_carlo_runs(sim_runs, df, initial_investment, rf)

portfolio_metrics = simulation_engine(df, weights_max, initial_investment, rf)
print('Best Portfolio Metrics Based on {} Monte Carlo Simulation Runs:'.format(sim_runs))
print('Weights for best portfolio metrics = {}'.format(weights_max))
print('Max Expected Portfolio Annual Return = {:.2f}%'.format(portfolio_metrics[0] * 100))
print('Max Portfolio Standard Deviation (Volatility) = {:.2f}%'.format(portfolio_metrics[1] * 100))
print('Max Sharpe Ratio = {:.2f}'.format(portfolio_metrics[2]))
print('Max Return on Investment = {:.2f}%'.format(portfolio_metrics[3]))
print('Max Portfolio Final Value = ${:.2f}'.format(portfolio_metrics[4]))   
    
## Output   
# Best Portfolio Metrics Based on 10000 Monte Carlo Simulation Runs:
# Weights for best portfolio metrics = [0.27482217 0.01623358 0.40617641 0.05014484 0.005778   0.07666848
#  0.02152889 0.00935884 0.09456712 0.04472166]
# Max Expected Portfolio Annual Return = 19.94%
# Max Portfolio Standard Deviation (Volatility) = 20.30%
# Max Sharpe Ratio = 0.83
# Max Return on Investment = 336.23%
# Max Portfolio Final Value = $43622552.14

#%%
# Plotting Efficient Frontier

# sim_runs = 10000
# initial_investment = 10000000
# rf = 0.03
# df = close_price_df

# weights_max, sr_max, sim_out_df = monte_carlo_runs(sim_runs, df, initial_investment, rf)

# portfolio_metrics = simulation_engine(df, weights_max, initial_investment, rf)

import plotly.graph_objects as go

fig = px.scatter(sim_out_df, x = 'Volatility', y = 'Portfolio_Return', color = 'Sharpe_Ratio', size = 'Sharpe_Ratio', hover_data = ['Sharpe_Ratio'] )
fig.add_trace(go.Scatter(x = [portfolio_metrics[1]], y = [portfolio_metrics[0]], mode = 'markers', name = 'Optimal Point', marker = dict(size=[40], color = 'red')))
fig.update_layout(coloraxis_colorbar = dict(y = 0.7, dtick = 5))
fig.update_layout({'plot_bgcolor': "white"})
fig.show()













