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























