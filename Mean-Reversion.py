'''
    Linear Regression Curves vs. Bollinger Bands
    If Close price is greater than average+n*deviation, go short
    If Close price is less than average+n*deviation, go long
    Both should close when you cross the average/mean
'''

import numpy as np
from scipy import stats
from quantopian.pipeline import Pipeline
from quantopian.algorithm import attach_pipeline, pipeline_output
from quantopian.pipeline.data.builtin import USEquityPricing
from quantopian.pipeline.factors import AverageDollarVolume


def initialize(context):
    # Create, register and name a pipeline in initialize.
    pipe = Pipeline()
    attach_pipeline(pipe, 'dollar_volume_10m_pipeline')

    # Construct a 100-day average dollar volume factor and add it to the pipeline.
    dollar_volume = AverageDollarVolume(window_length=100)
    pipe.add(dollar_volume, 'dollar_volume')

    #Create high dollar-volume filter to be the top 2% of stocks by dollar volume.
    high_dollar_volume = dollar_volume.percentile_between(99, 100)
    # Set the screen on the pipelines to filter out securities.
    pipe.set_screen(high_dollar_volume)

    context.dev_multiplier = 2
    context.max_notional = 1000000
    context.min_notional = -1000000
    context.days_traded = 0
    schedule_function(func=process_data_and_order, date_rule=date_rules.every_day())

def before_trading_start(context, data):
    # Pipeline_output returns the constructed dataframe.
    output = pipeline_output('dollar_volume_10m_pipeline')

    # sort the output. Most liquid stocks are at the top of the list,
    # and least liquid stocks are at the bottom
    sorted_output = output.sort('dollar_volume', ascending = False)

    context.my_securities = sorted_output.index

def process_data_and_order(context, data):
    context.days_traded += 1    

    dev_mult = context.dev_multiplier
    notional = context.portfolio.positions_value
    # Calls get_linear so that moving_average has something to reference by the time it is called
    linear = get_linear(context, data)
        
    # Only checks every 20 days
    if context.days_traded%20 == 0:
        try:
            for stock in context.my_securities:
                close = data.current(stock, "close")
                moving_average = linear[stock]
                stddev_history = data.history(stock, "price", 20, "1d")[:-1]
                moving_dev = stddev_history.std()
                band = moving_average + dev_mult*moving_dev
                # If close price is greater than band, short 5000 and if less, buy 5000
                if close > band and notional > context.min_notional:
                    order(stock, -5000)
                    log.debug("Shorting 5000 of " + str(stock))
                elif close < band and notional < context.max_notional:
                    order(stock, 5000)
                    log.debug("Going long 5000 of " + str(stock))
        except:
            return
  
# Linear regression curve that returns the intercept the curve
# Uses the past 20 days
def get_linear(context, data):
    days = [i for i in range(1,21)]
    stocks = {}
    for stock in context.my_securities:
        linear = stats.linregress(days, data.history(stock, "price", 20, "1d"))[1]
        stocks[stock] = linear
    return stocks
    
