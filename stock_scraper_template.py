from yahoo_finance import Share
import urllib2, string, time, os
from bs4 import BeautifulSoup, NavigableString

####This method isn't too important - just sets up bs4's html reader
def initOpener():
    opener = urllib2.build_opener()
    opener.addheaders = [('User-agent', 'Mozilla/5.0')]
    return opener



####This method will return a list of all the S&P 500's ticker strings.
def getSP500Dictionary():
    stockTickerUrl = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'

    usableStockTickerURL = initOpener().open(stockTickerUrl).read()
    
    stockTickerSoup = BeautifulSoup(usableStockTickerURL, 'html.parser')



####Input list of stock tickers and retrieve data from yahoo finance
def get_stock_info(tickers):




####Run the above methods below

SP500_stocks = getSP500Dictionary()
get_stock_info(SP500_stocks)
