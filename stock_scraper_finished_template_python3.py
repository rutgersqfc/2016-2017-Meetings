from yahoo_finance import Share
import urllib.request, string, time, os
from bs4 import BeautifulSoup, NavigableString


####This method isn't too important - just sets up bs4's html reader
def initOpener():
    opener = urllib.request.build_opener()
    opener.addheaders = [('User-agent', 'Mozilla/5.0')]
    return opener


####This method will return a list of all the S&P 500's ticker strings.
def getSP500Dictionary():
    stockTickerUrl = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'

    usableStockTickerURL = initOpener().open(stockTickerUrl).read()

    stockTickerSoup = BeautifulSoup(usableStockTickerURL, 'html.parser')

    stockTickerTable = stockTickerSoup.find('table')

    stockTickerRows = stockTickerTable.find_all('tr')

    SP500List = []


    for stockTickerRow in stockTickerRows:
        stockTickerColumns = stockTickerRow.find_all('td')
        counter = 1
        for element in stockTickerColumns:
	    ####Stock Ticker
            if (counter % 8) == 1:
                stockTicker = element.get_text().strip().encode('ascii', 'ignore')
                SP500List.append(stockTicker.decode('utf-8'))
            counter = counter + 1
    print(SP500List)
    return SP500List


####Input list of stock tickers and retrieve data from yahoo finance
def get_stock_info(tickers):
    for ticker in tickers:
        try:
            stock = Share(ticker)
            print((ticker + '\'s opening price today: ' + stock.get_open()))
            print((ticker + '\'s current price today: ' + stock.get_price()))
            print('\n')


        except Exception as e:
            print(('Could not retrieve data for ' + ticker + '\n'))


    return


####Run the above methods below

SP500_stocks = getSP500Dictionary()
get_stock_info(SP500_stocks)




