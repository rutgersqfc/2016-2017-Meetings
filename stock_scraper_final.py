from yahoo_finance import Share
from pprint import pprint
import urllib2, string, time, os
from bs4 import BeautifulSoup, NavigableString

####This method isn't too important - just sets up bs4's html reader
def initOpener():
    opener = urllib2.build_opener()
    opener.addheaders = [('User-agent', 'Mozilla/5.0')]
    return opener

####This method will return a list of all the S&P 500's ticker strings.
####It is also set up for you to retrieve other additional information.
def getSP500Dictionary():
    stockTickerUrl = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'

    usableStockTickerURL = initOpener().open(stockTickerUrl).read()

    stockTickerSoup = BeautifulSoup(usableStockTickerURL, 'html.parser')

    stockTickerTable = stockTickerSoup.find('table')

    ####TO FIND # OF ROWS
    stockTickerRows = stockTickerTable.find_all('tr')
    # print(len(stockTickerRows))

    baseWikiLink = 'https://en.wikipedia.org'

    SP500List = []

    ####SP500Table can be used if you want to map additional
    ####information to each ticker. An example is shown below.
    SP500Table = dict()

    ####FOR EACH ROW FIND ALL INFORMATION AND PUT IN SP500Table
    for stockTickerRow in stockTickerRows: 
	stockTickerColumns = stockTickerRow.find_all('td')
	counter = 1
	for element in stockTickerColumns:
	    ####Stock Ticker
            if (counter % 8) == 1:
		stockTicker = element.get_text().strip().encode('ascii', 'ignore')
		SP500List.append(stockTicker)
		SP500Table[stockTicker] = dict()
	    ####Company Name and Link
	    elif (counter % 8) == 2:
		SP500Table[stockTicker]['Name'] = element.get_text().strip()
		link = element.find('a')
		SP500Table[stockTicker]['Link'] = baseWikiLink + link['href']
	    ####Sector
	    elif (counter % 8) == 4:
		SP500Table[stockTicker]['Sector'] = element.get_text().strip()
	    ####Sub Sector	
	    elif (counter % 8) == 5:
		SP500Table[stockTicker]['SubSector'] = element.get_text().strip()
	    ####HQ
	    elif (counter % 8) == 6:
		SP500Table[stockTicker]['HQ'] = element.get_text().strip()
	    counter = counter + 1

    #pprint(SP500Table)
    return SP500List		

####Input list of stock tickers and retrieve data from yahoo finance
def get_stock_info(tickers):
    for ticker in tickers:
        try:
            stock = Share(ticker)
            print ticker + '\'s opening price today: ' + stock.get_open()
            print ticker + '\'s current price today: ' + stock.get_price()
    ##        print ticker + '\'s historical data: \n'
    ##        print stock.get_historical('2016-10-01', '2016-10-31')
            print '\n'
        except Exception as e:
            print 'Could not retrieve data for ' + ticker + '\n'

    return


####Run the above methods below

SP500_stocks = getSP500Dictionary()
get_stock_info(SP500_stocks)

