######Jim Cramer's FANG Stragegy
######Implemented By Jon Joh
######Rutgers QFC Spring 2016

def initialize(context):
    
    #### Portfolio of Facebook, Amazon, Netflix, and Alphabet (GOOG) (FANG)
    #### sid -> stock ID
    ### context is a global object
    context.myPortfolio = [sid(42950), sid(16841), sid(23709), sid(46631)]
    context.weight = 0.95/len(context.myPortfolio)

def handle_data(context, data):
    
    for security in context.myPortfolio:
        order_target_percent(security, context.weight)

# Commission, execution speed considerations
# This can also be hooked up to brokerages and AmeriTrade, RobinHood, etc accounts to engage in
# real time trading.
