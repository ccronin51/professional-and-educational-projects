''' Project 1 '''
# reading in data
df = pd.read_csv('eod-quotemedia.csv', parse_dates=['date'], index_col=False)

# makes each date an unique row and puts the tickers as columns
close = df.reset_index().pivot(index='date', columns='ticker', values='adj_close')

# resample the daily adjusted closing prices into monthly buckets, and select the last observation of each month
close_prices.resample(freq).last()

# log returns
def compute_log_returns(prices):
    return np.log(prices) - np.log(prices.shift())

# Prices:    
# Prices              NQE        ETIW          GCKA         AIL        RSRI
# 2008-08-31  21.05081048 17.01384381   10.98450376 11.24809343 12.96171273

monthly_close_returns = compute_log_returns(monthly_close)

# shift returns
def shift_returns(returns, shift_n):
    return returns.shift(shift_n)

prev_returns = shift_returns(monthly_close_returns, 1)
lookahead_returns = shift_returns(monthly_close_returns, -1)    

def get_top_n(prev_returns, top_n):
    index = prev_returns.index
    output = (prev_returns.stack().groupby(level=0).rank(ascending=False) <= top_n).unstack()
    output = output.reindex(index).fillna(0).astype(int)
    return output

# index = prev_returns.index
# print(prev_returns)
#                   JNLR         HPU        NZMJ        LDOX         DOW
# 2008-10-31  3.13172138  0.72709204  5.76874778  1.77557845  0.04098317
# 2008-11-30 -3.78816218 -0.67583590 -4.95433863 -1.67093250 -0.24929051    
# print(prev_returns.index)
# DatetimeIndex(['2008-08-31', '2008-09-30', '2008-10-31', '2008-11-30'], dtype='datetime64[ns]', freq=None

# output = (prev_returns.stack().groupby(level=0).rank(ascending=False) <= top_n).unstack()
#              JNLR    HPU   NZMJ  LDOX    DOW
# 2008-10-31   True  False   True  True  False
# 2008-11-30  False   True  False  True   True

# output = output.reindex(index).fillna(0).astype(int)
#             ZQY  IYW  FPMN  CSOF  OGEW
# 2008-08-31    0    0     0     0     0
# 2008-09-30    0    0     0     0     0
# 2008-10-31    1    0     1     1     0
# 2008-11-30    0    1     0     1     1

top_bottom_n = 50
df_long = get_top_n(prev_returns, top_bottom_n)
df_short = get_top_n(-1*prev_returns, top_bottom_n) 
# take say -2 log return or very low one, flippling the sign will make it a high positive to place a 1 next to

print("df long:")
# 2013-09-30  1    0    0     1     0    0    0    0     0    0  ...   0     0   
# 2013-10-31  0    1    0     0     0    0    0    0     1    0  ...   0     0   
print("df short:")
# 2013-09-30  0    1    0     0     0    0    1    0     0    0  ...   0     0   
# 2013-10-31  0    0    0     1     0    0    0    0     0    0  ...   0     0   
print("df long - short:")
# 2013-09-30  1   -1    0     1     0    0   -1    0     0    0  ...   0     0   
# 2013-10-31  0    1    0    -1     0    0    0    0     1    0  ...   0     0   

def portfolio_returns(df_long, df_short, lookahead_returns, n_stocks):
    portfolio_returns = lookahead_returns*(df_long-df_short)/(n_stocks)
    return portfolio_returns   

# one-sample, one-sided t-test on the observed mean return, to see if we can reject  𝐻0
from scipy import stats
def analyze_alpha(expected_portfolio_returns_by_date):
    t, pv_two_sided = stats.ttest_1samp(expected_portfolio_returns_by_date,0)
    return t, pv_two_sided/2

