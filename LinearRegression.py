import pandas as pd
import numpy as np
import math,datetime
from sklearn import preprocessing, model_selection
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')
df = pd.read_csv('/home/suman/Desktop/data/agriculture-development-bank-data.csv')
df = df[['maxprice','minprice','closingprice','previousclosing','amount']]
df['HL_PCT'] = (df['maxprice']-df['minprice'])/df['minprice']*100.0
df['PCT_change'] = (df['closingprice']-df['previousclosing'])/df['previousclosing']*100.0
df = df[['closingprice','HL_PCT','PCT_change','amount']]
forecast_col = 'closingprice'

df.fillna(-99999, inplace= True)
forecast_out = int(math.ceil(0.02*len(df)))
df['label'] = df[forecast_col].shift(-forecast_out)


X = np.array(df.drop(['label'],1))
X = preprocessing.scale(X)
X_lately = X[-forecast_out:]
X = X[:-forecast_out]
df.dropna(inplace = True)

Y = preprocessing.scale(Y)
Y = np.array(df['label'])

X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size = 0.2)

clf = LinearRegression()
clf.fit(X_train, Y_train)
accuracy = clf.score(X_test, Y_test)

forecast_set = clf.predict(X_lately)
print(forecast_set, accuracy, forecast_out)

df['forecast'] = np.nan

last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix+one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)]+[i]

df['closingprice'].plot()
df['forcast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()