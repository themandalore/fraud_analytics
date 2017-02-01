 
import pandas as pd
import numpy as np
from scipy.stats import skew
from sklearn.linear_model import LinearRegression

directory = 'C:\\Code\\Fraud\\'
train = pd.read_csv(directory + "train.csv")
test = pd.read_csv(directory + "test.csv")
#Puts it together (easier for preprocessing)
all_data = pd.concat((train.loc[:,'MSSubClass':'SaleCondition'],
                      test.loc[:,'MSSubClass':'SaleCondition']))
 
#Preprocessing, I stole this from a tutorial, so we'll have to revisit

prices = train['SalePrice']
train.drop('SalePrice', axis=1, inplace=True)
numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index
skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness
skewed_feats = skewed_feats[skewed_feats > 0.75]
skewed_feats = skewed_feats.index
all_data[skewed_feats] = np.log1p(all_data[skewed_feats])
all_data = pd.get_dummies(all_data)
all_data = all_data.fillna(all_data.mean())

X = all_data.as_matrix()
X = np.nan_to_num(X)
X_train = X[ : int(train.shape[0]*0.8)]
prices_train = prices[ : int( train.shape[0] * 0.8 )]
X_dev = X[int(train.shape[0]*0.8) : train.shape[0]]
prices_dev = prices[int(train.shape[0]*0.8) : ]
X_test = X[train.shape[0] : ]
prices_train.shape

#Linear regression model
lr = LinearRegression()
lr.fit(X_train, prices_train)


#gets sum of squared errors and built in accuracy function
Y = lr.predict(X_dev)
sq_diff = np.square(np.log(prices_dev) - np.log(Y))
error = np.sqrt(np.sum(sq_diff) / prices_dev.shape[0])
accuracy = lr.score(X_dev,prices_dev)
print (accuracy , error)


#Outputs the right format

Y = lr.predict(X_test)
out = pd.DataFrame()
out['Id'] = [i for i in range(X_train.shape[0]+X_dev.shape[0]+1,X_train.shape[0]+X_dev.shape[0]+X_test.shape[0]+1)]
out['SalePrice'] = Y
out.to_csv('output.csv', index=False)