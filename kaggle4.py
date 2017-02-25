 
import pandas as pd
import numpy as np
from scipy.stats import skew
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import statsmodels
import seaborn as sns

directory = 'C:\\Code\\Fraud\\'
train = pd.read_csv(directory + "train.csv")
test = pd.read_csv(directory + "test.csv")
#Puts it together (easier for preprocessing)
all_data = pd.concat((train.loc[:,'MSSubClass':'SaleCondition'],
                      test.loc[:,'MSSubClass':'SaleCondition']))

# train = train.sort_values('LotFrontage')
# y = train['SalePrice']
# x=train['LotFrontage']
# plt.plot(x, y)
# plt.show()
#Preprocessing, I stole this from a tutorial, so we'll have to revisit

numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index
skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness
skewed_feats = skewed_feats[skewed_feats > 0.75]
skewed_feats = skewed_feats.index
all_data[skewed_feats] = np.log1p(all_data[skewed_feats])
all_data = pd.get_dummies(all_data)
all_data = all_data.fillna(all_data.mean())


for i in all_data:
	print (i)
	
import statsmodels.formula.api as sm
result = sm.ols(formula='SalePrice~LotFrontage + TotRmsAbvGrd + LotArea',data=train).fit()
print (result.params)
print (result.summary())
