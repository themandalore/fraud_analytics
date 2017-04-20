 
import pandas as pd
import numpy as np
from scipy.stats import skew
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from matplotlib.backends.backend_pdf import PdfPages
import operator
from scipy.stats import skew
from sklearn.cross_validation import KFold
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, Lasso
from math import sqrt

directory = 'C:\\Code\\Fraud\\'
test_name = 'test'
train_name = 'train'
Y_var = 'SalePrice'
train = pd.read_csv(directory + train_name +".csv")
test = pd.read_csv(directory + test_name +".csv")


y_train = np.log(train[Y_var]+1)
train.drop([Y_var], axis=1, inplace=True)
all_data = train.append(test)
numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index
skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness
skewed_feats = skewed_feats[skewed_feats > 0.75]
skewed_feats = skewed_feats.index
all_data[skewed_feats] = np.log1p(all_data[skewed_feats])
all_data = pd.get_dummies(all_data)
all_data = all_data.fillna(all_data.mean())
# print (twovars['BsmtFinType1'].unique())
# df = train.copy()

for  i in all_data:
	if i.isalpha():
		pass
	else:
		new_col = ''.join(filter(str.isalpha, i))
		all_data.rename(columns={i: new_col}, inplace=True)

# print (twovars.groupby('BsmtFinType1').count())
print ('Done with dataclean')

class SklearnWrapper(object):
	def __init__(self, clf, seed=0, params=True):
	    params['random_state'] = seed
	    self.clf = clf(**params)

	def train(self, x_train, y_train):
	    xxx = self.clf.fit(x_train, y_train)
	    return xxx.coef_

	def predict(self, x):
	    return self.clf.predict(x)

def get_oof(clf):
	oof_train = np.zeros((ntrain,))
	oof_test = np.zeros((ntest,))
	oof_test_skf = np.empty((NFOLDS, ntest))
	coefs = []
	errors = []

	for i, (train_index, test_index) in enumerate(kf):
		x_tr = x_train[train_index]
		y_tr = y_train[train_index]
		x_te = x_train[test_index]

		coo = clf.train(x_tr, y_tr)
		oof_train[test_index] = clf.predict(x_te)
		oof_test_skf[i, :] = clf.predict(x_test)
		coefs.append(coo)
		#errors.append(mean_squared_error(clf.coef_, w))

	oof_test[:] = oof_test_skf.mean(axis=0)
	return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1),coefs,errors

TARGET = 'SalePrice'
NFOLDS = 8
SEED = 0
NROWS = None
#pass in dependendents as array of names
ntrain = train.shape[0]
ntest = test.shape[0]
kf = KFold(ntrain, n_folds=NFOLDS, shuffle=True, random_state=SEED)
x_train = np.array(all_data[:ntrain])
x_test = np.array(all_data[ntrain:])

et_params = {
'n_jobs': 16,
'n_estimators': 100,
'max_features': 0.5,
'max_depth': 12,
'min_samples_leaf': 2,
}

rf_params = {
    'n_jobs': 16,
    'n_estimators': 1000,
    'max_features': 0.2,
    'max_depth': 20,
    'min_samples_leaf': 2,
}


rd_params={
    'alpha': 10
}


ls_params={
    'alpha': 0.005
}


et = SklearnWrapper(clf=ExtraTreesRegressor, seed=SEED, params=et_params)
rf = SklearnWrapper(clf=RandomForestRegressor, seed=SEED, params=rf_params)
rd = SklearnWrapper(clf=Ridge, seed=SEED, params=rd_params)
ls = SklearnWrapper(clf=Lasso, seed=SEED, params=ls_params)


et_oof_train, et_oof_test,et_coefs,et_errors = get_oof(et)
rf_oof_train, rf_oof_test,rf_coefs,rf_errors = get_oof(rf)
rd_oof_train, rd_oof_test,rd_coefs,rd_errofs = get_oof(rd)
ls_oof_train, ls_oof_test,ls_coefs,ls_errors = get_oof(ls)


print("ET-CV: {}".format(sqrt(mean_squared_error(y_train, et_oof_train))))
print("RF-CV: {}".format(sqrt(mean_squared_error(y_train, rf_oof_train))))
print("RD-CV: {}".format(sqrt(mean_squared_error(y_train, rd_oof_train))))
print("LS-CV: {}".format(sqrt(mean_squared_error(y_train, ls_oof_train))))


x_train = np.concatenate((et_oof_train, rf_oof_train, rd_oof_train, ls_oof_train), axis=1)
x_test = np.concatenate((et_oof_test, rf_oof_test, rd_oof_test, ls_oof_test), axis=1)

print("{},{}".format(x_train.shape, x_test.shape))
sorted_deps = 'need more'
print (sorted_deps)




def output(dataset,dependents,independents):

	vv = 1
	with PdfPages('multipage_pdf.pdf') as pdf:
		while vv <= 5:
			# Display results
			plt.figure(figsize=(20, 6))

			plt.subplot(121)
			ax = plt.gca()
			ax.plot(alphas, coefs)
			ax.set_xscale('log')
			plt.xlabel('alpha')
			plt.ylabel('weights')
			plt.title('Ridge coefficients as a function of the regularization')
			plt.axis('tight')

			plt.subplot(122)
			ax = plt.gca()
			ax.plot(alphas, errors)
			ax.set_xscale('log')
			plt.xlabel('alpha')
			plt.ylabel('error')
			plt.title('Coefficient error as a function of the regularization')
			plt.axis('tight')

			plt.show()

			dset = dataset.sort_values(dependents[vv][0])
			y = dset[independents]
			x = dset[dependents[vv][0]]
			plt.figure(figsize=(3, 3))
			plt.plot(x,y, 'r-o')
			plt.title(str(dependents[vv][0]))
			pdf.savefig()  # saves the current figure into a pdf page
			plt.close()
			vv =vv + 1


#output(cleaned_dataset,dependents,'SalePrice')