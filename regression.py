 
import pandas as pd
import numpy as np
from scipy.stats import skew
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import statsmodels
import statsmodels.formula.api as sm
import seaborn as sns
import datetime
from matplotlib.backends.backend_pdf import PdfPages
import operator


directory = 'C:\\Code\\Fraud\\'
test_name = 'test'
train_name = 'train'
Y_var = ['SalePrice']

def dataclean(train,test):
	all_data = pd.read_csv(directory + train +".csv")
	twovars = all_data[['BsmtFinType1','BsmtFinType2']]
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
	return (all_data)

def regression(dataset,dependents):
	#pass in dependendents as array of names
	for variables in dependents:
		vstring = variables + '~'
		print (vstring)
		new_deps = {}
		for i in dataset:
			if i == str(variables):
				pass
			else:
				vstring += '+' + i
		result = sm.ols(formula=vstring,data=dataset).fit()
		#print (result.params)
		print (result.summary())
		for key, value in result.pvalues.iteritems():
			new_deps[key] = value

		print ('Done with regression')
		print (new_deps)
		sorted_deps = sorted(new_deps.items(), key=operator.itemgetter(1))
		print (sorted_deps)
		print (sorted_deps[0][0])
		return sorted_deps

def output(dataset,dependents,independents):

	vv = 1
	with PdfPages('multipage_pdf.pdf') as pdf:
		while vv <= 5:
			dset = dataset.sort_values(dependents[vv][0])
			y = dset[independents]
			x = dset[dependents[vv][0]]
			plt.figure(figsize=(3, 3))
			plt.plot(x,y, 'r-o')
			plt.title(str(dependents[vv][0]))
			pdf.savefig()  # saves the current figure into a pdf page
			plt.close()
			vv =vv + 1

		# We can also set the file's metadata via the PdfPages object:
		d = pdf.infodict()
		d['Title'] = 'Multipage PDF Example'
		d['Author'] = u'Jouni K. Sepp\xe4nen'
		d['Subject'] = 'How to create a multipage pdf file and set its metadata'
		d['Keywords'] = 'PdfPages multipage keywords author title subject'
		d['CreationDate'] = datetime.datetime(2009, 11, 13)
		d['ModDate'] = datetime.datetime.today()


cleaned_dataset = dataclean(train_name,test_name)
dependents = regression(cleaned_dataset,Y_var)
output(cleaned_dataset,dependents,'SalePrice')



# Create the PdfPages object to which we will save the pages:
# The with statement makes sure that the PdfPages object is closed properly at
# the end of the block, even if an Exception occurs.
