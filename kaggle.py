 
from urllib import request
import datetime,re
import pandas as pd
 

directory = 'C:\\Code\\Fraud\\'
df_train = pd.DataFrame.from_csv(directory + "train.csv")
df_test = pd.DataFrame.from_csv(directory + "test.csv")

 
print (df_train.head())
print (df_test.head())

columns = df_train.columns.tolist()

print (columns)
