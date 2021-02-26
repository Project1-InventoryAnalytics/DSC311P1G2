#import relevant modules
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

desired_width = 600
pd.set_option('display.width',desired_width)
pd.set_option('display.max_columns',10)

Data_Dir = "F:/NKU-DSC311 project files/"
filename = "SalesFINAL12312016.csv"
df = pd.read_csv(f"{Data_Dir}{filename}")
print(df.shape)
#print(type(df))
list_col = list(df.columns)
print(list_col)
#print(df.head())

#add columns for month and year to original data frame
df['sls_year'] = pd.DatetimeIndex(df['SalesDate']).year
df['sls_month'] = pd.DatetimeIndex(df['SalesDate']).month

#add columns with Store Name parsed from 'InventoryId' string
df['str_name'] = df.InventoryId.str.extract(r'_([^_]+)_', expand=True)


salesByStore = df.groupby(['Store','str_name','sls_year','sls_month']).agg({'SalesDollars':['sum','mean']}).reset_index()
salesByStore.to_csv('TotalSalesByStore.csv')

print(salesByStore)
#TODO: use pivot or melt feature to create a report that inserts subtotal lines by store by year.
