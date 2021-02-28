#import relevant modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing as pp
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

desired_width = 600
pd.set_option('display.width',desired_width)
pd.set_option('display.max_columns',20)

'''
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
'''

def main():

    # load data
    df_2017_purchase_prices = pd.read_csv("Datasets\\2017PurchasePricesDec.csv")
    df_beg_inv_final_2016 = pd.read_csv("Datasets\\BegInvFINAL12312016.csv")
    df_end_inv_final_2016 = pd.read_csv("Datasets\\EndInvFINAL12312016.csv")
    df_invoice_purchases_2016 = pd.read_csv("Datasets\\InvoicePurchases12312016.csv")
    df_purchases_final_2016 = pd.read_csv("Datasets\\PurchasesFINAL12312016.csv")
    df_sales_final_2016 = pd.read_csv("Datasets\\SalesFINAL12312016.csv")

    # clean data - address missing values, outliers, unrealistic values, ...

    # prep/transform data for analysis - create new dfs, add colums to existing dfs, ...


if __name__ == '__main__':
    main()
