#import relevant modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn import preprocessing as pp
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

desired_width = 600
pd.set_option('display.width',desired_width)
pd.set_option('display.max_columns',20)
pd.set_option('display.max_rows', 300)

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

def investigate_data(df):
    # look at shape
    print("Data Shape:")
    print(df.shape)
    print()

    # look at data columns and data types
    print("Column Details:")
    print(df.dtypes)
    print()

    # look at general statistics
    print("Data Statistics:")
    print(df.describe())
    print()

    # check for missing values
    print("Missing Values Check:")
    print(df.isna().sum())
    print()

    # check for outliers - greater than 2 standard deviations from mean
    print("Outliers Below Mean:")
    print((df < (df.mean() - (2 * df.std()))).sum())
    print("Outliers Above Mean:")
    print((df > (df.mean() + (2 * df.std()))).sum())
    print()


def clean_2017_purchase_prices(df):
    # address missing values
    df_missing = df[df.isna().any(axis=1)]
    print(df_missing.head())
    # based on the lack of identifying information for the product, remove this row
    df_cleaned = df.dropna()

    # address outliers
    df_outliers = df_cleaned[(df_cleaned["Price"] > (df_cleaned["Price"].mean() + (2 * df_cleaned["Price"].std()))) |
        (df_cleaned["PurchasePrice"] > (df_cleaned["PurchasePrice"].mean() + (2 * df_cleaned["PurchasePrice"].std())))]
    df_outliers.to_csv("Datasets\\Outliers_2017PurchasePricesDec.csv")
    # based on reviewing this list the data seems plausible, so outliers to remain

    return df_cleaned

def clean_invoice_purchases(df):
    # address outliers
    df_outliers = df[(df["Dollars"] > (df["Dollars"].mean() + (2 * df["Dollars"].std()))) |
        (df["Freight"] > (df["Freight"].mean() + (2 * df["Freight"].std()))) |
        (df["Quantity"] > (df["Quantity"].mean() + (2 * df["Quantity"].std())))]
    df_outliers.to_csv("Datasets\\Outliers_InvoicePurchases12312016.csv")
    # based on reviewing this list the data seems plausible, so outliers to remain

def clean_purchases_final_2016(df):
    # address missing values
    df_missing = df[df.isna().any(axis=1)]
    print(df_missing.head())
    # reviewing product description and purchase price in 2017PurchasePricesDec.csv, all missing are 750mL
    df_cleaned = df.copy(deep=True)
    df_cleaned.fillna("750mL", inplace=True)

    return df_cleaned

def prep_2017_purchase_prices(df):
    # add columns for profit (price-purchasePrice) and profit margin (profit/price)
    df_prepped = df.copy(deep=True)
    df_prepped["Profit"] = (df_prepped["Price"] - df_prepped["PurchasePrice"])
    df_prepped["ProfitMargin"] = (df_prepped["Profit"] / df_prepped["Price"])

    return df_prepped

def prep_invoice_purchases_2016(df):
    # add columns for year and month
    df_prepped = df.copy(deep=True)
    df_prepped["Purchase_Paid_Year"] = pd.DatetimeIndex(df_prepped["PayDate"]).year
    df_prepped["Purchase_Paid_Month"] = pd.DatetimeIndex(df_prepped["PayDate"]).month

    return df_prepped

def prep_purchases_final_2016_by_store(df):
    # add columns for year and month
    df_prepped = df.copy(deep=True)
    df_prepped["Purchase_Paid_Year"] = pd.DatetimeIndex(df_prepped["PayDate"]).year
    df_prepped["Purchase_Paid_Month"] = pd.DatetimeIndex(df_prepped["PayDate"]).month

    # add column with store name parsed from InventoryID
    df_prepped["Store_Name"] = df_prepped["InventoryId"].str.extract(r'_([^_]+)_', expand=True)

    df_prepped_by_store = df_prepped.groupby(["Store", "Store_Name", "Purchase_Paid_Year", "Purchase_Paid_Month"]).aggregate({"Dollars": ["sum", "mean"]})

    return df_prepped_by_store

def main():

    # load data
    df_2017_purchase_prices = pd.read_csv("Datasets\\2017PurchasePricesDec.csv")
    df_beg_inv_final_2016 = pd.read_csv("Datasets\\BegInvFINAL12312016.csv")
    df_end_inv_final_2016 = pd.read_csv("Datasets\\EndInvFINAL12312016.csv")
    df_invoice_purchases_2016 = pd.read_csv("Datasets\\InvoicePurchases12312016.csv")
    df_purchases_final_2016 = pd.read_csv("Datasets\\PurchasesFINAL12312016.csv")
    df_sales_final_2016 = pd.read_csv("Datasets\\SalesFINAL12312016.csv")
    '''
    # investigate data
    print("Investigate 2017 Purchase Prices Dataset " + ("*" * 60))
    investigate_data(df_2017_purchase_prices)
    
    print("Investigate Beginning Inventory Final 2016 Dataset " + ("*" * 60))
    investigate_data(df_beg_inv_final_2016)

    print("Investigate Ending Inventory Final 2016 Dataset " + ("*" * 60))
    investigate_data(df_end_inv_final_2016)
    
    print("Investigate Invoice Purchases 2016 Dataset " + ("*" * 60))
    investigate_data(df_invoice_purchases_2016)
    
    print("Investigate Purchases Final 2016 Dataset " + ("*" * 60))
    investigate_data(df_purchases_final_2016)
    
    print("Investigate Sales Final 2016 Dataset " + ("*" * 60))
    investigate_data(df_sales_final_2016)
    '''
    # clean data - address missing values, outliers, unrealistic values, ...
    df_2017_purchase_prices_cleaned = clean_2017_purchase_prices(df_2017_purchase_prices)
    clean_invoice_purchases(df_invoice_purchases_2016)
    df_purchases_final_2016_cleaned = clean_purchases_final_2016(df_purchases_final_2016)

    # prep/transform data for analysis - create new dfs, add columns to existing dfs, ...
    df_2017_purchase_prices_prepped = prep_2017_purchase_prices(df_2017_purchase_prices_cleaned)
    df_invoice_purchases_2016_prepped = prep_invoice_purchases_2016(df_invoice_purchases_2016)
    df_purchases_final_2016_prepped_by_store = prep_purchases_final_2016_by_store(df_purchases_final_2016_cleaned)


if __name__ == '__main__':
    main()
