'''
DSC 311 Project 1
Group 2:
Jessica Lester
Procter Mercer
Tami Farber

Stage one of Spring 2021 DSC 311 project
Clean and prepare for analysis related sales, inventory,
and purchasing files for fictional client Bibitor LLC

'''
# import relevant modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn import preprocessing as pp

desired_width = 600
pd.set_option('display.width', desired_width)
pd.set_option('display.max_columns', 20)
pd.set_option('display.max_rows', 300)

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


def clean_2017_purchase_prices(df):
    # investigate dataset
    investigate_data(df)

    # address missing values
    df_missing = df[df.isna().any(axis=1)]
    df_missing.to_csv("Output\\Missing_Values_2017PurchasePricesDec.csv")
    # based on the lack of identifying information for the product, remove this row
    df_cleaned = df.dropna()

    # check for outliers using histogram for Price, Volume, PurchasePrice
    plt.hist(df_cleaned["Price"], bins=100)
    plt.ylabel("Counts")
    plt.xlabel("2017 Purchase Prices - Price")
    plt.show()

    plt.hist(df_cleaned["Volume"], bins=100)
    plt.ylabel("Counts")
    plt.xlabel("2017 Purchase Prices - Volume")
    plt.show()

    plt.hist(df_cleaned["PurchasePrice"], bins=100)
    plt.ylabel("Counts")
    plt.xlabel("2017 Purchase Prices - PurchasePrice")
    plt.show()

    # review/address outliers - Price, PurchasePrice
    df_outliers = df_cleaned[(df_cleaned["Price"] > (df_cleaned["Price"].mean() + (2 * df_cleaned["Price"].std()))) |
                             (df_cleaned["PurchasePrice"] > (df_cleaned["PurchasePrice"].mean() +
                                                             (2 * df_cleaned["PurchasePrice"].std())))]
    df_outliers.to_csv("Output\\Outliers_2017PurchasePricesDec.csv")
    # based on reviewing this list the data seems plausible, so outliers to remain

    return df_cleaned

def createStoreList(df):
    # function is used by inventory methods
    # to create a uniform list of stores included
    storeList = df.groupby(by=['Store', 'City'], as_index=False, sort=True)
    #print(storeList)

    return storeList


def clean_beg_inv_final_2016(df):
    # investigate dataset
    investigate_data(df)
    # Data Shape: (206529, 9)

    # check for outliers using histogram for onHand, Price
    plt.hist(df["onHand"], bins=100)
    plt.ylabel("Counts")
    plt.xlabel("Beg Inv Final 2016 - onHand")
    plt.show()

    plt.hist(df["Price"], bins=100)
    plt.ylabel("Counts")
    plt.xlabel("Beg Inv Final 2016 - Price")
    plt.show()

    # review/address outliers - onHand, Price
    df_outliers = df[(df["onHand"] > (df["onHand"].mean() + (2 * df["onHand"].std()))) |
                     (df["Price"] > (df["Price"].mean() + (2 * df["Price"].std())))]
    df_outliers.to_csv("Output\\Outliers_BegInvFINAL12312016.csv")

    # Data Shape: (206529, 9)
    # Based on check in investigate_data(df), mo missing values
    # ['InventoryId', 'Store', 'City', 'Brand', 'Description', 'Size', 'onHand', 'Price', 'startDate']

    # drop column 'InventoryId' is highly correlated
    # all information from this column is represented in the columns 'Store', 'City', 'Brand'
    df.drop(axis=1, columns='InventoryId', inplace=True)

    return df


def clean_end_inv_final_2016(df):
    # investigate dataset
    investigate_data(df)

    # check for outliers using histogram for onHand, Price
    plt.hist(df["onHand"], bins=100)
    plt.ylabel("Counts")
    plt.xlabel("Beg Inv Final 2016 - onHand")
    plt.show()

    plt.hist(df["Price"], bins=100)
    plt.ylabel("Counts")
    plt.xlabel("Beg Inv Final 2016 - Price")
    plt.show()

    # review/address outliers - onHand, Price
    df_outliers = df[(df["onHand"] > (df["onHand"].mean() + (2 * df["onHand"].std()))) |
                     (df["Price"] > (df["Price"].mean() + (2 * df["Price"].std())))]
    df_outliers.to_csv("Output\\Outliers_EndInvFINAL12312016.csv")

    # Data Shape: (224489, 9)
    # ['InventoryId', 'Store', 'City', 'Brand', 'Description', 'Size', 'onHand', 'Price', 'endDate']

    # drop column 'InventoryId' is highly correlated
    # all information from this column is represented in the columns 'Store', 'City', 'Brand'
    df.drop(axis=1, columns='InventoryId', inplace=True)

    # Missing Values Check: City    1284
    # address missing values
    df_missing = df[df.isna().any(axis=1)]
    print('Missing Values:')
    print(df_missing.head())
    print()
    print('\n\nEnding Inventory:\nThe total onHand count of the instances with missing values is {:,}'.format(
        df_missing['onHand'].sum()))
    # The total onHand count of the instances with missing values is 0
    # despite the lack of inventory for these instances, removed rows NaN in 'City'
    df_cleaned = df.copy(deep=True)
    df_cleaned.fillna("TYWARDREATH", inplace=True)



    return df_cleaned


def clean_invoice_purchases_2016(df):
    # investigate dataset
    investigate_data(df)

    # check for outliers using histogram for Quantity, Dollars, Freight
    plt.hist(df["Quantity"], bins=100)
    plt.ylabel("Counts")
    plt.xlabel("Invoice Purchases 2016 - Quantity")
    plt.show()

    plt.hist(df["Dollars"], bins=100)
    plt.ylabel("Counts")
    plt.xlabel("Invoice Purchases 2016 - Dollars")
    plt.show()

    plt.hist(df["Freight"], bins=100)
    plt.ylabel("Counts")
    plt.xlabel("Invoice Purchases 2016 - Freight")
    plt.show()

    # review/address outliers - Dollars, Freight, Quantity
    df_outliers = df[(df["Dollars"] > (df["Dollars"].mean() + (2 * df["Dollars"].std()))) |
                     (df["Freight"] > (df["Freight"].mean() + (2 * df["Freight"].std()))) |
                     (df["Quantity"] > (df["Quantity"].mean() + (2 * df["Quantity"].std())))]
    df_outliers.to_csv("Output\\Outliers_InvoicePurchases12312016.csv")
    # based on reviewing this list the data seems plausible, so outliers to remain

    return df


def clean_purchases_final_2016(df):
    # investigate dataset
    investigate_data(df)

    # address missing values
    df_missing = df[df.isna().any(axis=1)]
    df_missing.to_csv("Output\\Missing_Values_PurchasesFINAL12312016.csv")
    # reviewing product description and purchase price in 2017PurchasePricesDec.csv, all missing are 750mL
    df_cleaned = df.copy(deep=True)
    df_cleaned.fillna("750mL", inplace=True)

    # check for outliers using histogram for PurchasePrice, Quantity, Dollars
    plt.hist(df["PurchasePrice"], bins=100)
    plt.ylabel("Counts")
    plt.xlabel("Purchases Final 2016 - PurchasePrice")
    plt.show()

    plt.hist(df["Quantity"], bins=100)
    plt.ylabel("Counts")
    plt.xlabel("Purchases Final 2016 - Quantity")
    plt.show()

    plt.hist(df["Dollars"], bins=100)
    plt.ylabel("Counts")
    plt.xlabel("Purchases Final 2016 - Dollars")
    plt.show()

    # review/address outliers - PurchasePrice, Quantity, Dollars
    df_outliers = df_cleaned[(df_cleaned["PurchasePrice"] > (df_cleaned["PurchasePrice"].mean() +
                                                             (2 * df_cleaned["PurchasePrice"].std()))) |
                     (df_cleaned["Quantity"] > (df_cleaned["Quantity"].mean() + (2 * df_cleaned["Quantity"].std()))) |
                     (df_cleaned["Dollars"] > (df_cleaned["Dollars"].mean() + (2 * df_cleaned["Dollars"].std())))]
    df_outliers.to_csv("Output\\Outliers_PurchasesFINAL12312016.csv")

    return df_cleaned


def clean_sales_final_2016(df):
    # investigate dataset
    investigate_data(df)

    # check for outliers using histogram for SalesQuantity, SalesDollars, SalesPrice, Volume, ExciseTax
    plt.hist(df["SalesQuantity"], bins=100)
    plt.ylabel("Counts")
    plt.xlabel("Sales Final 2016 - SalesQuantity")
    plt.show()

    plt.hist(df["SalesDollars"], bins=100)
    plt.ylabel("Counts")
    plt.xlabel("Sales Final 2016 - SalesDollars")
    plt.show()

    plt.hist(df["SalesPrice"], bins=100)
    plt.ylabel("Counts")
    plt.xlabel("Sales Final 2016 - SalesPrice")
    plt.show()

    plt.hist(df["Volume"], bins=100)
    plt.ylabel("Counts")
    plt.xlabel("Sales Final 2016 - Volume")
    plt.show()

    plt.hist(df["ExciseTax"], bins=100)
    plt.ylabel("Counts")
    plt.xlabel("Sales Final 2016 - ExciseTax")
    plt.show()

    # review/address outliers - SalesQuantity, SalesDollars, SalesPrice, Volume, ExciseTax
    df_outliers = df[(df["SalesQuantity"] > (df["SalesQuantity"].mean() + (2 * df["SalesQuantity"].std()))) |
                             (df["SalesDollars"] > (df["SalesDollars"].mean() + (2 * df["SalesDollars"].std()))) |
                             (df["SalesPrice"] > (df["SalesPrice"].mean() + (2 * df["SalesPrice"].std()))) |
                             (df["Volume"] > (df["Volume"].mean() + (2 * df["Volume"].std()))) |
                             (df["ExciseTax"] > (df["ExciseTax"].mean() + (2 * df["ExciseTax"].std())))]
    df_outliers.to_csv("Output\\Outliers_SalesFINAL12312016.csv")

    return df


def prep_2017_purchase_prices_with_profit(df):
    # add columns for profit (price-purchasePrice) and profit margin (profit/price)
    df_prepped = df.copy(deep=True)
    df_prepped["Profit"] = (df_prepped["Price"] - df_prepped["PurchasePrice"])
    df_prepped["ProfitMargin"] = (df_prepped["Profit"] / df_prepped["Price"])

    return df_prepped


def prep_invoice_purchases_2016_by_month(df):
    # add columns for year and month
    df_prepped = df.copy(deep=True)
    df_prepped["Purchase_Paid_Year"] = pd.DatetimeIndex(df_prepped["PayDate"]).year
    df_prepped["Purchase_Paid_Month"] = pd.DatetimeIndex(df_prepped["PayDate"]).month

    return df_prepped


def prep_purchases_final_2016_by_store_by_month(df):
    # add columns for year and month
    df_prepped = df.copy(deep=True)
    df_prepped["Year"] = pd.DatetimeIndex(df_prepped["PayDate"]).year
    df_prepped["Month"] = pd.DatetimeIndex(df_prepped["PayDate"]).month

    # add column with store name parsed from InventoryId
    df_prepped["Store_Name"] = df_prepped["InventoryId"].str.extract(r'_([^_]+)_', expand=True)

    df_prepped_by_store = df_prepped.groupby(["Store", "Store_Name", "Year", "Month"]).agg(
        Purchase_Sum=pd.NamedAgg(column="Dollars", aggfunc="sum"),
        Purchase_Mean=pd.NamedAgg(column="Dollars", aggfunc="mean")).reset_index()

    return df_prepped_by_store


def prep_sales_final_2016_by_store_by_month(df):
    # add columns for year and month
    df_prepped = df.copy(deep=True)
    df_prepped["Year"] = pd.DatetimeIndex(df_prepped["SalesDate"]).year
    df_prepped["Month"] = pd.DatetimeIndex(df_prepped["SalesDate"]).month

    # add column with store name parsed from InventoryId
    df_prepped["Store_Name"] = df_prepped["InventoryId"].str.extract(r'_([^_]+)_', expand=True)

    df_prepped_by_store = df_prepped.groupby(["Store", "Store_Name", "Year", "Month"]).agg(
        Sales_Sum=pd.NamedAgg(column="SalesDollars", aggfunc="sum"),
        Sales_Mean=pd.NamedAgg(column="SalesDollars", aggfunc="mean")).reset_index()

    return df_prepped_by_store


def prep_earnings_by_store_by_month(df_purchases, df_sales):
    # full join df_purchases and df_sales
    df_earnings = pd.merge(df_purchases, df_sales, how="outer", left_on=["Store", "Store_Name", "Year", "Month"],
                           right_on=["Store", "Store_Name", "Year", "Month"])

    # address blank cells where there may not be any purchases or sales by replacing with 0.0000001
    # not using 0, because going to calculate ratios and do not want to divide by 0
    df_earnings.fillna(0.0000001, inplace=True)

    # add earnings and earnings/sales ratio
    df_earnings["Earnings_Sum"] = (df_earnings["Sales_Sum"] - df_earnings["Purchase_Sum"])
    df_earnings["Earnings_Mean"] = (df_earnings["Sales_Mean"] - df_earnings["Purchase_Mean"])
    df_earnings["Earnings_Sales_Ratio_Sum"] = (df_earnings["Earnings_Sum"] / df_earnings["Sales_Sum"])
    df_earnings["Earnings_Sales_Ratio_Mean"] = (df_earnings["Earnings_Mean"] / df_earnings["Sales_Mean"])

    return df_earnings


def prep_sales_final_2016_by_store_by_classification_by_month(df):
    # add columns for year and month
    df_prepped = df.copy(deep=True)
    df_prepped["Year"] = pd.DatetimeIndex(df_prepped["SalesDate"]).year
    df_prepped["Month"] = pd.DatetimeIndex(df_prepped["SalesDate"]).month

    # add column with store name parsed from InventoryId
    df_prepped["Store_Name"] = df_prepped["InventoryId"].str.extract(r'_([^_]+)_', expand=True)

    df_prepped_by_store_by_classification = df_prepped.groupby(["Store", "Store_Name", "Classification", "Year", "Month"]).agg(
        Sales_Sum=pd.NamedAgg(column="SalesDollars", aggfunc="sum"),
        Sales_Mean=pd.NamedAgg(column="SalesDollars", aggfunc="mean")).reset_index()

    return df_prepped_by_store_by_classification


def prep_sales_final_2016_by_store_by_classification_by_vendor_by_month(df):
    # add columns for year and month
    df_prepped = df.copy(deep=True)
    df_prepped["Year"] = pd.DatetimeIndex(df_prepped["SalesDate"]).year
    df_prepped["Month"] = pd.DatetimeIndex(df_prepped["SalesDate"]).month

    # add column with store name parsed from InventoryId
    df_prepped["Store_Name"] = df_prepped["InventoryId"].str.extract(r'_([^_]+)_', expand=True)

    df_prepped_by_store_by_classification_by_vendor = df_prepped.groupby(["Store", "Store_Name", "Classification", "VendorName", "Year", "Month"]).agg(
        Sales_Sum=pd.NamedAgg(column="SalesDollars", aggfunc="sum"),
        Sales_Mean=pd.NamedAgg(column="SalesDollars", aggfunc="mean")).reset_index()

    return df_prepped_by_store_by_classification_by_vendor


def prep_sales_final_2016_by_store_by_classification_by_description_by_month(df):
    # add columns for year and month
    df_prepped = df.copy(deep=True)
    df_prepped["Year"] = pd.DatetimeIndex(df_prepped["SalesDate"]).year
    df_prepped["Month"] = pd.DatetimeIndex(df_prepped["SalesDate"]).month

    # add column with store name parsed from InventoryId
    df_prepped["Store_Name"] = df_prepped["InventoryId"].str.extract(r'_([^_]+)_', expand=True)

    df_prepped_by_store_by_classification_by_description = df_prepped.groupby(["Store", "Store_Name", "Description", "VendorName", "Year", "Month"]).agg(
        Sales_Sum=pd.NamedAgg(column="SalesDollars", aggfunc="sum"),
        Sales_Mean=pd.NamedAgg(column="SalesDollars", aggfunc="mean")).reset_index()

    return df_prepped_by_store_by_classification_by_description


def prep_2016_beg_inv_by_store_value(df):
    df_prepped = df.copy(deep=True)

    # create column to store value of inventory onHand
    df_prepped['InvValue'] = df_prepped['onHand'] * df_prepped['Price']

    # groupby 'Store' and 'City' to calculate totals by store
    byStore = createStoreList(df_prepped)

    print('\n\nThe value of the total inventory at the start of 2016 is {:,}'.format(df_prepped['InvValue'].sum()))
    print('Value of inventory by store:\n')
    inventoryB_valueByStore_subtotals = byStore['InvValue'].sum()

    return inventoryB_valueByStore_subtotals

def prep_2016_beg_inv_by_store_count(df):
    df_prepped = df.copy(deep=True)

    # groupby 'Store' and 'City' to calculate totals by store
    byStore = createStoreList(df_prepped)

    print('\n\nThe count of the total inventory at the start of 2016 is {:,}'.format(df_prepped['onHand'].sum()))
    print('Inventory count by store:\n')
    inventoryB_countByStore_subtotals = byStore['onHand'].sum()

    return inventoryB_countByStore_subtotals

def prep_2016_end_inv_by_store_value(df):
    df_prepped = df.copy(deep=True)

    # create column to store value of inventory onHand
    df_prepped['InvValue'] = df_prepped['onHand'] * df_prepped['Price']

    # groupby 'Store' and 'City' to calculate totals by store
    byStore = createStoreList(df_prepped)

    print('\n\nThe value of the total inventory at the end of 2016 is {:,}'.format(df_prepped['InvValue'].sum()))
    print('Value of inventory by store:\n')
    inventoryE_valueByStore_subtotals = byStore['InvValue'].sum()

    return inventoryE_valueByStore_subtotals

def prep_2016_end_inv_by_store_count(df):
    df_prepped = df.copy(deep=True)

    # groupby 'Store' and 'City' to calculate totals by store
    byStore = createStoreList(df_prepped)

    print('\n\nThe count of the total inventory at the end of 2016 is {:,}'.format(df_prepped['onHand'].sum()))
    print('Inventory count by store:\n')
    inventoryE_countByStore_subtotals = byStore['onHand'].sum()

    return inventoryE_countByStore_subtotals


def main():
    # load data
    df_2017_purchase_prices = pd.read_csv("Datasets\\2017PurchasePricesDec.csv")
    df_beg_inv_final_2016 = pd.read_csv("Datasets\\BegInvFINAL12312016.csv")
    df_end_inv_final_2016 = pd.read_csv("Datasets\\EndInvFINAL12312016.csv")
    df_invoice_purchases_2016 = pd.read_csv("Datasets\\InvoicePurchases12312016.csv")
    df_purchases_final_2016 = pd.read_csv("Datasets\\PurchasesFINAL12312016.csv")
    df_sales_final_2016 = pd.read_csv("Datasets\\SalesFINAL12312016.csv")


    # investigate data and clean data
    print("Investigate and Clean 2017 Purchase Prices Dataset " + ("*" * 60))
    df_2017_purchase_prices_cleaned = clean_2017_purchase_prices(df_2017_purchase_prices)


    print("Investigate and Clean Beginning Inventory Final 2016 Dataset " + ("*" * 60))
    df_beg_inv_final_2016_cleaned = clean_beg_inv_final_2016(df_beg_inv_final_2016)

    print("Investigate and Clean Ending Inventory Final 2016 Dataset " + ("*" * 60))
    df_end_inv_final_2016_cleaned = clean_end_inv_final_2016(df_end_inv_final_2016)


    print("Investigate and Clean Invoice Purchases 2016 Dataset " + ("*" * 60))
    df_invoice_purchases_2016_cleaned = clean_invoice_purchases_2016(df_invoice_purchases_2016)

    print("Investigate and Clean Purchases Final 2016 Dataset " + ("*" * 60))
    df_purchases_final_2016_cleaned = clean_purchases_final_2016(df_purchases_final_2016)

    print("Investigate and Clean Sales Final 2016 Dataset " + ("*" * 60))
    df_sales_final_2016_cleaned = clean_sales_final_2016(df_sales_final_2016)

    # prep/transform data for analysis - create new dfs, add columns to existing dfs, ...
    df_2017_purchase_prices_with_profit = prep_2017_purchase_prices_with_profit(df_2017_purchase_prices_cleaned)
    df_2017_purchase_prices_with_profit.to_csv("Output\\2017_purchase_prices_with_profit.csv")

    df_invoice_purchases_2016_by_month = prep_invoice_purchases_2016_by_month(df_invoice_purchases_2016_cleaned)
    df_invoice_purchases_2016_by_month.to_csv("Output\\invoice_purchases_2016_by_month.csv")

    df_purchases_final_2016_by_store_by_month = prep_purchases_final_2016_by_store_by_month(
        df_purchases_final_2016_cleaned)
    df_purchases_final_2016_by_store_by_month.to_csv("Output\\purchases_final_2016_by_store_by_month.csv")

    df_sales_final_2016_by_store_by_month = prep_sales_final_2016_by_store_by_month(df_sales_final_2016_cleaned)
    df_sales_final_2016_by_store_by_month.to_csv("Output\\sales_final_2016_by_store_by_month.csv")

    df_earnings_by_store_by_month = prep_earnings_by_store_by_month(df_purchases_final_2016_by_store_by_month,
                                                                    df_sales_final_2016_by_store_by_month)
    df_earnings_by_store_by_month.to_csv("Output\\earnings_by_store_by_month.csv")

    df_sales_final_2016_by_store_by_classification_by_month = prep_sales_final_2016_by_store_by_classification_by_month(
        df_sales_final_2016_cleaned)
    df_sales_final_2016_by_store_by_classification_by_month.to_csv(
        "Output\\sales_final_2016_by_store_by_classification_by_month.csv")

    df_sales_final_2016_by_store_by_classification_by_vendor_by_month = prep_sales_final_2016_by_store_by_classification_by_vendor_by_month(
        df_sales_final_2016_cleaned)
    df_sales_final_2016_by_store_by_classification_by_vendor_by_month.to_csv(
        "Output\\sales_final_2016_by_store_by_classification_by_vendor_by_month.csv")

    df_sales_final_2016_by_store_by_classification_by_description_by_month = prep_sales_final_2016_by_store_by_classification_by_description_by_month(
        df_sales_final_2016_cleaned)
    df_sales_final_2016_by_store_by_classification_by_description_by_month.to_csv(
        "Output\\sales_final_2016_by_store_by_classification_by_description_by_month.csv")


    df_2016_beg_inv_by_store_asValue = prep_2016_beg_inv_by_store_value(df_beg_inv_final_2016_cleaned)
    df_2016_beg_inv_by_store_asValue.to_csv("Output\\final_2016_beg_inventory_by_store_asValue.csv")
    df_2016_beg_inv_by_store_asCount = prep_2016_beg_inv_by_store_count(df_beg_inv_final_2016_cleaned)
    df_2016_beg_inv_by_store_asCount.to_csv("Output\\final_2016_beg_inventory_by_store_asCount.csv")
    df_2016_end_inv_by_store_asValue = prep_2016_end_inv_by_store_value(df_end_inv_final_2016_cleaned)
    df_2016_end_inv_by_store_asValue.to_csv("Output\\final_2016_end_inventory_by_store_asValue.csv")
    df_2016_end_inv_by_store_asCount = prep_2016_end_inv_by_store_count(df_end_inv_final_2016_cleaned)
    df_2016_end_inv_by_store_asCount.to_csv("Output\\final_2016_end_inventory_by_store_asCount.csv")


if __name__ == '__main__':
    main()
