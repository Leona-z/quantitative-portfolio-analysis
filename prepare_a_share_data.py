import pandas as pd


# ============================================================
# Prepare A-Share Stock Data
# Stocks:
# 000938 - 紫光股份
# 603986 - 兆易创新
# 002185 - 华天科技
# ============================================================


# 1. Read the three CSV files
zgc = pd.read_csv("000938历史数据.csv")
gds = pd.read_csv("603986历史数据.csv")
htkj = pd.read_csv("002185历史数据.csv")


# 2. Keep only Date and Close
zgc = zgc[["日期", "收盘"]].copy()
gds = gds[["日期", "收盘"]].copy()
htkj = htkj[["日期", "收盘"]].copy()


# 3. Rename columns
zgc.columns = ["Date", "ZGC"]
gds.columns = ["Date", "GDS"]
htkj.columns = ["Date", "HTKJ"]


# 4. Convert Date to datetime
zgc["Date"] = pd.to_datetime(zgc["Date"])
gds["Date"] = pd.to_datetime(gds["Date"])
htkj["Date"] = pd.to_datetime(htkj["Date"])


# 5. Sort by date
zgc = zgc.sort_values("Date")
gds = gds.sort_values("Date")
htkj = htkj.sort_values("Date")


# 6. Merge the three stocks
data = pd.merge(zgc, gds, on="Date", how="inner")
data = pd.merge(data, htkj, on="Date", how="inner")


# 7. Sort the final dataset
data = data.sort_values("Date")
data = data.reset_index(drop=True)


# 8. Save the final dataset
data.to_csv(
    "stock_data_a_share.csv",
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# Check the result
# ============================================================

print("========================================")
print("A-share data preparation completed!")
print("========================================")

print("\nFirst 5 rows:")
print(data.head())

print("\nLast 5 rows:")
print(data.tail())

print("\nData shape:")
print(data.shape)

print("\nColumn names:")
print(data.columns)

print("\nMissing values:")
print(data.isnull().sum())

print("\nSaved as:")
print("stock_data_a_share.csv")