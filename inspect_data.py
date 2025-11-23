import pandas as pd
import os

excel_file = 'excel.xlsx'
if not os.path.exists(excel_file):
    print(f"File {excel_file} not found.")
    # Try excel_binary.xlsx
    if os.path.exists('excel_binary.xlsx'):
        print("Found excel_binary.xlsx instead.")
        excel_file = 'excel_binary.xlsx'
    else:
        exit()

print(f"Reading {excel_file}...")
df = pd.read_excel(excel_file)
print("Columns:", df.columns.tolist())
print("First few rows:")
print(df.head())
print("\nSecond column unique values:")
print(df.iloc[:, 1].unique())
print("\nSecond column dtype:", df.iloc[:, 1].dtype)
