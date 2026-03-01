# This script converts a CSV file to an XLSX file using pandas. You can use this to easily view and manipulate your data in Excel.

import pandas as pd

# Paths
csv_path = "train_label_1.csv"
xlsx_path = "train_label_1.xlsx"

# Read CSV
df = pd.read_csv(csv_path)

# Write to Excel
df.to_excel(xlsx_path, index=False)

print("Conversion completed successfully!")
