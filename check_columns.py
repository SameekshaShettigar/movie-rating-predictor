import pandas as pd

# Try reading the dataset
df = pd.read_csv("imdb.csv", encoding='latin1')

# Print all column names
print("📋 Column names in your CSV:")
print(df.columns.tolist())
