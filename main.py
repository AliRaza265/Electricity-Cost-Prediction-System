import pandas as pd 


# Understand the given data 
read_csv = pd.read_csv(r"data_set\electricity_cost_dataset.csv") 
print(read_csv.head())
print(read_csv.describe())
print(read_csv.info())
print(read_csv.columns)

# Data Filtering
find_missing_values = read_csv.isnull().sum()
find_duplicate_value = read_csv.duplicated().sum()
print(find_duplicate_value)



