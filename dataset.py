import pandas as pd

# Load dataset
file_path = r"C:\Users\Kavya R\Desktop\INT82\Data_Quality\Supermart Grocery Sales - Retail Analytics Dataset.csv"
 # Adjust path if necessary
data = pd.read_csv(file_path)

# Display basic info
print("Dataset Preview:\n", data.head())
print("\nColumns in Dataset:\n", data.columns)
print("\nMissing Values:\n", data.isnull().sum())
