import pandas as pd
from sklearn.ensemble import IsolationForest

# Load dataset
file_path = r'C:\Users\Kavya R\Desktop\INT82\Data_Quality\Supermart Grocery Sales - Retail Analytics Dataset.csv'
data = pd.read_csv(file_path)

# Convert 'Order Date' to datetime
data['Order Date'] = pd.to_datetime(data['Order Date'], format='%m-%d-%Y', errors='coerce')

# Ensure numeric columns
data['Sales'] = pd.to_numeric(data['Sales'], errors='coerce')
data['Discount'] = pd.to_numeric(data['Discount'], errors='coerce')
data['Profit'] = pd.to_numeric(data['Profit'], errors='coerce')

# ✅ Detect and mark outliers using Isolation Forest
iso = IsolationForest(contamination=0.01, random_state=42)
data['Outlier'] = iso.fit_predict(data[['Sales']])  # -1 = outlier, 1 = normal

# ✅ Save cleaned dataset
data.to_csv('cleaned_data.csv', index=False)
print("✅ Data Preprocessing Completed! 'cleaned_data.csv' updated with 'Outlier' column.")
