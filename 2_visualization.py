import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

# Load cleaned data
data = pd.read_csv('cleaned_data.csv')
data['Order Date'] = pd.to_datetime(data['Order Date'])

# Correlation Matrix
correlation_matrix = data[['Sales', 'Discount', 'Profit']].corr()
fig = px.imshow(correlation_matrix, text_auto=True, color_continuous_scale='Viridis')
fig.update_layout(title="Correlation Matrix for Sales, Discount, and Profit")
fig.show()

# Time-Series Analysis
monthly_sales = data.set_index('Order Date')['Sales'].resample('M').sum()
plt.figure(figsize=(10, 6))
plt.plot(monthly_sales, marker='o', linestyle='-', color='blue', label='Monthly Sales')
plt.title("Monthly Sales Trends")
plt.xlabel("Date")
plt.ylabel("Total Sales")
plt.grid(True)
plt.legend()
plt.show()
