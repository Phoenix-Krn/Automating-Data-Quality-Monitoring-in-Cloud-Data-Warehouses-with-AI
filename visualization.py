import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# ✅ Load cleaned data
DATA_FILE = r"C:\Users\Kavya R\Desktop\INT82\Data_Quality\cleaned_sales_data.csv"
data = pd.read_csv(DATA_FILE)

# ✅ Check column names
print("📌 Column Names in Dataset:", data.columns.tolist())

# 🔹 Use "Days" instead of "Order Date" for time-series analysis
if "Days" not in data.columns:
    raise KeyError("🚨 'Days' column not found in dataset! Check CSV file.")

# ✅ Correlation Matrix
correlation_matrix = data[["Sales", "Discount", "Profit"]].corr()
fig = px.imshow(correlation_matrix, text_auto=True, color_continuous_scale="Viridis")
fig.update_layout(title="Correlation Matrix for Sales, Discount, and Profit")
fig.show()

# ✅ Time-Series Analysis (Sales Trends Over Days)
plt.figure(figsize=(12, 6))
plt.plot(data["Days"], data["Sales"], marker="o", linestyle="-", color="blue", label="Sales Over Time")
plt.title("📊 Sales Trends Over Time", fontsize=14)
plt.xlabel("Days Since First Order", fontsize=12)
plt.ylabel("Total Sales", fontsize=12)
plt.xticks(rotation=45)
plt.grid(True, linestyle="--", alpha=0.7)
plt.legend()
plt.show()

# ✅ Boxplot for Outlier Detection in Sales
plt.figure(figsize=(8, 5))
sns.boxplot(y=data["Sales"], color="skyblue")
plt.title("📌 Outlier Detection: Sales Distribution")
plt.ylabel("Sales")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()
