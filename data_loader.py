import pandas as pd
import os

# ✅ Update the correct path to your dataset
RAW_DATA_FILE = r"C:\Users\Kavya R\Desktop\INT82\Data_Quality\Supermart Grocery Sales - Retail Analytics Dataset.csv"
CLEANED_DATA_FILE = r"C:\Users\Kavya R\Desktop\INT82\Data_Quality\cleaned_sales_data.csv"

def load_data():
    """Load raw dataset from CSV."""
    return pd.read_csv(RAW_DATA_FILE)

def clean_data(df):
    """Clean and preprocess the dataset."""
    # Handle missing values
    df.fillna(df.median(numeric_only=True), inplace=True)

    # Convert date to datetime
    df["Order Date"] = pd.to_datetime(df["Order Date"], errors="coerce")

    # Convert categorical columns to numerical representation
    df = pd.get_dummies(df, columns=["Category", "Region"], drop_first=True)

    return df

def save_cleaned_data(df):
    """Save cleaned data."""
    df.to_csv(CLEANED_DATA_FILE, index=False)
    print(f"✅ Cleaned data saved as {CLEANED_DATA_FILE}")

def get_summary(df):
    """Generate summary statistics."""
    return {
        "Total Sales": df["Sales"].sum(),
        "Total Orders": len(df),
        "Missing Values": df.isnull().sum().to_dict()
    }

if __name__ == "__main__":
    print("📊 Loading and cleaning data...\n")
    df = load_data()
    df = clean_data(df)
    save_cleaned_data(df)
    print("📌 Data Summary:", get_summary(df))
