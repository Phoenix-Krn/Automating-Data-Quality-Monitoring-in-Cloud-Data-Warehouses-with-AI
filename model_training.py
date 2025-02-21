import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os
import time

# File paths
RAW_DATA_FILE = r"C:\Users\Kavya R\Desktop\INT82\Data_Quality\Supermart Grocery Sales - Retail Analytics Dataset.csv"
CLEANED_DATA_FILE = r"C:\Users\Kavya R\Desktop\INT82\Data_Quality\cleaned_sales_data.csv"
MODEL_FILE = r"C:\Users\Kavya R\Desktop\INT82\Data_Quality\models\sales_prediction_model.pkl"

# Ensure directories exist
os.makedirs(os.path.dirname(MODEL_FILE), exist_ok=True)

def load_and_clean_data():
    """Load, clean, and save the dataset."""
    df = pd.read_csv(RAW_DATA_FILE)

    # Remove non-numeric columns
    drop_cols = ["Order ID", "Customer Name", "City", "State"]  # Non-numeric columns
    df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True)

    # Handle missing values
    df.fillna(df.median(numeric_only=True), inplace=True)

    # Convert Date to numerical value (days since start)
    df["Order Date"] = pd.to_datetime(df["Order Date"], errors="coerce")
    df["Days"] = (df["Order Date"] - df["Order Date"].min()).dt.days
    df.drop(columns=["Order Date"], inplace=True)

    # 🔹 Convert Categorical Columns into Numbers using One-Hot Encoding
    categorical_cols = ["Sub Category", "Category", "Region"]  
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # Save cleaned data
    df.to_csv(CLEANED_DATA_FILE, index=False)
    print(f"✅ Cleaned data saved to {CLEANED_DATA_FILE}")

    return df

def train_model():
    """Train a RandomForest model to predict sales."""
    df = load_and_clean_data()
    
    # Ensure 'Sales' column exists
    if "Sales" not in df.columns:
        raise KeyError("🚨 'Sales' column not found in dataset. Check CSV file!")

    # Features and target variable
    X = df.drop(columns=["Sales"])
    y = df["Sales"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the model
    model = RandomForestRegressor(n_estimators=100, random_state=42)

    print("🚀 Training the model...")
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    print(f"✅ Model training completed in {training_time:.2f} seconds.")

    # Predictions
    y_pred = model.predict(X_test)

    # Model evaluation
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"\n📊 Model Performance:\nMAE: {mae:.2f}\nRMSE: {rmse:.2f}\nR² Score: {r2:.2f}")

    # Save the trained model
    joblib.dump(model, MODEL_FILE)
    print(f"✅ Model saved as {MODEL_FILE}")

    # Feature Importance
    feature_importances = pd.DataFrame({"Feature": X.columns, "Importance": model.feature_importances_})
    feature_importances = feature_importances.sort_values(by="Importance", ascending=False)

    print("\n🔥 Top Features Influencing Sales:")
    print(feature_importances.head(5))

    # Visualization of predictions
    plt.figure(figsize=(10, 5))
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
    plt.xlabel("Actual Sales")
    plt.ylabel("Predicted Sales")
    plt.title("Actual vs Predicted Sales")
    plt.show()

if __name__ == "__main__":
    train_model()
