import pandas as pd
from sklearn.ensemble import IsolationForest
import joblib

# Load dataset
data = pd.read_csv("cleaned_data.csv")

# Train Isolation Forest Model
model = IsolationForest(contamination=0.05, random_state=42)
data["Outlier"] = model.fit_predict(data[['Sales', 'Discount', 'Profit']])

# Save Model
joblib.dump(model, "models/model.pkl")

# Save cleaned dataset
data.to_csv("cleaned_data.csv", index=False)
