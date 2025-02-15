import numpy as np

# Generate confusion matrix
y_true = data['Outlier']  # True labels
y_pred = model.predict(data[['Sales', 'Discount', 'Profit']])  # Predictions
cm = confusion_matrix(y_true, y_pred)

# Display confusion matrix
fig, ax = plt.subplots(figsize=(6,6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(ax=ax, cmap='Blues')
plt.title("Confusion Matrix")
plt.show()
