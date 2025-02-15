from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load trained model
model = joblib.load('trained_model.pkl')

# ✅ Add a home route to avoid 404 error
@app.route('/')
def home():
    return "Welcome to the AI Model API! Use /predict to send POST requests."

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()  # Receive JSON input
        df = pd.DataFrame(data)  # Convert to DataFrame
        prediction = model.predict(df)  # Make prediction
        return jsonify({'prediction': prediction.tolist()})  # Return JSON response
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)  # Run Flask server
