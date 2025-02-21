from flask import Flask, render_template, request, jsonify
import pandas as pd
import os

app = Flask(__name__)

DATA_FILE = "data/sales_data.csv"

# Load dataset
def load_data():
    return pd.read_csv(DATA_FILE)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/add_record', methods=['POST'])
def add_record():
    new_data = request.json
    df = load_data()
    df = df.append(new_data, ignore_index=True)
    df.to_csv(DATA_FILE, index=False)
    return jsonify({"message": "Record added successfully!"})

if __name__ == '__main__':
    app.run(debug=True)
