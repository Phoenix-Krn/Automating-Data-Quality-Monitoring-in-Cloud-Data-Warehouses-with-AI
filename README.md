# 🛠️ Automating Data Quality Monitoring in Cloud Data Warehouses with AI

🚀 **Project Overview:**  
This project automates data quality monitoring for cloud data warehouses using AI. It analyzes supermarket sales data, detects anomalies, ensures data integrity, and generates insights through an interactive dashboard and web application.

---

## 🌟 **Key Features**
1. **Data Quality Monitoring:** Real-time detection of missing values, outliers, and anomalies.  
2. **Interactive Dashboard:** Visualize key metrics, heatmaps, scatter plots, and sales trends.  
3. **Manual & CSV Upload:** Add new records via form or upload sales datasets.  
4. **AI Model:** Trained RandomForest model for sales prediction and anomaly detection.  
5. **Automated Reports:** Generate Power BI reports for insights and alerts.  

---

## 📊 **Tech Stack**
- **Backend:** Python, Flask  
- **Frontend:** HTML, Dash (Plotly)  
- **Data Processing:** Pandas, NumPy, Scikit-learn  
- **Visualization:** Matplotlib, Seaborn, Plotly  
- **Database:** CSV as flat-file storage  
- **Deployment:** IBM Cloud  

---

## 📂 **Project Structure**
```
📦 Data_Quality
├─ 📁 data              # Dataset & cleaned data
│  ├─ Supermart_Grocery_Sales.csv
│  └─ cleaned_data.csv
├─ 📁 models            # Trained model for predictions
│  └─ sales_prediction_model.pkl
├─ 📁 templates         # HTML templates for web app
│  ├─ index.html        # Dashboard homepage
│  └─ analysis.html     # Visualization page
├─ 📁 static            # CSS & JS for frontend
│  └─ style.css
├─ 🔖 app.py            # Main Flask app for dashboard
├─ 🔖 data_preprocessing.py  # Data cleaning script
├─ 🔖 model_training.py # Train and evaluate AI model
├─ 🔖 visualization.py  # Generate interactive charts
├─ 🔖 dashboard.py      # Dash-based dashboard
├─ 🔖 requirements.txt  # Python dependencies
└─ 🔖 README.md         # Project documentation
```

---

## ⚙️ **Setup Instructions**

1. **Clone Repository:**
```bash
git clone https://github.com/Phoenix-Krn/Automating-Data-Quality-Monitoring-in-Cloud-Data-Warehouses-with-AI.git
cd Data_Quality
```

2. **Create Virtual Environment (Optional but Recommended):**
```bash
python -m venv env
source env/bin/activate  # For Windows: .\env\Scripts\activate
```

3. **Install Dependencies:**
```bash
pip install -r requirements.txt
```

4. **Run Data Preprocessing:**
```bash
python data_preprocessing.py
```

5. **Train AI Model:**
```bash
python model_training.py
```

6. **Start Flask Web App:**
```bash
python app.py
```

7. **Access Dashboard:**  
Open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser.

---

## 📈 **Usage**
1. **Upload Sales Dataset:** Upload a CSV or manually add new records.  
2. **Monitor Data Quality:** View trends, heatmaps, and anomalies.  
3. **Analyze Predictions:** Check AI-generated sales predictions.  
4. **Export Reports:** Download insights in PDF/Excel format.  

---

## 🚀 **Deployment (IBM Cloud)**
1. Create an IBM Cloud account.  
2. Deploy app using **IBM Code Engine**.  
3. Connect Flask app to **Watson Studio** for AI.  

---

## 🔮 **Future Scope**
1. Real-time Streaming with Kafka or Spark.  
2. Advanced AI models for anomaly detection.  
3. Automated email alerts for quality breaches.  

---

## 🤝 **Contributing**
1. Fork the repo.  
2. Create a branch (`git checkout -b feature/new-feature`).  
3. Commit changes (`git commit -m "Added new feature"`).  
4. Push to GitHub (`git push origin feature/new-feature`).  
5. Open a Pull Request.  

---

## 📜 **Troubleshooting**
1. **Environment issues?** Recreate the virtual environment.  
2. **Large file warning?** Use Git LFS to track `.pkl` files.  
3. **Dashboard not loading?** Ensure `app.py` runs without errors.  

---

## 📳 **License**
This project is licensed under the **MIT License**.

---

💡 **Questions or Feedback?**  
Feel free to raise an issue or connect via [GitHub Issues](https://github.com/Phoenix-Krn/Automating-Data-Quality-Monitoring-in-Cloud-Data-Warehouses-with-AI/issues).

---

🚀 **Let's build smarter data pipelines together!**

