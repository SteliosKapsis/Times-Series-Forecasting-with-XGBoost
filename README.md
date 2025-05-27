
# PJME Energy Use Time Series Forecasting

This project focuses on forecasting **PJME energy consumption** using historical hourly data, leveraging **XGBoost regression**, **time series feature engineering**, and **cross-validation** techniques.

## 📂 Project Structure
```
📁 data/
    └── PJME_hourly.csv          # Raw time series data
📁 models/
    └── model.json               # Saved trained model
📄 app.py                        # Main script (kept)
📄 requirements.txt              # Python dependencies
📄 .gitignore                    # Files/folders to ignore
```

## 🚀 Key Features
- **Data Cleaning**: Removes outliers from energy consumption data.
- **Feature Engineering**:
  - Time-based features (hour, dayofweek, month, etc.)
  - Lag features (previous year, two years, and three years)
- **TimeSeriesSplit Cross-Validation**: Robust model evaluation across multiple temporal folds.
- **Model Training**: Utilizes XGBoost with early stopping and hyperparameter tuning.
- **Future Forecasting**: Generates future dataframes and forecasts up to a year ahead.
- **Error Analysis**: Calculates RMSE scores and daily average errors for interpretability.
- **Model Saving & Loading**: Supports saving trained models and reloading for reuse.
- **Visualizations**:
  - Time series plots
  - Data distribution
  - Cross-validation fold visualizations
  - Forecast plots

## 🛠 Setup Instructions
1️⃣ Clone the repository:
```bash
git clone https://github.com/SteliosKapsis/Times-Series-Forecasting-with-XGBoost
cd pjme-energy-forecasting
```

2️⃣ Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3️⃣ Install dependencies:
```bash
pip install -r requirements.txt
```

4️⃣ Run the main script:
```bash
python app.py
```

## 📦 Dependencies
The project uses the following Python libraries:
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- xgboost

## 🔒 .gitignore
The following files/folders are ignored from version control:
```
venv/
model.json
```

## 💡 Future Improvements
- Hyperparameter tuning using Grid Search or Random Search
- Integration of weather and external data as features
- Deployment as a web service or interactive dashboard

---

© 2025 PJME Energy Forecasting Project
