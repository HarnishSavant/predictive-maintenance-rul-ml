# 🚀 Predictive Maintenance System — Remaining Useful Life (RUL) Prediction

Machine Learning–based Predictive Maintenance system developed using the **NASA C-MAPSS Turbofan Engine Dataset** to estimate the **Remaining Useful Life (RUL)** of aircraft engines using sensor data.

This project demonstrates a complete **end-to-end ML pipeline** including data preprocessing, exploratory analysis, model comparison, evaluation, and deployment using a Flask web application.

---

## 📌 Project Overview

Unplanned equipment failure causes significant operational and financial loss in industrial systems. Traditional maintenance approaches are either reactive or time-based, leading to inefficiencies.

This project applies **Machine Learning** to predict engine failure before it occurs by estimating Remaining Useful Life (RUL) from real-time sensor measurements.

✅ Enables condition-based maintenance
✅ Reduces downtime and maintenance cost
✅ Supports data-driven decision-making

---

## 🧠 Problem Statement

Given multivariate sensor readings from turbofan engines, predict:

> **Remaining Useful Life (RUL)** — number of cycles before engine failure.

The model learns degradation patterns from historical run-to-failure data.

---

## 📂 Dataset

**Source:** NASA Prognostics Center of Excellence
**Dataset:** C-MAPSS FD001

* 100 training engines
* 21 sensor measurements
* 3 operational settings
* Time-series degradation simulation

---

## ⚙️ Tech Stack

**Programming**

* Python 3

**Data Processing**

* Pandas
* NumPy

**Visualization**

* Matplotlib
* Seaborn

**Machine Learning**

* Scikit-learn

**Deployment**

* Flask
* HTML / CSS / JavaScript

---

## 🔄 Machine Learning Pipeline

1. Data Loading & Cleaning
2. Exploratory Data Analysis (EDA)
3. RUL Calculation
4. Feature Selection (Variance Analysis)
5. Feature Scaling (MinMaxScaler)
6. Model Training & Comparison
7. Performance Evaluation
8. Model Serialization (Pickle)
9. Flask Deployment

---

## 🤖 Models Implemented

* Linear Regression
* K-Nearest Neighbors (KNN)
* Decision Tree Regressor
* Random Forest Regressor ✅ (Best Model)
* Bagging Regressor
* Gradient Boosting Regressor

---

## 📊 Model Performance (Test Set)

| Model             | RMSE      | MAE       | R² Score |
| ----------------- | --------- | --------- | -------- |
| Linear Regression | 20.83     | 16.57     | 0.73     |
| KNN               | 20.17     | 14.47     | 0.74     |
| Decision Tree     | 19.50     | 14.59     | 0.76     |
| **Random Forest** | **17.22** | **12.20** | **0.81** |
| Bagging           | 17.94     | 12.46     | 0.80     |
| Gradient Boosting | 17.81     | 12.64     | 0.80     |

---

## 🏆 Final Model Selection

**Random Forest Regressor** was selected because:

* Captures non-linear degradation patterns
* Handles sensor interactions effectively
* Strong generalization performance
* Stable predictions across engines

---

## 🌐 Deployment

The trained model is deployed using a **Flask Web Application**.

### Features:

* Real-time RUL prediction
* Sensor input interface
* Health status indicator
* Instant inference

---

## 📁 Project Structure

```
predictive-maintenance-rul-ml/
│
├── data/
├── notebooks/
├── models/
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ▶️ Installation & Setup

### 1️⃣ Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/predictive-maintenance-rul-ml.git
cd predictive-maintenance-rul-ml
```

### 2️⃣ Create Virtual Environment

```bash
python -m venv venv
```

Activate:

**Windows**

```bash
venv\Scripts\activate
```

**Mac/Linux**

```bash
source venv/bin/activate
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run Application

```bash
python app.py
```

Open browser:

```
http://127.0.0.1:5000
```

---

## 📈 Key Learnings

* Predictive maintenance workflow design
* Time-series degradation modelling
* Feature engineering for sensor data
* Ensemble learning techniques
* Model deployment using Flask

---

## 🔮 Future Improvements

* Deep Learning (LSTM for temporal modeling)
* Hyperparameter tuning
* Docker deployment
* Cloud deployment (GCP/AWS)
* Real-time streaming inference

---

## 👨‍💻 Author

**Harnish Savant**
M.Sc. Data Science & Spatial Analytics (Geo-Intelligence)
Machine Learning | Geospatial Analytics | Generative AI

GitHub: https://github.com/HarnishSavant
LinkedIn: https://www.linkedin.com/in/harnish-savant-0217b3270/

---

## ⭐ If you found this project useful

Give it a ⭐ on GitHub!
