# 🏠 House Price Prediction

A machine learning project that predicts median house values using the Boston Housing dataset. The project follows a complete machine learning workflow, including data exploration, preprocessing, regression modeling, evaluation, and model saving.

---

## 📌 Project Overview

House prices depend on several factors such as crime rate, number of rooms, accessibility to highways, property taxes, and socioeconomic indicators.

The goal of this project is to build a regression model that predicts the median house value (`medv`) based on these features.

The project follows a complete machine learning workflow:

- Data loading and inspection
- Exploratory Data Analysis (EDA)
- Correlation analysis
- Feature and target separation
- Train-test splitting
- Feature scaling
- Linear Regression
- Random Forest Regression
- Model evaluation
- Model comparison
- Saving the trained model

---

## 📊 Dataset

The project uses the `BostonHousing.csv` dataset.

The dataset contains:

- **506 observations**
- **13 input features**
- **1 target variable**

## Features

| Feature | Description |
|---|---|
| `crim` | Per-capita crime rate |
| `zn` | Proportion of residential land zoned for lots over 25,000 sq. ft. |
| `indus` | Proportion of non-retail business acres |
| `chas` | Charles River dummy variable |
| `nox` | Nitric oxide concentration |
| `rm` | Average number of rooms per dwelling |
| `age` | Proportion of owner-occupied units built before 1940 |
| `dis` | Weighted distance to employment centres |
| `rad` | Accessibility to radial highways |
| `tax` | Property-tax rate |
| `ptratio` | Pupil-teacher ratio |
| `b` | Demographic-related index |
| `lstat` | Lower-status population percentage |

## Target

`medv` — Median house value.

---

## 🔎 Exploratory Data Analysis

The dataset was initially inspected to understand its structure, data types, distributions, and relationships between variables.

The dataset contains:

- 506 rows
- 14 columns
- 13 numerical input features
- 1 numerical target variable
- No missing values

Exploratory analysis included:

- Dataset information and statistical summary
- Target variable distribution
- Feature-target relationships
- Correlation analysis
- Correlation heatmap
- Scatter plots

## Important Observations

The correlation analysis showed several noticeable relationships with the target variable.

- `rm` has a positive relationship with `medv`, meaning higher average numbers of rooms are generally associated with higher median house values.
- `lstat` has a negative relationship with `medv`, meaning higher values of the lower-status population percentage are generally associated with lower median house values.
- Other features also show varying degrees of positive or negative correlation with the target.

---

## ⚙️ Machine Learning Workflow

### 1. Feature and Target Separation

The target variable `medv` was separated from the input features.

```python
X = df.drop("medv", axis=1)
y = df["medv"]
```

---

### 2. Train-Test Split

The dataset was divided into training and testing sets using an 80:20 split.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

This resulted in:

- Training set: **404 observations**
- Testing set: **102 observations**

---

### 3. Feature Scaling

Feature scaling was applied to the input features for the Linear Regression model using `StandardScaler`.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

The scaler was fitted only on the training data and then applied to the test data to prevent data leakage.

---

## 🤖 Machine Learning Models

Two regression models were implemented and compared.

### Linear Regression

Linear Regression was used as the baseline regression model.

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
```

### Random Forest Regression

Random Forest Regression was implemented as a non-linear regression model.

```python
from sklearn.ensemble import RandomForestRegressor

rf_model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)
```

Feature scaling was not required for the Random Forest model.

---

## 📈 Model Evaluation

The models were evaluated using three regression metrics:

### Mean Absolute Error (MAE)

MAE measures the average absolute difference between the actual and predicted values.

Lower MAE indicates smaller average prediction errors.

### Mean Squared Error (MSE)

MSE measures the average squared difference between actual and predicted values.

Lower MSE indicates smaller prediction errors.

### R² Score

R² indicates how much of the variation in the target variable is explained by the model.

A higher R² indicates that the model explains more of the variation in the target data.

---

## 📊 Results

The models were evaluated on the 20% test set using `random_state=42`.

| Model | MAE | MSE | R² Score |
|---|---:|---:|---:|
| Linear Regression | 3.1891 | 24.2911 | 0.6688 |
| Random Forest | 2.0413 | 8.5102 | 0.8840 |

The Random Forest model produced lower MAE and MSE and a higher R² score than the Linear Regression model on this particular test split.

These results represent the performance obtained from the specified train-test split and should not be interpreted as a universal measure of model performance.

---

## 💾 Saving the Model

The trained Random Forest model can be saved using `joblib`.

```python
import joblib

joblib.dump(
    rf_model,
    "C:/Users/Shreyanshi/ml/House-price-prediction/models/house_price_random_forest.pkl"
)
```

The saved model can later be loaded without retraining:

```python
rf_model = joblib.load(
    "C:/Users/Shreyanshi/ml/House-price-prediction/models/house_price_random_forest.pkl"
)
```

---

## 📁 Project Structure

```text
House-price-prediction/
│
├── data/
│   └── BostonHousing.csv
│
├── models/
│   └── house_price_random_forest.pkl
│
├── notebooks/
│   └── 01_house_price_prediction.ipynb
│
├── src/
│
├── .gitignore
├── README.md
└── requirements.txt
```

The local Conda environment and Jupyter checkpoint files are excluded from version control using `.gitignore`.

---

## 🛠️ Technologies Used

- Python
- Jupyter Notebook
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib
- Git
- GitHub

---

## 🚀 How to Run

### 1. Clone the Repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd House-price-prediction
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Launch Jupyter Notebook

```bash
jupyter notebook
```

Open the following notebook:

```text
notebooks/01_house_price_prediction.ipynb
```

Run the notebook cells from top to bottom.

---

## 🔮 Future Improvements

Possible extensions to the project include:

- Hyperparameter tuning for Random Forest
- Cross-validation
- Testing additional regression algorithms
- Feature importance analysis
- Interactive house-price prediction interface
- Streamlit-based deployment
- User-input based house-price predictions

---

# 🎯 Project Objective

This project demonstrates the fundamental workflow of a supervised machine learning regression problem:

**Data → Exploration → Preprocessing → Training → Prediction → Evaluation → Model Saving**

It provides practical experience with tabular data, regression models, feature preprocessing, evaluation metrics, and model persistence.

---

# 👩‍💻 Author

**Shreyanshi Rana**

This project was developed as part of a machine learning internship project to understand and implement a complete regression workflow using Python and Scikit-learn.