# Lab 04 — Supervised Machine Learning (Linear & Logistic Regression)

This laboratory covers regression and classification modeling using **scikit-learn**, data exploration, feature preparation, and model evaluation metrics.

---

## 1. Laboratory Notebooks

### 1. `Simple Linear Regression.ipynb`
- **Objective:** Predict salary based on years of experience ($y = \beta_0 + \beta_1 x$).
- **Dataset:** `Salary_dataset.csv` (Features: `YearsExperience`, Target: `Salary`).
- **Implementation:**
  - 80/20 train-test split via `train_test_split(..., random_state=42)`.
  - Ordinary Least Squares model fitting with `LinearRegression()`.
  - Coefficient ($\beta_1$) and intercept ($\beta_0$) extraction.
  - Test set prediction and visualization of regression line vs. actual scatter points.
  - Evaluation: MAE, MSE, RMSE calculation.

### 2. `Multi Linear Regression.ipynb`
- **Objective:** Predict residential house prices based on multi-dimensional housing attributes.
- **Dataset:** `house_price_regression_dataset.csv`.
- **Features:** `Square_Footage`, `Num_Bedrooms`, `Num_Bathrooms`, `Year_Built`, `Lot_Size`, `Garage_Size`, `Neighborhood_Quality`.
- **Target:** `House_Price`.
- **Evaluation Metrics:**
  - Mean Absolute Error (MAE)
  - Mean Squared Error (MSE)
  - Root Mean Squared Error (RMSE)
  - Coefficient of Determination ($R^2$ Score)

### 3. `diabetes-data-exploration-and-logistic-regressi.ipynb`
- **Objective:** Binary classification of diabetes diagnosis ($0$ = Non-Diabetic, $1$ = Diabetic).
- **Dataset:** `diabetes.csv`.
- **Features:** `Age`, `BMI`, `BloodPressure`, `Insulin`, `Glucose`, `DiabetesPedigreeFunction`.
- **Target:** `Outcome`.
- **Implementation:**
  - Logistic Regression classification with `max_iter=1000`.
  - Accuracy evaluation, Confusion Matrix visualization with Seaborn heatmap.
  - Individual patient probability scoring.
  - Model serialization to `Mymdoel.pkl` for API integration in LAB06.
