import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('healthcare_dataset.csv')

# Show top rows
print(df.head())

# Show basic info
print(df.info())

# Summary stats
print(df.describe(include='all'))

# Select features for regression
features = ['Age', 'Gender', 'Blood Type', 'Medical Condition', 'Admission Type', 'Test Results']
target = 'Billing Amount'

# Copy relevant data
data = df[features + [target]].copy()

# Encode categorical variables
label_encoders = {}
for col in ['Gender', 'Blood Type', 'Medical Condition', 'Admission Type', 'Test Results']:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Split into train and test sets
X = data[features]
y = data[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Linear Regression model
dt_model = DecisionTreeRegressor()
dt_model.fit(X_train, y_train)

# Predict
y_pred_dt = dt_model.predict(X_test)

# Evaluation
mse_dt = mean_squared_error(y_test, y_pred_dt)
r2_dt = r2_score(y_test, y_pred_dt)

# Train Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predict
y_pred_rf = rf_model.predict(X_test)

# Evaluation
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

# Output results
print(f"Decision Tree Prediction price: {y_pred_dt}")
print(f"Random Forest Prediction price: {y_pred_rf}")
print(f"Decision Tree - MSE: {mse_dt}, R2: {r2_dt}")
print(f"Random Forest - MSE: {mse_rf}, R2: {r2_rf}")

# print(f"Prediction price: {y_pred}")
# print(f"Model slope(coefficient): {dt_model.coef_}")
# print(f"Model intercept: {dt_model.intercept_}")

# print(f"MSE: {mse}")
# print(f"R2: {r2}")

# Visualization
# plt.figure(figsize=(8, 5))
# sns.scatterplot(x=y_test, y=y_pred, alpha=0.5)
# plt.xlabel("Actual Billing Amount")
# plt.ylabel("Predicted Billing Amount")
# plt.title("Actual vs Predicted Billing Amount (Linear Regression)")
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# mse, r2

# Visualization
plt.figure(figsize=(8, 5))

# Scatterplot for Decision Tree
sns.scatterplot(x=y_test, y=y_pred_dt, alpha=0.5, label="Decision Tree", color='blue')
sns.scatterplot(x=y_test, y=y_pred_rf, alpha=0.5, label="Random Forest", color='green')

plt.xlabel("Actual Billing Amount")
plt.ylabel("Predicted Billing Amount")
plt.title("Actual vs Predicted Billing Amount (Decision Tree vs Random Forest)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Returning evaluation metrics for both models
mse_dt, r2_dt, mse_rf, r2_rf
