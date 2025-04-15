import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
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
# data = df[features + [target]].copy()

# # Encode categorical variables
# label_encoders = {}
# for col in ['Gender', 'Blood Type', 'Medical Condition', 'Admission Type', 'Test Results']:
#     le = LabelEncoder()
#     data[col] = le.fit_transform(data[col])
#     label_encoders[col] = le

# # Split into train and test sets
# X = data[features]
# y = data[target]

X = df[features]
y = df[target]

categorical_cols = ['Gender', 'Blood Type', 'Medical Condition', 'Admission Type', 'Test Results']
numerical_cols = ['Age']

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ],
    remainder='passthrough'
)


# Train Linear Regression model
regressor = RandomForestRegressor()

# Create a pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', regressor)
])

#Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)


print(f"Prediction price: {y_pred}")
# print(f"Model slope(coefficient): {regressor.coef_}")
# print(f"Model intercept: {regressor.intercept_}")

print(f"MSE: {mse}")
print(f"R2: {r2}")

# Visualization
plt.figure(figsize=(8, 5))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.5)
plt.xlabel("Actual Billing Amount")
plt.ylabel("Predicted Billing Amount")
plt.title("Actual vs Predicted Billing Amount (Linear Regression)")
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
sns.histplot(y_test - y_pred, kde=True, bins=50, color='purple')
plt.title("Prediction Error Distribution")
plt.xlabel("Prediction Error (Actual - Predicted)")
plt.grid(True)
plt.tight_layout()
plt.show()

mse, r2

