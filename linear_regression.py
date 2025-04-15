import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

#dataset
data = {
    "area": [1000, 1500, 2000, 2500, 3000],
    "price": [200000, 250000, 300000, 350000, 400000]
}

df = pd.DataFrame(data)

X = df[['area']] #independent variable
y = df['price'] #dependent variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=45)

print(f"X_train: {X_train}")
print(f"X_test: {X_test}")
print(f"y_train: {y_train}")
print(f"y_train: {y_test}")

#Linear regression Model train
model = LinearRegression()
model.fit(X_train, y_train)

#Pridict
y_pred = model.predict(X_test)


print(f"Prediction price: {y_pred}")
print(f"Model slope(coefficient): {model.coef_}")
print(f"Model intercept: {model.intercept_}")


#plotting
plt.scatter(X, y, color='blue')
plt.plot(X, model.predict(X), color='red')
plt.xlabel("Area")
plt.ylabel("Price ($)")
plt.title("Linear Regression - Area vs Price")
plt.show()

