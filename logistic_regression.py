import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

#load custom dataset
data = {
    'Pclass': [3, 1, 3, 1, 3],
    # 'Age': [28, 32, 22, 25, 36],
    'Age': [22, 38, 26, 35, 28],
    'Survival': [0, 1, 1, 1, 0]
}

df = pd.DataFrame(data)

#feature and target
X = df[['Pclass', 'Age']]
y = df['Survival']

#scatter plot of the data
plt.figure(figsize=(6, 4))
sns.scatterplot(data=df, x='Age', y='Pclass', hue='Survival', palette='coolwarm', s=100)
plt.title("Passenger class vs Age")
plt.gca().invert_yaxis()
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(f"Survival Prediction: {y_pred}")
print(f"Confusion matrix:\n {confusion_matrix(y_test, y_pred)}")
print(f"Classification report:\n {classification_report(y_test, y_pred)}")

#confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[0, 1], yticklabels=[0, 1])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()




