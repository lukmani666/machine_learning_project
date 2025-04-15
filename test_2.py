import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

def housing_model_price_pridiction():
    #load dataset
    df = pd.read_csv("housing.csv")
    print(f"Dataset head: {df.head()}")
    print(f"Shape of dataset: {df.shape}")
    print(f"Features in dataset: {df.columns}")
    print(f"Describe: {df.describe()}")


if __name__ == "__main__":
    housing_model_price_pridiction()