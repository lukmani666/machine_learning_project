import numpy as np

# 1D Array
arr = np.array([1, 2, 3])
print(arr)


# 2D Array
arr2 = np.array([[1, 2], [3, 4]])
print(arr2)

print(arr[0])     
print(arr[1:])
print(arr2[1, 0])


a = np.array([1, 2])
b = np.array([3, 4])

a + b       # [4 6]
a * b       # [3 8] (element-wise multiplication)
np.dot(a, b) # 1*3 + 2*4 = 11 (dot product)
np.mean(a)   # 1.5 (average)



import pandas as pd

s = pd.Series([10, 20, 30])
print(s)


data = {
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Age': [25, 30, 35],
    'City': ['Lagos', 'Abuja', 'Kano']
}
df = pd.DataFrame(data)
print(df)


#4. Inspecting Data

df.head()        # First 5 rows
df.info()        # Data types and non-null counts
df.describe()    # Stats summary of numerical columns

#5. Accessing Data

df['Name']       # Access single column
df[['Name', 'Age']]  # Multiple columns
df.loc[0]        # Row by label (index)
df.iloc[1]       # Row by position


#Filtering Rows

df[df['Age'] > 25]


#1. Grouping Data with groupby()
# Used to split, apply a function, and combine. Common for aggregation like sum, mean, etc.

data = {
    'Department': ['IT', 'HR', 'IT', 'Finance', 'HR'],
    'Employee': ['A', 'B', 'C', 'D', 'E'],
    'Salary': [50000, 60000, 55000, 70000, 62000]
}

df = pd.DataFrame(data)

# Group by Department and get average salary
grouped = df.groupby('Department')['Salary'].mean()
print(grouped)


#2. Merging DataFrames with merge()
# Used to combine two DataFrames based on a common column (like SQL joins).

df1 = pd.DataFrame({
    'Employee': ['A', 'B', 'C'],
    'Department': ['IT', 'HR', 'Finance']
})

df2 = pd.DataFrame({
    'Employee': ['A', 'B', 'C'],
    'Salary': [50000, 60000, 70000]
})


#Create sample with missing data:

df = pd.DataFrame({
    'Name': ['Alice', 'Bob', 'Charlie', 'David'],
    'Age': [25, np.nan, 30, np.nan],
    'City': ['Lagos', 'Abuja', np.nan, 'Kano']
})

#Detect missing values

df.isnull()

#Drop rows with missing data

df.dropna()


#Fill missing data with default value

df.fillna({'Age': df['Age'].mean(), 'City': 'Unknown'})


#dropna() removes any rows with NaN (missing) values.

#fillna() replaces missing values with something meaningful.


#Removing Duplicates

df.drop_duplicates()



