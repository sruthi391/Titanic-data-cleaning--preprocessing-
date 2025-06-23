
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

# Load the dataset
df = pd.read_csv('Titanic-Dataset.csv')

# Display basic info
print("Dataset Info:")
df.info()

# Display null values
print("\nNull Values Before Handling:")
print(df.isnull().sum())

# Handle missing values
# Impute 'Age' with the median
imputer_age = SimpleImputer(strategy='median')
df['Age'] = imputer_age.fit_transform(df[['Age']])

# Impute 'Embarked' with the most frequent value (mode)
imputer_embarked = SimpleImputer(strategy='most_frequent')
df['Embarked'] = imputer_embarked.fit_transform(df[['Embarked']]).ravel()

# Drop 'Cabin' column due to high number of missing values
df.drop('Cabin', axis=1, inplace=True)

print("\nNull Values After Handling:")
print(df.isnull().sum())

# Convert categorical features into numerical using encoding
# Identify categorical columns for encoding
categorical_cols = ['Sex', 'Embarked']
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Drop irrelevant columns that are not useful for machine learning
df.drop(['PassengerId', 'Name', 'Ticket'], axis=1, inplace=True)

# Normalize/standardize the numerical features
# Identify numerical columns for scaling
numerical_cols = ['Age', 'Fare', 'SibSp', 'Parch']
scaler = StandardScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

print("\nFirst 5 rows of the preprocessed DataFrame:")
print(df.head())

# Save the preprocessed data to a new CSV file
df.to_csv('Titanic_Preprocessed_Data.csv', index=False)

# Visualize outliers using boxplots and remove them
numerical_cols_for_outliers = ['Age', 'Fare', 'SibSp', 'Parch']

# Create boxplots for numerical features to visualize outliers
plt.figure(figsize=(15, 10))
for i, col in enumerate(numerical_cols_for_outliers):
    plt.subplot(2, 2, i+1)
    sns.boxplot(y=df[col])
    plt.title(f'Boxplot of {col}')
plt.tight_layout()
plt.savefig('Outlier_Boxplots_Before_Removal.png')

# Remove outliers using IQR method
for col in numerical_cols_for_outliers:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

# Create boxplots after outlier removal to confirm
plt.figure(figsize=(15, 10))
for i, col in enumerate(numerical_cols_for_outliers):
    plt.subplot(2, 2, i+1)
    sns.boxplot(y=df[col])
    plt.title(f'Boxplot of {col} After Outlier Removal')
plt.tight_layout()
plt.savefig('Outlier_Boxplots_After_Removal.png')

# Display the shape of the DataFrame after outlier removal
print(f"\nShape of DataFrame after outlier removal: {df.shape}")

# Save the final preprocessed data after outlier removal
df.to_csv('Titanic_Final_Preprocessed_Data.csv', index=False)
