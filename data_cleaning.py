
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy import stats

# Load dataset
df = pd.read_csv("Titanic-Dataset.csv")  # Replace with actual file if needed

# 1. Handle missing values
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df.drop(columns=['Cabin', 'Ticket', 'Name'], inplace=True)

# 2. Encode categorical variables
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

# 3. Normalize numerical features
scaler = StandardScaler()
numerical_features = ['Age', 'Fare', 'SibSp', 'Parch']
df[numerical_features] = scaler.fit_transform(df[numerical_features])

# 4. Visualize outliers
plt.figure(figsize=(10, 6))
sns.boxplot(data=df[numerical_features])
plt.title("Boxplots for Numerical Features")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("boxplot.png")

# 5. Remove outliers using Z-score
z_scores = np.abs(stats.zscore(df[numerical_features]))
df_cleaned = df[(z_scores < 3).all(axis=1)]

# 6. Save the cleaned dataset
df_cleaned.to_csv("cleaned_titanic_dataset.csv", index=False)
