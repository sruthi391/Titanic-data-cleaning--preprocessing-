
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the cleaned dataset
df = pd.read_csv("cleaned_titanic_dataset.csv")

# Select numerical features
numerical_features = ['Age', 'Fare', 'SibSp', 'Parch']

# Generate boxplot
plt.figure(figsize=(10, 6))
sns.boxplot(data=df[numerical_features])
plt.title("Boxplots for Numerical Features After Cleaning")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("boxplot.png")
print("Saved boxplot.png")
