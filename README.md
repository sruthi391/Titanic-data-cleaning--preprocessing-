# Titanic-data-cleaning--preprocessing-
Data cleaning and preprocessing of the Titanic dataset for machine learning tasks. 
 
## Objective 
This project demonstrates the essential steps to clean and preprocess data for machine 
learning using the Titanic dataset. 
 
## Tools Used 
- Python 
- Pandas, NumPy 
- Seaborn, Matplotlib 
- Scikit-learn 
 
## Steps Performed 
 
1. Data Exploration: Inspected null values and data types. 
2. Missing Value Handling: 
   - Imputed `Age` using median. 
   - Filled `Embarked` with mode. 
   - Dropped `Cabin`, `Name`, `Ticket` due to high missing rates or low relevance. 
3. Encoding: 
   - Converted `Sex` to binary. 
   - Applied one-hot encoding for `Embarked`. 
4. Normalization: 
   - Standardized `Age`, `Fare`, `SibSp`, and `Parch` using `StandardScaler`. 
5. Outlier Detection: 
   - Used boxplots and Z-score method to remove outliers. 
 
## Output 
- `cleaned_titanic_dataset.csv`: Final dataset ready for ML. 
- `boxplot.png`: Outlier visualization.
