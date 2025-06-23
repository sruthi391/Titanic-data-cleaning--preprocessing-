# Titanic Data Cleaning & Preprocessing

This repository contains the Python code and preprocessed data for cleaning and preparing the raw Titanic dataset for machine learning.

## Objective

The main objective of this project is to demonstrate fundamental data cleaning and preprocessing techniques, including:
1.  Handling missing values.
2.  Converting categorical features to numerical.
3.  Normalizing/standardizing numerical features.
4.  Identifying and removing outliers.

## Files Provided

* `main.py`: The Python script containing all the data cleaning and preprocessing steps.
* `Titanic_Final_Preprocessed_Data.csv`: The final preprocessed dataset after handling missing values, encoding categorical features, scaling numerical features, and removing outliers.
* `Titanic_Preprocessed_Data.csv`: An intermediate preprocessed dataset after handling missing values, encoding categorical features, and scaling numerical features, but *before* outlier removal.
* `Outlier_Boxplots_Before_Removal.png`: Boxplots visualizing outliers in numerical features before their removal.
* `Outlier_Boxplots_After_Removal.png`: Boxplots visualizing numerical features after outlier removal.
* `requirements.txt`: A list of Python libraries required to run `main.py`.

## How to Use

1.  **Download the dataset:** Ensure you have `Titanic-Dataset.csv` in the same directory as `main.py`.
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Run the script:**
    ```bash
    python main.py
    ```

This will generate the preprocessed CSV files and the boxplot images in the same directory.

## Steps Performed in `main.py`

1.  **Data Loading & Initial Exploration:**
    * Loads `Titanic-Dataset.csv` into a Pandas DataFrame.
    * Displays basic information (`.info()`) and counts of null values.

2.  **Missing Value Handling:**
    * 'Age' column: Missing values are imputed with the median age.
    * 'Embarked' column: Missing values are imputed with the most frequent embarkation point (mode).
    * 'Cabin' column: Dropped due to a very high percentage of missing values.

3.  **Feature Engineering & Cleaning:**
    * 'Sex' and 'Embarked' columns are one-hot encoded to convert them into numerical representations.
    * Irrelevant columns like 'PassengerId', 'Name', and 'Ticket' are dropped.

4.  **Feature Scaling (Standardization):**
    * Numerical features ('Age', 'Fare', 'SibSp', 'Parch') are standardized using `StandardScaler` to bring them to a common scale (mean=0, variance=1).

5.  **Outlier Detection and Removal:**
    * Boxplots are generated to visualize outliers in numerical features before removal.
    * Outliers are identified and removed using the Interquartile Range (IQR) method (values outside 1.5 * IQR from Q1 and Q3 are removed).
    * Boxplots are generated again to show the distribution after outlier removal.

This README provides a comprehensive overview of the data preprocessing pipeline.
