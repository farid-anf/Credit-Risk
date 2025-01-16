The dataset contains 1,000 entries and 11 columns. Here's a summary:

### Columns:
1. **Unnamed: 0**: Index column (likely unnecessary for analysis).
2. **Age**: Borrower's age (integer).
3. **Sex**: Borrower's gender (categorical: male, female).
4. **Job**: Job category (integer).
5. **Housing**: Type of housing (categorical: own, free, rent).
6. **Saving accounts**: Amount in savings accounts (categorical, with some missing values).
7. **Checking account**: Amount in checking accounts (categorical, with some missing values).
8. **Credit amount**: Loan amount (integer).
9. **Duration**: Loan duration in months (integer).
10. **Purpose**: Purpose of the loan (categorical: car, education, etc.).
11. **Risk**: Target variable (categorical: good, bad).

### Observations:
- **Missing Values**: Some missing values in "Saving accounts" and "Checking account."
- **Target Variable**: "Risk" is the classification target.

### Next Steps:
1. **Clean the Data**:
   - Handle missing values.
   - Drop the unnecessary "Unnamed: 0" column.
   - Encode categorical variables.

2. **Exploratory Data Analysis**:
   - Visualize the distribution of features.
   - Investigate correlations with the target variable.

3. **Build Machine Learning Model**:
   - Use ensemble methods (e.g., Random Forest, XGBoost) for classification.

Let me clean the data and start the analysis.

![image](image1)


### Summary Statistics:
- **Age**: Ranges from 19 to 75 years (mean: 35.5).
- **Job**: Four job categories (0-3).
- **Credit Amount**: Loan amounts range from 250 to 18,424 (mean: 3,271.26).
- **Duration**: Loan terms vary from 4 to 72 months (mean: 20.9).

### Risk Distribution:
- **Good Risk**: 700 instances.
- **Bad Risk**: 300 instances (imbalanced target distribution).

### Visualization:
The "Risk" variable shows an imbalance, with "good" being more common. This imbalance might need addressing in the machine learning phase using techniques like oversampling or adjusting class weights.

Next, let's explore feature relationships and build a machine learning model.

```Python
import matplotlib.pyplot as plt
import seaborn as sns

# Drop the unnecessary column
credit_data = credit_data.drop(columns=['Unnamed: 0'])

# Handle missing values: Replace NaN with 'unknown' for categorical columns
credit_data['Saving accounts'] = credit_data['Saving accounts'].fillna('unknown')
credit_data['Checking account'] = credit_data['Checking account'].fillna('unknown')

# Encode categorical variables using one-hot encoding for features and label encoding for the target variable
credit_data_encoded = pd.get_dummies(
    credit_data,
    columns=['Sex', 'Housing', 'Saving accounts', 'Checking account', 'Purpose'],
    drop_first=True
)
credit_data_encoded['Risk'] = credit_data['Risk'].map({'good': 0, 'bad': 1})

# Summary statistics and distribution of the target variable
summary_stats = credit_data.describe()
risk_distribution = credit_data['Risk'].value_counts()

# Visualize the distribution of Risk
plt.figure(figsize=(6, 4))
sns.countplot(x='Risk', data=credit_data, palette='coolwarm')
plt.title('Distribution of Risk')
plt.show()

summary_stats, risk_distribution
```


```Python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Split the data into features (X) and target (y)
X = credit_data_encoded.drop(columns=['Risk'])
y = credit_data_encoded['Risk']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Initialize a Random Forest classifier
rf_model = RandomForestClassifier(random_state=42, class_weight='balanced')

# Train the model
rf_model.fit(X_train, y_train)

# Make predictions
y_pred = rf_model.predict(X_test)
y_prob = rf_model.predict_proba(X_test)[:, 1]

# Evaluate the model
conf_matrix = confusion_matrix(y_test, y_pred)
classif_report = classification_report(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

conf_matrix, classif_report, roc_auc
```

### Model Evaluation:
1. **Confusion Matrix**:
   - True Negatives (Good Risk correctly classified): 125
   - False Positives (Good Risk misclassified as Bad): 15
   - False Negatives (Bad Risk misclassified as Good): 38
   - True Positives (Bad Risk correctly classified): 22

2. **Classification Report**:
   - **Precision**:
     - Good Risk: 77%
     - Bad Risk: 59%
   - **Recall**:
     - Good Risk: 89%
     - Bad Risk: 37%
   - **F1-Score**:
     - Good Risk: 83%
     - Bad Risk: 45%
   - Overall Accuracy: 73%
   - Macro Average F1-Score: 64%

3. **ROC AUC Score**:
   - 0.75 (indicating reasonable separability of the classes).

### Observations:
- The model performs better in predicting "Good Risk" compared to "Bad Risk," likely due to class imbalance.
- Precision for "Bad Risk" is moderate, but recall is lower, meaning some "Bad Risk" instances are being missed.
- The overall ROC AUC score indicates the model captures discriminatory power between the two classes but can be improved.

### Next Steps:
1. Address class imbalance (e.g., using oversampling or synthetic data generation like SMOTE).
2. Tune hyperparameters of the Random Forest model or use other ensemble methods (e.g., XGBoost, LightGBM).
3. Feature importance analysis to understand key predictors of risk.

Would you like to proceed with these enhancements?
