### Fetching Data
```Python
# Load and inspect the dataset to understand its structure and contents.
import pandas as pd

# Load the uploaded file
file_path = '.../german_credit_data.csv'
credit_data = pd.read_csv(file_path)

# Display the first few rows and dataset information
credit_data_info = credit_data.info()
credit_data_head = credit_data.head()

credit_data_info, credit_data_head
```


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

```Python
import numpy as np

# Define defaulted borrowers
credit_data['Default'] = (credit_data['Risk'] == 'bad').astype(int)

# 1. Calculate Probability of Default (PD)
total_borrowers = len(credit_data)
defaulted_borrowers = credit_data['Default'].sum()
PD = defaulted_borrowers / total_borrowers

# 2. Simulate recovery rates (assume recovery between 30% and 70% of the credit amount)
np.random.seed(42)
credit_data['Recovery Rate'] = np.random.uniform(0.3, 0.7, size=total_borrowers)
credit_data['Recovery Amount'] = credit_data['Credit amount'] * credit_data['Recovery Rate']

# 3. Calculate Loss Given Default (LGD) for each borrower
credit_data['LGD'] = np.where(
    credit_data['Default'] == 1,
    (credit_data['Credit amount'] - credit_data['Recovery Amount']) / credit_data['Credit amount'],
    0
)

# 4. Exposure at Default (EAD) is simply the credit amount for defaulted borrowers
credit_data['EAD'] = np.where(credit_data['Default'] == 1, credit_data['Credit amount'], 0)

# Summarize results
LGD_avg = credit_data.loc[credit_data['Default'] == 1, 'LGD'].mean()
EAD_total = credit_data['EAD'].sum()

PD, LGD_avg, EAD_total

```

It seems like I can’t do more advanced data analysis right now. Please try again later. If you'd like, I can guide you step-by-step on how to implement SMOTE and re-evaluate the model on your system. Let me know how you'd like to proceed!

```Python
from imblearn.over_sampling import SMOTE

# Apply SMOTE to balance the classes in the training set
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

# Train a new Random Forest model on the balanced dataset
rf_model_balanced = RandomForestClassifier(random_state=42, class_weight='balanced')
rf_model_balanced.fit(X_train_balanced, y_train_balanced)

# Make predictions on the test set
y_pred_balanced = rf_model_balanced.predict(X_test)
y_prob_balanced = rf_model_balanced.predict_proba(X_test)[:, 1]

# Evaluate the model
conf_matrix_balanced = confusion_matrix(y_test, y_pred_balanced)
classif_report_balanced = classification_report(y_test, y_pred_balanced)
roc_auc_balanced = roc_auc_score(y_test, y_prob_balanced)

conf_matrix_balanced, classif_report_balanced, roc_auc_balanced
```
### Ensemble:

```Python
from sklearn.ensemble import VotingClassifier
from xgboost import XGBClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler

# Standardize the features for the deep learning model
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_balanced)
X_test_scaled = scaler.transform(X_test)

# Convert labels to categorical for deep learning
y_train_dl = to_categorical(y_train_balanced, num_classes=2)
y_test_dl = to_categorical(y_test, num_classes=2)

# Build the deep learning model
dl_model = Sequential([
    Dense(64, activation='relu', input_dim=X_train_scaled.shape[1]),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(2, activation='softmax')
])

dl_model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the deep learning model
dl_model.fit(X_train_scaled, y_train_dl, epochs=20, batch_size=32, verbose=0, validation_split=0.2)

# Evaluate the deep learning model
dl_probs = dl_model.predict(X_test_scaled)[:, 1]
dl_preds = (dl_probs > 0.5).astype(int)

# Train the XGBoost model
xgb_model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X_train_balanced, y_train_balanced)

# XGBoost predictions
xgb_probs = xgb_model.predict_proba(X_test)[:, 1]
xgb_preds = xgb_model.predict(X_test)

# Combine all models in a voting ensemble
ensemble_model = VotingClassifier(
    estimators=[
        ('rf', rf_model_balanced),
        ('xgb', xgb_model),
        ('dl', rf_model_balanced)  # Placeholder for deep learning (adjust later if needed)
    ],
    voting='soft'
)
ensemble_model.fit(X_train_balanced, y_train_balanced)

# Ensemble predictions
ensemble_probs = ensemble_model.predict_proba(X_test)[:, 1]
ensemble_preds = ensemble_model.predict(X_test)

# Evaluate ensemble model
conf_matrix_ensemble = confusion_matrix(y_test, ensemble_preds)
classif_report_ensemble = classification_report(y_test, ensemble_preds)
roc_auc_ensemble = roc_auc_score(y_test, ensemble_probs)

conf_matrix_ensemble, classif_report_ensemble, roc_auc_ensemble
```

```Python
[[118  22]
 [ 27  33]]
              precision    recall  f1-score   support

           0       0.81      0.84      0.83       140
           1       0.60      0.55      0.57        60

    accuracy                           0.76       200
   macro avg       0.71      0.70      0.70       200
weighted avg       0.75      0.76      0.75       200

0.7527380952380952

```

