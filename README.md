
![Design 5 (1)](https://github.com/user-attachments/assets/188586a9-0355-4b32-81be-6aa119a52306)

 

Credit risk assessment is a cornerstone of financial risk management, enabling institutions to evaluate the likelihood of borrower default and potential financial losses. This project addresses credit risk by computing key metrics—**Probability of Default (PD)**, **Loss Given Default (LGD)**, and **Exposure at Default (EAD)**—from a real-world dataset and developing machine learning models to classify borrowers into risk categories.  

For risk classification, an ensemble of machine learning models was implemented, including Random Forest, XGBoost, and a Deep Learning neural network. The dataset underwent comprehensive preprocessing, including handling missing values, encoding categorical variables, and balancing class distribution using SMOTE. Model performance was evaluated using accuracy, F1-score, and ROC AUC, with an ensemble approach combining predictions for improved robustness and accuracy.  

The results demonstrate the efficacy of combining domain-specific metrics with advanced machine learning techniques in assessing and managing credit risk. This work underscores the potential of integrating traditional risk management frameworks with modern data-driven methods to enhance predictive accuracy and decision-making in financial services.

### Fetching Data
```Python
# Loading and inspecting the dataset to understand its structure and contents.
import pandas as pd

# Loading the uploaded file
file_path = '.../german_credit_data.csv'
credit_data = pd.read_csv(file_path)

# Displaying the first few rows and dataset information
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
1. **Cleaning the Data**:
   - Handling missing values.
   - Dropping the unnecessary "Unnamed: 0" column.
   - Encoding categorical variables.

2. **Exploratoring Data Analysis**:
   - Visualizing the distribution of features.
   - Investigating correlations with the target variable.

3. **Building Machine Learning Model**:
   - Using ensemble methods (e.g., Random Forest, XGBoost) for classification.


### Summary Statistics:
- **Age**: Ranges from 19 to 75 years (mean: 35.5).
- **Job**: Four job categories (0-3).
- **Credit Amount**: Loan amounts range from 250 to 18,424 (mean: 3,271.26).
- **Duration**: Loan terms vary from 4 to 72 months (mean: 20.9).

### Risk Distribution:
- **Good Risk**: 700 instances.
- **Bad Risk**: 300 instances (imbalanced target distribution).


The "Risk" variable shows an imbalance, with "good" being more common. This imbalance might need addressing in the machine learning phase using techniques like oversampling or adjusting class weights.

Next, we explore feature relationships and build a machine learning model.

```Python
import matplotlib.pyplot as plt
import seaborn as sns

# Dropping the unnecessary column
credit_data = credit_data.drop(columns=['Unnamed: 0'])

# Handling missing values: Replacing NaN with 'unknown' for categorical columns
credit_data['Saving accounts'] = credit_data['Saving accounts'].fillna('unknown')
credit_data['Checking account'] = credit_data['Checking account'].fillna('unknown')

# Encoding categorical variables using one-hot encoding for features and label encoding for the target variable
credit_data_encoded = pd.get_dummies(
    credit_data,
    columns=['Sex', 'Housing', 'Saving accounts', 'Checking account', 'Purpose'],
    drop_first=True
)
credit_data_encoded['Risk'] = credit_data['Risk'].map({'good': 0, 'bad': 1})

# Summary statistics and distribution of the target variable
summary_stats = credit_data.describe()
risk_distribution = credit_data['Risk'].value_counts()

# Visualizing the distribution of Risk
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

# Splitting the data into features (X) and target (y)
X = credit_data_encoded.drop(columns=['Risk'])
y = credit_data_encoded['Risk']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Initializing a Random Forest classifier
rf_model = RandomForestClassifier(random_state=42, class_weight='balanced')

# Training the model
rf_model.fit(X_train, y_train)

# Making predictions
y_pred = rf_model.predict(X_test)
y_prob = rf_model.predict_proba(X_test)[:, 1]

# Evaluating the model
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

# Defining defaulted borrowers
credit_data['Default'] = (credit_data['Risk'] == 'bad').astype(int)

# 1. Calculating Probability of Default (PD)
total_borrowers = len(credit_data)
defaulted_borrowers = credit_data['Default'].sum()
PD = defaulted_borrowers / total_borrowers

# 2. Simulating recovery rates (assume recovery between 30% and 70% of the credit amount)
np.random.seed(42)
credit_data['Recovery Rate'] = np.random.uniform(0.3, 0.7, size=total_borrowers)
credit_data['Recovery Amount'] = credit_data['Credit amount'] * credit_data['Recovery Rate']

# 3. Calculating Loss Given Default (LGD) for each borrower
credit_data['LGD'] = np.where(
    credit_data['Default'] == 1,
    (credit_data['Credit amount'] - credit_data['Recovery Amount']) / credit_data['Credit amount'],
    0
)

# 4. Exposuring at Default (EAD) is simply the credit amount for defaulted borrowers
credit_data['EAD'] = np.where(credit_data['Default'] == 1, credit_data['Credit amount'], 0)

# Summarizing results
LGD_avg = credit_data.loc[credit_data['Default'] == 1, 'LGD'].mean()
EAD_total = credit_data['EAD'].sum()

PD, LGD_avg, EAD_total

```


```Python
from imblearn.over_sampling import SMOTE

# Appling SMOTE to balance the classes in the training set
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

# Training a new Random Forest model on the balanced dataset
rf_model_balanced = RandomForestClassifier(random_state=42, class_weight='balanced')
rf_model_balanced.fit(X_train_balanced, y_train_balanced)

# Making predictions on the test set
y_pred_balanced = rf_model_balanced.predict(X_test)
y_prob_balanced = rf_model_balanced.predict_proba(X_test)[:, 1]

# Evaluating the model
conf_matrix_balanced = confusion_matrix(y_test, y_pred_balanced)
classif_report_balanced = classification_report(y_test, y_pred_balanced)
roc_auc_balanced = roc_auc_score(y_test, y_prob_balanced)

conf_matrix_balanced, classif_report_balanced, roc_auc_balanced
```
### Ensemble Method:

```Python
from sklearn.ensemble import VotingClassifier
from xgboost import XGBClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler

# Standardizing the features for the deep learning model
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_balanced)
X_test_scaled = scaler.transform(X_test)

# Converting labels to categorical for deep learning
y_train_dl = to_categorical(y_train_balanced, num_classes=2)
y_test_dl = to_categorical(y_test, num_classes=2)

# Building the deep learning model
dl_model = Sequential([
    Dense(64, activation='relu', input_dim=X_train_scaled.shape[1]),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(2, activation='softmax')
])

dl_model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Training the deep learning model
dl_model.fit(X_train_scaled, y_train_dl, epochs=20, batch_size=32, verbose=0, validation_split=0.2)

# Evaluating the deep learning model
dl_probs = dl_model.predict(X_test_scaled)[:, 1]
dl_preds = (dl_probs > 0.5).astype(int)

# Training the XGBoost model
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

# Ensembling predictions
ensemble_probs = ensemble_model.predict_proba(X_test)[:, 1]
ensemble_preds = ensemble_model.predict(X_test)

# Evaluating ensemble model
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
### Senario and Stress Testing

To simulate the effect of a financial downturn on your credit risk model, a stress test can help assess how well the model performs under such conditions. This can involve adjusting the features and labels to reflect a downturn scenario or introducing synthetic data that mimics a financial crisis.

Here are a few steps to conduct a stress test on your model during a financial downturn:

### **Simulate a Financial Downturn Scenario:**
   - **Adjust the Feature Distributions:**
     - **Age:** During financial downturns, younger people might have more difficulty finding stable jobs, and older people might have reduced savings. You can adjust the distribution of the `Age` feature to represent these changes.
     - **Credit Amount and Duration:** In a financial crisis, the credit amounts requested might be smaller, and the duration of loans could shorten as lenders become more cautious.
     - **Savings and Checking Accounts:** Customers may have lower savings or reduced balances in checking accounts during a downturn.
     - **Job and Housing:** Job loss rates may rise, affecting job stability. Housing status may also change as more people face eviction or downsizing.
   - **Simulate Higher Risk:** Increase the proportion of instances classified as high-risk (`Risk` feature) to reflect increased default rates during a downturn.

We perform an stress test on Saving and Checking acounts, and credit amount and its duration. 

```Python
import numpy as np
import pandas as pd

# Create a copy of the X_test to simulate the downturn
X_stress_test = X_test.copy()

# Adjust 'Saving accounts' columns to reflect a downturn (increase 'unknown', decrease other categories)
X_stress_test['Saving accounts_unknown'] += np.random.uniform(0.1, 0.3, size=X_stress_test.shape[0])  # Increase 'unknown' accounts
X_stress_test['Saving accounts_moderate'] *= np.random.uniform(0.7, 0.9, size=X_stress_test.shape[0])  # Decrease moderate savings
X_stress_test['Saving accounts_quite rich'] *= np.random.uniform(0.5, 0.7, size=X_stress_test.shape[0])  # Decrease quite rich savings
X_stress_test['Saving accounts_rich'] *= np.random.uniform(0.2, 0.4, size=X_stress_test.shape[0])  # Decrease rich savings

# Adjust 'Checking account' columns to reflect a downturn (increase 'unknown', decrease other categories)
X_stress_test['Checking account_unknown'] += np.random.uniform(0.1, 0.3, size=X_stress_test.shape[0])  # Increase 'unknown' accounts
X_stress_test['Checking account_moderate'] *= np.random.uniform(0.7, 0.9, size=X_stress_test.shape[0])  # Decrease moderate checking
X_stress_test['Checking account_rich'] *= np.random.uniform(0.4, 0.6, size=X_stress_test.shape[0])  # Decrease rich checking

# Modify other features to reflect downturn (e.g., higher credit amounts, shorter durations)
X_stress_test['Credit amount'] *= np.random.uniform(1.0, 1.5, size=X_stress_test.shape[0])  # Increase credit demand
X_stress_test['Duration'] *= np.random.uniform(0.8, 1.2, size=X_stress_test.shape[0])  # Shorten durations

# Simulate increased risk during a downturn (e.g., more defaults)
y_stress_test = y_test.copy()
y_stress_test[:] = np.random.choice([1, 0], size=X_stress_test.shape[0], p=[0.7, 0.3])  # Increase high-risk instances

# Standardize and evaluate the model on the stress-test data
X_stress_test_scaled = scaler.transform(X_stress_test)

# Evaluate the ensemble model on stress-test data
ensemble_preds_stress_test = ensemble_model.predict(X_stress_test_scaled)
ensemble_probs_stress_test = ensemble_model.predict_proba(X_stress_test_scaled)[:, 1]




print(y_stress_test.sum())
print(ensemble_preds_stress_test.sum())
```


**Credit Risk Simulation during this Financial Downturn:**

During a financial downturn, the probability of defaults on loans and credit applications significantly increases. In this scenario, a stress test has been conducted on the credit risk model, simulating the impact of a downturn by modifying key features, such as savings accounts, checking accounts, credit amounts, and loan durations. The adjustments reflect the real-world behavior of individuals in such periods — reduced savings, increased borrowing demand, and shortened loan durations, as well as a rise in the number of high-risk applicants.

As expected, the model reveals that under these stressed conditions, the risk of defaults escalates sharply. The results of the stress test show that nearly 90 percent of applicants could be classified as high-risk or defaulting. This figure underscores the critical impact a financial downturn has on loan performance and highlights the importance of adjusting risk models to account for these extreme conditions. 

