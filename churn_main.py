import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from joblib import dump

# Load the dataset
telecom_cust = pd.read_csv('Telco_Customer_Churn.csv')

# Data preprocessing
# Fix TotalCharges
telecom_cust['TotalCharges'] = pd.to_numeric(telecom_cust['TotalCharges'], errors='coerce')
telecom_cust['TotalCharges'].fillna(0, inplace=True)

# Convert 'Churn' to binary labels
label_encoder = LabelEncoder()
telecom_cust['Churn'] = label_encoder.fit_transform(telecom_cust['Churn'])

# Encode additional categorical columns
categorical_cols = [
    'InternetService', 'Contract', 'Partner',
    'Dependents', 'PhoneService', 'PaperlessBilling'
]

for col in categorical_cols:
    telecom_cust[col] = label_encoder.fit_transform(telecom_cust[col])

# ---- Select 10 Features ----
selected_features = [
    'tenure',
    'InternetService',
    'Contract',
    'MonthlyCharges',
    'TotalCharges',
    'SeniorCitizen',
    'Partner',
    'Dependents',
    'PhoneService',
    'PaperlessBilling'
]

X = telecom_cust[selected_features]
y = telecom_cust['Churn']

# Train SVM model
model = SVC(kernel='rbf', probability=True, random_state=101)
model.fit(X, y)

# Save the model
dump(model, 'svm_churn_model.joblib')

print("SVM model trained with 10 features and saved successfully!")
