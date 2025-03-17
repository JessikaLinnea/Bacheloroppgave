import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE  # Fix class imbalance


import matplotlib.pyplot as plt
import numpy as np




# Load the training dataset
training_data = pd.read_csv('Dataset_extended.csv', dtype=str, sep=";")  # Load as strings to prevent conversion errors





# Convert boolean values (True/False stored as strings) into integers (0/1)
bool_columns = ['Extreme weather', 'Ongoing conflicts', 'Strike']
for col in bool_columns:
    training_data[col] = training_data[col].map({'TRUE': 1, 'FALSE': 0}).astype(int)

# Convert numeric fields to float
numeric_columns = ['Stock change over 1 month (%)', 'ESG score']
for col in numeric_columns:
    training_data[col] = training_data[col].str.replace(',', '.').astype(float)

# Keep original categorical fields for reference
categorical_columns = ['Maintain (on hold)', 'Reason code', 'Credit rating']
original_data = training_data.copy()  # Preserve original dataset for reference


# Encode categorical features (One-Hot Encoding)
X_train = pd.get_dummies(training_data.drop('Score', axis=1), columns=categorical_columns)
y_train = training_data['Score']




# Check class distribution before applying SMOTE
print("Class Distribution Before Balancing:")
print(y_train.value_counts())

# Apply SMOTE with k_neighbors set to 1 to avoid errors
smote = SMOTE(sampling_strategy='auto', k_neighbors=1, random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Check class distribution after balancing
print("Class Distribution After Balancing:")
print(pd.Series(y_train_resampled).value_counts())

# Split into training and test sets
X_train_split, X_test_split, y_train_split, y_test_split = train_test_split(
    X_train_resampled, y_train_resampled, test_size=0.2, random_state=42
)

# Train the model with class balancing
model = RandomForestClassifier(class_weight="balanced", random_state=42)
model.fit(X_train_split, y_train_split)

# Predictions
y_pred = model.predict(X_test_split)
accuracy = accuracy_score(y_test_split, y_pred)
report = classification_report(y_test_split, y_pred, zero_division=1)

# Print results
print(f"Model Accuracy: {accuracy}")
print("Classification Report:")
print(report)

# Input new data for prediction (keeping strings for reference)
new_data = {
    'Maintain (on hold)': 'No',
    'Reason code': 'NULL',
    'Credit rating': 'A',
    'Purchase order': 4,
    'Non conformance': 1,
    'Extreme weather': True,
    'Stock change over 1 month (%)': 7.24,
    'Stock price (USD)': 422.71,
    'EU': False,
    'Workdays next 30 days': 21,
    'Ongoing conflicts': False,
    'ESG score': 73.84,
    'Strike': True
}

# Convert new data into DataFrame
new_data_df = pd.DataFrame([new_data])


# Convert boolean fields in new data to 0/1
for col in bool_columns:
    new_data_df[col] = new_data_df[col].map({True: 1, False: 0}).astype(int)



# Encode categorical features (One-Hot Encoding for new data)
new_data_df_encoded = pd.get_dummies(new_data_df, columns=categorical_columns)

# Ensure it has the same columns as training data
new_data_df_encoded = new_data_df_encoded.reindex(columns=X_train.columns, fill_value=0)

# Predict using the trained model
new_data_prediction = model.predict(new_data_df_encoded)
print(f"Prediction for the new data: {new_data_prediction[0]}")








# Hent feature importance fra den trente modellen
feature_importances = model.feature_importances_
feature_names = X_train.columns

# Sorter viktige funksjoner i synkende rekkefølge
indices = np.argsort(feature_importances)[::-1]

# Plott feature importance for å visualisere 
plt.figure(figsize=(12, 6))
plt.title("Feature Importance")
plt.bar(range(len(feature_importances)), feature_importances[indices], align="center")
plt.xticks(range(len(feature_importances)), np.array(feature_names)[indices], rotation=90)
plt.xlabel("Funksjoner")
plt.ylabel("Viktighet")
plt.show()
