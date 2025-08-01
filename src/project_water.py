# -*- coding: utf-8 -*-
"""Project_water.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1Qo3SjAuk4gHYMlVFNuSf1srr0BEAOsJO
"""

import pandas as pd
from sklearn.impute import KNNImputer

# Step 1: Load the Dataset
file_path = '/content/drive/MyDrive/new_Project/water_quality_aquaponic_dataset.csv'
df = pd.read_csv(file_path)
print('Dataset Loaded Successfully!')
print(df.head(10))

# Step 2: Data Cleaning - Handling Missing Values using MICE (Approximated by KNN Imputer)
imputer = KNNImputer(n_neighbors=5)
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
print('Missing Values Handled Successfully!')
print(df_imputed.head(10))

# Step 3: Adding Separate Multiclass Columns based on Parameter Ranges
df_imputed['Plant'] = df_imputed.apply(lambda row: 1 if (5.5 <= row['pH'] <= 7.5 and 16 <= row['Temperature'] <= 30 and row['Dissolved Oxygen'] > 3 and row['Ammonia'] < 30 and row['Nitrite'] < 1) else 0, axis=1)

df_imputed['Bacteria'] = df_imputed.apply(lambda row: 1 if (6 <= row['pH'] <= 8.5 and 14 <= row['Temperature'] <= 34 and 4 <= row['Dissolved Oxygen'] <= 8 and row['Ammonia'] < 3 and row['Nitrite'] < 1) else 0, axis=1)

df_imputed['Warm Water Fish'] = df_imputed.apply(lambda row: 1 if (6 <= row['pH'] <= 8.5 and 22 <= row['Temperature'] <= 32 and 4 <= row['Dissolved Oxygen'] <= 6 and row['Ammonia'] < 3 and row['Nitrate'] < 400 and row['Nitrite'] < 1) else 0, axis=1)

df_imputed['Cold Water Fish'] = df_imputed.apply(lambda row: 1 if (6 <= row['pH'] <= 8.5 and 10 <= row['Temperature'] <= 21 and 6 <= row['Dissolved Oxygen'] <= 8 and row['Ammonia'] < 1 and row['Nitrate'] < 400 and row['Nitrite'] < 0.1) else 0, axis=1)

print('Multiclass columns added to the dataset successfully!')
print(df_imputed.head(10))

# Step 4: Adding Output Column using MSB Concept (Casting to int to avoid ValueError)
def calculate_output(row):
    binary_str = f"{int(row['Plant'])}{int(row['Bacteria'])}{int(row['Warm Water Fish'])}{int(row['Cold Water Fish'])}"
    return int(binary_str, 2)

df_imputed['Output'] = df_imputed.apply(calculate_output, axis=1)
print('Output column added to the dataset successfully!')
print(df_imputed.head())

print(df_imputed.columns)

# Step 5: Finding Number of Unique Patterns and Their Counts
unique_patterns = df_imputed[['Plant', 'Bacteria', 'Warm Water Fish', 'Cold Water Fish']].value_counts().reset_index()
unique_patterns.columns = ['Plant', 'Bacteria', 'Warm Water Fish', 'Cold Water Fish', 'Count']
num_unique_patterns = len(unique_patterns)
print(f'Number of unique patterns in multiclass columns: {num_unique_patterns}')
print('Unique patterns and their counts:')
print(unique_patterns)

from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestClassifier

# Define features and target columns
features = ['pH', 'Dissolved Oxygen', 'Temperature', 'Ammonia', 'Nitrite', 'Nitrate']
targets = ['Plant', 'Bacteria', 'Warm Water Fish', 'Cold Water Fish']

X = df_imputed[features]
y = df_imputed[targets]

# Initialize a classifier (Random Forest in this example)
model = RandomForestClassifier(random_state=42)

# Apply RFECV with all 6 features included
rfecv = RFECV(estimator=model, step=1, min_features_to_select=6, cv=5, scoring='accuracy')
rfecv.fit(X, y)

# Check which features are selected and their ranking
selected_features = [feature for feature, selected in zip(features, rfecv.support_) if selected]
feature_ranking = rfecv.ranking_

print("Selected Features:", selected_features)
print("Feature Ranking:", feature_ranking)

from imblearn.over_sampling import SMOTE

# Use the 'output' column as the target for SMOTE
y = df_imputed['Output']  # Single target column for SMOTE

# Apply G-SMOTE to handle class imbalance
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Check the class distribution before and after SMOTE
print("Before SMOTE:\n", y.value_counts())
print("\nAfter SMOTE:\n", pd.Series(y_resampled).value_counts())

# model training
# Import Required Libraries
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

# Initialize Individual Classifiers
model1 = GradientBoostingClassifier(random_state=42)
model2 = AdaBoostClassifier(random_state=42)
model3 = RandomForestClassifier(random_state=42)
model4 = DecisionTreeClassifier(random_state=42)

# Train Each Model Separately
model1.fit(X_resampled, y_resampled)
print("Gradient Boosting Model Trained")

model2.fit(X_resampled, y_resampled)
print("AdaBoost Model Trained")

model3.fit(X_resampled, y_resampled)
print("Random Forest Model Trained")

model4.fit(X_resampled, y_resampled)
print("Decision Tree Model Trained")

#using voting classifier
# Import Voting Classifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Combine Models using Voting Classifier (Hard Voting)
voting_clf = VotingClassifier(
    estimators=[
        ('gb', model1),
        ('ada', model2),
        ('rf', model3),
        ('dt', model4)
    ],
    voting='hard'  # Use 'hard' voting to combine predictions
)

# Train the Voting Classifier on Resampled Data
voting_clf.fit(X_resampled, y_resampled)
print("Voting Classifier Model Trained")

# Make Predictions on the Training Data
y_pred = voting_clf.predict(X_resampled)

# (Optional) Model Evaluation
print("\nConfusion Matrix:\n", confusion_matrix(y_resampled, y_pred))
print("\nClassification Report:\n", classification_report(y_resampled, y_pred))
print("\nAccuracy Score:", accuracy_score(y_resampled, y_pred))

#each model performance
# Import Metrics
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# List of Models with Names
models = [
    ('Gradient Boosting', model1),
    ('AdaBoost', model2),
    ('Random Forest', model3),
    ('Decision Tree', model4)
]

# Evaluate Each Model Individually
for name, model in models:
    print(f"\nModel: {name}")

    # Make Predictions on Training Data
    y_pred = model.predict(X_resampled)

    # Display Metrics
    print("Confusion Matrix:\n", confusion_matrix(y_resampled, y_pred))
    print("\nClassification Report:\n", classification_report(y_resampled, y_pred))
    print("Accuracy Score:", accuracy_score(y_resampled, y_pred))
    print("-" * 50)

#on test set
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Retrain the Voting Classifier on the training set
voting_clf.fit(X_train, y_train)
print("Voting Classifier Model Trained on Training Set")

# Evaluate the model on the test set
y_pred_test = voting_clf.predict(X_test)

# Display evaluation metrics
print("\nConfusion Matrix (Test Set):\n", confusion_matrix(y_test, y_pred_test))
print("\nClassification Report (Test Set):\n", classification_report(y_test, y_pred_test))
print("\nAccuracy Score (Test Set):", accuracy_score(y_test, y_pred_test))

import joblib

# Save the trained Voting Classifier model
model_filename = 'voting_classifier_model.pkl'
joblib.dump(voting_clf, model_filename)

print(f'Model saved as: {model_filename}')

# Code to download the model file in Google Colab
from google.colab import files
files.download(model_filename)

"""graph--->

1.Confusion Matrix (Multiclass Version)->
"""

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.show()

"""2. Classification Report (as Table or Heatmap)"""

from sklearn.metrics import classification_report
import seaborn as sns

report = classification_report(y_test, y_pred, output_dict=True)
sns.heatmap(pd.DataFrame(report).iloc[:-1, :].T, annot=True, cmap="YlGnBu")
plt.title("Classification Report")
plt.show()

"""3.Feature Importance Plot"""

import matplotlib.pyplot as plt
import numpy as np

# Assuming 'model' is your DecisionTreeClassifier and 'X_train' is your training data
importances = model.feature_importances_
feature_names = X_train.columns

plt.figure(figsize=(8, 6))
plt.barh(feature_names, importances, color='skyblue')
plt.xlabel("Feature Importance Score")
plt.title("Feature Importances from Decision Tree")
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

"""4.Accuracy vs. Class Bar Plot"""

from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt

# Define your prediction mapping again (if not already in memory)
prediction_mapping = {
    0: 'Not Suitable for Any',
    1: 'Only Cold Water Fish',
    2: 'Only Warm Water Fish',
    3: 'Warm & Cold Water Fish',
    4: 'Only Bacteria',
    5: 'Bacteria & Cold Water Fish',
    6: 'Bacteria & Warm Water Fish',
    7: 'Bacteria, Warm & Cold Water Fish',
    8: 'Only Plant',
    9: 'Plant & Cold Water Fish',
    10: 'Plant & Warm Water Fish',
    11: 'Plant, Warm & Cold Water Fish',
    12: 'Plant & Bacteria',
    13: 'Plant, Bacteria & Cold Water Fish',
    14: 'Plant, Bacteria & Warm Water Fish',
    15: 'Suitable for All'
}

# Calculate per-class accuracy
unique_classes = np.unique(y_test)
accuracies = []

for label in unique_classes:
    indices = y_test == label
    class_acc = accuracy_score(y_test[indices], y_pred[indices])
    accuracies.append(class_acc)

# Plot
plt.figure(figsize=(10, 6))
plt.bar([prediction_mapping[label] for label in unique_classes], accuracies, color='mediumseagreen')
plt.xticks(rotation=90)
plt.ylabel("Accuracy")
plt.title("Per-Class Accuracy of the Model")
plt.tight_layout()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()