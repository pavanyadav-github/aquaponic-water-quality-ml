# src/train_voting_classifier.py
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE
import pandas as pd
import joblib

from src.preprocessing import load_and_preprocess

def main():
    # 1. Load & preprocess
    X, y, df = load_and_preprocess('water_quality_aquaponic_dataset.csv')

    # 2. Handle class imbalance
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    print("Before SMOTE:\n", y.value_counts())
    print("\nAfter SMOTE:\n", pd.Series(y_resampled).value_counts())

    # 3. Initialize models
    model1 = GradientBoostingClassifier(random_state=42)
    model2 = AdaBoostClassifier(random_state=42)
    model3 = RandomForestClassifier(random_state=42)
    model4 = DecisionTreeClassifier(random_state=42)

    # 4. Voting Classifier
    voting_clf = VotingClassifier(
        estimators=[('gb', model1), ('ada', model2), ('rf', model3), ('dt', model4)],
        voting='hard'
    )
    voting_clf.fit(X_resampled, y_resampled)
    print("Voting Classifier trained successfully!")

    # 5. Evaluate on training data
    y_pred = voting_clf.predict(X_resampled)
    print("\nConfusion Matrix:\n", confusion_matrix(y_resampled, y_pred))
    print("\nClassification Report:\n", classification_report(y_resampled, y_pred))
    print("\nAccuracy Score:", accuracy_score(y_resampled, y_pred))

    # 6. Train/Test Split evaluation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    voting_clf.fit(X_train, y_train)
    y_pred_test = voting_clf.predict(X_test)
    print("\nTest Confusion Matrix:\n", confusion_matrix(y_test, y_pred_test))
    print("\nTest Classification Report:\n", classification_report(y_test, y_pred_test))
    print("\nTest Accuracy:", accuracy_score(y_test, y_pred_test))

    # 7. Save model
    joblib.dump(voting_clf, 'voting_classifier_model.pkl')
    print("Model saved as: voting_classifier_model.pkl")

if __name__ == "__main__":
    main()
