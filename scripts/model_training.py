import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    f1_score,
    precision_recall_curve,
    auc,
    confusion_matrix,
    ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt

def split_data(X, y):
    """Splits data into training and testing sets, stratified by the target variable."""
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

def train_logistic_regression(X_train, y_train):
    """Trains a Logistic Regression model."""
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, y_train)
    return model

def train_random_forest(X_train, y_train):
    """Trains a Random Forest Classifier."""
    # class_weight='balanced' can be an alternative or complement to SMOTE
    model = RandomForestClassifier(random_state=42, n_estimators=100, class_weight='balanced')
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test, model_name="Model"):
    """
    Evaluates the model using F1-score, AUC-PR, and Confusion Matrix.
    Returns a dictionary of the metrics.
    """
    # Get predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Calculate metrics
    f1 = f1_score(y_test, y_pred)
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    auc_pr = auc(recall, precision)

    print(f"--- Evaluation Metrics for {model_name} ---")
    print(f"F1-Score: {f1:.4f}")
    print(f"AUC-PR (Area Under Precision-Recall Curve): {auc_pr:.4f}")
    print("\nConfusion Matrix:")

    # Plot Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix - {model_name}")
    plt.show()

    return {"f1_score": f1, "auc_pr": auc_pr}