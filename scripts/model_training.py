import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve, auc, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from preprocess import preprocess_pipeline


def prepare_data(fraud_path, ip_path):
    # Preprocess data using the pipeline from Task 1
    X, y, encoder, scaler, _ = preprocess_pipeline(fraud_path, ip_path)
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    return X_train, X_test, y_train, y_test

def train_and_evaluate_model(model, model_name, X_train, X_test, y_train, y_test):
    # Train the model
    model.fit(X_train, y_train)
    # Predict probabilities for AUC-PR
    y_scores = model.predict_proba(X_test)[:, 1]
    # Compute AUC-PR
    precision, recall, _ = precision_recall_curve(y_test, y_scores)
    auc_pr = auc(recall, precision)
    # Compute F1-Score
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred)
    # Compute Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    # Plot Confusion Matrix
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f'confusion_matrix_{model_name.lower().replace(" ", "_")}.png')
    plt.close()
    return auc_pr, f1, cm

def main():
    # Paths to datasets
    fraud_path = "./data/Fraud_Data.csv"
    ip_path = "./data/IpAddress_to_Country.csv"
    
    # Prepare data
    X_train, X_test, y_train, y_test = prepare_data(fraud_path, ip_path)
    
    # Initialize models
    log_reg = LogisticRegression(random_state=42, max_iter=1000)
    rf = RandomForestClassifier(random_state=42, n_estimators=100)
    
    # Train and evaluate Logistic Regression
    auc_pr_lr, f1_lr, cm_lr = train_and_evaluate_model(log_reg, "Logistic Regression", X_train, X_test, y_train, y_test)
    print(f"Logistic Regression - AUC-PR: {auc_pr_lr:.4f}, F1-Score: {f1_lr:.4f}")
    print("Confusion Matrix:\n", cm_lr)
    
    # Train and evaluate Random Forest
    auc_pr_rf, f1_rf, cm_rf = train_and_evaluate_model(rf, "Random Forest", X_train, X_test, y_train, y_test)
    print(f"Random Forest - AUC-PR: {auc_pr_rf:.4f}, F1-Score: {f1_rf:.4f}")
    print("Confusion Matrix:\n", cm_rf)
    
    # Model selection justification
    best_model = "Random Forest" if auc_pr_rf > auc_pr_lr else "Logistic Regression"
    print(f"\nBest Model: {best_model}")
    print("Justification: The chosen model has higher AUC-PR, indicating better performance in handling the imbalanced dataset, which is critical for fraud detection. Random Forest is typically more robust due to its ensemble nature, capturing complex patterns, while Logistic Regression offers interpretability for business stakeholders.")

if __name__ == "__main__":
    main()