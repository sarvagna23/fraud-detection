import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

def preprocess(df: pd.DataFrame):
    # Scale Time and Amount (V1-V28 are already scaled by PCA)
    scaler = StandardScaler()
    df['Amount_scaled'] = scaler.fit_transform(df[['Amount']])
    df['Time_scaled'] = scaler.fit_transform(df[['Time']])
    df.drop(columns=['Time', 'Amount'], inplace=True)

    # Split features and target
    X = df.drop(columns=['Class'])
    y = df['Class']

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Train size: {len(X_train):,} | Test size: {len(X_test):,}")
    print(f"Fraud in train before SMOTE: {y_train.sum():,}")

    # SMOTE to handle class imbalance
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    print(f"Train size after SMOTE: {len(X_train_res):,}")
    print(f"Fraud in train after SMOTE: {y_train_res.sum():,}")

    return X_train_res, X_test, y_train_res, y_test

if __name__ == "__main__":
    from ingest import load_data
    df = load_data()
    X_train, X_test, y_train, y_test = preprocess(df)
    print("Preprocessing complete")