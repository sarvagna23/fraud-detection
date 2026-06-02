import pandas as pd
import os

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'creditcard.csv')

def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Loaded {len(df):,} records")
    print(f"Fraud cases: {df['Class'].sum():,} ({df['Class'].mean()*100:.2f}%)")
    print(f"Columns: {list(df.columns)}")
    return df

def validate_data(df: pd.DataFrame) -> bool:
    required_columns = [f'V{i}' for i in range(1, 29)] + ['Time', 'Amount', 'Class']
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        print(f"Missing columns: {missing}")
        return False
    null_count = df.isnull().sum().sum()
    if null_count > 0:
        print(f"Found {null_count} null values")
        return False
    print("Data validation passed")
    return True

if __name__ == "__main__":
    df = load_data()
    validate_data(df)