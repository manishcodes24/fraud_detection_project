import pandas as pd

def load_and_inspect_data(filepath):
    """
    Loads the dataset and prints basic info.
    """
    df = pd.read_csv(filepath)

    print("Dataset Loaded.")
    print("Shape:", df.shape)
    print("\nColumns:", df.columns.tolist())
    print("\nMissing Values:\n", df.isnull().sum())

    return df

def save_cleaned_data(df, output_path):
    """
    Saves the cleaned/preprocessed DataFrame to a CSV file.
    """
    df.to_csv(output_path, index=False)
    print(f"\nCleaned data saved to: {output_path}")
