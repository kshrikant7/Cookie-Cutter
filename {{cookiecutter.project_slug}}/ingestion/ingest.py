import pandas as pd
from sklearn.model_selection import train_test_split

def ingest_data():
    dataset_path = "/home/sigmoid/Downloads/retail_retail_based.csv"
    data = pd.read_csv(dataset_path)
    print(f"Data loaded from {dataset_path}")
    
    # Split the data into current_data and reference_data
    current_data, reference_data = train_test_split(data, test_size=0.2, random_state=42)
    
    return current_data, reference_data