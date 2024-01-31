import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml

def ingest_data():
    if "{{cookiecutter.ml_task}}" == "classification":
        data_bunch = fetch_openml(name="miceprotein", version=4)
        data = pd.DataFrame(data=data_bunch['data'], columns=data_bunch['feature_names'])
        data['target'] = data_bunch['target']
        # Split the data into current_data and reference_data
        current_data, reference_data = train_test_split(data, test_size=0.2, random_state=42)
    elif "{{cookiecutter.ml_task}}" == "timeseries":
        dataset_path = "/home/sigmoid/Downloads/retail_retail_based.csv"
        data = pd.read_csv(dataset_path)
        print(f"Data loaded from {dataset_path}")
    
    # Sort the data by the date column
    # Assume the date is in the first column
        data = data.sort_values(by=data.columns[0])
        # Split the data based on the date
        split_point = int(len(data) * 0.8)
        current_data = data.iloc[:split_point, :]
        reference_data = data.iloc[split_point:, :]
    else:
        print("Invalid machine learning task.")
        return None, None
    
    return current_data, reference_data