# def check_prediction_drift(predictions):
#     # Define a reference value or range of values
#     reference_value = 0.5

#     # Calculate the mean prediction
#     mean_prediction = predictions.mean()

#     # Check if the mean prediction is significantly different from the reference value
#     if abs(mean_prediction - reference_value) > 0.1:
#         print("Warning: Prediction drift detected. The model's predictions have significantly changed.")


import pandas as pd
import numpy as np

def check_prediction_drift(predictions, historical_data=None):
    if "classification" == "classification":
        # Check if the distribution of predictions has significantly changed from the historical data
        if historical_data is None:
            print("No historical data provided.")
            return
        reference_distribution = historical_data.value_counts(normalize=True)
        prediction_distribution = pd.Series(predictions).value_counts(normalize=True)
        drift = abs(reference_distribution - prediction_distribution).sum()
    elif "classification" == "timeseries":
        # Define the reference value as the mean of the historical data
        if historical_data is None:
            print("No historical data provided.")
            return
        reference_value = historical_data.mean()
        # Calculate the mean prediction
        mean_prediction = np.mean(predictions)
        drift = abs(mean_prediction - reference_value)
    else:
        print("Invalid machine learning task.")
        return

    # Check if the drift is significant
    if drift > 0.1:
        print("Warning: Prediction drift detected. The model's predictions have significantly changed.")