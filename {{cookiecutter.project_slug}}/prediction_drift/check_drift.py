def check_prediction_drift(predictions):
    # Define a reference value or range of values
    reference_value = 0.5

    # Calculate the mean prediction
    mean_prediction = predictions.mean()

    # Check if the mean prediction is significantly different from the reference value
    if abs(mean_prediction - reference_value) > 0.1:
        print("Warning: Prediction drift detected. The model's predictions have significantly changed.")