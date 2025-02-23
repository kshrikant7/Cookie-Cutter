import numpy as np

# Load the model from disk

# def make_prediction(model, X_test):
#     # user_input = input("Please enter your input for prediction: ")
#     # Convert the user input to the format expected by your model
#     # This is just an example, you'll need to adjust this to match your actual model
#     # user_input = np.array([float(user_input)]).reshape(1, -1)
#     # exog_future = None  # Define the variable exog_future
#     # Make a prediction based on the user input
#     prediction = model.predict(n_periods=10)
#     print(f"Prediction: {prediction}")
#     return prediction


def make_prediction(model, X_test):
    X_test = X_test.reset_index(drop=True)
    
    if "{{cookiecutter.ml_task}}" == "classification":
        # Make predictions
        prediction = model.predict(X_test)
        print(f"Prediction: {prediction}")
    elif "{{cookiecutter.ml_task}}" == "timeseries":
        # Make predictions
        prediction = model.predict(start=0, end=len(X_test) - 1)
        print(f"Prediction: {prediction}")
    else:
        print("Invalid machine learning task.")
    
    return prediction