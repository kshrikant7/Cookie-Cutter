import numpy as np

# Load the model from disk'

def make_prediction(model, X_test):
    user_input = input("Please enter your input for prediction: ")
    # Convert the user input to the format expected by your model
    # This is just an example, you'll need to adjust this to match your actual model
    user_input = np.array([float(user_input)]).reshape(1, -1)
    # Make a prediction based on the user input
    prediction = model.predict(user_input)
    print(f"Prediction made for input {user_input}: {prediction}")