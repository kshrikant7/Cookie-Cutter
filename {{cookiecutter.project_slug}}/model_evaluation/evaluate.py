from sklearn.metrics import mean_squared_error

def evaluate_model(model, X_test, y_test):
    X_test = X_test.reset_index(drop=True)
    # Make predictions
    y_pred = model.predict(start=0, end=len(X_test) - 1)

    # Print mean squared error
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")