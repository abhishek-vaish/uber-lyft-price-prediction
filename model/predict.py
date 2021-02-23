from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def predict(random_forest, X_test, y_test):
    """
    Perform the prediction over the testing dataset.
    This function is used to perform the prediction and evaluate the 
    scoring metrics over r2_score, mean_squared_error and mean_absolute_error.
    Parameters
    ----------
    random_forest: train model
        machine learning model
    X_test: 2D-array
        testing dataset without target
    y_test: 1D-array
        testing target
    """
    print("Predicting the dataset...")
    # predict the value over testing dataset
    y_preds = random_forest.predict(X_test)
    # r2_scored metrics
    r2 = r2_score(y_test, y_preds)
    # mse scoring metrics
    mse = mean_squared_error(y_test, y_preds)
    # mae scoring metrics
    mae = mean_absolute_error(y_test, y_preds)
    # print all the scoring metrics
    print({
        "R2_score": r2,
        "Mean Square Error": mse,
        "Mean Absolute Error": mae
    })
