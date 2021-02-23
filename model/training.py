from sklearn.ensemble import RandomForestRegressor

def training(X_train, y_train):
    """
    Train the machine learning model.
    This function is used to train the regression model.
    Parameters
    ----------
    X_train: 2D-array
        training dataset without target
    y_train: 1D-array 
        training target
    Return
    ------
    random_forest: RandomForestRegressor
        train model
    """
    print("Train the model...")
    random_forest = RandomForestRegressor()
    random_forest.fit(X_train, y_train)
    
    return random_forest