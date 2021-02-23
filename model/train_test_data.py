import numpy as np
from sklearn.model_selection import train_test_split



def train_test_data(df):
    """
    Prepare the training and testing dataset.
    This function is used to split the dataset into X and y.
    Then split the dataset into training and testing dataset.
    Parameters
    ----------
    df: csv file
        take csv file contains only numerical dateset
    Returns
    -------
    X_train: 2D-array
        training dataset without target
    y_train: 2D-array
        training target
    X_test: 1D-array
        testing dataset without target
    y_test: 1D-array
        testing target
    """
    print("Preparing X and y dataset...")
    # prepare the X dataset
    X = df.drop('price', axis=1)
    # prepare the y dataset
    y = df['price']

    print("Splitting the dataset...")
    # assign the random seed
    np.random.seed(42)
    # split the data into training and testing dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    # returns the training and testing dataset
    return X_train, X_test, y_train, y_test
