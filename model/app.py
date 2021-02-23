from data_manipulation import data_manipulation
from train_test_data import train_test_data
from training import training
from predict import predict
import pickle

if __name__ == "__main__":
    df = data_manipulation('../data/cab_rides.csv')
    X_train, X_test, y_train, y_test = train_test_data(df)
    model = training(X_train, y_train)
    predict(random_forest=model, 
        X_test=X_test,
        y_test=y_test)
    pickle.dump(model, open("../model/random_forest_regressor.pkl", "wb"))

