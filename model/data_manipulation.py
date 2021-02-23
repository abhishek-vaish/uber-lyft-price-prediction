import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

def data_manipulation(data_set):
    """
    Prepare the dataset for training the regression model.
    This function takes the csv file location as a parameter
    Parameters
    ----------
    data_set: str
        data file location
    Returns
    -------
    data_set: csv file
        turns all the categorical data into numerical data.
    """
    print("Loading the Dataset...")
    df = pd.read_csv(data_set)
    print("Data Manipulation...")
    # remove all the null value from the dataset
    df.dropna(inplace=True)
    # create the variable that contains the combine source destination
    source_destination = df.source + '-' + df.destination
    # create the column and assign to the above created variable to the 
    df['source_destination'] = source_destination
    # since we create a seperate column for source destination, now we drop that
    df.drop(['source', 'destination'], axis=1, inplace=True)

    # create a dictionary to convert cab_type data to number
    cab_type={
        'Lyft': 0,
        'Uber': 1
    }
    
    # map the above dictionary to the cab_type dataset
    df.cab_type = df['cab_type'].map(cab_type)
    # drop the id column since it has lot of unique variables
    df.drop('id', axis=1, inplace=True)

    print("One Hot Encoding...")
    # create a list for categorical labels
    categorical_label = ['product_id', 'source_destination', 'name']
    # create a OneHotEncoder object
    one_hot_encoding = OneHotEncoder()
    # perform the columnTransformer and use oneHotEncoder as a transformer
    transformer = ColumnTransformer([('one_hot', one_hot_encoding, categorical_label)], remainder='passthrough')
    # fit the transformer with the dataframe
    transform_df = transformer.fit_transform(df).toarray()

    # get the feature name
    columns = transformer.get_feature_names()
    new_column = []
    # get the column name using iteration and append it to new_column
    for i in range(len(columns)-5):
        new_column.append(columns[i][12:])
    for i in range(len(columns)-5, len(columns)):
        new_column.append(columns[i])
    
    print("Transforming data...")
    # create a dataset using the transformed dataframe
    data_set = pd.DataFrame(transform_df)
    # replace the column with the new_column list that we created
    data_set.columns = new_column
    # return the manipulated dataset
    return data_set