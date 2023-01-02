from os.path import exists
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import pickle
from xgboost import XGBRegressor



def train():
    # read csv
    sentences = np.load('sentence_embeddings.npy')
    embedding = pd.DataFrame(sentences)
    # print(embedding)
    data = pd.read_csv('train.csv')
    data_numerical = data.drop(['text', 'retweets_count', 'urls', 'mentions', 'hashtags'], axis=1)
    pd.concat([embedding, data_numerical], axis=1)
    Y = data['retweets_count']

    X_train, X_test, Y_train, Y_test = train_test_split(embedding, Y, test_size=0.25, shuffle=True)
    if exists("XGBRegressor_embedding.pkl"):
        with open("XGBRegressor_embedding.pkl", "rb") as file:
            regressor = pickle.load(file)
    else:
        regressor = XGBRegressor()

    # train
    regressor.fit(X_train, Y_train)

    # test
    predictions = regressor.predict(X_test)

    error = mean_absolute_error(y_true=Y_test, y_pred=predictions)

    print('Prediction error:', error)

    with open("XGBRegressor_embedding.pkl", "wb") as file:
        pickle.dump(regressor, file)


def test():
    # read csv
    data = pd.read_csv('evaluation.csv')
    data_set = data.drop(['text', 'urls', 'mentions', 'hashtags'], axis=1)
    TweetID = data['TweetID']
    print(len(TweetID))

    if exists("XGBRegressor.pkl"):
        with open("XGBRegressor.pkl", "rb") as file:
            regressor = pickle.load(file)
    else:
        regressor = XGBRegressor()

    # predict
    predictions = regressor.predict(data_set)
    print(len(predictions))

    with open("result.csv", "w") as result:
        result.write('TweetID,retweets_count\n')
        for i in range(117990):
            result.write(str(TweetID[i]))
            result.write(',')
            result.write(str(predictions[i]))
            result.write('\n')


if __name__ == '__main__':
    train()
    test()