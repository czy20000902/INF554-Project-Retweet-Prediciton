from os.path import exists
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import pickle
from xgboost import XGBRegressor
import warnings

warnings.filterwarnings('ignore')


def train():
    params = {"learning_rate": 0.1,
              "max_depth": 3,
              "colsample_bytree": 1,
              "n_estimators": 200}

    # read csv
    data = pd.read_csv('train.csv')
    X = data.drop(['text', 'retweets_count', 'urls', 'mentions', 'hashtags'], axis=1)
    Y = data['retweets_count']

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5, shuffle=True)
    model_path = 'XGBRegressor.pkl'
    if exists(model_path):
        with open(model_path, "rb") as file:
            regressor = pickle.load(file)
    else:
        regressor = XGBRegressor(eval_metric="mae", **params)

    # train
    regressor.fit(X_train, Y_train, eval_set=[(X_test, Y_test)],  early_stopping_rounds=10)

    # test
    predictions = regressor.predict(X_test)

    error = mean_absolute_error(y_true=Y_test, y_pred=predictions)

    print('Mean Absolute Error:', error)
    with open("XGBRegressor.pkl", "wb") as file:
        pickle.dump(regressor, file)


def test():
    # read csv
    data = pd.read_csv('evaluation.csv')
    data_set = data.drop(['text', 'urls', 'mentions', 'hashtags'], axis=1)
    TweetID = data['TweetID']

    model_path = 'XGBRegressor.pkl'
    if exists(model_path):
        with open(model_path, "rb") as file:
            regressor = pickle.load(file)
    else:
        raise ValueError("Model not found")

    # predict
    predictions = regressor.predict(data_set)
    predictions = [int(y) for y in predictions]

    with open("result.csv", "w") as result:
        result.write('TweetID,retweets_count\n')
        for i in range(len(data_set)):
            result.write(str(TweetID[i]))
            result.write(',')
            result.write(str(predictions[i]))
            result.write('\n')


if __name__ == '__main__':
    train()
    test()
