import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

TEST_DATA_PERCENTAGE = 0.2

'''Main'''
def main():
    # read data
    df = pd.read_csv('BitcoinHeistData.csv')

    # get features
    x = df[['address',
            'year',
            'day',
            'length',
            'weight',
            'count',
            'looped',
            'neighbors',
            'income'
        ]]
    
    print(f"Features:\n{x}\n")

    # get label
    y = df['label']
    print(f"Labels:\n{y}\n")

    # split data
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=TEST_DATA_PERCENTAGE, random_state=0)

    # call Random Forest classifier
    random_forest_classifier(X_train, y_train, X_test, y_test)

    # call KNN classifier

'''Random Forest Decision Tree Classifier'''
def random_forest_classifier(X_train, y_train, X_test, y_test):
    # Random Forest model
    rf = RandomForestClassifier()

    # train model
    rf.fit(X_train, y_train)

    # predict
    y_pred = rf.predict(X_test)

    # prediction results
    rf.score(X_test, y_test)

'''K-Nearest Neigbor Classifier'''
def knn_classifier():
    # KNN model
    print()

'''Call main'''
if __name__ == "__main__":
    main()
