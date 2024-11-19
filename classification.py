import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier

TEST_DATA_PERCENTAGE = 0.2
NUMBER_OF_NEIGHBORS = 10

'''Main'''
def main():
    # read data
    df = pd.read_csv('BitcoinHeistData.csv')

    # check for missing values
    print(f"Check for missing values:\n{df.isnull().sum()}\n")

    # drop address column
    if 'address' in df.columns:
        df.drop(columns=['address'], inplace=True)

    # get features
    x = df.drop(columns=['label'])
    print(f"Features:\n{x}\n")

    # get label
    y = df['label']
    print(f"Labels:\n{y}\n")

    # split data
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=TEST_DATA_PERCENTAGE, random_state=0)

    # call Random Forest classifier
    random_forest_classifier(X_train, y_train, X_test, y_test)

    # call KNN classifier
    knn_classifier(X_train, y_train, X_test, y_test)

'''Random Forest Decision Tree Classifier'''
def random_forest_classifier(X_train, y_train, X_test, y_test):
    # Random Forest model
    rf = RandomForestClassifier(random_state=0)

    # Random Forest model with hyperparameters
    # rf = RandomForestClassifier(
    #     n_estimators=1000,          # number of trees
    #     max_depth=10,               # maximum depth of trees
    #     min_samples_split=10,       # minimum samples to split a node
    #     criterion='entropy',        # criterion for split quality
    #     n_jobs=1,                   # use all available processors
    #     random_state=0
    # )

    # train model
    print("\nTraining Random Forest Model...\n")
    rf.fit(X_train, y_train)

    # prediction
    y_pred = rf.predict(X_test)

    # performance results
    print("\nRandom Forest Performance Results:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100}%\n")
    print(f"\nClassification Report:\n{classification_report(y_test, y_pred)}\n")

'''Preprocess data'''
def preprocess(X_train, X_test):
    scaler = MinMaxScaler(feature_range=(0,1))
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.fit_transform(X_test)
    return X_train_scaled, X_test_scaled

'''K-Nearest Neigbor Classifier'''
def knn_classifier(X_train, y_train, X_test, y_test):
    # KNN model
    knn = KNeighborsClassifier(n_neighbors=NUMBER_OF_NEIGHBORS)

    # preprocess data
    X_train_scaled, X_test_scaled = preprocess(X_train, X_test)

    # train model
    print("\nTraining KNN Model...")
    knn.fit(X_train_scaled, y_train)

    # prediction
    y_pred = knn.predict(X_test_scaled)

    # performance results
    print("\nK-Nearest Neighbor Performance Results:")
    print(f"Number of neighbors: {NUMBER_OF_NEIGHBORS}")
    print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100}%\n")
    print(f"\nClassification Report:\n{classification_report(y_test, y_pred)}\n")

'''Call main'''
if __name__ == "__main__":
    main()
