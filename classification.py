import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix, classification_report

TEST_DATA_PERCENTAGE = 0.2
DATASET_FRACTION = 0.25

'''Main'''
def main():
    # read data
    df = pd.read_csv('BitcoinHeistData.csv')

    # check for missing values
    print(f"Check for missing values:\n{df.isnull().sum()}\n")

    # drop address column
    if 'address' in df.columns:
        df.drop(columns=['address'], inplace=True)

    # downsample dataset
    print("Downsampling the dataset...\n")
    df = df.sample(frac=DATASET_FRACTION, random_state=0)
    
    # encode categorical variables
    # print("Encoding categorical values...\n")
    # categorical_cols = ['address', 'label'] # replace with actual categorical columns
    # df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

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
    # knn_classifier(X_train, y_train, X_test, y_test)

'''Random Forest Decision Tree Classifier'''
def random_forest_classifier(X_train, y_train, X_test, y_test):
    # Random Forest model
    # rf = RandomForestClassifier()

    # Random Forest model with hyperparameters
    rf = RandomForestClassifier(
        n_estimators=100,           # number of trees
        max_depth=10,               # maximum depth of trees
        min_samples_split=10,        # minimum samples to split a node
        criterion='entropy',        # criterion for split quality
        n_jobs=1,                   # use all available processors
        random_state=0
    )

    # train model
    print("\nTraining Random Forest Model...\n")
    rf.fit(X_train, y_train)

    # predict
    y_pred = rf.predict(X_test)

    # prediction results
    print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100}%\n")
    print(f"\nClassification Report:\n{classification_report(y_test, y_pred)}\n")

'''K-Nearest Neigbor Classifier'''
def knn_classifier(X_train, y_train, X_test, y_test):
    # KNN model
    print()

'''Call main'''
if __name__ == "__main__":
    main()
