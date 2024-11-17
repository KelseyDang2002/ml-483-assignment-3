import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

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

    # get features
    x = df.drop(columns=['label'])
    print(f"Features:\n{x}\n")

    # get label
    # y = df['label']
    # print(f"Labels:\n{y}\n")

    # split data
    # X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=TEST_DATA_PERCENTAGE, random_state=0)

'''K-means Clustering Model'''
def k_means_clustering():
    print()

'''Expectation-Maximization (EM) Clustering Model'''
def em_clustering():
    print()

'''Call main'''
if __name__ == "__main__":
    main()
