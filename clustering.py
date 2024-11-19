import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import mean_squared_error, adjusted_rand_score, silhouette_score

NUMBER_OF_CLUSTERS = 18
NUMBER_OF_COMPONENTS = 3

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

    # normalize features
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    # find optimal number of clusters
    find_optimal_k(x_scaled)

    # call K-means clustering
    k_means_clustering(x_scaled, y)

    # call EM clustering
    em_clustering(x_scaled, y)

'''Find optimal number of clusters'''
def find_optimal_k(x_scaled):
    print("\nFinding the optimal number of clusters...\n")
    inertia = []
    k_values = range(1, 19) # test k from 1 to 18
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=0)
        kmeans.fit(x_scaled)
        inertia.append(kmeans.inertia_)

    # plot elbow curve
    plt.figure(figsize=(8, 4))
    plt.plot(k_values, inertia, marker='o')
    plt.title('Elbow Method for Optimal k')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia')
    plt.show()

'''K-means Clustering Model'''
def k_means_clustering(x_scaled, y):
    # K-means model with hyperparameters
    kmeans = KMeans(
        n_clusters=NUMBER_OF_CLUSTERS,
        max_iter=1000,
        random_state=0
    )

    # train model
    print("\nClustering with K-means Model...")
    kmeans.fit(x_scaled)

    # compute predictions (based on centroids)
    predicted_labels = kmeans.labels_
    predicted_centroids = kmeans.cluster_centers_[predicted_labels]

    # performance results
    mse = mean_squared_error(x_scaled, predicted_centroids)
    ari = adjusted_rand_score(y, predicted_labels)
    print("\nK-means Performance Results:")
    print(f"Number of clusters: {NUMBER_OF_CLUSTERS}")
    print(f"Mean Squared Error (internal index): {mse}")
    print(f"Absolute Rand Index (external index): {ari}\n")

'''Expectation-Maximization (EM) Clustering Model'''
def em_clustering(x_scaled, y):
    # EM model
    em = GaussianMixture(n_components=NUMBER_OF_COMPONENTS, random_state=0)

    # train model
    print("\nClustering with EM (Gaussian Mixture) Model...")
    em.fit(x_scaled)

    # predict labels
    predicted_labels = em.predict(x_scaled)

    # get the means (centroids) for each cluster
    predicted_centroids = em.means_[predicted_labels]

    # performance results
    mse = mean_squared_error(x_scaled, predicted_centroids)
    ari = adjusted_rand_score(y, predicted_labels)
    print("\nEM (Gaussian Mixture) Performance Results:")
    print(f"Number of componentss: {NUMBER_OF_COMPONENTS}")
    print(f"Mean Squared Error (internal index): {mse}")
    print(f"Absolute Rand Index (external index): {ari}\n")

'''Call main'''
if __name__ == "__main__":
    main()
