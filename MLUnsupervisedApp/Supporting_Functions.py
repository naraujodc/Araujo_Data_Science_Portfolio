# import necessary packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage, dendrogram, set_link_color_palette
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

# --------------------------
# GENERAL HELPER FUNCTIONS
# --------------------------
# functions that will be used throughout many pages

# turn sklearn toy datasets into pandas dataframes
def toy_to_df(load_function):
    bunch = load_function()
    df = pd.DataFrame(bunch.data, columns=bunch.feature_names)
    df["target"] = bunch.target
    return df

# data preprocessing
def data_preprocessing(df, target):
    # drop rows with missing data
    df = df.dropna()
    # encode categorical variables
    categorical_cols = [col for col in df.columns
                        if (pd.api.types.is_categorical_dtype(df[col].dtype)
                        or pd.api.types.is_object_dtype(df[col].dtype))
                        and col != target]
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    # define feature and target variables
    if target is not None:
        X = df.drop(target, axis=1)
        y = df[target]
    else:
        X = df
        y = None
    # center and scale features
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)
    return df, X_std, y

# principal component analysis
def data_pca(X_std, n_components):
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_std)
    explained_variance = pca.explained_variance_ratio_
    exp_variance_cumsum = np.cumsum(explained_variance)
    return X_pca, exp_variance_cumsum

# 2D scatterplot of real labels with PCA
def pca_true_scatterplot(df, X_pca, target, y):

    # get target names
    if target is not None:
        target_names = df[target].unique()

        # plot
        fig = plt.figure(figsize=(8, 6))
        colors = ['powderblue', 'midnightblue', 'palegreen', 'mediumpurple', 'teal',
                  'darkseagreen', 'slateblue', 'dodgerblue', 'greenyellow', 'darkturquoise']
        for i, target_name in enumerate(target_names):
            plt.scatter(X_pca[y == target_name, 0], X_pca[y == target_name, 1], color=colors[i], alpha=0.7,
                        label=target_name, edgecolor='k', s=60)
        
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('True Labels: 2D PCA Projection')
    plt.legend(loc='best')
    plt.grid(True)
    
    return fig

# --------------------------
# PCA PAGE HELPER FUNCTIONS
# --------------------------

# plot scree plot
def scree_plot(df, X_std, target):

    # compute full PCA
    if target is None:
        length = len(df.columns)
    else: 
        length = len(df.columns) - 1

    # avoid too many PCs for better visualization
    if length > 15:
        max_length = 15
    else: 
        max_length = length
    
    # compute PCA with all components
    pca_full = PCA(n_components=max_length).fit(X_std)
    explained = pca_full.explained_variance_ratio_*100
    components = np.arange(1, (len(explained) + 1))
    cumulative = np.cumsum(explained)

    # create the combined plot
    fig, ax1 = plt.subplots(figsize=(8, 6))

    # bar plot for individual variance explained
    bar_color = 'steelblue'
    ax1.bar(components, explained, color=bar_color, alpha=0.8, label='Individual Variance')
    ax1.set_xlabel('Principal Component')
    ax1.set_ylabel('Individual Variance Explained (%)', color=bar_color)
    ax1.tick_params(axis='y', labelcolor=bar_color)
    ax1.set_xticks(components)
    ax1.set_xticklabels([f"PC{i}" for i in components])

    # find the maximum explained variance and adjust y_max
    max_explained = max(explained)
    label_height = 2
    y_max = max(max_explained * 1.2 + label_height, 10)
    ax1.set_ylim(0, y_max)

    # add percentage labels on each bar
    for i, v in enumerate(explained):
        ax1.text(components[i], v + 1, f"{v:.1f}%", ha='center',
                 va='bottom', fontsize=10, color='black')

    # create a second y-axis for cumulative variance explained
    ax2 = ax1.twinx()
    line_color = 'crimson'
    ax2.plot(components, cumulative, color=line_color, marker='o', label='Cumulative Variance')
    ax2.set_ylabel('Cumulative Variance Explained (%)', color=line_color)
    ax2.tick_params(axis='y', labelcolor=line_color)
    ax2.set_ylim(0, 100)

    # remove grid lines
    ax1.grid(False)
    ax2.grid(False)

    # combine legends from both axes and position the legend inside the plot (middle right)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right', bbox_to_anchor=(0.85, 0.5))

    plt.title('PCA: Variance Explained', pad=20)
    plt.tight_layout()
    
    return fig

# plot scatterplot
def pca_scatterplot(df, target, X_pca, y):

    # get target names
    if target is not None:
        target_names = df[target].unique()

        # plot
        fig = plt.figure(figsize=(8, 6))
        colors = ['powderblue', 'midnightblue', 'palegreen', 'mediumpurple', 'teal', 'darkseagreen', 'slateblue', 'dodgerblue']
        for i, target_name in enumerate(target_names):
            plt.scatter(X_pca[y == target_name, 0], X_pca[y == target_name, 1], color=colors[i], alpha=0.7,
                        label=target_name, edgecolor='k', s=60)
            
    else:
        fig = plt.figure(figsize=(8,6))
        plt.scatter(X_pca[:, 0], X_pca[:, 1], color='powderblue', alpha=0.7,
                   edgecolor='k', s=60)
        
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('PCA: 2D Projection of Dataset')
    plt.legend(loc='best')
    plt.grid(True)
    
    return fig

# plot biplot
def pca_biplot(X_std, X_pca, df, target, y):
    # compute the loadings
    pca = PCA(n_components=2)
    pca.fit(X_std)
    loadings = pca.components_.T
    scaling_factor = 50.0  # increased scaling factor by 5 times

    # get target and feature names
    if target is not None:
        target_names = df[target].unique()
        features = df.drop(target, axis=1)
        feature_names = features.columns

    # plot the PCA scores as before
        fig = plt.figure(figsize=(8, 6))
        colors = ['powderblue', 'midnightblue', 'palegreen', 'mediumpurple', 'teal', 'darkseagreen', 'slateblue', 'dodgerblue']
        for i, target_name in enumerate(target_names):
            plt.scatter(X_pca[y == target_name, 0], X_pca[y == target_name, 1], color=colors[i], alpha=0.7,
                        label=target_name, edgecolor='k', s=60)
            
    else:
        features = df
        feature_names = features.columns
        fig = plt.figure(figsize=(8,6))
        plt.scatter(X_pca[:, 0], X_pca[:, 1], color='powderblue', alpha=0.7,
                   edgecolor='k', s=60)
        
    # plot the loadings as arrows
    for i, feature in enumerate(feature_names):
        plt.arrow(0, 0, scaling_factor * loadings[i, 0], scaling_factor * loadings[i, 1],
                color='crimson', width=0.02, head_width=0.1)
        plt.text(scaling_factor * loadings[i, 0] * 1.1, scaling_factor * loadings[i, 1] * 1.1,
                feature, color='crimson', ha='center', va='center')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('Biplot: PCA Scores and Loadings')
    plt.legend(loc='best')
    plt.grid(True)

    return fig

# --------------------------
# KMEANS PAGE HELPER FUNCTIONS
# --------------------------

# apply k-means clustering
def apply_kmeans(X_std, k):
    kmeans = KMeans(n_clusters=k, random_state=42)
    clusters = kmeans.fit_predict(X_std)
    return clusters

# 2D scatterplot of clustering results with PCA
def pca_kmeans_scatterplot(X_pca, clusters, k):
    fig = plt.figure(figsize=(8, 6))
    colors = ['powderblue', 'midnightblue', 'palegreen', 'mediumpurple', 'teal',
              'darkseagreen', 'slateblue', 'dodgerblue', 'greenyellow', 'darkturquoise']
    unique_clusters = np.unique(clusters)
    color_map = {label: colors[i % len(colors)] for i, label in enumerate(unique_clusters)}

    for cluster_label in unique_clusters:
        plt.scatter(X_pca[clusters == cluster_label, 0], X_pca[clusters == cluster_label, 1],
                    color=color_map[cluster_label], alpha=0.7, label=f'Cluster {cluster_label}', edgecolor='k', s=60)

    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title(f'KMeans Clustering: 2D PCA Projection (k={k})')
    plt.legend(loc='best')
    plt.grid(True)
    
    return fig

# calculate within-cluster sum of squares and silhouette score
def wcss_silhouette(X_std):
    ks = range(2,11)
    wcss = []
    silhouette_scores = []
    for k in ks:
        km = KMeans(k, random_state=42)
        km.fit(X_std)
        wcss.append(km.inertia_)
        labels = km.labels_
        silhouette_scores.append(silhouette_score(X_std, labels))
    return wcss, silhouette_scores

# elbow plot
def elbow_plot(wcss):
    fig = plt.figure(figsize=(8,6))
    ks = range(2,11)
    plt.plot(ks, wcss, marker='o', color='steelblue')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
    plt.title('Elbow Method for Optimal k')
    plt.grid(True)
    return fig

# silhouette scores plot
def silhouette_plot(silhouette_scores):
    fig = plt.figure(figsize=(8,6))
    ks = range(2,11)
    plt.plot(ks, silhouette_scores, marker='o', color='teal')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score for Optimal k')
    plt.grid(True)
    return fig

# --------------------------
# HIERARCHICAL CLUSTERING PAGE HELPER FUNCTIONS
# --------------------------

# create dendrogram
def plot_dendrogram(X_std, linkage_choice, linkage_method, y):
    # create linkage matrix
    Z = linkage(X_std, method=linkage_method)

    if y is not None:
        labels = y.to_list()
    else:
        labels=None
    
    fig = plt.figure(figsize=(12,6))
    set_link_color_palette(['steelblue', 'midnightblue', 'palegreen', 'mediumpurple', 'teal', 'darkseagreen', 'slateblue', 'dodgerblue'])
    dendrogram(Z, labels=labels)
    plt.ylabel('Linkage Distance')
    plt.xlabel('Data Points')
    plt.title(f'Hierarchical Clustering Dendrogram (Linkage: {linkage_choice})')

    return fig

# apply agglomerative hierarchical clustering
def agg_hierarchical(k, linkage_method, X_std, df_processed):
    agg = AgglomerativeClustering(n_clusters=k, linkage=linkage_method)
    df_processed["cluster"] = agg.fit_predict(X_std)
    cluster_labels = df_processed["cluster"].to_list()
    return agg, cluster_labels

# plot clustering results with PCA
def plot_clustering(X_std, cluster_labels):
    # reduce to 2 dimensions for plotting
    X_pca, exp_variance_cumsum = data_pca(X_std=X_std, n_components=2)
    # create scatterplot
    fig = plt.figure(figsize=(8,6))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels,
                          cmap='tab20c',
                          s=60, edgecolor='k', alpha=0.7)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('Agglomerative Hierarchical Clustering (via PCA)')
    plt.legend(*scatter.legend_elements(), title="Clusters")
    plt.grid(True)
    
    return fig

# silhouette plot for best k
def silhouette_agg(X_std, linkage_method, linkage_choice):
    ks = range(2, 11)
    sil_scores = []

    for k in ks:
        # fit hierarchical clustering
        labels = AgglomerativeClustering(n_clusters=k, linkage=linkage_method).fit_predict(X_std)

        score = silhouette_score(X_std, labels)
        sil_scores.append(score)

    # Plot the curve
    fig = plt.figure(figsize=(7,4))
    plt.plot(list(ks), sil_scores, marker="o")
    plt.xticks(list(ks))
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Average Silhouette Score")
    plt.title(f"Silhouette Analysis for Agglomerative ({linkage_choice}) Clustering")
    plt.grid(True, alpha=0.3)
    return fig