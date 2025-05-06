# import necessary packages
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from Unsupervised_ML import *
from sklearn.datasets import load_breast_cancer, load_iris, load_wine
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# -------------------------
# PAGE CONFIGURATIONS
# -------------------------
st.set_page_config(
    page_title="K-Means Clustering",
    page_icon=":material/graph_6:",
    layout="wide"
)

# -------------------------
# INITIAL PAGE STRUCTURE
# -------------------------

st.title(":material/graph_6: K-Means Clustering")
st.write("""
         **K-Means Clustering** is an **unsupervised machine learning** technique used to group data points into clusters based on their similiarities.
         It works by iteratively choosing a k number of **centroids** and assigning data points to the group corresponding to the closest centroid.
         Centroids are recalculated and data points are reassigned in order to minimize within-cluster variances.

         In this page, you can:
         - Upload your own dataset (labeled or unlabeled) or use a demo dataset;
         - Choose the number of clusters (k) you want to form;
         - Visualize the scatterplot of your clusters and compare them with the true labels (if the data is labeled);
         - Learn different methods for finding the best k for unlabeled data.
         """)

# -------------------------
# HELPER FUNCTIONS
# -------------------------

# apply k-means clustering
def apply_kmeans(X_std, k):
    kmeans = KMeans(n_clusters=k, random_state=42)
    clusters = kmeans.fit_predict(X_std)
    return clusters

# 2D scatterplot of clustering results with PCA
def pca_kmeans_scatterplot(X_pca, clusters):
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

# 2D scatterplot of real labels with PCA
def pca_true_scatterplot(df, X_pca, target):

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

# -------------------------
# SIDEBAR STRUCTURE
# -------------------------

# dataset upload option
st.sidebar.markdown("## Dataset Selection")
dataset_upload = st.sidebar.file_uploader(label="Upload your own dataset",
                                          type="csv")

# use uploaded dataset if user inputs one
if dataset_upload is not None:
    dataset = pd.read_csv(dataset_upload)
    is_labeled = st.sidebar.checkbox(label="Is the dataset labeled?")
    if is_labeled == True:
        target = st.sidebar.selectbox(label="What is the target variable?",
                                      options=dataset.columns,
                                      index=None)
    else:
        target = None

# dataset demo option
dataset_demo = None
if dataset_upload is None:
    st.sidebar.markdown("#### No dataset? Use a demo")
    dataset_demo = st.sidebar.selectbox(label="Demo datasets",
                                        options=["Breast Cancer", "Iris", "Wine"],
                                        index=None)

# use demo datasets if user chooses one
if dataset_demo == "Breast Cancer":
    dataset = toy_to_df(load_breast_cancer)
    target = "target"
elif dataset_demo == "Iris":
    dataset = toy_to_df(load_iris)
    target = "target"
elif dataset_demo == "Wine":
    dataset = toy_to_df(load_wine)
    target = "target"

# hyperparameter tuning after dataset selection
if dataset_demo or dataset_upload is not None:
    st.sidebar.header("Hyperparameter Tuning")

    # select k number of clusters
    k = st.sidebar.select_slider(label="Number of clusters (k)",
                                 options=range(2,11),
                                 value=2)

# -------------------------
# MAIN PAGE STRUCTURE
# -------------------------

# only execute if dataset is selected
if (dataset_demo or dataset_upload is not None) and k is not None:
    # preprocess the data
    processed_df, X_std, y = data_preprocessing(df=dataset, target=target)
    # compute kmeans clustering
    clusters = apply_kmeans(X_std=X_std, k=k)

    # reduce dimensionality with PCA for visualization
    X_pca, exp_variance_cumsum = data_pca(X_std=X_std, n_components=2)

    # show scatterplot by itself if there's no target variable
    if target is None:
        # columns for better sizing
        col1, col2, col3 = st.columns([0.1, 0.8, 0.1])
        with col2:
            st.subheader("K-Means Clustering Results")
            st.pyplot(fig=pca_kmeans_scatterplot(X_pca=X_pca, clusters=clusters))
            with st.popover("What does this mean?"):
                st.write("""
                         This scatterplot shows the clusters formed by K-Means Clustering projected on a 2D plane generated by
                         Principal Component Analysis (PCA). This gives us an idea of how the clusters are separated.
                         ***If your dataset is labeled, select the target variable in the sidebar to view the true labels as a comparison.***
                         """)

    # create two columns if there is a target variable
    else:
        col1, col2 = st.columns(2)

        with col1:
            # scatterplot of clusters in column 1
            st.subheader("K-Means Clustering Results")
            st.pyplot(fig=pca_kmeans_scatterplot(X_pca=X_pca, clusters=clusters))
            with st.popover("What does this mean?"):
                st.write("""
                         This scatterplot shows the clusters formed by K-Means Clustering projected on a 2D plane generated by
                         Principal Component Analysis (PCA). This gives us an idea of how the clusters are separated.
                         """)
            
            # classification report
            #st.subheader("Classification Report")
            #accuracy, report = evaluate_kmeans(df=dataset, target=target, clusters=clusters)
            #report_df = pd.DataFrame(report).T
            #st.dataframe(report_df)

        with col2:
            # scatterplot of true labels in column 2
            st.subheader("True Clusters in the Data")
            st.pyplot(fig=pca_true_scatterplot(df=dataset, X_pca=X_pca, target=target))
            with st.popover("What does this mean?"):
                st.write("""
                         This scatterplot shows the true clusters in the data projected on a 2D plane generated by
                         Principal Component Analysis (PCA). This allows us to gauge the performance of K-Means Clustering.
                         ***How similar are the true labels to the clusters generated by K-Means?***
                         """)
                
            # print accuracy score
            #st.write(f"##### :primary[K-Means Clustering was able to classify the data into] {k} :primary[clusters with] {accuracy*100:.2f}% :primary[accuracy.]")

    # section for understanding how to choose k
    st.subheader("Choosing the optimal number of clusters (k)")
    st.write("""
             When working with real-world unlabeled data, it may be hard to decide how many clusters we should try to find in the data.
             For example, for a market segmentation analysis, how many different costumer groups could we have? How many is _too_ many?
             Two good ways to handle this are the **elbow method** and the **silhouette score**.
             """)
    
    # create columns to display methods
    col1, col2 = st.columns(2)

    # compute WCSS and silhouette scores
    wcss, silhouette_scores = wcss_silhouette(X_std=X_std)

    with col1:
        # display elbow method
        st.write("#### Elbow Method")
        st.write("""
                 We plot the **Within-Clusters Sum of Squares (WCSS)** against a range of possible values of k.
                 The lower the WCSS, the better &mdash; so we look for the **elbow point**, where the rate of decline of WCSS drastically changes.
                 """)
        st.pyplot(fig=elbow_plot(wcss=wcss))


    with col2:
        # display silhouette score
        st.write("#### Silhouette Score")
        st.write("""
                 We plot the **silhouette score** against a range of possible values of k. This score indicates how similar an object is to its
                 own cluster compared to the other clusters &mdash; so the **higher** the silhouette score, the better.
                 """)
        st.pyplot(fig=silhouette_plot(silhouette_scores=silhouette_scores))

    st.write(":bulb: Try adjusting k according to these methods to see how the clusters look!")



# prompt dataset selection
elif dataset_upload is None and dataset_demo is None:
    st.write("#### :primary[Please select a dataset to start.]")

# create section for common questions
st.subheader("Common Questions")

# create columns for help buttons
col1, col2 = st.columns(2)

with col1:
    with st.expander("How is the data preprocessed for K-Means Clustering?"):
        st.write("""
                 - First, rows with missing data are excluded. There are different ways to handle missing data, but deletion was
                 chosen here since the app can be used with any dataset and it would be difficult to make specific decisions.
                 - The data is centered so that each feature has a mean of zero.
                 - Then, each feature is scaled by its standard deviation.

                 These steps ensure that each feature makes an equal contribution to the model. Otherwise, features with naturally
                 larger ranges would disproportionally influence the results.
                 After K-Means Clustering, rincipal Compoment Analysis (PCA) was used to generate 2D visualizations of the data.
                 """)
    
with col2:
    with st.expander("What is clustering?"):
        st.write("""
                 **Clustering** is a type of unsupervised machine learning technique used to classify unlabeled data into groups
                 based on similarities in their features.
                 This is very useful for real-world applications where data is not labeled, such as market segmentation and medical imaging analysis.
                 """)