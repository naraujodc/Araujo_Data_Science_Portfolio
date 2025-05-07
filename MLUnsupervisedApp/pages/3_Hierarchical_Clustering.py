# import necessary packages
import streamlit as st
import pandas as pd

from sklearn.datasets import load_breast_cancer, load_iris, load_wine
from Supporting_Functions import *

# -------------------------
# PAGE CONFIGURATIONS
# -------------------------
st.set_page_config(
    page_title="Hierarchical Clustering",
    page_icon=":material/graph_1:",
    layout="wide"
)

# -------------------------
# INITIAL PAGE STRUCTURE
# -------------------------

st.title(":material/graph_1: Hierarchical Clustering")
st.write("""
         **Hierarchical Clustering** is an **unsupervised machine learning** technique used to group data points into clusters by creating a
         hierarchy of clusters. This hierarchy is often represented in a **dendrogram**, a tree-like structure.
         It can be divisive (top-down) or agglomerative (bottom-up). Here, we use **Agglomerative Hierarchical Clustering**.
         Each object starts as its own cluster and the algorithm progressively groups the two closest clusters until there is only one.

         In this page, you can:
         - Upload your own dataset (labeled or unlabeled) or use a demo dataset;
         - Choose the number of clusters (k) you want to form;
         - Choose the linkage method to calculate the distance between clusters;
         - Visualize the dendrogram of your dataset;
         - Visualize the scatterplot of your clusters and compare them with the true labels (if the data is labeled);
         - Learn how to find the best k for unlabeled data using silhouette scores.
         """)

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
    
    # select linkage method
    linkage_choice = st.sidebar.selectbox(label="Linkage method",
                                   options=["Single", "Complete", "Ward"],
                                   index=None)
    if linkage_choice == "Single":
        linkage_method = "single"
    elif linkage_choice == "Complete":
        linkage_method = "complete"
    elif linkage_choice == "Ward":
        linkage_method = "ward"

# -------------------------
# MAIN PAGE STRUCTURE
# -------------------------

# check for adequate input
if (dataset_demo or dataset_upload is not None) and linkage_choice is not None:

    # preprocess data
    processed_df, X_std, y = data_preprocessing(df=dataset, target=target)
    
    # plot dendrogram
    st.subheader("Dendrogram")
    st.write("""
             The dendrogram is a useful tool to understand how the clusters are grouped. Since Hierarchical Clustering
             does not require you to pre-determine a number of clusters (k), **visually inspecting the dendrogram** is
             a good way to gauge an appropriate number of clusters for your data.
             """)
    st.pyplot(fig=plot_dendrogram(X_std=X_std, linkage_choice=linkage_choice, linkage_method=linkage_method, y=y))
    with st.popover("What does this mean?"):
        st.write("""
                 This dendrogram shows how the clusters were progressively grouped by Hierarchical Clustering.
                 At the bottom, you see each individual object starting as its own cluster.
                 The height at which two clusters are combined into one indicates the **linkage distance** between them.
                 """)

    # apply agglomerative hierarchical clustering
    agg, cluster_labels = agg_hierarchical(k=k, linkage_method=linkage_method, X_std=X_std, df_processed=processed_df)

    # show scatterplot by itself if the data is unlabeled
    if target is None:
        col1, col2, col3 = st.columns([0.1, 0.8, 0.1])
        with col2:
            # plot clusters
            st.subheader("Hierarchical Clustering")
            st.pyplot(fig=plot_clustering(X_std=X_std, cluster_labels=cluster_labels))
            with st.popover("What does this mean?"):
                st.write("""
                         This scatterplot shows the clusters formed by Hierarchical Clustering projected on a 2D plane generated by
                         Principal Component Analysis (PCA). This gives us an idea of how the clusters are separated.
                         ***If your dataset is labeled, select the target variable in the sidebar to view the true labels as a comparison.***
                         """)

    else:
        col1, col2 = st.columns(2)
        with col1:
            # plot clusters
            st.subheader("Hierarchical Clustering")
            st.pyplot(fig=plot_clustering(X_std=X_std, cluster_labels=cluster_labels))
            with st.popover("What does this mean?"):
                st.write("""
                         This scatterplot shows the clusters formed by Hierarchical Clustering projected on a 2D plane generated by
                         Principal Component Analysis (PCA). This gives us an idea of how the clusters are separated.
                         """)

        with col2:
            # apply PCA
            X_pca, exp_variance_cumsum = data_pca(X_std=X_std, n_components=2)
            # plot true clusters
            st.subheader("True Clusters in the Data")
            st.pyplot(fig=pca_true_scatterplot(df=dataset, X_pca=X_pca, target=target, y=y))
            with st.popover("What does this mean?"):
                st.write("""
                         This scatterplot shows the true clusters in the data projected on a 2D plane generated by
                         Principal Component Analysis (PCA). This allows us to gauge the performance of Hierarchical Clustering.
                         ***How similar are the true labels to the clusters generated by Hierarchical Clustering?***
                         """)

    # section for understanding how to choose k
    st.subheader("Choosing the optimal number of clusters (k)")
    st.write("""
             When working with real-world unlabeled data, it may be hard to decide how many clusters we should try to find in the data.
             For example, for a market segmentation analysis, how many different costumer groups could we have? How many is _too_ many?
             
             Besides **visual inspection of the dendrogram**, a good way to do this is by using the silhouette score.
             """)
    
    col1, col2 = st.columns(2)

    with col1:
        st.write("#### Silhouette Score")
        st.write("""
                 We plot the **silhouette score** against a range of possible values of k. This score indicates how similar an object is to its
                 own cluster compared to the other clusters &mdash; so the **higher** the silhouette score, the better.

                 Silhouette scores closer to **+1** indicate good clustering, while scores closer to **-1** indicate that there is a
                 poor clustering structure, which is worse than random assignment of samples to clusters.
                 """)

    with col2:
        # plot silhouette scores
        st.pyplot(fig=silhouette_agg(X_std=X_std, linkage_choice=linkage_choice, linkage_method=linkage_method))

    # option to view dataset information
    with st.expander("**View Dataset Information**"):
        if dataset_demo == "Breast Cancer":
            st.write("""
                    **Breast cancer wisconsin (diagnostic) dataset:** This is one of the toy datasets from the Python package scikit-learn.
                    It is used to predict whether a tumor is malignant or benign according to 30 predictive variables.
                    """)
        elif dataset_demo == "Iris":
            st.write("""
                    **Iris plants dataset:** This is one of the toy datasets from the Python package scikit-learn.
                    It is used to predict whether an Iris plant is from the species Setosa, Versicolour, or Virginica according to 4 predictive variables.
                    """)
        elif dataset_demo == "Wine":
            st.write("""
                    **Wine recognition dataset:** This is one of the toy datasets from the Python package scikit-learn.
                    It is used to predict whether a wine was manufactured by cultivator 0, 1, or 2 in Italy using 13 predictive variables.
                    """)
        col1, col2 = st.columns(2)
        with col1:
            st.write("#### First 5 Rows of the Dataset")
            st.dataframe(dataset.head())
        with col2:
            st.write("#### Statistical Summary")
            st.dataframe(dataset.describe())

# prompt dataset selection
elif dataset_upload is None and dataset_demo is None:
    st.write("#### :primary[Please select a dataset to start.]")

# prompt linkage method selection
elif (dataset_upload or dataset_demo is not None) and linkage_choice is None:
    st.write("#### :primary[Please select a linkage method to start.]")

# create section for common questions
st.subheader("Common Questions")

# create columns for help buttons
col1, col2 = st.columns(2)

with col1:
    with st.expander("How is the data preprocessed for Hierarchical Clustering?"):
        st.write("""
                 - First, rows with missing data are excluded. There are different ways to handle missing data, but deletion was
                 chosen here since the app can be used with any dataset and it would be difficult to make specific decisions.
                 - The data is centered so that each feature has a mean of zero.
                 - Then, each feature is scaled by its standard deviation.

                 These steps ensure that each feature makes an equal contribution to the model. Otherwise, features with naturally
                 larger ranges would disproportionally influence the results.
                 After Hierarchical Clustering, Principal Compoment Analysis (PCA) was used to generate 2D visualizations of the data.
                 """)
    
with col2:
    with st.expander("What is clustering?"):
        st.write("""
                 **Clustering** is a type of unsupervised machine learning technique used to classify unlabeled data into groups
                 based on similarities in their features.
                 This is very useful for real-world applications where data is not labeled, such as market segmentation and medical imaging analysis.
                 """)
        
with st.expander("What is a linkage method?"):
    st.write("""
             **Linkage methods** define how the distance between two clusters is measured, which determines how the clusters are merged.
             There are many options to choose from, but here you have three choices:
             - **Single (Nearest Neighbor):** The distance between two clusters is the **shortest distance between any two points**
             in the two clusters. It tends to create long, chain-like clusters and it is highly sensitive to noise/outliers.
             - **Complete (Furthest Neighbor):** The distance between two clusters is the **longest distance between any two points**
             in the two clusters. It tends to form more compact, spherical clusters and it is less sensitive to noise/outliers.
             - **Ward's Method (Minimum Variance):** At each step, it merges the two clusters that result in the smallest increase in the
             within-cluster variance. It tends to form clusters with similar numbers of data points.
             """)