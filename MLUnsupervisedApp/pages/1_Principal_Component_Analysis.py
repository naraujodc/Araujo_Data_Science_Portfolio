# import necessary packages
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.preprocessing import StandardScaler
from Unsupervised_ML import *
from sklearn.datasets import load_breast_cancer, load_iris, load_wine


# -------------------------
# PAGE CONFIGURATIONS
# -------------------------
st.set_page_config(
    page_title="Principal Component Analysis",
    page_icon=":material/graph_3:",
    layout="wide"
)

# -------------------------
# INITIAL PAGE STRUCTURE
# -------------------------

st.title(":material/graph_3: Principal Component Analysis")

st.write("""
         Principal Component Analysis (PCA) is an unsupervised machine learning technique used for **dimensionality reduction**.
         In a nutshell, PCA finds the **principal components**, which are linear combinations of the original variables, that capture the most **variance** in the data.
         
         In this page, you can:
         - Upload your own dataset (labeled or unlabeled) or choose one of the demo datasets;
         - Choose the number of principal components you want your dataset to be reduced to;
         - Visualize the scatterplot and biplot for two PCs, and the scree plot for any number of PCs.
         """)


# -------------------------
# HELPER FUNCTIONS
# -------------------------

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
def pca_scatterplot(df, target):

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
def pca_biplot(X_std, X_pca, df, target):
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
    is_labeled = st.sidebar.checkbox(label="Click if dataset is labeled")
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

# get number of features to determine max number of PCs
if dataset_upload or dataset_demo is not None:
    # preprocess the data
    processed_df, X_std, y = data_preprocessing(df=dataset, target=target)

    # select hyperparameter after dataset
    st.sidebar.header("Hyperparameter Tuning")

    # choice for number of components
    n_components = st.sidebar.select_slider(label="Number of principal components",
                                            options=range(1,len(dataset.columns)),
                                            value=2)
    st.sidebar.caption("Note: Selecting 2 components allows for more visualizations.")

# -------------------------
# MAIN PAGE STRUCTURE
# -------------------------

# only display if dataset was chosen
if dataset_upload or dataset_demo is not None:

    # show total variance explained by PCA
    X_pca, exp_variance_cumsum = data_pca(X_std=X_std, n_components=n_components)
    st.write(f"#### :primary[A total of] {exp_variance_cumsum[-1]*100:.2f}% :primary[of the variance in the data is explained by] {n_components} :primary[principal component(s).]")

    # only show scatterplot and biplot if there are 2 PCs and target variable is selected
    if n_components == 2:
        # create two columns
        col1, col2 = st.columns(2)

        # display scatterplot
        with col1:
            st.subheader("Scatterplot")
            st.pyplot(fig=pca_scatterplot(df=dataset, target=target))
            with st.popover("What does this mean?"):
                st.write("""
                         This plots the data into a two-dimensional system defined by the first two principal components.
                         This allows us to see how the data is spread out and whether these PCs help us separate distinct groups.
                         If the dataset was labeled, the colors of each data point correspond to their true labels.
                         """)
                
        # display biplot
        with col2:
            st.subheader("Biplot")
            st.pyplot(fig=pca_biplot(X_std=X_std, X_pca=X_pca, df=dataset, target=target))
            with st.popover("What does this mean?"):
                st.write("""
                         The biplot overlays the **loadings** (original feature vectors) on the scatterplot.
                         This gives us an idea of the direction and contribution of each original feature in the reduced space.
                         """)

    # explain why the scatterplot doesn't show up with more PCs
    else:
        with st.popover("Why can't I see the other plots?"):
            st.write("""
                     One of the main advantages of PCA is the ability to create **2D visualizations** that capture the information
                     contained in multidimensional data. To do this, you need to select 2 principal components. The scatterplot and biplot
                     will allow you to see the distribution of the data and the contributions of the original features.
                     """)

    # show scree plot
    st.subheader("Scree Plot")
    st.pyplot(fig=scree_plot(df=dataset, X_std=X_std, target=target))
    with st.popover("What does this mean?"):
        st.write("""
                 This plots the cumulative proportion of variance in the data explained by the principal components
                 and the variance individually explained by each component. This can help us decide how many PCs to keep.
                 - **Elbow method:** Beyond the point where the plot forms an "elbow," additional principal components make
                 very marginal contributions to the cumulative explained variance.
                 """)

elif dataset_upload is None and dataset_demo is None:
    st.write("#### :primary[Please select a dataset to start.]")

# create section for common questions
st.subheader("Common Questions")

# create columns for help buttons
col1, col2 = st.columns(2)

with col1:
    with st.expander("How is the data preprocessed for PCA?"):
        st.write("""
                 - First, rows with missing data are excluded. There are different ways to handle missing data, but deletion was
                 chosen here since the app can be used with any dataset and it would be difficult to make specific decisions.
                 - The data is centered so that each feature has a mean of zero.
                 - Then, each feature is scaled by its standard deviation.

                 These steps ensure that each feature makes an equal contribution to the model. Otherwise, features with naturally
                 larger ranges would disproportionally influence the results.
                 """)
    
with col2:
    with st.expander("What is dimensionality reduction?"):
        st.write("""
                 **Dimensionality reduction** is a type of unsupervised machine learning technique used to reduce the number of features
                 of a high-dimensional dataset while retaining the most important properties of the data.
                 This is a great way to handle the **curse of dimensionality**, preparing the data for better model performance and
                 easier visualization, and improving computational efficiency.
                 """)