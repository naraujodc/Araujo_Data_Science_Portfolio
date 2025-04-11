# import necessary packages
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import graphviz

from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, roc_auc_score
from sklearn.datasets import load_breast_cancer, load_iris, load_wine

# ------------------------------------
# Application Information
# ------------------------------------
st.title("Decision Tree Hyperparameter Tuning")
st.markdown("""
### About this Application
This interactive application demonstrates the performance of a Decision Tree classifier with different hyperparameters. You can:
- Input your own dataset or choose one of the demo datasets available.
- Choose a metric to use as the criterion for splitting the data at each node.
- Tune the model's hyperparameters.
- Automatically find the best tree according to a scoring parameter of your choice.
""")

# ------------------------------------
# Helper Functions
# ------------------------------------

# turn sklearn toy datasets into pandas dataframes
def toy_to_df(load_function):
    bunch = load_function()
    df = pd.DataFrame(bunch.data, columns=bunch.feature_names)
    df["target"] = bunch.target
    return df

# data preprocessing
def preprocess(df):
    # encode categorical variables
    categorical_cols = [col for col in df.columns
                        if pd.api.types.is_categorical_dtype(df[col].dtype)
                        or pd.api.types.is_object_dtype(df[col].dtype)]
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    # define features and target
    X = df.drop(target, axis=1)
    y = df[target]
    return df, X, y

# split dataset
def split_data(X, y, test_size = 0.2, random_state=None):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

# train decision tree
def train_decision_tree(X_train, y_train, criterion, max_depth, min_samples_split, min_samples_leaf):
    model = DecisionTreeClassifier(criterion=criterion,
                                   max_depth=max_depth,
                                   min_samples_split=min_samples_split,
                                   min_samples_leaf=min_samples_leaf)
    model.fit(X_train, y_train)
    return model

# plot confusion matrix
def plot_confusion_matrix(cm):
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    st.pyplot(plt)
    plt.clf()

# plot ROC and AUC

# plot the decision tree
def plot_decision_tree(model,df,target,X_train):
    dot_data = tree.export_graphviz(model,
                                    feature_names=X_train.columns,
                                    class_names = df[target].unique().astype('str'),
                                    filled=True)
    graph = graphviz.Source(dot_data)
    st.graphviz_chart(graph)

# ------------------------------------
# Sidebar Structure
# ------------------------------------

# dataset upload option
st.sidebar.markdown("## Dataset Selection")
dataset_upload = st.sidebar.file_uploader(label="Upload your own dataset",
                         type="csv")

# dataset demo option
st.sidebar.markdown("#### No dataset? Use a demo")
dataset_demo = st.sidebar.selectbox(label="Demo datasets",
                                    options=["Breast Cancer", "Iris", "Wine"],
                                    index=None)

# use uploaded dataset if user inputs one
if dataset_upload is not None:
    dataset = pd.read_csv(dataset_upload)
    target = st.sidebar.selectbox(label="What is the target variable?",
                                  options=dataset.columns)
# use demo datasets if user chooses one
elif dataset_demo == "Breast Cancer":
    dataset = toy_to_df(load_breast_cancer)
    target = "target"
elif dataset_demo == "Iris":
    dataset = toy_to_df(load_iris)
    target = "target"
elif dataset_demo == "Wine":
    dataset = toy_to_df(load_wine)
    target = "target"

# select hyperparameters
st.sidebar.markdown("## Hyperparameter Selection")

# select hyperparameter: criterion
criterion = st.sidebar.selectbox(label="Criterion",
                                 options=["Gini Index", "Entropy", "Log Loss"])
if criterion == "Gini Index":
    criterion = "gini"
elif criterion == "Entropy":
    criterion = "entropy"
elif criterion == "Log Loss":
    criterion = "log_loss"

# select hyperparameter: max depth
max_depth = st.sidebar.select_slider(label="Maximum depth of tree",
                                     options=range(1, 21))

# select hyperparameter: min samples split
min_samples_split = st.sidebar.select_slider(label="Minimum samples to split",
                                             options=range(2,21,2))

# select hyperparemeter: min samples leaf
min_samples_leaf = st.sidebar.select_slider(label="Minimum samples for leaf",
                                            options=range(1, 21))

# ------------------------------------
# Main Page Structure
# ------------------------------------

if dataset_upload or dataset_demo is not None:

    # preprocess and split df, train model
    processed_df, X, y = preprocess(dataset)
    X_train, X_test, y_train, y_test = split_data(X, y)
    dt_model = train_decision_tree(X_train=X_train, y_train=y_train, criterion=criterion, max_depth=max_depth,
                                   min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf)

    # classification report
    st.subheader("Classification Report")

    # predict and evaluate model
    y_pred = dt_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"**Accuracy:** {accuracy:.2f}")

    # display classification report
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report_dict).T
    st.dataframe(report_df)

    # create two columns for side-by-side display
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Confusion matrix")
        cm = confusion_matrix(y_test, y_pred)
        plot_confusion_matrix(cm)
    
    with col2:
        st.subheader("ROC and AUC plot")
        
    # show decision tree
    st.subheader("Decision Tree")
    plot_decision_tree(model=dt_model, df=dataset, target=target,X_train=X_train)

else:
    st.markdown("#### Please select a dataset to start.")