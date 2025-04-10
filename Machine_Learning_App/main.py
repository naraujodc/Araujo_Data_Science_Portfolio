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

# data preprocessing
def preprocess(dataset):
    # read in dataset
    df = pd.read_csv(dataset)
    # encode categorical variables
    categorical_cols = [col for col in df.columns
                        if pd.api.types.is_categorical_dtype(df[col].dtype)
                        or pd.api.types.is_object_dtype(df[col].dtype)]
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    # define features and target
    X = df.drop(target, axis=1)
    y = target
    return df, X, y

# split dataset
def split_data(X, y, test_size = 0.2, random_state=None):
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size=test_size,
                                                        random_state=random_state)
    return X_train, X_test, y_train, y_test

# train decision tree
def train_decision_tree(X_train, y_train, criterion, max_depth, min_samples_split, min_samples_leaf):
    model = DecisionTreeClassifier(criterion=criterion,
                                   max_depth=max_depth,
                                   min_samples_split=min_samples_split,
                                   min_samples_leaf=min_samples_leaf)
    model.fit(X_train, y_train)
    return model

# plot confusion matrix

# plot ROC and AUC

# plot the decision tree

# ------------------------------------
# Sidebar Structure
# ------------------------------------
st.sidebar.markdown("## Choose a dataset")
dataset_upload = st.sidebar.file_uploader(label="Upload your own dataset",
                         type="csv")
st.sidebar.markdown("#### No dataset? Use a demo")
dataset_demo = st.sidebar.selectbox(label="Demo datasets",
                                    options=[])
if dataset_upload != None:
    dataset = dataset_upload
else:
    dataset = dataset_demo