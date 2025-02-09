import streamlit as st
import pandas as pd

# ======================
# setting up the web page
# ======================

# set page configurations
st.set_page_config(
    page_title = "Netflix Movies and TV Shows",
    page_icon = ":popcorn:",
    layout = "wide"
)

# display web page title
st.title("Netflix Movies and TV Shows")

# create button linking to original dataset
st.link_button("View dataset on Kaggle",
               "https://www.kaggle.com/datasets/anandshaw2001/netflix-movies-and-tv-shows?resource=download")

# write a brief description
st.markdown(
    """
    This dataset lists all movies and TV shows available on Netflix.
    Here, you can navigate through the data and filter it by different attributes
    such as media type, categories, and rating.
    """
)

# ======================
# setting up the dataframe
# ======================

# add dataset as a pandas dataframe
df = pd.read_csv("data/netflix_titles.csv", dtype = {"Release Year": str})

# rename columns
df = df.rename(columns = {
    "type" : "Type",
    "title" : "Title",
    "director" : "Director",
    "cast" : "Cast",
    "country" : "Country",
    "date_added" : "Date Added",
    "release_year" : "Release Year",
    "rating" : "Rating",
    "duration" : "Duration",
    "listed_in" : "Categories",
    "description" : "Description"
})

# set show title as the index
df.set_index("Title", inplace=True)

# delete show_id column
df = df.drop("show_id", axis=1)

# reorder columns
df = df.iloc[:, [0, 8, 6, 5, 7, 3, 4, 1, 2, 9]]

# convert strings into lists for list columns
df["Categories"] = df["Categories"].str.split(", ")
df["Country"] = df["Country"].str.split(", ")
df["Cast"] = df["Cast"].str.split(", ")

# convert release year column into string to avoid commas in years
df["Release Year"] = df["Release Year"].astype(str)

# ======================
# creating filters for the dataframe
# ======================

# setting the filtered df before filters
filtered_df = df

# create columns to place the widgets
col1, col2, col3 = st.columns([0.2, 0.55, 0.25])

## type
# creating a segmented control widget for type
media_types = df["Type"].unique()

with col1:
    media_type = st.segmented_control(
        "Filter by media type:",
        options = media_types,
        selection_mode = "multi"
    )

# filter dataset according to chosen type
if media_type:
    filtered_df = df[df["Type"].isin(media_type)]

## categories
# creating a multiselect button for categories
categories = df["Categories"].explode().unique()
categories = sorted(categories)

with col2:
    category = st.multiselect(
        "Filter by categories:",
        options = categories
    )

# filter dataset according to categories
if category:
    # check if any of the categories in the row match any selected category
    filtered_df = df[df["Categories"].apply(lambda x: any(cat in x for cat in category))]

## rating
# creating a multiselect button for ratings
# using a list to order it by restriction
ratings = ["G", "PG", "PG-13", "R", "NC-17",
"TV-Y", "TV-Y7", "TV-Y7 FV", "TV-G", "TV-PG", "TV-14", "TV-MA"]

with col3:
    rating = st.multiselect(
        "Filter by rating:",
        options = ratings
    )

# filter dataset according to ratings
if rating:
    filtered_df = df[df["Rating"].isin(rating)]

# ======================
# displaying the dataframe
# ======================

# display dataframe
st.dataframe(filtered_df)

# link to my github
st.markdown("Developed by [Natália Araújo](https://github.com/naraujodc)")