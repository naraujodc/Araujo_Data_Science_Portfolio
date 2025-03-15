# Tidy Data Project
## Table of Contents:
[1. Project Overview (With Visual Examples)](#project-overview-with-visual-examples)\
[2. Dataset Description](#dataset-description)\
[3. Instructions for Use](#instructions-for-use)\
[4. References](#references)

## Project Overview (With Visual Examples)
The purpose of this project is to apply the principles of tidy data to a messy dataset and demonstrate how tidy data can be easily used for data analysis and visualization.

This project is based on the principles of tidy data outlined by [Hadley Wickham](https://vita.had.co.nz/papers/tidy-data.pdf), namely:
- Each variable must form a column.
- Each observation must form a row.
- Each type of observational unit must form a table.

By organizing messy datasets into the standard data structure outlined above, we can easily apply simple tools for three important elements of data analysis: **manipulation**, **visualization**, and **modeling**.

Here, I have cleaned up a messy dataset and organized it into a tidy dataframe. Then, I applied different methods of manipulation (especially aggregation and sorting) to use the dataframe for exploratory data analysis and visualization.

Below are some examples of what you will find in greater detail in the Jupyter Notebook for this project.

### Data Cleaning and Tidying
The original dataset contained values as column headers (e.g. "male" and "female", which are values of the variable "gender"), and two variables combined ("gender" and "sport").

I solved these issues with a few simple lines of code:
```
# create a copy of the original dataframe
df_medals = df.copy()

# reorganize dataframe so that each variable is in a separate column
df_medals = pd.melt(df_medals,
                    id_vars=["medalist_name"],
                    value_vars=df_medals.columns[1:],
                    var_name="sport",
                    value_name="medal")

# split sport column so that each variable within it has its own column
df_medals[["gender", "sport"]] = df_medals["sport"].str.split("_", expand = True)

# drop rows containing null data
df_medals = df_medals.dropna()
```
With this, I transformed a messy dataset into a tidy dataframe in a way that can be visualized as:


### Data Manipulation and Visualization

## Dataset Description

## Instructions for Use

## References
Below I have linked some references that guided this work and/or that may be useful for further reading into the topic of tidy data and the techniques I have used.
- [***Tidy Data***](https://vita.had.co.nz/papers/tidy-data.pdf), by Hadley Wickham
- [Pandas Cheat Sheet](https://pandas.pydata.org/Pandas_Cheat_Sheet.pdf)
- [From Data to Viz](https://www.data-to-viz.com/)
