# Tidy Data Project
![Python](https://img.shields.io/badge/-Python-ffe873?style=flat&logo=python)&nbsp;
![Pandas](https://img.shields.io/badge/pandas%20-%23150458.svg?&style=flat&logo=pandas&logoColor=white)&nbsp;
![Seaborn](https://img.shields.io/badge/Seaborn-79b6bc)&nbsp;
![Matplotlib](https://img.shields.io/badge/Matplotlib-1e3f5a)&nbsp;
![highlight_text](https://img.shields.io/badge/highlight_text-be5103)

## Table of Contents:
[1. Project Overview (With Visual Examples)](#project-overview-with-visual-examples)\
[2. Files](#files)\
[3. Dataset Description](#dataset-description)\
[4. Instructions for Use](#instructions-for-use)\
[5. References](#references)

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
With this, I transformed a messy dataset into a tidy dataframe in a way that can be understood as:
<p align="center">
<img src="https://github.com/naraujodc/Araujo_Data_Science_Portfolio/blob/main/TidyData-Project/images/tidy_data_example.jpg" />

### Data Manipulation and Visualization
Then, I used a few aggregation techniques to manipulate the data and construct useful visualizations such as the following:
<p align="center">
<img src="https://github.com/naraujodc/Araujo_Data_Science_Portfolio/blob/main/TidyData-Project/images/medals_by_sport.png" width="450" height="350"/>
<img src="https://github.com/naraujodc/Araujo_Data_Science_Portfolio/blob/main/TidyData-Project/images/medals_by_gender_type.png" width="450" height="350"/>
</p>

- The standard format of tidy data was essential to manipulate the data and produce the visualizations above. For more detail, refer to the code in the Jupyter Notebook.

I also demonstrate how pivot tables may be a useful way to aggregate data for analysis, especially when the data contains duplicates. Here, for example, multiple medalists are associated with the same sport. Thus, I used a pivot table to count how many medals were received in each sport using an aggregation function. I chose to use the values for "gender" as the new columns to separate the medal counts by gender.

With a few simple lines of code, I produced the pivot table described above:
```
# aggregate data to count number of medals by sport, gender, and medal type
df_sport = pd.pivot_table(df_medals,
                          index="sport",
                          columns="gender",
                          values="medal",
                          aggfunc="value_counts")
```
The pivot table reorganized my data into a format that can be used to quickly present the data with more straightforward conclusions. This technique may be particularly useful for presenting data to stakeholders.

<p align="center">
<img src="https://github.com/naraujodc/Araujo_Data_Science_Portfolio/blob/main/TidyData-Project/images/pivot_example.jpg" />
</p>

Using the new pivot table, I created a visualization showing the percentage of medals in each sport that were received by female and male athletes.

<p align="center">
<img src="https://github.com/naraujodc/Araujo_Data_Science_Portfolio/blob/main/TidyData-Project/images/medal_gender_dist_sport.png" />
</p>

#### Final Remarks
In this project, I have demonstrated the importance of tidying up data before an analysis according to [Hadley Wickham's](https://vita.had.co.nz/papers/tidy-data.pdf) framework. With simple steps, I was able to transform a messy dataframe into a tidy one, following a standard data structure. This allowed me to apply several methods to analyze and visualize my data using different packages.

In addition, I have shown how reshaping and aggregating tidy data may be useful to easily highlight different conclusions in tables and visualizations.

I highly recommend reading the **Jupyter Notebook** associated with this project. It contains detailed explanations of all the decisions I made while tidying and manipulating the data, all the conclusions I took from the exploratory visualizations, and all the code I used.

## Files
- `TidyData.ipynb`: This notebook contains the code and detailed explanations for the project.
- `olympics_08_medalists.csv`: Raw data file.
- `images`: This directory stores the plots generated in the Jupyter Notebook and other images used in the documentation.
- `README.md`: Project documentation.
- `requirements.txt`: List of required Python libraries.

## Dataset Description
The dataset used contains information about the Olympic Medalists from 2008 and was adapted from [this dataset](https://edjnet.github.io/OlympicsGoNUTS/2008/) authored by Giorgio Comai for EDJNet.

Unlike the traditional medal tables used during the Olympic Games, this dataset counts each person receiving a medal as a medal. For example, in team sports, each member of a medalist team is counted as a medalist.

### Messy dataframe:
_1875 rows, 71 columns_

**Variables:**
- `medalist_name`: name of each athlete receiving a medal (categorical)
- `gender_sport`: type of medal received for a certain sport associated with a gender; there are 70 such columns, e.g. `male_archery` (categorical)

### Tidy dataframe:
_1875 rows, 4 columns_

**Variables:**

- `medalist_name`: name of each athlete receiving a medal (categorical)
- `sport`: sport for which the athlete received a medal (categorical)
- `medal`: type of medal received, i.e. gold, silver, or bronze (categorical)
- `gender`: gender of the medalist (categorical)

## Instructions for Use
**Note:** If you only want to read the Jupyter Notebook without running it, simply click on the `TidyData.ipynb` file in this repository.
### Installation:
1. Install Anaconda.
2. Clone this repository and cd into it.
```
git clone https://github.com/naraujodc/Araujo_Data_Science_Portfolio
```
```
cd TidyData-Project
```
3. In a new environment `(new_env)`, install the dependencies.
```
pip install -r requirements.txt
```
4. Open and run the notebook `TidyData.ipynb`.

## References
Below I have linked some references that guided this work and/or that may be useful for further reading into the topic of tidy data and the techniques I have used.
- [_Tidy Data_](https://vita.had.co.nz/papers/tidy-data.pdf), by Hadley Wickham
- [Pandas Cheat Sheet](https://pandas.pydata.org/Pandas_Cheat_Sheet.pdf)
- [From Data to Viz](https://www.data-to-viz.com/)
