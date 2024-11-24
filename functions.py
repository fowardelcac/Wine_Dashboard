import re
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
import streamlit as st
from textblob import TextBlob
import altair as alt
import seaborn as sns
import matplotlib.pyplot as plt


def separate_by_continent(row):
    """Assign a continent based on the country provided in the 'row' dictionary.

    This function receives a dictionary row that must contain a key "country" with the name of a country, and it returns the continent to which that country belongs. The continents are determined based on predefined lists of countries for each continent.

    Arguments: row (dict): A dictionary that contains at least the key "country", which is the name of the country to evaluate.

    Returns: str: The continent to which the country belongs. Possible values are: - "Europe" - "Asia" - "North America" - "South America" - "Oceania" - "Africa" - "Other" if the country is not in the predefined lists.
    """
    europe = [
        "Austria",
        "Bosnia and Herzegovina",
        "Bulgaria",
        "Croatia",
        "Cyprus",
        "Czech Republic",
        "England",
        "France",
        "Germany",
        "Greece",
        "Italy",
        "Luxembourg",
        "Portugal",
        "Hungary",
        "Macedonia",
        "Moldova",
        "Romania",
        "Serbia",
        "Slovakia",
        "Slovenia",
        "Spain",
        "Switzerland",
        "Turkey",
        "Ukraine",
        "Georgia",
    ]
    asia = ["Armenia", "China", "India", "Israel", "Lebanon"]
    northAmerica = ["Canada", "US", "Mexico"]
    sudAmerica = ["Argentina", "Brazil", "Chile", "Peru", "Uruguay"]
    oceania = ["Australia", "New Zealand"]
    africa = ["South Africa", "Morocco", "Egypt"]
    if row["country"] in europe:
        val = "Europe"
    elif row["country"] in asia:
        val = "Asia"
    elif row["country"] in northAmerica:
        val = "North America"
    elif row["country"] in sudAmerica:
        val = "South America"
    elif row["country"] in oceania:
        val = "Oceania"
    elif row["country"] in africa:
        val = "Africa"
    else:
        val = "Other"
    return val


def random_imputer(df_filter: pd.DataFrame, col: str):
    """
    Imputes missing (NaN) values in a column of a DataFrame with random values
    taken from the non-null values present in that same column.

    Arguments:
    df_filter (pd.DataFrame): The DataFrame containing the column with missing values.
    col (str): The name of the column where imputations will be made.

    Returns:
    pd.DataFrame: The DataFrame with missing values in the specified column
                  imputed with random values from the non-null values in the column.
    """
    non_null_values = df_filter[col].dropna().values

    df_filter.loc[:, col] = df_filter[col].apply(
        lambda x: np.random.choice(non_null_values) if pd.isnull(x) else x
    )
    return df_filter


def analyze_sentiment(text):
    """
    Analyzes the sentiment of a text and classifies it as "Positive", "Neutral", or "Negative".

    The function uses the TextBlob library to perform sentiment analysis on the provided
    text. Based on the sentiment polarity points, the function classifies the text into
    one of three categories:
    - "Negative" if the polarity is negative.
    - "Neutral" if the polarity is zero.
    - "Positive" if the polarity is positive.

    Arguments:
    text (str): The text to be analyzed for sentiment.

    Returns:
    str: The sentiment classification of the text, which can be:
        - "Negative" for negative sentiment.
        - "Neutral" for neutral sentiment.
        - "Positive" for positive sentiment.

    Example:
    sentiment = analyze_sentiment("I love this product!")
    print(sentiment)  # Output: "Positive"
    """
    analysis = TextBlob(text)
    points = analysis.sentiment.polarity
    if points < 0:
        return "Negative"
    elif points == 0:
        return "Neutral"
    else:
        return "Positive"


def get_year(texto):
    """
    Extracts a year from a text, either in a four-digit format (19xx or 20xx)
    or as an age expressed in "X Years" or "X Years Old".

    The function first attempts to find a four-digit year (e.g., 1999 or 2024).
    If no year is found, it tries to extract an age (in the format "X Years" or "X Years Old")
    and calculates the corresponding year based on the age, assuming the current year is 2024.

    Arguments:
    text (str): The text in which a year or age will be searched for.

    Returns:
    str or None: The extracted year as a string if a valid year is found, or the age converted to a year.
                 Returns None if no match is found.
    """

    match = re.search(r"\b(19|20)\d{2}\b", texto)
    if match:
        return match.group()

    match = re.search(r"(\d+)\s*(Años|Years\s*Old)", texto, re.IGNORECASE)
    if match:
        return str(int(2024) - int(match.group(1)))
    return None


def imputer(df_filter, col):
    """
    Imputes missing (NaN) values in a column of a DataFrame using the median
    of the existing values in that column.

    The function uses `SimpleImputer` from `sklearn` to replace NaN values in the
    specified column with the median of the non-null values present in that column.

    Arguments:
    df_filter (pd.DataFrame): The DataFrame where the missing values will be imputed.
    col (str): The name of the column in which the missing values will be imputed.

    Returns:
    pd.DataFrame: The DataFrame with the missing values in the specified column
                  imputed with the median.
    """

    imputer = SimpleImputer(strategy="median")
    df_filter.loc[:, col] = imputer.fit_transform(df_filter[[col]])
    return df_filter


def load_data():
    """
    Loads a wine dataset from a URL and returns a Pandas DataFrame.

    The function downloads a CSV file containing information about wines, loads it into a
    Pandas DataFrame, and removes an unnecessary column called 'Unnamed: 0', which is typically
    generated as an index when CSV files are saved with an index.

    Returns:
    pd.DataFrame: The DataFrame containing the CSV data, with the 'Unnamed: 0' column removed.
    """

    df = pd.read_csv(
        "https://media.githubusercontent.com/media/fowardelcac/Tp2_sem/refs/heads/main/winemag-data-130k-v2.csv"
    ).drop("Unnamed: 0", axis=1)
    return df


def process(df: pd.DataFrame):
    """
    Performs a series of transformations and cleaning operations on a wine DataFrame.

    This function performs several operations on the provided DataFrame, such as removing
    duplicates, handling missing values, imputing data, and creating new columns based on
    data analysis. The steps include:

    1. Remove duplicate rows.
    2. Remove rows with missing values in the 'country' and 'designation' columns.
    3. Add a 'continent' column based on the country using the `separate_by_continent` function.
    4. Impute missing values in the 'price' column using random values from the column.
    5. Create a 'year' column from the 'title' column, extracting the year using the `get_year` function.
    6. Impute missing values in the 'year' column.
    7. Calculate the 'age' column based on the current year (2024 - 'year').
    8. Add a 'sentiment' column based on the 'description' column, using the `analyze_sentiment` function.
    9. Create a 'points vs price' column that represents the relationship between points and price.

    Arguments:
    df (pd.DataFrame): A Pandas DataFrame containing wine data.

    Returns:
    pd.DataFrame: A processed DataFrame with the mentioned transformations, keeping only the relevant columns.
    """

    df = df.copy()
    df.drop_duplicates(inplace=True)
    df_filter = df.dropna(subset=["country", "description"])

    df_filter.loc[:, "continent"] = df_filter.apply(separate_by_continent, 1)

    df_filter = random_imputer(df_filter, "price")

    df_filter.loc[:, "year"] = df_filter["title"].apply(get_year)
    df_filter = imputer(df_filter, "year")
    df_filter.loc[:, "age"] = 2024 - df_filter.year.astype(int)

    df_filter.loc[:, "sentiment"] = df_filter["description"].apply(analyze_sentiment)
    df_filter.loc[:, "points vs price"] = df_filter["points"] / df_filter["price"]

    return df_filter.filter(
        [
            "continent",
            "country",
            "province",
            "designation",
            "title",
            "variety",
            "winery",
            "description",
            "sentiment",
            "points",
            "price",
            "points vs price",
            "year",
            "age",
        ],
        axis=1,
    )


def continents(df: pd.DataFrame):
    """
    Displays a bar chart visualizing the number of wines by continent.

    This function takes a DataFrame with a column called 'continent' that should contain the names
    of the continents for the wines. It then generates a bar chart using Altair to show the number
    of wines present in each continent, ordered from highest to lowest.

    The chart is rendered using Streamlit and displayed with the title "Production by Continent".

    Arguments:
    df (pd.DataFrame): A Pandas DataFrame that must contain a column called 'continent',
                       which holds the continents of the wines.
    """
    st.markdown("### Production by continent")
    continente_counts = df["continent"].value_counts().reset_index()
    continente_counts.columns = ["Continent", "Amount"]

    chart = (
        alt.Chart(continente_counts)
        .mark_bar()
        .encode(
            x=alt.X("Continent", sort="-y", axis=alt.Axis(labelAngle=-45)),
            y="Amount",
            color="Continent",
        )
    )

    st.altair_chart(chart, use_container_width=True)


def main_producers(df: pd.DataFrame):
    """
    Displays a bar chart of the 15 countries with the highest wine production.

    This function takes a DataFrame with a column called 'country' that should contain the names
    of the countries for the wines. It then generates a bar chart showing the top 15 countries
    with the highest number of wines, ordered from highest to lowest, and visualizes it in a
    Streamlit application.

    The chart is generated using the Altair library and displayed with the title "Countries with the Highest Production".

    Arguments:
    df (pd.DataFrame): A Pandas DataFrame that must contain a column called 'country',
                       which holds the countries of the wines.

    Returns:
    None: The function does not return anything, but generates an interactive chart visualized in a Streamlit app.
    """

    st.markdown("### Countries with biggest production")
    country_counts = df["country"].value_counts().head(15).reset_index()
    country_counts.columns = ["Country", "Amount"]
    chart = (
        alt.Chart(country_counts)
        .mark_bar()
        .encode(
            x=alt.X("Country", sort="-y", axis=alt.Axis(labelAngle=-45)),
            y="Amount",
            color="Country",
        )
    )

    st.altair_chart(chart, use_container_width=True)


def main_provinces(df: pd.DataFrame):
    """
    Displays a bar chart of the 15 provinces with the highest wine production.

    This function takes a DataFrame with the columns 'province' (province) and 'country' (country),
    then generates a bar chart showing the top 15 provinces with the highest number of wines,
    ordered from highest to lowest. The chart is visualized in a Streamlit application.

    The chart is generated using the Altair library and displayed with the title "Provinces with the Highest Production".

    Arguments:
    df (pd.DataFrame): A Pandas DataFrame that must contain the 'province' and 'country' columns,
                       which indicate the province and country of each wine.

    Returns:
    None: The function does not return anything, but generates an interactive chart visualized in a Streamlit app.
    """

    st.markdown("### Provinces with the biggest production")
    province_counts = (
        df.groupby(["province", "country"])["province"]
        .count()
        .sort_values(ascending=False)
        .head(15)
        .reset_index(name="Count")
    )
    province_counts.columns = ["Province", "Country", "Amount"]
    chart = (
        alt.Chart(province_counts)
        .mark_bar()
        .encode(
            x=alt.X("Province", sort="-y", axis=alt.Axis(labelAngle=-45)),
            y="Amount",
            color="Country",
        )
    )

    st.altair_chart(chart, use_container_width=True)


def expensive_countries(df: pd.DataFrame):
    """
    Displays a bar chart of the 15 countries with the highest average wine prices.

    This function takes a DataFrame with a column called 'country' and a column called 'price',
    calculates the average wine price per country, and generates a bar chart showing the top 15 countries
    with the highest average wine prices. The chart is visualized in a Streamlit application.

    The chart is generated using the Altair library and displayed with the title "Top 15 Countries with the Highest
    Average Wine Prices".

    Arguments:
    df (pd.DataFrame): A Pandas DataFrame that must contain the 'country' and 'price' columns,
                       which indicate the country and price of the wines.

    Returns:
    None: The function does not return anything, but generates an interactive chart visualized in a Streamlit app.
    """

    st.markdown("### The 15 Countries with the Highest Average Wine Prices")

    country_by_points = (
        df.groupby("country")["price"]
        .mean()
        .sort_values(ascending=False)
        .head(15)
        .reset_index()
    )
    country_by_points.columns = ["Country", "Average price"]

    chart = (
        alt.Chart(country_by_points)
        .mark_bar()
        .encode(
            x=alt.X("Country", sort="-y", axis=alt.Axis(labelAngle=-45)),
            y="Average price",
            color="Country",
        )
    )

    st.altair_chart(chart, use_container_width=True)


def countries_with_the_best_ratio(df: pd.DataFrame):
    """
    Displays a bar chart of the 10 countries with the best price-to-points ratio.

    This function takes a DataFrame with the columns 'country' and 'points vs price' (points/Price ratio),
    and calculates the best points-Price ratio per country. It then generates a bar chart showing the top 10 countries
    with the highest points-Price ratio, ordered from highest to lowest. The chart is visualized in a Streamlit application.

    The chart is generated using the Altair library and displayed with the title "Countries with the Best points-Price Ratio".

    Arguments:
    df (pd.DataFrame): A Pandas DataFrame that must contain the 'country' and 'points vs price' columns,
                       which indicate the country and points-Price ratio of the wines.

    Returns:
    None: The function does not return anything, but generates an interactive chart visualized in a Streamlit app.
    """

    st.markdown("### Countries with the Best points-Price Ratio")
    country_by_points = (
        df.groupby("country")["points vs price"]
        .max()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )
    country_by_points.columns = ["Country", "points-Price Ratio"]

    chart = (
        alt.Chart(country_by_points)
        .mark_bar()
        .encode(
            x=alt.X("Country", sort="-y", axis=alt.Axis(labelAngle=-45)),
            y="points-Price Ratio",
            color="Country",
        )
    )

    st.altair_chart(chart, use_container_width=True)


def countries_comparisson(df: pd.DataFrame):
    """
    Displays a comparison between the 10 countries with the highest average pointss, average prices,
    and best points/price ratio.

    This function calculates three metrics for each country:
    1. **Average points** of wines by country.
    2. **Average price** of wines by country.
    3. **points/price ratio** (best ratio per country).

    It then displays these metrics in three separate tables in a Streamlit application, using the
    `points`, `price`, and `points vs price` columns from the DataFrame.

    Arguments:
    df (pd.DataFrame): A Pandas DataFrame that must contain the 'country', 'points', 'price',
                       and 'points vs price' columns, which represent the country, wine points,
                       price, and the points/price ratio.

    Returns:
    None: The function does not return anything, but generates three tables displayed in a Streamlit app.
    """

    by_points = (
        df.groupby("country")["points"]
        .mean()
        .sort_values(ascending=False)
        .round(2)
        .head(10)
        .to_frame()
    )
    by_price = (
        df.groupby("country")["price"]
        .mean()
        .sort_values(ascending=False)
        .round(2)
        .head(10)
        .to_frame()
    )
    by_ratio = (
        df.groupby("country")["points vs price"]
        .max()
        .sort_values(ascending=False)
        .round(2)
        .head(10)
        .to_frame()
    )

    col1, col2, col3 = st.columns(3, vertical_alignment="center")
    with col1:
        st.text("Countries with the Highest Average Wine Points")
        col1.table(by_points)
    with col2:
        st.text("Countries with the Highest Average Wine Prices")
        col2.table(by_price)
    with col3:
        st.text("Countries with the Best Points/Price Ratio")
        col3.table(by_ratio)


def wineries_cols(df: pd.DataFrame):
    """
    Displays three tables comparing wineries based on different metrics: average points, points/price ratio,
    and production quantity.

    The function calculates the following metrics for each winery:
    1. **Average points** of the wines produced by each winery.
    2. **Best points/price ratio** (maximum ratio per winery).
    3. **Highest production quantity** (wineries with the largest number of wines).

    These metrics are displayed in three separate tables in a Streamlit application, arranged in three horizontal columns.

    Arguments:
    df (pd.DataFrame): A Pandas DataFrame that must contain the 'winery', 'points', and 'points vs price' columns,
                       which represent the winery, wine points, and the points/price ratio, respectively.

    Returns:
    None: The function does not return anything, but generates three tables displayed in a Streamlit app.
    """

    col1, col2, col3 = st.columns(3, vertical_alignment="center")
    with col1:
        st.text("Wineries with highest punctuation:")
        col1.dataframe(
            df.groupby("winery")["points"]
            .mean()
            .sort_values(ascending=False)
            .head(10)
            .to_frame()
        )
    with col2:
        st.text("Wineries with the Best Points vs. Price Ratio")
        col2.dataframe(
            df.groupby("winery")["points vs price"]
            .max()
            .sort_values(ascending=False)
            .head(10)
            .to_frame()
        )
    with col3:
        st.text("Wineries with the biggest production")
        col3.dataframe(
            df.winery.value_counts().sort_values(ascending=False).to_frame().head(10)
        )


def sentiment(df: pd.DataFrame):
    """
    Displays the wineries with the highest number of positive wine descriptions.

    This function filters the DataFrame to obtain only the descriptions classified as "Positive"
    by a sentiment analysis algorithm. It then groups the results by winery and displays the top 10 wineries
    with the highest number of positive descriptions.

    Arguments:
    df (pd.DataFrame): A Pandas DataFrame that must contain the 'winery' and 'sentiment' columns,
                       where 'sentiment' contains values like "Positive", "Negative", or "Neutral".

    Returns:
    None: The function does not return anything, but generates a table displayed in a Streamlit app
          showing the wineries and the number of positive descriptions.
    """

    st.markdown("### Wineries with the Most Positive Reviews")
    st.text(
        "** After using a sentiment analysis algorithm on the descriptions, it was determined whether they are positive, negative, or neutral."
    )
    rdo = df[df.sentiment == "Positive"]
    st.dataframe(
        rdo.groupby("winery")["sentiment"]
        .count()
        .sort_values(ascending=False)
        .head(10)
        .to_frame()
    )


def price_scatter(df: pd.DataFrame):
    """
    Displays a graphical analysis of the relationship between wine price and age,
    along with a numerical correlation between the selected variables.

    This function performs a correlation analysis and plots the data for price, age, and wine points
    to investigate the possible relationship between wine price and age.

    Arguments:
    df (pd.DataFrame): A Pandas DataFrame that must contain the 'price', 'age', and 'points' columns,
                       which represent the price, age, and points of the wines, respectively.

    Returns:
    None: The function does not return anything, but generates a table and an interactive chart displayed
          in a Streamlit app.
    """
    st.header("There is no relationship between price and age.", divider="gray")

    st.text(
        "By performing a correlation analysis, it can be stated that there is no relationship between the age of the wine and its price."
    )
    df = df.drop("points vs price", axis=1)
    df_corr = df.corr(numeric_only=True)
    st.dataframe(df_corr)

    st.markdown("### Graphically:")

    data = df.filter(["price", "age", "points"], axis=1)
    data.columns = ["Price", "Age", "Points"]
    chart = (
        alt.Chart(data)
        .mark_circle()
        .encode(
            x="Price",
            y="Age",
            color=alt.Color("Points", scale=alt.Scale(scheme="viridis")),
            tooltip=["Price", "Age", "Points"],
        )
        .interactive()
    )

    st.altair_chart(chart, use_container_width=True)


def price_distribution(df: pd.DataFrame):
    """
    Displays the distribution of wine prices, filtering values up to the 0.9 decile.

    This function filters the wine prices to show only those that are less than or equal to a threshold
    (the 0.9 decile, in this case, 65). It then displays a price distribution chart using a histogram
    with a probability density estimate (KDE).

    Arguments:
    df (pd.DataFrame): A Pandas DataFrame that must contain the 'price' column, which represents the price of the wines.

    Returns:
    None: The function does not return anything, but generates a distribution chart displayed in a Streamlit app.
    """
    st.header("Price Distribution", divider="gray")
    st.text("** Filtering values up to the 0.9 decile.")
    price_filter = df.loc[df.price <= 65]["price"]

    fig, ax = plt.subplots()
    sns.histplot(
        price_filter, kde=True, bins=30, label="Price Distribution (<= 65)", ax=ax
    )
    plt.xlabel("Price", fontsize=14)
    plt.ylabel("Frquency", fontsize=14)
    st.pyplot(fig)
