from functions import *


@st.cache_data
def get_data():
    """
    Loads and processes the dataset.

    This function loads the dataset using the `load_data()` function and then processes it
    with the `process()` function. The processed result is cached to improve performance
    in future executions.

    Returns:
        pd.DataFrame: A DataFrame with the processed data.
    """
    return process(load_data())


def geogrhapic(df: pd.DataFrame):
    """
    Displays the geographical information of the dataset.

    This function displays different sections of geographical information in the user
    interface, including continents, top producers, top provinces, expensive countries,
    countries with the best price-quality ratio, and a comparison between countries.
    It also allows the user to filter the data by country through an interactive interface.

    Args:
        df (pd.DataFrame): The DataFrame containing the data to be visualized.
    """

    st.header("Geographic information", divider="gray")
    continents(df)
    main_producers(df)
    main_provinces(df)
    expensive_countries(df)
    countries_with_the_best_ratio(df)
    countries_comparisson(df)
    st.markdown("### Filter by country")
    options = st.multiselect("Select countries", df["country"].unique())
    if options:
        filtered_df = df[df["country"].isin(options)]
        st.dataframe(filtered_df)
    else:
        st.dataframe(df)


def wineries_results(df: pd.DataFrame):
    """
    Displays information about the wineries in the dataset.

    This function displays information about the wineries, including the relevant columns
    and the results of sentiment analysis. It also allows the user to filter the data by
    winery through an interactive interface.

    Args:
        df (pd.DataFrame): The DataFrame containing data about the wineries.
    """
    st.header("About wineries", divider="gray")

    wineries_cols(df)
    sentiment(df)

    st.markdown("### Filter by winery")
    options = st.multiselect(
        "Select wineries", df["winery"].unique()
    )
    if options:
        filtered_df = df[df["winery"].isin(options)]
        st.dataframe(filtered_df)
    else:
        st.dataframe(df)


def about_price(df: pd.DataFrame):
    """
    Displays information about the price distribution and relationship.

    This function displays two interactive visualizations: the price distribution
    and a scatter plot related to the prices of the products.

    Args:
        df (pd.DataFrame): The DataFrame containing the data to analyze the prices.
    """

    price_distribution(df)
    price_scatter(df)


def app():
    """
    Main function that runs the application.

    This function loads the data through `get_data()` and presents the user interface
    in Streamlit with three main sections: a preview of the dataset, geographical information,
    winery results, and price information.
    """
    df = get_data()

    st.title("Dataset Visualization")
    st.header("Dataset Preview", divider="gray")
    st.dataframe(df)

    geogrhapic(df)
    wineries_results(df)
    about_price(df)


app()
