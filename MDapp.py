import altair as alt
import pandas as pd
import streamlit as st


DATA_PATH = "data/movies_genres_summary.csv"

# Show the page title and description.
st.set_page_config(page_title="Movies dataset", page_icon="🎬")
st.title("🎬 Movies dataset")
st.write(
    """
    This app visualizes data from [The Movie Database (TMDB)](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata).
    It shows which movie genre performed best at the box office over the years. Just 
    click on the widgets below to explore!
    """
)

# Load the data from a CSV. We're caching this so it doesn't reload every time the app
# reruns (e.g. if the user interacts with the widgets).
@st.cache_data
def load_data():
    try:
        df = pd.read_csv(DATA_PATH)
        return df
    except FileNotFoundError:
        st.error("Could not find the data file at 'data/movies_genres_summary.csv'. Make sure it exists and try again.")
        st.stop()


df = load_data()

# Show a multiselect widget with the genres using `st.sidebar.multiselect`.
st.sidebar.header("Filters")
genres = st.sidebar.multiselect(
    "Genres",
    df.genre.unique(),
    [],
)

# Don't render anything until the user picks at least one genre.
if not genres:
    st.info("Select a genre in the sidebar to get started.")
    st.stop()

# Show a slider widget with the years using `st.sidebar.slider`.
min_year = int(df["year"].min())
max_year = int(df["year"].max())
years = st.sidebar.slider("Years", min_year, max_year, (min_year, max_year))

# Let the user pick a metric.
metric = st.sidebar.radio(
    "Metric",
    ["Gross earnings ($)", "IMDb score", "Audience score"],
)

# Let the user pick a chart type.
chart_type = st.sidebar.radio("Chart type", ["Line", "Bar"])

metric_col = {
    "Gross earnings ($)": "gross",
    "IMDb score": "imdb_score",
    "Audience score": "vote_average",
}[metric]

# Filter the dataframe based on the widget input and reshape it.
df_filtered = df[(df["genre"].isin(genres)) & (df["year"].between(years[0], years[1]))]

df_reshaped = df_filtered.pivot_table(
    index="year", columns="genre", values=metric_col, aggfunc="mean", fill_value=0
)
df_reshaped = df_reshaped.sort_values(by="year", ascending=False)

# Display the data as a table using `st.dataframe`.
st.dataframe(
    df_reshaped,
    width='stretch',
    column_config={"year": st.column_config.TextColumn("Year")},
)

st.download_button(
    label="Download as CSV",
    data=df_reshaped.to_csv(),
    file_name=f"movies_{metric_col}_filtered.csv",
    mime="text/csv",
)

# Show a summary card per genre with its peak year and metric value.
st.subheader(f"Peak year by genre — {metric}")
cols = st.columns(len(genres))
for col, genre in zip(cols, genres):
    if genre in df_reshaped.columns:
        peak_year = df_reshaped[genre].idxmax()
        peak_val = df_reshaped[genre].max()
        if metric_col == "gross":
            display_val = f"${peak_val:,.0f}"
        else:
            display_val = f"{peak_val:.1f}"
        col.metric(
            label=f"{genre} - {peak_year}",
            value=display_val,
            #delta=str(peak_year),
        )

# Display the data as an Altair chart using `st.altair_chart`.
df_chart = pd.melt(
    df_reshaped.reset_index(), id_vars="year", var_name="genre", value_name=metric_col
)

mark = alt.Chart(df_chart).mark_line() if chart_type == "Line" else alt.Chart(df_chart).mark_bar(opacity=0.6)

chart = (
    mark
    .encode(
        x=alt.X("year:N", title="Year"),
        y=alt.Y(f"{metric_col}:Q", title=metric, stack=False),
        #y=alt.Y("gross:Q", title="Gross earnings ($)", stack=False),
        color=alt.Color("genre:N", legend=alt.Legend(title="Genre")),
        tooltip=[
            alt.Tooltip("year:N", title="Year"),
            alt.Tooltip("genre:N", title="Genre"),
            alt.Tooltip(f"{metric_col}:Q", title=metric, format=",.2f"),
        ],
    )
    .properties(height=320)
)
st.altair_chart(chart, width='stretch')
