#!/usr/bin/env python
# coding: utf-8

# ## Streamlit + GitHub

# ### Import Libraries

# In[ ]:


import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib as plt


# ### Load Data

# In[ ]:


ratings = pd.read_csv("rating.csv", encoding="ISO-8859-1")
movies = pd.read_csv("movies.csv", encoding="ISO-8859-1")


# ### Preprocess

# In[9]:


ratings["year"] = pd.to_datetime(ratings["timestamp"], unit="s").dt.year
movies["genres_list"] = movies["genres"].str.split("|")
df = pd.merge(ratings, movies, on="movieId", how="left")


# ### Sidebar Filters

# In[14]:


st.sidebar.title("Movie Explorer (Ages 18–35)")
st.sidebar.markdown("Filter by genre to explore rating trends")

all_genres = sorted({genre for sublist in movies["genres_list"] for genre in sublist})
selected_genres = st.sidebar.multiselect("Select genres:", all_genres, default=["Comedy", "Action"])

df_filtered = df[df["genres_list"].apply(lambda x: any(g in x for g in selected_genres))]


# ### Page Title

# In[17]:


st.title(" Interactive Movie Ratings Dashboard")
st.markdown("**Audience:** Ages 18–35  \n*Purpose:* Discover genre trends, popularity, and ratings")


# ### Plot 1 – Most Rated Genres

# In[ ]:


genre_counts = (
    df.explode("genres_list")
    .groupby("genres_list")
    .agg(avg_rating=("rating", "mean"), rating_count=("rating", "count"))
    .reset_index()
)

fig1 = px.bar(
    genre_counts.sort_values("rating_count", ascending=False).head(10),
    x="genres_list", y="rating_count", color="avg_rating",
    color_continuous_scale="Turbo",
    title="Most Rated Genres with Average Ratings",
    labels={"genres_list": "Genre", "rating_count": "Number of Ratings", "avg_rating": "Avg Rating"}
)
st.plotly_chart(fig1, use_container_width=True)


# ### Plot 2 – Ratings Distribution

# In[21]:


fig2 = px.histogram(
    df_filtered, x="rating", nbins=20,
    title="Ratings Distribution for Selected Genres",
    color_discrete_sequence=["#636EFA"]
)
fig2.update_layout(bargap=0.2)
st.plotly_chart(fig2, use_container_width=True)

# Step 8: Plot 3 – Animated Ratings Over Time
genre_year_summary = (
    df.explode("genres_list")
    .groupby(["genres_list", "year"])
    .agg(avg_rating=("rating", "mean"), rating_count=("rating", "count"))
    .reset_index()
    .rename(columns={"genres_list": "genre"})
)

fig3 = px.bar(
    genre_year_summary[genre_year_summary["genre"].isin(selected_genres)],
    x="genre", y="rating_count", color="avg_rating", animation_frame="year",
    color_continuous_scale="Turbo",
    title="Genre Popularity Over Time (Animated)",
    labels={"genre": "Genre", "rating_count": "Ratings Count", "avg_rating": "Avg Rating"}
)
st.plotly_chart(fig3, use_container_width=True)


# ### Insights

# In[ ]:


st.markdown("---")
st.markdown("### Why This Dataset Fits Machine Learning")
st.markdown("""
- **Genre Preferences:** Consistent patterns across genre and rating trends  
- **User Behavior:** Millions of data points help build recommender systems  
- **Scalability:** Well-structured for collaborative filtering or clustering  
- **Business Use:** E-commerce/movie platforms can personalize content
""")


# In[ ]:





# In[ ]:




