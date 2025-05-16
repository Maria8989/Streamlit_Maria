#!/usr/bin/env python
# coding: utf-8

# # Machine Learniarning for Bussines 

# ## Data Visualizationn

# ### Import Libraries

# In[17]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objs as go
import warnings
warnings.filterwarnings("ignore")


# ### Loading the datasets

# In[19]:


tags = pd.read_csv('tags.csv', encoding='ISO-8859-1')
movies = pd.read_csv('movies.csv', encoding='ISO-8859-1')
ratings = pd.read_csv('rating.csv', encoding='ISO-8859-1')


# ### Exploring the data

# In[21]:


print("Tags Preview:")
print(tags.head())

print("Movies Preview:")
print(movies.head())

print("Ratings Preview:")
print(ratings.head())

# ### Shape

# In[23]:


print(f"Tags shape: {tags.shape}")
print(f"Movies shape: {movies.shape}")
print(f"Ratings shape: {ratings.shape}")


# ### Detecting missing values

# In[25]:


print("Missing values in tags:\n", tags.isnull().sum())
print("Missing values in movies:\n", movies.isnull().sum())
print("Missing values in ratings:\n", ratings.isnull().sum())


# ### Duplicates

# In[27]:


print(f"Duplicate rows in tags: {tags.duplicated().sum()}")
print(f"Duplicate rows in movies: {movies.duplicated().sum()}")
print(f"Duplicate rows in ratings: {ratings.duplicated().sum()}")


# ### Merge Movies with Ratings

# In[29]:


df = pd.merge(ratings, movies, on='movieId')
df['genres_list'] = df['genres'].str.split('|')


# ### Data Prep for Visualization

# In[60]:


# --- Explode genres ---
df_genre = df.explode('genres_list')

# --- Ratings per Genre ---
genre_summary = df_genre.groupby('genres_list').agg(
    count=('rating', 'count'),
    avg_rating=('rating', 'mean')
).reset_index().sort_values(by='count', ascending=False)


# ### Interactive Visual with a Genre Slicer (Plotly)

# In[37]:


import plotly.express as px

fig = px.bar(
    genre_summary,
    x='genres_list',
    y='count',
    color='avg_rating',
    color_continuous_scale='Turbo',
    title='Most Rated Genres & Their Average Ratings',
    labels={
        'genres_list': 'Genre',
        'count': 'Number of Ratings',
        'avg_rating': 'Average Rating'
    }
)

fig.update_layout(
    plot_bgcolor='white',
    font=dict(size=14, family="Helvetica"),
    title_x=0.5
)

fig.show()


# In[39]:


fig.write_html("interactive_genre_bar.html")


# ### Interactive visual Genre popularity Over time

# In[56]:


import plotly.express as px

# 1. Explode genres_list so each genre is on its own row
df_exploded = df.explode('genres_list')

# 2. Rename the column for clarity
df_exploded = df_exploded.rename(columns={'genres_list': 'genre'})

# 3. Extract the year from the timestamp (for animation)
df_exploded['year'] = pd.to_datetime(df_exploded['timestamp'], unit='s').dt.year

# 4. Group by year and genre to count ratings and get average
genre_year_summary = (
    df_exploded.groupby(['year', 'genre'])
    .agg(
        rating_count=('rating', 'count'),
        avg_rating=('rating', 'mean')
    )
    .reset_index()
)


fig = px.bar(
    genre_year_summary,
    x='genre',
    y='rating_count',
    color='avg_rating',
    animation_frame='year',
    title="Genre Popularity Over Time ",
    labels={'genre': 'Genre', 'rating_count': 'Ratings Count', 'avg_rating': 'Avg Rating'},
    color_continuous_scale='Turbo'
)

fig.update_layout(
    plot_bgcolor='white',
    font=dict(size=14),
    title_x=0.5
)

fig.show()


# In[51]:


fig.write_html("animated_genre_popularity.html")

print("Interactive dashboard saved as 'animated_genre_popularity.html'")


# In[ ]:





# In[71]:


import streamlit as st
import pandas as pd
import plotly.express as px

# --- Load Data ---
ratings = pd.read_csv("rating.csv", encoding="ISO-8859-1")
movies = pd.read_csv("movies.csv", encoding="ISO-8859-1")

# --- Preprocessing ---
ratings["year"] = pd.to_datetime(ratings["timestamp"], unit="s").dt.year
movies["genres_list"] = movies["genres"].str.split("|")
df = pd.merge(ratings, movies, on="movieId", how="left")

# --- Sidebar ---
st.sidebar.title("Movie Explorer for Ages 18–35")
st.sidebar.markdown("Use the filters to explore genre trends and ratings.")

all_genres = sorted({genre for sublist in movies["genres_list"] for genre in sublist})
selected_genres = st.sidebar.multiselect("Select genres:", all_genres, default=["Comedy", "Action"])

# --- Filter by selected genres ---
df_filtered = df[df["genres_list"].apply(lambda x: any(g in x for g in selected_genres))]

# --- Title ---
st.title(" Interactive Movie Ratings Dashboard")
st.markdown("**Target Audience:** Ages 18–35  |  *Built for insights, engagement, and ML readiness*")

# --- Plot 1: Top Genres by Ratings Count ---
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
    labels={"genres_list": "Genre", "rating_count": "Number of Ratings", "avg_rating": "Avg Rating"},
    title="Most Rated Genres with Average Ratings"
)
st.plotly_chart(fig1, use_container_width=True)

# --- Plot 2: Histogram of Ratings for Selected Genres ---
fig2 = px.histogram(
    df_filtered, x="rating", nbins=20,
    title="Ratings Distribution for Selected Genres",
    color_discrete_sequence=["#636EFA"]
)
fig2.update_layout(bargap=0.2)
st.plotly_chart(fig2, use_container_width=True)

# --- Plot 3: Animated Genre Popularity Over Time ---
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

# --- Insights Footer ---
st.markdown("---")
st.markdown("### Why This Dataset Fits Machine Learning")
st.markdown("""
- **Genre Preferences:** Clear trends in how different genres are rated.
- **User Behavior:** Millions of ratings allow collaborative filtering and user segmentation.
- **Scalability:** Structure supports clustering, content filtering, and hybrid models.
- **Retail Relevance:** Useful for e-commerce/movie platforms to personalize recommendations.
""")


# In[ ]:





# In[ ]:





# ### Most Rated Genres

# #### Popularity of genres across all users.

# In[63]:


from collections import Counter

genre_counts = Counter([genre for sublist in df['genres_list'] for genre in sublist])
genre_df = pd.DataFrame(genre_counts.items(), columns=['Genre', 'Count']).sort_values('Count', ascending=False)

sns.barplot(data=genre_df.head(10), x='Count', y='Genre')
plt.title("Top 10 Most Rated Genres")
plt.xlabel("Number of Ratings")
plt.ylabel("Genre")
plt.show()


# ### Movie ratings by Gender average and count

# In[58]:


import plotly.express as px

# Genre summary
genre_summary = df_exploded.groupby('genre').agg(
    Average_Rating=('rating', 'mean'),
    Rating_Count=('rating', 'count')
).reset_index()

# Reshape for interactive plot
genre_summary_long = genre_summary.melt(
    id_vars='genre',
    value_vars=['Average_Rating', 'Rating_Count'],
    var_name='Metric',
    value_name='Value'
)

# Plot
fig = px.bar(
    genre_summary_long,
    x='genre',
    y='Value',
    color='Metric',
    barmode='group',
    facet_col='Metric',
    title='Summary of Movie Ratings by Genre',
    labels={'Value': 'Metric Value', 'genre': 'Genre'}
)
fig.show()


# In[ ]:


fig.write_html("interactive_genre_plot.html")


# In[ ]:





# In[ ]:





# ### Average Ratings per Genre

# #### Identify which genres users rate higher.

# In[ ]:


# Explode genres for per-genre aggregation
df_exploded = df.explode('genres_list')
avg_genre_rating = df_exploded.groupby('genres_list')['rating'].mean().sort_values(ascending=False).reset_index()

sns.barplot(data=avg_genre_rating, x='rating', y='genres_list')
plt.title("Average Rating per Genre")
plt.xlabel("Average Rating")
plt.ylabel("Genre")
plt.show()


# ### User Rating Behavior

# #### Total ratings per user (activity level)

# In[ ]:


user_activity = df.groupby('userId').size().reset_index(name='RatingCount')
sns.histplot(user_activity['RatingCount'], bins=30, kde=True)
plt.title("User Activity Distribution")
plt.xlabel("Number of Ratings")
plt.ylabel("Number of Users")
plt.show()


# ### Ratings Distribution
# 

# #### General understanding of how often users give high/low scores

# In[ ]:


sns.histplot(df['rating'], bins=10, kde=True)
plt.title("Ratings Distribution")
plt.xlabel("Rating")
plt.ylabel("Frequency")
plt.show()


# In[ ]:





# In[ ]:




