
# Movie Recommendation System using NLP and Streamlit

# 1. Import Libraries
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# 2. Load and Preprocess Data
movies = pd.read_csv("movies_2024.csv")

# Clean data
movies.columns = ['title', 'description']
movies['title'] = movies['title'].str.extract(r'\d+\.\s*(.*)')  # Remove numbering
movies.dropna(subset=['description'], inplace=True)

# 3. NLP Vectorization
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['description'])

cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()

# 4. Recommendation Function
def recommend(title, cosine_sim=cosine_sim):
    idx = indices.get(title)
    if idx is None:
        return ["Movie not found."]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]
    movie_indices = [i[0] for i in sim_scores]
    return movies['title'].iloc[movie_indices].tolist()

# 5. Streamlit UI
st.title("Movie Recommendation System")
movie_name = st.text_input("Enter a movie title:")

if st.button("Recommend"):
    recommendations = recommend(movie_name)
    st.subheader("Recommended Movies:")
    for rec in recommendations:
        st.write(rec)
