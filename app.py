import streamlit as st
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Configuration
st.set_page_config(layout="wide")
st.title("🎬 Movie Recommender System")

# Load data
@st.cache_data
def load_data():
    try:
        movies = pd.read_csv(r"C:\Users\user\Movie recommender system\dataset.csv")
        movies = movies.dropna(subset=['title', 'overview', 'genre'])
        
        # Create better features for recommendations
        movies['clean_title'] = movies['title'].str.lower().str.replace('[^\w\s]', '', regex=True)
        movies['combined_features'] = movies['overview'] + ' ' + movies['genre'].str.replace(',', ' ')
        
        # Build recommendation model
        tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
        tfidf_matrix = tfidf.fit_transform(movies['combined_features'])
        similarity = linear_kernel(tfidf_matrix, tfidf_matrix)
        
        return movies, similarity
    except Exception as e:
        st.error(f"Data loading failed: {str(e)}")
        return None, None

movies, similarity = load_data()

if movies is not None:
    # Movie selection
    selected_movie = st.selectbox("Select a movie", movies['title'].unique())
    
    if st.button("Recommend"):
        try:
            # Find closest matching title
            input_movie = selected_movie.lower().replace('[^\w\s]', '')
            matches = movies[movies['clean_title'] == input_movie]
            
            if len(matches) == 0:
                st.warning("No exact match found. Showing similar titles:")
                matches = movies[movies['clean_title'].str.contains(input_movie)]
            
            if len(matches) > 0:
                idx = matches.index[0]
                sim_scores = sorted(enumerate(similarity[idx]), key=lambda x: x[1], reverse=True)[1:6]
                
                st.success(f"Movies similar to {selected_movie}:")
                cols = st.columns(5)
                
                for i, (movie_idx, score) in enumerate(sim_scores):
                    movie = movies.iloc[movie_idx]
                    with cols[i]:
                        st.write(f"🎬 {movie['title']}")
                        st.write(f"⭐ {movie.get('vote_average', 'N/A')}")
                        st.write(f"Genre: {movie['genre']}")
                        st.write(f"Match: {score:.1%}")
            else:
                st.warning("No similar movies found. Try another title.")
                
        except Exception as e:
            st.error(f"Error: {str(e)}")

# Add some debug info
# with st.expander("Debug Info"):
#     st.write(f"Total movies: {len(movies) if movies is not None else 0}")
#     if movies is not None:
#         st.write("Sample data:", movies[['title', 'genre']].head(3))