import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# 1. Load data - PROPER PATH HANDLING
try:
    # Method 1: Raw string (recommended)
    movies = pd.read_csv(r"C:\Users\user\Movie recommender system\dataset.csv")
    
    # OR Method 2: Double backslashes
    # movies = pd.read_csv("C:\\Users\\user\\Movie recommender system\\dataset.csv")
    
except Exception as e:
    print("Failed to load CSV:", str(e))
    exit()

# 2. Clean data - ESSENTIAL
print("\nData cleaning:")
print("Initial rows:", len(movies))
movies = movies.dropna(subset=['overview', 'genre'])  # Drop rows with missing text
movies = movies.fillna('')  # Fill other NAs with empty strings
print("Valid rows after cleaning:", len(movies))

# 3. Create combined features
movies['combined'] = movies['overview'].astype(str) + " " + movies['genre'].astype(str)

# 4. Save movies list
movies.to_pickle("movies_list.pkl")
print("\nSaved movies_list.pkl")

# 5. Generate similarity matrix
print("\nGenerating similarity matrix...")
tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
tfidf_matrix = tfidf.fit_transform(movies['combined'])
similarity = linear_kernel(tfidf_matrix, tfidf_matrix)

with open("similarity.pkl", 'wb') as f:
    pickle.dump(similarity, f)
print("Saved similarity.pkl")

print("\nDone! Files generated successfully.")
print("Now you can run your Streamlit app.")