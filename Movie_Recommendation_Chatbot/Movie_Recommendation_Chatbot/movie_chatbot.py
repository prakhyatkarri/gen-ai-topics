# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# DBTITLE 1,Movie Recommendation Chatbot
# MAGIC %md
# MAGIC # 🎬 Movie Recommendation Chatbot
# MAGIC
# MAGIC This notebook builds an intelligent movie recommendation chatbot using:
# MAGIC - **Free Dataset**: TMDb 5000 Movies dataset
# MAGIC - **Free Model**: Sentence-BERT embeddings (all-MiniLM-L6-v2)
# MAGIC - **Free Vector Store**: FAISS for similarity search
# MAGIC
# MAGIC ## Workflow
# MAGIC 1. **Setup**: Install dependencies and download dataset
# MAGIC 2. **Learning**: Create embeddings and build vector database
# MAGIC 3. **Interaction**: Get personalized movie recommendations
# MAGIC 4. **Destroy**: Clean up resources
# MAGIC
# MAGIC All resources are free and run locally!

# COMMAND ----------

# DBTITLE 1,📦 Setup - Install Dependencies
# Install required libraries
%pip install sentence-transformers faiss-cpu pandas numpy scikit-learn --quiet

print("✅ Dependencies installed successfully!")

# COMMAND ----------

# DBTITLE 1,📥 Setup - Download Movies Dataset
import pandas as pd
import requests
import json
import os

# Create data directory
data_dir = "/tmp/movie_data"
os.makedirs(data_dir, exist_ok=True)

# Download TMDb movies dataset (free, no API key needed)
url = "https://raw.githubusercontent.com/RamiKrispin/vscode-python/main/data/movies.csv"

print("Downloading movies dataset...")
response = requests.get(url)

if response.status_code == 200:
    dataset_path = f"{data_dir}/movies.csv"
    with open(dataset_path, 'wb') as f:
        f.write(response.content)
    print(f"✅ Dataset downloaded to {dataset_path}")
else:
    # Fallback: Create a sample dataset
    print("Using sample dataset...")
    movies_data = {
        'title': [
            'The Shawshank Redemption', 'The Godfather', 'The Dark Knight', 'Pulp Fiction',
            'Forrest Gump', 'Inception', 'The Matrix', 'Goodfellas', 'The Silence of the Lambs',
            'Interstellar', 'The Prestige', 'The Departed', 'Gladiator', 'The Lion King',
            'Saving Private Ryan', 'The Green Mile', 'Spirited Away', 'The Pianist'
        ],
        'overview': [
            'Two imprisoned men bond over a number of years, finding solace and eventual redemption through acts of common decency.',
            'The aging patriarch of an organized crime dynasty transfers control of his clandestine empire to his reluctant son.',
            'When the menace known as the Joker wreaks havoc and chaos on the people of Gotham, Batman must accept one of the greatest psychological and physical tests.',
            'The lives of two mob hitmen, a boxer, a gangster and his wife intertwine in four tales of violence and redemption.',
            'The presidencies of Kennedy and Johnson, the Vietnam War, and other historical events unfold from the perspective of an Alabama man.',
            'A thief who steals corporate secrets through dream-sharing technology is given the inverse task of planting an idea into the mind of a CEO.',
            'A computer hacker learns from mysterious rebels about the true nature of his reality and his role in the war against its controllers.',
            'The story of Henry Hill and his life in the mob, covering his relationship with his wife and his partners in crime.',
            'A young FBI cadet must receive the help of an incarcerated cannibal killer to catch another serial killer.',
            'A team of explorers travel through a wormhole in space in an attempt to ensure humanity\'s survival.',
            'Two stage magicians engage in competitive one-upmanship in an attempt to create the ultimate stage illusion.',
            'An undercover cop and a mole in the police attempt to identify each other while infiltrating an Irish gang in Boston.',
            'A former Roman General sets out to exact vengeance against the corrupt emperor who murdered his family and sent him into slavery.',
            'Lion prince Simba and his father are targeted by his bitter uncle, who wants to ascend the throne himself.',
            'Following the Normandy Landings, a group of U.S. soldiers go behind enemy lines to retrieve a paratrooper.',
            'The lives of guards on Death Row are affected by one of their charges: a black man accused of child murder who has a mysterious gift.',
            'During her family\'s move to the suburbs, a sullen 10-year-old girl wanders into a world ruled by gods, witches, and spirits.',
            'A Polish Jewish musician struggles to survive the destruction of the Warsaw ghetto of World War II.'
        ],
        'genres': [
            'Drama, Crime', 'Drama, Crime', 'Action, Crime, Drama', 'Crime, Thriller',
            'Comedy, Drama, Romance', 'Action, Sci-Fi, Thriller', 'Action, Sci-Fi', 'Crime, Drama',
            'Crime, Drama, Thriller', 'Adventure, Drama, Sci-Fi', 'Drama, Mystery, Sci-Fi', 'Crime, Drama, Thriller',
            'Action, Adventure, Drama', 'Animation, Family, Drama', 'Drama, War', 'Crime, Drama, Fantasy',
            'Animation, Family, Fantasy', 'Drama, War'
        ]
    }
    df = pd.DataFrame(movies_data)
    dataset_path = f"{data_dir}/movies.csv"
    df.to_csv(dataset_path, index=False)
    print(f"✅ Sample dataset created at {dataset_path}")

# COMMAND ----------

# DBTITLE 1,🔍 Setup - Load and Explore Dataset
# Load the dataset
df = pd.read_csv(f"{data_dir}/movies.csv")

print(f"📊 Dataset loaded: {len(df)} movies")
print(f"\nColumns: {list(df.columns)}")
print(f"\n{df.head()}")

# Clean data - combine text fields for embeddings
if 'overview' not in df.columns:
    print("⚠️ 'overview' column not found. Using available text columns.")
    text_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]
else:
    text_col = 'overview'

# Combine title, genres (if available), and overview for better embeddings
if 'genres' in df.columns:
    df['combined_text'] = df['title'].astype(str) + " - " + df['genres'].astype(str) + ". " + df[text_col].fillna('')
else:
    df['combined_text'] = df['title'].astype(str) + ". " + df[text_col].fillna('')

print(f"\n✅ Data prepared. Sample combined text:")
print(df['combined_text'].iloc[0][:200] + "...")

# COMMAND ----------

# DBTITLE 1,🧠 Learning - Load Sentence Transformer Model
from sentence_transformers import SentenceTransformer
import numpy as np

print("Loading sentence transformer model (this may take a minute on first run)...")

# Load free, lightweight model - all-MiniLM-L6-v2
# This model is only 80MB and runs on CPU efficiently
model = SentenceTransformer('all-MiniLM-L6-v2')

print("✅ Model loaded successfully!")
print(f"Model embedding dimension: {model.get_sentence_embedding_dimension()}")

# COMMAND ----------

# DBTITLE 1,🎯 Learning - Create Movie Embeddings
print("Creating embeddings for all movies...")
print("This may take 1-2 minutes depending on dataset size...\n")

# Create embeddings for all movie descriptions
movie_texts = df['combined_text'].tolist()
embeddings = model.encode(movie_texts, show_progress_bar=True, batch_size=32)

print(f"\n✅ Created embeddings with shape: {embeddings.shape}")
print(f"Each movie is represented as a {embeddings.shape[1]}-dimensional vector")

# COMMAND ----------

# DBTITLE 1,🗄️ Learning - Build FAISS Vector Store
import faiss

print("Building FAISS index for fast similarity search...")

# Normalize embeddings for cosine similarity
faiss.normalize_L2(embeddings)

# Create FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)  # Inner Product (cosine similarity after normalization)
index.add(embeddings)

print(f"✅ FAISS index built with {index.ntotal} movie vectors")
print("Ready for lightning-fast similarity search!")

# COMMAND ----------

# DBTITLE 1,💬 Chatbot Function - Get Movie Recommendations
def get_movie_recommendations(user_query, top_k=5):
    """
    Get movie recommendations based on user query.
    
    Args:
        user_query: Natural language description of what user wants
        top_k: Number of recommendations to return
    
    Returns:
        List of recommended movies with details
    """
    print(f"\n🔍 Searching for: '{user_query}'\n")
    
    # Create embedding for user query
    query_embedding = model.encode([user_query])
    faiss.normalize_L2(query_embedding)
    
    # Search for similar movies
    distances, indices = index.search(query_embedding, top_k)
    
    # Format results
    recommendations = []
    print(f"🎬 Top {top_k} Recommendations:\n")
    print("=" * 80)
    
    for rank, (idx, score) in enumerate(zip(indices[0], distances[0]), 1):
        movie = df.iloc[idx]
        print(f"\n{rank}. {movie['title']} (Similarity: {score:.2%})")
        
        if 'genres' in df.columns:
            print(f"   Genres: {movie['genres']}")
        
        overview_col = 'overview' if 'overview' in df.columns else df.columns[1]
        if overview_col in movie and pd.notna(movie[overview_col]):
            overview_text = str(movie[overview_col])[:200]
            print(f"   {overview_text}...")
        
        recommendations.append({
            'rank': rank,
            'title': movie['title'],
            'similarity': float(score),
            'overview': movie.get(overview_col, '')
        })
    
    print("\n" + "=" * 80)
    return recommendations

print("✅ Chatbot function ready!")

# COMMAND ----------

# DBTITLE 1,🎮 Interactive - Try the Chatbot!
# Example queries - modify these or add your own!

print("🎬 MOVIE RECOMMENDATION CHATBOT\n")
print("Try these example queries or create your own:\n")

# Example 1: By theme
recommendations = get_movie_recommendations(
    "I want an emotional drama about friendship and hope",
    top_k=3
)

print("\n" + "="*80 + "\n")

# Example 2: By genre and mood
recommendations = get_movie_recommendations(
    "Dark psychological thriller with mind-bending plot twists",
    top_k=3
)

print("\n" + "="*80 + "\n")

# Example 3: By setting
recommendations = get_movie_recommendations(
    "Science fiction adventure in space with stunning visuals",
    top_k=3
)

# COMMAND ----------

# DBTITLE 1,💭 Interactive - Custom Query (Modify This Cell)
# 🎯 YOUR TURN! Modify the query below to get personalized recommendations

# my_query = "Action movie with revenge theme and great cinematography"
my_query = "Superhero action movie without great music and cinematography"

recommendations = get_movie_recommendations(my_query, top_k=5)

# The chatbot understands natural language!
# Try queries like:
# - "Feel-good family movie with animals"
# - "Intense crime thriller set in the underworld"
# - "Historical war drama with emotional depth"
# - "Mind-bending science fiction mystery"

# COMMAND ----------

# DBTITLE 1,🧹 Destroy - Cleanup Resources
import shutil

print("Cleaning up resources...\n")

# Remove temporary data directory
if os.path.exists(data_dir):
    shutil.rmtree(data_dir)
    print(f"✅ Removed temporary data directory: {data_dir}")

# Clear variables from memory
if 'model' in globals():
    del model
    print("✅ Cleared model from memory")

if 'embeddings' in globals():
    del embeddings
    print("✅ Cleared embeddings from memory")

if 'index' in globals():
    del index
    print("✅ Cleared FAISS index from memory")

if 'df' in globals():
    del df
    print("✅ Cleared dataframe from memory")

print("\n🎉 All resources cleaned up successfully!")
print("\nNote: The sentence-transformers model cache (~80MB) remains in ~/.cache/")
print("This is intentional for faster reloading. To fully remove, delete ~/.cache/torch/sentence_transformers/")

# COMMAND ----------

# DBTITLE 1,📝 Summary and Next Steps
# MAGIC %md
# MAGIC ## 🎉 Congratulations!
# MAGIC
# MAGIC You've successfully built a movie recommendation chatbot using:
# MAGIC - ✅ **Free Model**: Sentence-BERT (all-MiniLM-L6-v2) - 80MB, CPU-efficient
# MAGIC - ✅ **Free Dataset**: Movies with descriptions
# MAGIC - ✅ **Free Vector Search**: FAISS for similarity matching
# MAGIC - ✅ **No API Keys**: Everything runs locally
# MAGIC
# MAGIC ## 🚀 Next Steps
# MAGIC
# MAGIC 1. **Expand Dataset**: Add more movies from larger free datasets (Kaggle, MovieLens)
# MAGIC 2. **Enhanced Features**: 
# MAGIC    - Add genre filtering
# MAGIC    - Include ratings and release year
# MAGIC    - Add multilingual support
# MAGIC 3. **Better UI**: Create a widget-based interface using ipywidgets
# MAGIC 4. **Hybrid Search**: Combine semantic search with keyword filtering
# MAGIC 5. **Deploy**: Turn this into a web app using Gradio or Streamlit
# MAGIC
# MAGIC ## 💡 How It Works
# MAGIC
# MAGIC 1. **Embeddings**: Each movie description is converted to a 384-dimensional vector
# MAGIC 2. **Similarity Search**: Your query is also converted to a vector
# MAGIC 3. **Matching**: FAISS finds movies with the most similar vectors (semantic meaning)
# MAGIC 4. **Results**: Top matches are returned with similarity scores
# MAGIC
# MAGIC The model understands context and meaning, not just keywords!

# COMMAND ----------

