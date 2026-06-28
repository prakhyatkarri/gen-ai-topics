# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# DBTITLE 1,Notebook Introduction
# MAGIC %md
# MAGIC # Semantic Search Learning with Wikipedia Articles
# MAGIC
# MAGIC This notebook demonstrates semantic search fundamentals using 100 Wikipedia articles. You'll learn:
# MAGIC
# MAGIC * How to generate text embeddings using sentence-transformers
# MAGIC * How to store embeddings in Delta tables
# MAGIC * How to implement cosine similarity search
# MAGIC * The difference between semantic and keyword search
# MAGIC
# MAGIC **All examples run on free serverless compute (CPU-based)**

# COMMAND ----------

# DBTITLE 1,Setup Section
# MAGIC %md
# MAGIC ## Setup Section
# MAGIC Installing dependencies and preparing the dataset

# COMMAND ----------

# DBTITLE 1,Install sentence-transformers
# Install sentence-transformers library (free, open-source)
# This provides pre-trained models for generating embeddings
%pip install sentence-transformers wikipedia-api --quiet

# COMMAND ----------

# DBTITLE 1,Restart Python environment
# Restart Python to load newly installed packages
dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Import libraries
# Import required libraries
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
from pyspark.sql.functions import col
from pyspark.sql.types import ArrayType, FloatType, StructType, StructField, StringType, IntegerType
import wikipediaapi
import time

# COMMAND ----------

# DBTITLE 1,Create sample Wikipedia dataset
# Create a sample dataset of 100 Wikipedia articles
# Using popular topics across different categories

wiki_topics = [
    # Science & Technology (20 articles)
    "Artificial intelligence", "Machine learning", "Quantum computing", "Blockchain", "Python (programming language)",
    "Neural network", "Deep learning", "Data science", "Cloud computing", "Cybersecurity",
    "Internet", "Smartphone", "Electric vehicle", "Solar energy", "Climate change",
    "DNA", "Vaccine", "Antibiotic", "Astronomy", "Black hole",
    
    # History (20 articles)
    "World War II", "Renaissance", "Industrial Revolution", "Ancient Egypt", "Roman Empire",
    "Napoleon", "Abraham Lincoln", "Albert Einstein", "Leonardo da Vinci", "Isaac Newton",
    "American Revolution", "French Revolution", "Cold War", "Great Depression", "Age of Enlightenment",
    "Maya civilization", "Silk Road", "Vikings", "Ottoman Empire", "Mongol Empire",
    
    # Geography (20 articles)
    "Mount Everest", "Amazon rainforest", "Sahara", "Pacific Ocean", "Antarctica",
    "Great Barrier Reef", "Nile", "Himalayas", "Grand Canyon", "Yellowstone National Park",
    "New York City", "Tokyo", "Paris", "London", "Sydney",
    "Africa", "Asia", "Europe", "North America", "South America",
    
    # Arts & Culture (20 articles)
    "Music", "Painting", "Sculpture", "Literature", "Poetry",
    "Jazz", "Rock music", "Classical music", "Hip hop", "Opera",
    "Shakespeare", "Beethoven", "Mozart", "Van Gogh", "Picasso",
    "Cinema", "Photography", "Dance", "Theater", "Architecture",
    
    # Sports & Entertainment (20 articles)
    "Football", "Basketball", "Tennis", "Cricket", "Olympic Games",
    "FIFA World Cup", "Swimming", "Athletics", "Golf", "Baseball",
    "Michael Jordan", "Muhammad Ali", "Serena Williams", "Lionel Messi", "Usain Bolt",
    "Netflix", "YouTube", "Video game", "Chess", "Poker"
]

print(f"Prepared {len(wiki_topics)} Wikipedia topics for embedding generation")

# COMMAND ----------

# DBTITLE 1,Fetch Wikipedia content
# Fetch article summaries from Wikipedia
# Using summaries instead of full text for faster processing

wiki_wiki = wikipediaapi.Wikipedia(
    user_agent='SemanticSearchLearning/1.0',
    language='en'
)

articles_data = []

for idx, topic in enumerate(wiki_topics):
    try:
        page = wiki_wiki.page(topic)
        if page.exists():
            # Get first 500 characters of the summary for manageable embedding size
            summary = page.summary[:500] if len(page.summary) > 500 else page.summary
            articles_data.append({
                'article_id': idx + 1,
                'title': page.title,
                'text': summary
            })
        time.sleep(0.1)  # Be respectful to Wikipedia API
    except Exception as e:
        print(f"Error fetching {topic}: {e}")
        continue

# Create pandas DataFrame
df_pandas = pd.DataFrame(articles_data)
print(f"Successfully fetched {len(df_pandas)} Wikipedia articles")
print(f"\nSample article:")
display(df_pandas.head(3))

# COMMAND ----------

# DBTITLE 1,Setup Delta table location
# Define table name for Unity Catalog (free, managed storage)
# Unity Catalog provides ACID guarantees and works on serverless compute

table_name = "wikipedia_embeddings"
print(f"Delta table will be created as: {table_name}")

# COMMAND ----------

# DBTITLE 1,Learning Section Header
# MAGIC %md
# MAGIC ## Learning Section: Generate Embeddings & Implement Search
# MAGIC
# MAGIC ### What are Embeddings?
# MAGIC
# MAGIC **Embeddings** are numerical representations of text that capture semantic meaning. Each piece of text is converted into a vector (list of numbers) where:
# MAGIC - Similar meanings have similar vectors
# MAGIC - Vector distance measures semantic similarity
# MAGIC - Typical embeddings have 384-768 dimensions
# MAGIC
# MAGIC **Why are they useful?**
# MAGIC - Enable semantic search (meaning-based, not just keyword matching)
# MAGIC - Power recommendation systems
# MAGIC - Enable clustering and classification
# MAGIC - Work across languages (with multilingual models)

# COMMAND ----------

# DBTITLE 1,Load embedding model
# Load a pre-trained sentence transformer model
# Model: 'all-MiniLM-L6-v2' - small (80MB), fast, produces 384-dimensional embeddings
# Perfect for learning and CPU-based compute

print("Loading embedding model...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print(f"Model loaded! Embedding dimension: {model.get_sentence_embedding_dimension()}")

# COMMAND ----------

# DBTITLE 1,Generate embeddings
# Generate embeddings for all articles
# This converts each article's text into a 384-dimensional vector

print("Generating embeddings for all articles...")
texts = df_pandas['text'].tolist()
embeddings = model.encode(texts, show_progress_bar=True)

# Add embeddings to the DataFrame
df_pandas['embedding'] = embeddings.tolist()

print(f"\nGenerated {len(embeddings)} embeddings")
print(f"Each embedding has {len(embeddings[0])} dimensions")

# COMMAND ----------

# DBTITLE 1,Show example embedding
# Let's examine an example embedding vector
example_article = df_pandas.iloc[0]

print(f"Article Title: {example_article['title']}")
print(f"\nArticle Text (first 200 chars): {example_article['text'][:200]}...")
print(f"\nEmbedding Vector (first 10 dimensions): {example_article['embedding'][:10]}")
print(f"\nEmbedding shape: {len(example_article['embedding'])} dimensions")
print(f"\nEmbedding statistics:")
print(f"  - Min value: {min(example_article['embedding']):.4f}")
print(f"  - Max value: {max(example_article['embedding']):.4f}")
print(f"  - Mean value: {np.mean(example_article['embedding']):.4f}")

# COMMAND ----------

# DBTITLE 1,Delta Table Section
# MAGIC %md
# MAGIC ### Save Embeddings to Delta Table
# MAGIC
# MAGIC Delta Lake provides ACID transactions, time travel, and efficient storage for our embeddings.

# COMMAND ----------

# DBTITLE 1,Convert to Spark DataFrame and save
# Convert pandas DataFrame to Spark DataFrame
# Define schema explicitly to ensure correct types

schema = StructType([
    StructField("article_id", IntegerType(), False),
    StructField("title", StringType(), False),
    StructField("text", StringType(), False),
    StructField("embedding", ArrayType(FloatType()), False)
])

df_spark = spark.createDataFrame(df_pandas, schema=schema)

# Write to Delta table as a managed Unity Catalog table
# Using workspace.default catalog (free, accessible on serverless)
table_name = "wikipedia_embeddings"
print(f"Writing to Delta table: {table_name}...")
df_spark.write.format("delta").mode("overwrite").saveAsTable(table_name)
print(f"✓ Successfully saved {df_spark.count()} articles to Delta table")

# COMMAND ----------

# DBTITLE 1,Create Delta table reference
# Table is already created as a managed table
# No additional setup needed
print(f"✓ Table '{table_name}' is ready for querying")

# COMMAND ----------

# DBTITLE 1,Query and verify data
# Query the Delta table to verify the data
print("Querying Delta table...\n")

df_loaded = spark.table(table_name)

# Show sample records (without embedding column for readability)
print("Sample articles:")
display(df_loaded.select("article_id", "title", "text").limit(5))

# Show table statistics
print(f"\nTable Statistics:")
print(f"  - Total articles: {df_loaded.count()}")
print(f"  - Columns: {len(df_loaded.columns)}")
print(f"  - Schema:")
df_loaded.printSchema()

# COMMAND ----------

# DBTITLE 1,Semantic Search Section
# MAGIC %md
# MAGIC ### Implement Semantic Search
# MAGIC
# MAGIC **Cosine Similarity** measures the similarity between two vectors based on the angle between them:
# MAGIC - Score of 1.0 = identical direction (most similar)
# MAGIC - Score of 0.0 = perpendicular (unrelated)
# MAGIC - Score of -1.0 = opposite direction (dissimilar)
# MAGIC
# MAGIC We'll use this to find articles most semantically similar to a search query.

# COMMAND ----------

# DBTITLE 1,Create cosine similarity function
# Define cosine similarity function
def cosine_similarity(vec1, vec2):
    """
    Calculate cosine similarity between two vectors.
    Returns a score between -1 and 1 (higher = more similar)
    """
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    return dot_product / (norm1 * norm2)

# Test the function
test_vec1 = [1, 0, 0]
test_vec2 = [1, 0, 0]
test_vec3 = [0, 1, 0]

print(f"Cosine similarity between identical vectors: {cosine_similarity(test_vec1, test_vec2):.4f}")
print(f"Cosine similarity between perpendicular vectors: {cosine_similarity(test_vec1, test_vec3):.4f}")

# COMMAND ----------

# DBTITLE 1,Create semantic search function
# Create a semantic search function
def semantic_search(query, top_k=5):
    """
    Perform semantic search on the Wikipedia articles.
    
    Args:
        query: Search query string
        top_k: Number of top results to return
    
    Returns:
        DataFrame with top K most similar articles
    """
    # Generate embedding for the query
    query_embedding = model.encode(query)
    
    # Load all articles with embeddings
    df_articles = spark.table(table_name).toPandas()
    
    # Calculate similarity scores
    similarities = []
    for idx, row in df_articles.iterrows():
        article_embedding = row['embedding']
        similarity = cosine_similarity(query_embedding, article_embedding)
        similarities.append({
            'article_id': row['article_id'],
            'title': row['title'],
            'text': row['text'][:200] + '...',  # Truncate for display
            'similarity_score': similarity
        })
    
    # Convert to DataFrame and sort by similarity
    results_df = pd.DataFrame(similarities)
    results_df = results_df.sort_values('similarity_score', ascending=False).head(top_k)
    results_df = results_df.reset_index(drop=True)
    results_df.index = results_df.index + 1  # Start index at 1
    
    return results_df

print("✓ Semantic search function created")

# COMMAND ----------

# DBTITLE 1,Example search: Space and astronomy
# Example 1: Search for space-related articles
print("="*80)
print("SEMANTIC SEARCH EXAMPLE 1: Space and astronomy")
print("="*80)
query1 = "space exploration and celestial objects"
print(f"Query: '{query1}'\n")
results1 = semantic_search(query1, top_k=5)
display(results1)

# COMMAND ----------

# DBTITLE 1,Example search: AI and technology
# Example 2: Search for technology articles
print("="*80)
print("SEMANTIC SEARCH EXAMPLE 2: Artificial intelligence and computing")
print("="*80)
query2 = "artificial intelligence and computer algorithms"
print(f"Query: '{query2}'\n")
results2 = semantic_search(query2, top_k=5)
display(results2)

# COMMAND ----------

# DBTITLE 1,Example search: Historical figures
# Example 3: Search for historical figures
print("="*80)
print("SEMANTIC SEARCH EXAMPLE 3: Historical leaders and inventors")
print("="*80)
query3 = "famous leaders and inventors from history"
print(f"Query: '{query3}'\n")
results3 = semantic_search(query3, top_k=5)
display(results3)

# COMMAND ----------

# DBTITLE 1,Semantic vs Keyword Search
# MAGIC %md
# MAGIC ### Semantic Search vs Keyword Search
# MAGIC
# MAGIC Let's compare semantic search with traditional keyword matching to see the difference.

# COMMAND ----------

# DBTITLE 1,Compare semantic vs keyword search
# Keyword search function (simple text matching)
def keyword_search(query, top_k=5):
    """
    Simple keyword-based search (for comparison)
    """
    df_articles = spark.table(table_name).toPandas()
    query_lower = query.lower()
    
    matches = []
    for idx, row in df_articles.iterrows():
        text_lower = (row['title'] + ' ' + row['text']).lower()
        # Count keyword matches
        match_count = sum(word in text_lower for word in query_lower.split())
        if match_count > 0:
            matches.append({
                'article_id': row['article_id'],
                'title': row['title'],
                'text': row['text'][:200] + '...',
                'keyword_matches': match_count
            })
    
    results_df = pd.DataFrame(matches)
    if len(results_df) > 0:
        results_df = results_df.sort_values('keyword_matches', ascending=False).head(top_k)
        results_df = results_df.reset_index(drop=True)
        results_df.index = results_df.index + 1
    
    return results_df

# Compare searches
test_query = "intelligent machines and automated systems"

print("="*80)
print(f"COMPARISON: '{test_query}'")
print("="*80)

print("\nSemantic Search Results (meaning-based):")
semantic_results = semantic_search(test_query, top_k=5)
display(semantic_results[['title', 'similarity_score']])

print("\nKeyword Search Results (exact word matching):")
keyword_results = keyword_search(test_query, top_k=5)
if len(keyword_results) > 0:
    display(keyword_results[['title', 'keyword_matches']])
else:
    print("No exact keyword matches found!")

print("\nInsight: Semantic search finds relevant articles even without exact keyword matches!")

# COMMAND ----------

# DBTITLE 1,Visualization Section
# MAGIC %md
# MAGIC ### Visualize Search Results
# MAGIC
# MAGIC Let's visualize the similarity scores to understand the search quality.

# COMMAND ----------

# DBTITLE 1,Visualize similarity scores
import matplotlib.pyplot as plt

# Perform a search for visualization
vis_query = "music and musical instruments"
vis_results = semantic_search(vis_query, top_k=10)

# Create bar chart
fig, ax = plt.subplots(figsize=(12, 6))
colors = plt.cm.viridis(vis_results['similarity_score'] / vis_results['similarity_score'].max())

bars = ax.barh(range(len(vis_results)), vis_results['similarity_score'], color=colors)
ax.set_yticks(range(len(vis_results)))
ax.set_yticklabels(vis_results['title'])
ax.set_xlabel('Cosine Similarity Score', fontsize=12)
ax.set_title(f'Top 10 Results for: "{vis_query}"', fontsize=14, fontweight='bold')
ax.set_xlim(0, 1.0)

# Add score labels on bars
for i, (bar, score) in enumerate(zip(bars, vis_results['similarity_score'])):
    ax.text(score + 0.01, i, f'{score:.3f}', va='center', fontsize=9)

plt.tight_layout()
plt.grid(axis='x', alpha=0.3)
display(plt.show())

print(f"\nVisualization shows similarity scores ranging from {vis_results['similarity_score'].min():.3f} to {vis_results['similarity_score'].max():.3f}")

# COMMAND ----------

# DBTITLE 1,Cleanup Section Header
# MAGIC %md
# MAGIC ## Cleanup Section
# MAGIC
# MAGIC Time to clean up all resources used in this demo.

# COMMAND ----------

# DBTITLE 1,Drop Delta table
# Drop the Delta table
print("Dropping Delta table...")
spark.sql(f"DROP TABLE IF EXISTS {table_name}")
print(f"✓ Table '{table_name}' dropped")

# COMMAND ----------

# DBTITLE 1,Clean up temporary files
# No temporary files to clean up
# The table is managed by Unity Catalog
print("✓ No additional cleanup needed (managed table)")

# COMMAND ----------

# DBTITLE 1,Completion message
# MAGIC %md
# MAGIC ## Notebook Complete!
# MAGIC
# MAGIC ### What You Learned:
# MAGIC
# MAGIC 1. **Embeddings**: How to convert text into numerical vectors that capture semantic meaning
# MAGIC 2. **Sentence Transformers**: Using pre-trained models for efficient embedding generation
# MAGIC 3. **Delta Tables**: Storing embeddings with ACID guarantees in Delta Lake format
# MAGIC 4. **Semantic Search**: Implementing cosine similarity to find semantically similar content
# MAGIC 5. **Comparison**: Understanding the advantage of semantic search over keyword matching
# MAGIC
# MAGIC ### Key Takeaways:
# MAGIC
# MAGIC * Semantic search finds relevant content based on **meaning**, not just exact words
# MAGIC * Embeddings enable powerful AI applications: search, recommendations, clustering
# MAGIC * Open-source models like sentence-transformers make this accessible and free
# MAGIC * Delta Lake provides reliable storage for embedding data
# MAGIC
# MAGIC ### Next Steps:
# MAGIC
# MAGIC * Try different embedding models (e.g., 'all-mpnet-base-v2' for higher quality)
# MAGIC * Experiment with different search queries
# MAGIC * Scale to larger datasets (thousands or millions of documents)
# MAGIC * Explore Databricks Vector Search for production-scale semantic search
# MAGIC * Add filters and hybrid search (combining semantic + keyword)
# MAGIC
# MAGIC **All resources have been cleaned up!**