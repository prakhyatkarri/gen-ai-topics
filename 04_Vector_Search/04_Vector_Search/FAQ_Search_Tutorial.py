# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# DBTITLE 1,FAQ Search with Vector Embeddings - Tutorial
# MAGIC %md
# MAGIC # FAQ Search with Vector Embeddings 🔍
# MAGIC
# MAGIC This notebook teaches you how to build a **semantic search system** using Databricks documentation as a knowledge base.
# MAGIC
# MAGIC ## What You'll Learn
# MAGIC - Load and process documentation as a knowledge base
# MAGIC - Create vector embeddings using free models
# MAGIC - Build a semantic FAQ search system
# MAGIC - Query and retrieve top-K relevant documents
# MAGIC
# MAGIC ## Architecture
# MAGIC ```
# MAGIC User Question → Embedding → Vector Search → Top 5 Results
# MAGIC ```
# MAGIC
# MAGIC ## Free Resources Used
# MAGIC - ✅ Sentence-Transformers (all-MiniLM-L6-v2)
# MAGIC - ✅ NumPy for vector operations
# MAGIC - ✅ Public Databricks documentation
# MAGIC
# MAGIC ---

# COMMAND ----------

# DBTITLE 1,📦 SETUP: Install Required Libraries
# Install sentence-transformers for embeddings (free, open-source)
%pip install --upgrade sentence-transformers
%pip install beautifulsoup4 requests --quiet

dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,📚 SETUP: Import Dependencies
import numpy as np
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import pickle
import os
from typing import List, Tuple
import warnings
warnings.filterwarnings('ignore')

print("✅ All libraries imported successfully!")

# COMMAND ----------

# DBTITLE 1,📖 SETUP: Sample Databricks Documentation
# Sample Databricks FAQ documentation
# In production, you would scrape docs.databricks.com or use Databricks documentation API

databricks_docs = [
    {
        "question": "How do I create a Unity Catalog?",
        "answer": "To create a Unity Catalog: 1) Navigate to the Data icon in the sidebar, 2) Click 'Create Catalog', 3) Enter a catalog name, 4) Optionally add a comment, 5) Click 'Create'. You need account admin or metastore admin privileges. Unity Catalog is a unified governance solution for data and AI assets."
    },
    {
        "question": "What is Delta Lake?",
        "answer": "Delta Lake is an open-source storage framework that brings ACID transactions to Apache Spark and big data workloads. It provides scalable metadata handling, time travel, schema enforcement, and unifies streaming and batch data processing. Delta Lake runs on top of your existing data lake."
    },
    {
        "question": "How do I create a cluster?",
        "answer": "To create a cluster: 1) Click 'Compute' in the sidebar, 2) Click 'Create Compute', 3) Specify cluster name, 4) Choose Databricks runtime version, 5) Configure node types and autoscaling, 6) Set advanced options if needed, 7) Click 'Create Compute'. Clusters can be single-node or multi-node."
    },
    {
        "question": "What is a notebook in Databricks?",
        "answer": "A Databricks notebook is a web-based interface for creating and running code, queries, and narrative text. Notebooks support multiple languages (Python, SQL, Scala, R) in the same notebook. They provide collaboration features, visualization tools, and integrate with Databricks workflows and jobs."
    },
    {
        "question": "How do I schedule a job?",
        "answer": "To schedule a job: 1) Go to 'Workflows' in the sidebar, 2) Click 'Create Job', 3) Add tasks (notebook, JAR, Python script), 4) Configure task parameters, 5) Set up a schedule using cron syntax or UI, 6) Configure cluster settings, 7) Add alerts and notifications, 8) Click 'Create'. Jobs can run on a schedule or be triggered manually."
    },
    {
        "question": "What is MLflow?",
        "answer": "MLflow is an open-source platform for managing the ML lifecycle, including experimentation, reproducibility, and deployment. It provides tracking for parameters, metrics, and artifacts, a model registry for versioning, and model serving capabilities. MLflow is pre-integrated with Databricks."
    },
    {
        "question": "How do I create a table?",
        "answer": "To create a table in Databricks: Use SQL CREATE TABLE statement or DataFrame.write methods. For Unity Catalog: CREATE TABLE catalog.schema.table_name. You can create managed tables (Databricks manages data) or external tables (you manage data location). Supports various formats: Delta, Parquet, CSV, JSON."
    },
    {
        "question": "What are Databricks Workflows?",
        "answer": "Databricks Workflows (formerly Jobs) is a fully managed orchestration service for data and ML pipelines. It allows you to schedule notebooks, Python scripts, JARs, and Delta Live Tables. Features include task dependencies, error handling, notifications, and integration with Git for CI/CD."
    },
    {
        "question": "How do I connect to external databases?",
        "answer": "Connect to external databases using: 1) JDBC/ODBC drivers, 2) Spark connectors, 3) Partner Connect, or 4) Unity Catalog external connections. Example: spark.read.jdbc(url, table, properties). Supported databases include PostgreSQL, MySQL, SQL Server, Oracle, Snowflake, and more."
    },
    {
        "question": "What is Databricks SQL?",
        "answer": "Databricks SQL is a serverless data warehouse built on the lakehouse architecture. It provides SQL analytics capabilities with features like SQL endpoints (compute), queries, dashboards, and alerts. It's optimized for BI workloads and integrates with popular BI tools like Tableau and Power BI."
    },
    {
        "question": "How do I use AutoML?",
        "answer": "To use Databricks AutoML: 1) Go to 'Machine Learning' workspace, 2) Click 'AutoML' experiment, 3) Select your dataset, 4) Choose prediction target and problem type (classification/regression), 5) Configure training settings, 6) Click 'Start AutoML'. It automatically trains multiple models, tunes hyperparameters, and generates a notebook with the best model."
    },
    {
        "question": "What is a metastore in Unity Catalog?",
        "answer": "A metastore is the top-level container for metadata in Unity Catalog. It stores information about catalogs, schemas, tables, views, and permissions. Each Databricks account has one metastore per region. The metastore manages data governance, access control, and audit logging across workspaces."
    },
    {
        "question": "How do I share data with Delta Sharing?",
        "answer": "Delta Sharing enables secure data sharing across organizations without copying data. As a provider: 1) Create a share, 2) Add tables to the share, 3) Grant access to recipients. As a recipient: Use the provided activation link and credentials. Delta Sharing works with any platform that supports the open protocol."
    },
    {
        "question": "What are Databricks clusters?",
        "answer": "Clusters are groups of compute resources that run data engineering, data science, and analytics workloads. Types include: All-purpose clusters (interactive work), Job clusters (automated workloads), and SQL warehouses (SQL analytics). Clusters support autoscaling, spot instances, and custom configurations."
    },
    {
        "question": "How do I use Git with Databricks?",
        "answer": "Databricks integrates with Git providers (GitHub, GitLab, Bitbucket, Azure DevOps). To use: 1) Go to 'Repos', 2) Click 'Add Repo', 3) Authenticate with your Git provider, 4) Clone repository, 5) Create branches, commit changes, and sync. This enables version control, CI/CD, and team collaboration on notebooks and code."
    }
]

print(f"✅ Loaded {len(databricks_docs)} documentation entries")
print(f"\nSample entry:")
print(f"Q: {databricks_docs[0]['question']}")
print(f"A: {databricks_docs[0]['answer'][:100]}...")

# COMMAND ----------

# DBTITLE 1,🧠 SETUP: Initialize Embedding Model
# Load a lightweight, free sentence transformer model
# all-MiniLM-L6-v2: 384 dimensions, fast, good for semantic search

print("Loading embedding model...")
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
print(f"✅ Model loaded: {model.get_sentence_embedding_dimension()} dimensions")

# Test the model
test_sentence = "What is Unity Catalog?"
test_embedding = model.encode(test_sentence)
print(f"\nTest embedding shape: {test_embedding.shape}")
print(f"First 5 values: {test_embedding[:5]}")

# COMMAND ----------

# DBTITLE 1,⚡ SETUP: Create Vector Embeddings
# Create embeddings for all documentation entries
# We'll embed the question + answer together for better semantic matching

print("Creating embeddings for documentation...")

doc_texts = []
for doc in databricks_docs:
    # Combine question and answer for richer semantic representation
    combined_text = f"{doc['question']} {doc['answer']}"
    doc_texts.append(combined_text)

# Generate embeddings
doc_embeddings = model.encode(doc_texts, show_progress_bar=True)

print(f"\n✅ Created {len(doc_embeddings)} embeddings")
print(f"Embedding shape: {doc_embeddings.shape}")
print(f"Memory usage: {doc_embeddings.nbytes / 1024:.2f} KB")

# COMMAND ----------

# DBTITLE 1,🔍 LEARNING: Build FAQ Search Function
def semantic_search(query: str, top_k: int = 5) -> List[Tuple[dict, float]]:
    """
    Perform semantic search on the documentation.
    
    Args:
        query: User's question
        top_k: Number of top results to return
    
    Returns:
        List of (document, similarity_score) tuples
    """
    # 1. Encode the query
    query_embedding = model.encode(query)
    
    # 2. Calculate cosine similarity with all documents
    # Cosine similarity = dot product of normalized vectors
    query_norm = query_embedding / np.linalg.norm(query_embedding)
    doc_norms = doc_embeddings / np.linalg.norm(doc_embeddings, axis=1, keepdims=True)
    similarities = np.dot(doc_norms, query_norm)
    
    # 3. Get top-k most similar documents
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    # 4. Prepare results
    results = []
    for idx in top_indices:
        results.append((
            databricks_docs[idx],
            float(similarities[idx])
        ))
    
    return results

print("✅ FAQ Search function ready!")

# COMMAND ----------

# DBTITLE 1,🎯 LEARNING: Example 1 - Unity Catalog Question
# Example query: How do I create a Unity Catalog?

query = "How do I create a Unity Catalog?"
print(f"🔍 Query: {query}")
print("=" * 80)

results = semantic_search(query, top_k=5)

for i, (doc, score) in enumerate(results, 1):
    print(f"\n📄 Result #{i} (Similarity: {score:.4f})")
    print(f"Question: {doc['question']}")
    print(f"Answer: {doc['answer'][:150]}...")
    print("-" * 80)

# COMMAND ----------

# DBTITLE 1,🎯 LEARNING: Example 2 - Job Scheduling
# Example query: How can I automate my workflow?

query = "How can I automate my workflow and run it on schedule?"
print(f"🔍 Query: {query}")
print("=" * 80)

results = semantic_search(query, top_k=5)

for i, (doc, score) in enumerate(results, 1):
    print(f"\n📄 Result #{i} (Similarity: {score:.4f})")
    print(f"Question: {doc['question']}")
    print(f"Answer: {doc['answer'][:150]}...")
    print("-" * 80)

# COMMAND ----------

# DBTITLE 1,🎯 LEARNING: Example 3 - Machine Learning
# Example query: Automatic model training

query = "How can I automatically train machine learning models?"
print(f"🔍 Query: {query}")
print("=" * 80)

results = semantic_search(query, top_k=5)

for i, (doc, score) in enumerate(results, 1):
    print(f"\n📄 Result #{i} (Similarity: {score:.4f})")
    print(f"Question: {doc['question']}")
    print(f"Answer: {doc['answer'][:150]}...")
    print("-" * 80)

# COMMAND ----------

# DBTITLE 1,🎯 LEARNING: Interactive Query (Try Your Own!)
# Try your own query!
# Modify the query below to search the documentation

my_query = "How do I connect to external databases?"

print(f"🔍 Your Query: {my_query}")
print("=" * 80)

results = semantic_search(my_query, top_k=5)

for i, (doc, score) in enumerate(results, 1):
    print(f"\n📄 Result #{i} (Similarity: {score:.4f})")
    print(f"Question: {doc['question']}")
    print(f"Answer: {doc['answer']}")
    print("-" * 80)

# COMMAND ----------

# DBTITLE 1,📊 LEARNING: Analyze Similarity Distribution
# Analyze how well the semantic search works

test_queries = [
    "How do I create a Unity Catalog?",
    "What is Delta Lake?",
    "Machine learning model training",
    "Data sharing between organizations"
]

print("Similarity Score Analysis")
print("=" * 80)

for query in test_queries:
    results = semantic_search(query, top_k=1)
    top_doc, top_score = results[0]
    print(f"\nQuery: {query}")
    print(f"Best Match: {top_doc['question']}")
    print(f"Score: {top_score:.4f}")
    print(f"Quality: {'🟢 Excellent' if top_score > 0.7 else '🟡 Good' if top_score > 0.5 else '🔴 Poor'}")

# COMMAND ----------

# DBTITLE 1,💡 Key Concepts Explained
# MAGIC %md
# MAGIC ## 🧠 How FAQ Search Works
# MAGIC
# MAGIC ### 1. **Vector Embeddings**
# MAGIC - Text is converted to dense numerical vectors (384 dimensions)
# MAGIC - Similar meanings → Similar vectors in vector space
# MAGIC - Example: "Create catalog" and "Make catalog" are close in vector space
# MAGIC
# MAGIC ### 2. **Cosine Similarity**
# MAGIC - Measures angle between two vectors
# MAGIC - Range: -1 (opposite) to 1 (identical)
# MAGIC - Formula: `cos(θ) = (A · B) / (||A|| × ||B||)`
# MAGIC
# MAGIC ### 3. **Semantic Search vs Keyword Search**
# MAGIC
# MAGIC | Aspect | Keyword Search | Semantic Search |
# MAGIC |--------|----------------|------------------|
# MAGIC | Method | Exact text match | Meaning-based |
# MAGIC | Example | "create catalog" matches only "create catalog" | Matches "make catalog", "setup catalog", etc. |
# MAGIC | Synonyms | ❌ Misses | ✅ Captures |
# MAGIC | Context | ❌ Ignores | ✅ Understands |
# MAGIC
# MAGIC ### 4. **Why Top-K Results?**
# MAGIC - User question might be ambiguous
# MAGIC - Multiple relevant answers possible
# MAGIC - Allows user to choose best match
# MAGIC - Common: top-5 or top-10
# MAGIC
# MAGIC ---

# COMMAND ----------

# DBTITLE 1,🚀 Production Enhancements
# MAGIC %md
# MAGIC ## 🔧 How to Scale This to Production
# MAGIC
# MAGIC ### 1. **Use Databricks Vector Search**
# MAGIC ```python
# MAGIC from databricks.vector_search.client import VectorSearchClient
# MAGIC
# MAGIC # Managed vector database with automatic sync
# MAGIC client = VectorSearchClient()
# MAGIC index = client.create_index(
# MAGIC     name="databricks_faq_index",
# MAGIC     embedding_model="databricks-bge-large-en",
# MAGIC     sync_mode="continuous"
# MAGIC )
# MAGIC ```
# MAGIC
# MAGIC ### 2. **Larger Knowledge Base**
# MAGIC - Scrape full docs.databricks.com
# MAGIC - Include community forums, blog posts
# MAGIC - Add code examples, error messages
# MAGIC
# MAGIC ### 3. **Better Embeddings**
# MAGIC - Use Databricks Foundation Model APIs
# MAGIC - Models: BGE, MPNet, Instructor
# MAGIC - Higher dimensions → Better accuracy
# MAGIC
# MAGIC ### 4. **Hybrid Search**
# MAGIC - Combine semantic + keyword search
# MAGIC - Rerank results with cross-encoder
# MAGIC - Filter by metadata (date, category)
# MAGIC
# MAGIC ### 5. **RAG (Retrieval Augmented Generation)**
# MAGIC - Retrieve relevant docs (this tutorial)
# MAGIC - Pass to LLM as context
# MAGIC - Generate natural language answer
# MAGIC
# MAGIC ---

# COMMAND ----------

# DBTITLE 1,🗑️ DESTROY: Cleanup Resources
# Clean up to free memory
import gc

print("Cleaning up resources...")

# Clear large objects
del model
del doc_embeddings
del databricks_docs

# Force garbage collection
gc.collect()

print("✅ Cleanup complete! All resources freed.")
print("\n💡 This notebook used only free resources:")
print("   - Open-source embedding model")
print("   - NumPy for vector operations")
print("   - No external API calls or costs")

# COMMAND ----------

# DBTITLE 1,🎓 Summary & Next Steps
# MAGIC %md
# MAGIC ## 🎉 Congratulations!
# MAGIC
# MAGIC You've learned how to build a semantic FAQ search system!
# MAGIC
# MAGIC ### What You Accomplished
# MAGIC ✅ Loaded Databricks documentation as a knowledge base  
# MAGIC ✅ Created vector embeddings using sentence-transformers  
# MAGIC ✅ Built a semantic search function with cosine similarity  
# MAGIC ✅ Retrieved top-5 relevant documents for user queries  
# MAGIC ✅ Used 100% free, open-source resources  
# MAGIC
# MAGIC ### Key Takeaways
# MAGIC 1. **Vector embeddings** capture semantic meaning
# MAGIC 2. **Cosine similarity** measures relevance
# MAGIC 3. **Top-K retrieval** provides multiple relevant results
# MAGIC 4. **Semantic search** understands intent, not just keywords
# MAGIC
# MAGIC ### Next Steps
# MAGIC 1. **Scale up**: Try Databricks Vector Search for production
# MAGIC 2. **Add RAG**: Combine with LLMs for answer generation
# MAGIC 3. **Improve data**: Expand documentation coverage
# MAGIC 4. **Fine-tune**: Use domain-specific embedding models
# MAGIC 5. **Hybrid search**: Combine semantic + keyword matching
# MAGIC
# MAGIC ### Resources
# MAGIC - [Databricks Vector Search Documentation](https://docs.databricks.com/vectors/)
# MAGIC - [Sentence Transformers](https://www.sbert.net/)
# MAGIC - [Unity Catalog Documentation](https://docs.databricks.com/unity-catalog/)
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC **Questions?** Try modifying the interactive query cell above with your own questions!

# COMMAND ----------

