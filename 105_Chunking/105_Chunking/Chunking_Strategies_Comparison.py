# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# DBTITLE 1,Introduction to Chunking Strategies
# MAGIC %md
# MAGIC # Chunking Strategies Comparison
# MAGIC
# MAGIC This notebook explores different chunking strategies for text processing in RAG (Retrieval-Augmented Generation) systems.
# MAGIC
# MAGIC ## What is Chunking?
# MAGIC
# MAGIC Chunking is the process of breaking down large documents into smaller, manageable pieces for embedding and retrieval. The chunk size significantly impacts:
# MAGIC - **Retrieval quality**: Smaller chunks may be more precise but lose context
# MAGIC - **Number of chunks**: More chunks mean more embeddings to store and search
# MAGIC - **Response quality**: Chunk size affects the context provided to LLMs
# MAGIC
# MAGIC ## Comparison Overview
# MAGIC
# MAGIC We'll compare three chunk sizes:
# MAGIC 1. **256 tokens** - Small, precise chunks
# MAGIC 2. **512 tokens** - Medium-sized chunks (common default)
# MAGIC 3. **1024 tokens** - Larger chunks with more context
# MAGIC
# MAGIC ## Measurements
# MAGIC
# MAGIC For each strategy, we'll measure:
# MAGIC - Number of chunks generated
# MAGIC - Retrieval accuracy (semantic similarity)
# MAGIC - Coverage and context preservation

# COMMAND ----------

# DBTITLE 1,Setup: Install Dependencies
# Install required libraries (free resources)
%pip install tiktoken sentence-transformers scikit-learn --quiet
dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Setup: Import Libraries
import tiktoken
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

print("✓ All libraries imported successfully")

# COMMAND ----------

# DBTITLE 1,Setup: Create Sample Dataset
# Create a sample dataset about machine learning concepts
# This simulates a knowledge base document

sample_document = """
Machine Learning Fundamentals

Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed. 
The field encompasses various approaches including supervised learning, unsupervised learning, and reinforcement learning.

Supervised Learning: In supervised learning, algorithms learn from labeled training data. The algorithm makes predictions based on input data and is corrected when its predictions are incorrect. 
Common supervised learning algorithms include linear regression, logistic regression, decision trees, random forests, and neural networks. 
These algorithms are used for classification and regression tasks. Classification assigns inputs to discrete categories, while regression predicts continuous values.

Unsupervised Learning: Unsupervised learning works with unlabeled data to find hidden patterns or intrinsic structures. 
The algorithm must discover the patterns on its own without any guidance. Common techniques include clustering algorithms like K-means and hierarchical clustering, 
and dimensionality reduction methods like Principal Component Analysis (PCA) and t-SNE. These methods are valuable for exploratory data analysis, 
customer segmentation, and feature engineering.

Reinforcement Learning: This paradigm involves an agent learning to make decisions by performing actions in an environment to maximize cumulative reward. 
The agent receives feedback in the form of rewards or penalties and learns the optimal policy through trial and error. 
Applications include game playing, robotics, resource management, and autonomous systems.

Neural Networks and Deep Learning: Neural networks are computing systems inspired by biological neural networks. 
They consist of interconnected nodes (neurons) organized in layers. Deep learning uses neural networks with multiple hidden layers to learn hierarchical representations. 
Convolutional Neural Networks (CNNs) excel at image processing, while Recurrent Neural Networks (RNNs) and Transformers are effective for sequential data like text and time series.

Model Evaluation and Validation: Proper evaluation is crucial for machine learning systems. Common metrics include accuracy, precision, recall, F1-score for classification, 
and mean squared error, R-squared for regression. Cross-validation techniques like k-fold cross-validation help assess model generalization. 
Overfitting occurs when a model learns training data too well but fails on new data, while underfitting happens when the model is too simple to capture patterns.

Feature Engineering: The process of selecting, modifying, or creating features from raw data significantly impacts model performance. 
Techniques include normalization, standardization, one-hot encoding for categorical variables, and creating interaction features. 
Domain knowledge often guides effective feature engineering.

Model Deployment and MLOps: Deploying machine learning models to production requires careful consideration of scalability, monitoring, and maintenance. 
MLOps practices combine machine learning, DevOps, and data engineering to streamline the ML lifecycle. 
This includes version control for models and data, continuous integration and deployment, model monitoring for drift, and automated retraining pipelines.
"""

print(f"Sample document created: {len(sample_document)} characters")
print(f"Preview: {sample_document[:200]}...")

# COMMAND ----------

# DBTITLE 1,Setup: Initialize Tokenizer and Embedding Model
# Initialize tokenizer (using GPT-2 tokenizer as standard)
tokenizer = tiktoken.get_encoding("cl100k_base")  # OpenAI's tokenizer

# Initialize a free, lightweight embedding model
print("Loading embedding model (this may take a moment)...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # Free, efficient model

print("✓ Tokenizer and embedding model initialized")
print(f"✓ Document has {len(tokenizer.encode(sample_document))} tokens")

# COMMAND ----------

# DBTITLE 1,Learning: Define Chunking Function
def chunk_text(text, chunk_size, overlap=50):
    """
    Split text into chunks based on token count with overlap.
    
    Args:
        text: Input text to chunk
        chunk_size: Target size of each chunk in tokens
        overlap: Number of overlapping tokens between chunks
    
    Returns:
        List of text chunks
    """
    tokens = tokenizer.encode(text)
    chunks = []
    
    start = 0
    while start < len(tokens):
        # Get chunk of tokens
        end = min(start + chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]
        
        # Decode back to text
        chunk_text = tokenizer.decode(chunk_tokens)
        chunks.append(chunk_text)
        
        # Move start position (with overlap)
        start += chunk_size - overlap
        
        # Break if we've covered all tokens
        if end == len(tokens):
            break
    
    return chunks

print("✓ Chunking function defined")

# COMMAND ----------

# DBTITLE 1,Learning: Create Chunks for Each Strategy
# Define chunk sizes to compare
chunk_sizes = [256, 512, 1024]

# Store results
chunking_results = {}

for size in chunk_sizes:
    print(f"\nCreating chunks with size {size} tokens...")
    chunks = chunk_text(sample_document, chunk_size=size, overlap=50)
    
    # Calculate actual token counts
    token_counts = [len(tokenizer.encode(chunk)) for chunk in chunks]
    
    chunking_results[size] = {
        'chunks': chunks,
        'num_chunks': len(chunks),
        'token_counts': token_counts,
        'avg_tokens': np.mean(token_counts),
        'min_tokens': min(token_counts),
        'max_tokens': max(token_counts)
    }
    
    print(f"  ✓ Generated {len(chunks)} chunks")
    print(f"  ✓ Avg tokens per chunk: {np.mean(token_counts):.1f}")
    print(f"  ✓ Token range: [{min(token_counts)}, {max(token_counts)}]")

print("\n✓ All chunking strategies completed")

# COMMAND ----------

# DBTITLE 1,Learning: Generate Embeddings
# Generate embeddings for all chunks
print("Generating embeddings for all chunks...\n")

for size in chunk_sizes:
    print(f"Processing {size}-token chunks...")
    chunks = chunking_results[size]['chunks']
    
    # Generate embeddings
    embeddings = embedding_model.encode(chunks, show_progress_bar=False)
    chunking_results[size]['embeddings'] = embeddings
    
    print(f"  ✓ Generated {len(embeddings)} embeddings of dimension {embeddings.shape[1]}")

print("\n✓ All embeddings generated")

# COMMAND ----------

# DBTITLE 1,Learning: Define Test Queries
# Create test queries to evaluate retrieval quality
test_queries = [
    "What is supervised learning and what algorithms are used?",
    "Explain reinforcement learning and its applications",
    "How do neural networks work in deep learning?",
    "What metrics are used to evaluate machine learning models?",
    "What is feature engineering and why is it important?"
]

print(f"Created {len(test_queries)} test queries for evaluation")
for i, query in enumerate(test_queries, 1):
    print(f"  {i}. {query}")

# COMMAND ----------

# DBTITLE 1,Measurement: Evaluate Retrieval Quality
# Evaluate retrieval quality for each chunking strategy
print("Evaluating retrieval quality...\n")

retrieval_results = []

for size in chunk_sizes:
    print(f"Evaluating {size}-token chunks...")
    embeddings = chunking_results[size]['embeddings']
    chunks = chunking_results[size]['chunks']
    
    query_scores = []
    
    for query in test_queries:
        # Encode query
        query_embedding = embedding_model.encode([query])[0]
        
        # Calculate similarity with all chunks
        similarities = cosine_similarity([query_embedding], embeddings)[0]
        
        # Get top-1 retrieval score
        max_similarity = np.max(similarities)
        best_chunk_idx = np.argmax(similarities)
        
        query_scores.append({
            'query': query,
            'max_similarity': max_similarity,
            'best_chunk_idx': best_chunk_idx,
            'best_chunk_preview': chunks[best_chunk_idx][:100] + '...'
        })
    
    avg_similarity = np.mean([s['max_similarity'] for s in query_scores])
    
    chunking_results[size]['query_scores'] = query_scores
    chunking_results[size]['avg_retrieval_score'] = avg_similarity
    
    print(f"  ✓ Average retrieval score: {avg_similarity:.4f}")
    
    # Store for comparison table
    retrieval_results.append({
        'Chunk Size': f"{size} tokens",
        'Num Chunks': chunking_results[size]['num_chunks'],
        'Avg Tokens/Chunk': f"{chunking_results[size]['avg_tokens']:.1f}",
        'Avg Retrieval Score': f"{avg_similarity:.4f}"
    })

print("\n✓ Retrieval quality evaluation completed")

# COMMAND ----------

# DBTITLE 1,Measurement: Display Comparison Results
# Display comprehensive comparison
results_df = pd.DataFrame(retrieval_results)

print("\n" + "="*70)
print(" CHUNKING STRATEGY COMPARISON RESULTS")
print("="*70 + "\n")

print(results_df.to_string(index=False))
print("\n" + "="*70)

# Key insights
print("\n📊 KEY INSIGHTS:\n")

best_retrieval_size = chunk_sizes[np.argmax([chunking_results[s]['avg_retrieval_score'] for s in chunk_sizes])]
print(f"✓ Best retrieval quality: {best_retrieval_size} tokens")
print(f"  (Score: {chunking_results[best_retrieval_size]['avg_retrieval_score']:.4f})\n")

min_chunks_size = chunk_sizes[np.argmin([chunking_results[s]['num_chunks'] for s in chunk_sizes])]
print(f"✓ Fewest chunks (storage efficient): {min_chunks_size} tokens")
print(f"  ({chunking_results[min_chunks_size]['num_chunks']} chunks)\n")

print("💡 RECOMMENDATIONS:\n")
print("• 256 tokens: Best for precise retrieval, but creates more chunks (higher storage)")
print("• 512 tokens: Good balance between context and precision (recommended default)")
print("• 1024 tokens: Best for context-heavy tasks, fewer chunks, but may be too broad")
print("\n• Choose based on your use case:")
print("  - QA systems → Smaller chunks (256-512)")
print("  - Summarization → Larger chunks (512-1024)")
print("  - Hybrid retrieval → Medium chunks (512)")

# COMMAND ----------

# DBTITLE 1,Measurement: Visualize Per-Query Performance
# Show detailed per-query performance
print("\n" + "="*70)
print(" PER-QUERY RETRIEVAL PERFORMANCE")
print("="*70 + "\n")

for i, query in enumerate(test_queries):
    print(f"Query {i+1}: '{query}'\n")
    
    for size in chunk_sizes:
        score = chunking_results[size]['query_scores'][i]['max_similarity']
        bar_length = int(score * 50)  # Scale to 50 chars
        bar = '█' * bar_length + '░' * (50 - bar_length)
        
        print(f"  {size:4d} tokens: {bar} {score:.4f}")
    
    print()

# COMMAND ----------

# DBTITLE 1,Measurement: Analyze Trade-offs
# Analyze storage vs quality trade-offs
print("\n" + "="*70)
print(" STORAGE VS QUALITY TRADE-OFF ANALYSIS")
print("="*70 + "\n")

for size in chunk_sizes:
    num_chunks = chunking_results[size]['num_chunks']
    retrieval_score = chunking_results[size]['avg_retrieval_score']
    embedding_dim = chunking_results[size]['embeddings'].shape[1]
    
    # Estimate storage (simplified)
    storage_mb = (num_chunks * embedding_dim * 4) / (1024 * 1024)  # 4 bytes per float32
    
    print(f"{size} tokens:")
    print(f"  • Number of chunks: {num_chunks}")
    print(f"  • Retrieval quality: {retrieval_score:.4f}")
    print(f"  • Estimated storage: {storage_mb:.3f} MB")
    print(f"  • Chunks per query: ~{num_chunks/len(test_queries):.1f} searchable units\n")

print("💡 Storage Insight: Larger chunks reduce storage requirements but may sacrifice retrieval precision.")

# COMMAND ----------

# DBTITLE 1,Destroy: Cleanup
# Clean up variables to free memory
print("Cleaning up resources...\n")

try:
    del embedding_model
    del tokenizer
    del chunking_results
    del sample_document
    print("✓ Embedding model released")
    print("✓ Tokenizer released")
    print("✓ Chunking results cleared")
    print("✓ Sample document cleared")
    print("\n✓ All resources cleaned up successfully")
    print("\n📌 Note: No persistent storage was used. All computations were in-memory.")
except Exception as e:
    print(f"⚠ Cleanup warning: {e}")

# COMMAND ----------

# DBTITLE 1,Summary and Next Steps
# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC This notebook demonstrated:
# MAGIC 1. ✅ **Setup**: Created sample data and initialized free embedding models
# MAGIC 2. ✅ **Learning**: Implemented chunking strategies (256, 512, 1024 tokens)
# MAGIC 3. ✅ **Measurement**: Evaluated retrieval quality and chunk counts
# MAGIC 4. ✅ **Comparison**: Analyzed trade-offs between strategies
# MAGIC 5. ✅ **Cleanup**: Released all resources
# MAGIC
# MAGIC ## Key Takeaways
# MAGIC
# MAGIC - **Chunk size matters**: It directly impacts both retrieval precision and storage requirements
# MAGIC - **No one-size-fits-all**: Choose chunk size based on your specific use case
# MAGIC - **Test with real queries**: Always evaluate on representative queries from your domain
# MAGIC - **Consider overlap**: Overlapping chunks can improve retrieval but increase storage
# MAGIC
# MAGIC ## Next Steps
# MAGIC
# MAGIC 1. **Experiment with your own data**: Replace `sample_document` with your documents
# MAGIC 2. **Try different overlap sizes**: Test overlap values from 0 to 100 tokens
# MAGIC 3. **Use advanced chunking**: Explore semantic chunking, sentence-based splitting
# MAGIC 4. **Implement hybrid search**: Combine dense (embedding) and sparse (keyword) retrieval
# MAGIC 5. **Scale to production**: Use vector databases like Databricks Vector Search
# MAGIC
# MAGIC ## Resources
# MAGIC
# MAGIC - [Databricks Vector Search Documentation](https://docs.databricks.com/en/generative-ai/vector-search.html)
# MAGIC - [LangChain Text Splitters](https://python.langchain.com/docs/modules/data_connection/document_transformers/)
# MAGIC - [Sentence Transformers](https://www.sbert.net/)

# COMMAND ----------

