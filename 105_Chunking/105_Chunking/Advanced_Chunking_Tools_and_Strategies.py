# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# DBTITLE 1,Introduction to Advanced Chunking Tools
# MAGIC %md
# MAGIC # Advanced Chunking Tools and Strategies
# MAGIC
# MAGIC This notebook explores **different chunking tools and strategies** beyond simple token-based splitting. We'll compare various approaches to intelligently split documents for RAG systems.
# MAGIC
# MAGIC ## Chunking Tools Covered
# MAGIC
# MAGIC 1. **Recursive Character Text Splitter** - Hierarchical splitting by separators
# MAGIC 2. **Sentence-Based Chunker** - Respects sentence boundaries
# MAGIC 3. **Semantic Chunker** - Groups by semantic similarity
# MAGIC 4. **Markdown-Aware Chunker** - Preserves document structure
# MAGIC 5. **Fixed-Size with Overlap** - Traditional token-based (baseline)
# MAGIC
# MAGIC ## Why Different Strategies?
# MAGIC
# MAGIC - **Context preservation**: Some strategies maintain logical document structure
# MAGIC - **Semantic coherence**: Keep related information together
# MAGIC - **Domain-specific needs**: Technical docs vs narratives require different approaches
# MAGIC - **Query performance**: Different strategies excel at different query types
# MAGIC
# MAGIC ## Architecture
# MAGIC
# MAGIC This notebook follows a clean lifecycle:
# MAGIC - **Setup**: Install tools, load models, prepare data (free resources only)
# MAGIC - **Learning**: Implement and compare 5 different chunking strategies
# MAGIC - **Measurement**: Evaluate retrieval quality and chunk characteristics
# MAGIC - **Destroy**: Clean up all resources

# COMMAND ----------

# DBTITLE 1,Setup: Install Dependencies
# Install required libraries (all free and open-source)
%pip install langchain langchain-text-splitters sentence-transformers scikit-learn nltk --quiet
dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Setup: Import Libraries
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# LangChain text splitters
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
    CharacterTextSplitter
)

# Sentence processing
import nltk
from nltk.tokenize import sent_tokenize

# Embeddings and similarity
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering

# Download NLTK data
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

print("✓ All libraries imported successfully")

# COMMAND ----------

# DBTITLE 1,Setup: Create Sample Documents
# Create a comprehensive technical document with structure
# This simulates a real technical documentation page

sample_document = """
# Introduction to Vector Databases

Vector databases are specialized database systems designed to store and query high-dimensional vectors efficiently. They have become essential infrastructure for modern AI applications, particularly in retrieval-augmented generation (RAG) systems and semantic search.

## What Are Vectors?

In the context of machine learning, vectors are numerical representations of data. Text, images, audio, and other data types can be converted into vectors through embedding models. These embeddings capture semantic meaning in a way that computers can process and compare.

For example, the sentence "machine learning" might be represented as a 384-dimensional vector like [0.23, -0.45, 0.67, ...]. Similar concepts have similar vector representations, measured by metrics like cosine similarity.

## Key Features of Vector Databases

Vector databases offer several critical capabilities:

### Similarity Search

The primary operation is finding vectors most similar to a query vector. This enables semantic search where you can find conceptually related content, not just keyword matches. Traditional databases struggle with this use case because they're optimized for exact matches and range queries.

### Scalability

Modern vector databases can handle billions of vectors while maintaining sub-second query latency. They achieve this through specialized indexing algorithms like HNSW (Hierarchical Navigable Small World) and IVF (Inverted File Index).

### Hybrid Search

Many vector databases support combining dense vector search with traditional sparse keyword search (BM25). This hybrid approach often produces better results than either method alone, especially for queries with specific terminology.

## Popular Vector Databases

Several options exist in the market:

- **Databricks Vector Search**: Integrated directly with Delta tables and Unity Catalog, supporting both managed embeddings and self-managed indices
- **Pinecone**: Cloud-native vector database with managed infrastructure
- **Weaviate**: Open-source with built-in vectorization modules
- **Milvus**: High-performance open-source option designed for production AI
- **Qdrant**: Rust-based database with advanced filtering capabilities

## Use Cases

Vector databases power many modern applications:

1. **Question Answering Systems**: Retrieve relevant context from large document collections
2. **Recommendation Engines**: Find similar products, content, or users
3. **Anomaly Detection**: Identify outliers by measuring distance from normal patterns
4. **Image Search**: Find visually similar images using vision embeddings
5. **Code Search**: Semantic code search across large repositories

## Implementation Considerations

When implementing a vector database solution, consider:

- **Embedding quality**: The embedding model quality directly impacts retrieval accuracy
- **Index type**: Choose between accuracy (flat) and speed (approximate) based on your scale
- **Chunk size**: For text, chunking strategy significantly affects retrieval quality
- **Metadata filtering**: Pre-filter by metadata before vector search to improve relevance
- **Update frequency**: Some databases handle real-time updates better than others

## Conclusion

Vector databases are foundational infrastructure for AI applications. As embedding models improve and use cases expand, these specialized databases will become even more critical. Choose a solution that aligns with your scale, latency requirements, and integration needs.
"""

print(f"✓ Sample document created: {len(sample_document)} characters")
print(f"✓ Preview: {sample_document[:200].strip()}...")

# COMMAND ----------

# DBTITLE 1,Setup: Initialize Embedding Model
# Load a lightweight, free embedding model
print("Loading embedding model (this may take a moment)...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # 384-dim, fast, free

print("✓ Embedding model loaded")
print(f"✓ Embedding dimension: {embedding_model.get_sentence_embedding_dimension()}")

# COMMAND ----------

# DBTITLE 1,Learning: Strategy 1 - Recursive Character Splitter
# Strategy 1: RecursiveCharacterTextSplitter
# Tries to split by paragraphs, then sentences, then characters hierarchically

print("Strategy 1: Recursive Character Text Splitter\n" + "="*60)

recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    length_function=len,
    separators=["\n\n", "\n", ". ", " ", ""]  # Hierarchical separators
)

recursive_chunks = recursive_splitter.split_text(sample_document)

print(f"✓ Generated {len(recursive_chunks)} chunks")
print(f"✓ Chunk sizes: min={min(len(c) for c in recursive_chunks)}, max={max(len(c) for c in recursive_chunks)}")
print(f"\nFirst chunk preview:\n{recursive_chunks[0][:200]}...")

# COMMAND ----------

# DBTITLE 1,Learning: Strategy 2 - Sentence-Based Chunker
# Strategy 2: Sentence-based chunking
# Groups sentences together up to a character limit, respecting sentence boundaries

print("Strategy 2: Sentence-Based Chunker\n" + "="*60)

def sentence_based_chunker(text, target_size=500, overlap_sentences=1):
    """
    Chunk text by grouping sentences, respecting sentence boundaries.
    """
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_size = 0
    
    for i, sentence in enumerate(sentences):
        sentence_len = len(sentence)
        
        if current_size + sentence_len > target_size and current_chunk:
            # Save current chunk
            chunks.append(" ".join(current_chunk))
            
            # Start new chunk with overlap
            if overlap_sentences > 0:
                current_chunk = current_chunk[-overlap_sentences:]
                current_size = sum(len(s) for s in current_chunk)
            else:
                current_chunk = []
                current_size = 0
        
        current_chunk.append(sentence)
        current_size += sentence_len
    
    # Add final chunk
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks

sentence_chunks = sentence_based_chunker(sample_document, target_size=500, overlap_sentences=1)

print(f"✓ Generated {len(sentence_chunks)} chunks")
print(f"✓ Chunk sizes: min={min(len(c) for c in sentence_chunks)}, max={max(len(c) for c in sentence_chunks)}")
print(f"\nFirst chunk preview:\n{sentence_chunks[0][:200]}...")

# COMMAND ----------

# DBTITLE 1,Learning: Strategy 3 - Semantic Chunker
# Strategy 3: Semantic chunking using embedding similarity
# Groups sentences with high semantic similarity together

print("Strategy 3: Semantic Chunker\n" + "="*60)

def semantic_chunker(text, similarity_threshold=0.5, max_chunk_size=800):
    """
    Chunk text by grouping semantically similar sentences.
    Uses hierarchical clustering on sentence embeddings.
    """
    sentences = sent_tokenize(text)
    
    if len(sentences) < 2:
        return [text]
    
    # Get embeddings for all sentences
    print("  Computing sentence embeddings...")
    sentence_embeddings = embedding_model.encode(sentences)
    
    # Compute similarity matrix
    similarity_matrix = cosine_similarity(sentence_embeddings)
    
    # Convert similarity to distance for clustering
    distance_matrix = 1 - similarity_matrix
    
    # Hierarchical clustering
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=1-similarity_threshold,
        metric='precomputed',
        linkage='average'
    )
    
    labels = clustering.fit_predict(distance_matrix)
    
    # Group sentences by cluster
    chunks = []
    for cluster_id in range(max(labels) + 1):
        cluster_sentences = [sentences[i] for i, label in enumerate(labels) if label == cluster_id]
        chunk_text = " ".join(cluster_sentences)
        
        # Split large chunks if needed
        if len(chunk_text) > max_chunk_size:
            # Fall back to sentence-based splitting for this cluster
            sub_chunks = sentence_based_chunker(chunk_text, target_size=max_chunk_size, overlap_sentences=0)
            chunks.extend(sub_chunks)
        else:
            chunks.append(chunk_text)
    
    return chunks

semantic_chunks = semantic_chunker(sample_document, similarity_threshold=0.5, max_chunk_size=800)

print(f"✓ Generated {len(semantic_chunks)} semantic chunks")
print(f"✓ Chunk sizes: min={min(len(c) for c in semantic_chunks)}, max={max(len(c) for c in semantic_chunks)}")
print(f"\nFirst chunk preview:\n{semantic_chunks[0][:200]}...")

# COMMAND ----------

# DBTITLE 1,Learning: Strategy 4 - Markdown-Aware Chunker
# Strategy 4: Markdown header-based chunking
# Preserves document structure by splitting at markdown headers

print("Strategy 4: Markdown-Aware Chunker\n" + "="*60)

headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]

markdown_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on,
    strip_headers=False
)

markdown_chunks_docs = markdown_splitter.split_text(sample_document)

# Extract just the text content
markdown_chunks = [doc.page_content for doc in markdown_chunks_docs]

# Further split large sections
text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=50)
final_markdown_chunks = []
for chunk in markdown_chunks:
    if len(chunk) > 600:
        sub_chunks = text_splitter.split_text(chunk)
        final_markdown_chunks.extend(sub_chunks)
    else:
        final_markdown_chunks.append(chunk)

markdown_chunks = final_markdown_chunks

print(f"✓ Generated {len(markdown_chunks)} structure-aware chunks")
print(f"✓ Chunk sizes: min={min(len(c) for c in markdown_chunks)}, max={max(len(c) for c in markdown_chunks)}")
print(f"\nFirst chunk preview:\n{markdown_chunks[0][:200]}...")

# COMMAND ----------

# DBTITLE 1,Learning: Strategy 5 - Fixed-Size Baseline
# Strategy 5: Simple fixed-size chunking (baseline)
# Splits text into fixed-size chunks with overlap, no structure awareness

print("Strategy 5: Fixed-Size Chunker (Baseline)\n" + "="*60)

fixed_splitter = CharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separator=" ",
    length_function=len
)

fixed_chunks = fixed_splitter.split_text(sample_document)

print(f"✓ Generated {len(fixed_chunks)} fixed-size chunks")
print(f"✓ Chunk sizes: min={min(len(c) for c in fixed_chunks)}, max={max(len(c) for c in fixed_chunks)}")
print(f"\nFirst chunk preview:\n{fixed_chunks[0][:200]}...")

# COMMAND ----------

# DBTITLE 1,Measurement: Generate Embeddings for All Strategies
# Store all chunking results
chunking_strategies = {
    'Recursive Character': recursive_chunks,
    'Sentence-Based': sentence_chunks,
    'Semantic': semantic_chunks,
    'Markdown-Aware': markdown_chunks,
    'Fixed-Size (Baseline)': fixed_chunks
}

print("Generating embeddings for all strategies...\n")

strategy_embeddings = {}

for strategy_name, chunks in chunking_strategies.items():
    print(f"Processing {strategy_name}...")
    embeddings = embedding_model.encode(chunks, show_progress_bar=False)
    strategy_embeddings[strategy_name] = embeddings
    print(f"  ✓ Generated {len(embeddings)} embeddings of dimension {embeddings.shape[1]}")

print("\n✓ All embeddings generated")

# COMMAND ----------

# DBTITLE 1,Measurement: Define Test Queries
# Create diverse test queries to evaluate retrieval quality
test_queries = [
    "What is a vector database and why is it important?",
    "Explain how similarity search works in vector databases",
    "What are the key features and capabilities of vector databases?",
    "Tell me about Databricks Vector Search",
    "How do I choose between different indexing algorithms?",
    "What are common use cases for vector databases?"
]

print(f"Created {len(test_queries)} test queries for evaluation:")
for i, query in enumerate(test_queries, 1):
    print(f"  {i}. {query}")

# COMMAND ----------

# DBTITLE 1,Measurement: Evaluate Retrieval Quality
# Evaluate retrieval quality for each strategy
print("Evaluating retrieval quality across all strategies...\n")

results = []

for strategy_name, chunks in chunking_strategies.items():
    embeddings = strategy_embeddings[strategy_name]
    
    query_scores = []
    
    for query in test_queries:
        # Encode query
        query_embedding = embedding_model.encode([query])[0]
        
        # Calculate similarity with all chunks
        similarities = cosine_similarity([query_embedding], embeddings)[0]
        
        # Get top-1 retrieval score
        max_similarity = np.max(similarities)
        query_scores.append(max_similarity)
    
    avg_retrieval_score = np.mean(query_scores)
    
    results.append({
        'Strategy': strategy_name,
        'Num Chunks': len(chunks),
        'Avg Chunk Size': int(np.mean([len(c) for c in chunks])),
        'Min Size': min(len(c) for c in chunks),
        'Max Size': max(len(c) for c in chunks),
        'Retrieval Score': f"{avg_retrieval_score:.4f}",
        'Score_numeric': avg_retrieval_score
    })

print("✓ Evaluation completed\n")

# COMMAND ----------

# DBTITLE 1,Measurement: Display Comparison Results
# Display comprehensive comparison
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('Score_numeric', ascending=False)
results_df_display = results_df.drop('Score_numeric', axis=1)

print("\n" + "="*90)
print(" CHUNKING STRATEGY COMPARISON RESULTS")
print("="*90 + "\n")

print(results_df_display.to_string(index=False))
print("\n" + "="*90)

# Key insights
print("\n📊 KEY INSIGHTS:\n")

best_strategy = results_df.iloc[0]['Strategy']
best_score = results_df.iloc[0]['Score_numeric']

print(f"✓ Best retrieval quality: {best_strategy}")
print(f"  (Score: {best_score:.4f})\n")

fewest_chunks = results_df.loc[results_df['Num Chunks'].idxmin()]
print(f"✓ Most efficient (fewest chunks): {fewest_chunks['Strategy']}")
print(f"  ({fewest_chunks['Num Chunks']} chunks)\n")

print("💡 STRATEGY RECOMMENDATIONS:\n")
print("• Recursive Character: Great general-purpose choice, respects natural breaks")
print("• Sentence-Based: Best for QA systems, maintains complete thoughts")
print("• Semantic: Excellent for topical clustering, computationally intensive")
print("• Markdown-Aware: Perfect for structured docs (documentation, wikis)")
print("• Fixed-Size: Simple baseline, may break mid-sentence")
print("\n🎯 WHEN TO USE EACH:\n")
print("  - Technical docs with headers → Markdown-Aware")
print("  - Q&A / Chatbots → Sentence-Based or Recursive")
print("  - Topic-based search → Semantic")
print("  - Simple/fast pipeline → Fixed-Size")
print("  - Unknown document type → Recursive (safe default)")

# COMMAND ----------

# DBTITLE 1,Measurement: Visualize Trade-offs
# Visualize the trade-offs between strategies
print("\n" + "="*90)
print(" STRATEGY TRADE-OFF VISUALIZATION")
print("="*90 + "\n")

for _, row in results_df.iterrows():
    strategy = row['Strategy']
    score = row['Score_numeric']
    chunks = row['Num Chunks']
    
    # Create visual bars
    score_bar_len = int(score * 50)
    score_bar = '█' * score_bar_len + '░' * (50 - score_bar_len)
    
    chunk_bar_len = min(int(chunks * 3), 50)  # Scale for visibility
    chunk_bar = '█' * chunk_bar_len + '░' * (50 - chunk_bar_len)
    
    print(f"{strategy:25s}")
    print(f"  Retrieval Quality: {score_bar} {score:.4f}")
    print(f"  Num Chunks:        {chunk_bar} {chunks}")
    print()

# COMMAND ----------

# DBTITLE 1,Measurement: Chunk Quality Analysis
# Analyze chunk quality characteristics
print("\n" + "="*90)
print(" CHUNK QUALITY CHARACTERISTICS")
print("="*90 + "\n")

for strategy_name, chunks in chunking_strategies.items():
    print(f"{strategy_name}:")
    
    # Check for mid-sentence breaks (heuristic: chunks ending without punctuation)
    clean_endings = sum(1 for c in chunks if c.strip()[-1] in '.!?;:')
    clean_ending_pct = (clean_endings / len(chunks)) * 100
    
    # Check for header preservation
    has_headers = sum(1 for c in chunks if c.strip().startswith('#'))
    
    # Size variance (lower is more consistent)
    sizes = [len(c) for c in chunks]
    size_variance = np.std(sizes) / np.mean(sizes) if np.mean(sizes) > 0 else 0
    
    print(f"  • Clean endings: {clean_ending_pct:.1f}% (chunks end with punctuation)")
    print(f"  • Header preservation: {has_headers} chunks contain markdown headers")
    print(f"  • Size consistency: {1/(1+size_variance):.2f} (1.0 = perfectly uniform)")
    print(f"  • Avg chunk length: {np.mean(sizes):.0f} chars")
    print()

print("🔍 Insights: Higher 'clean endings' means better sentence boundary respect.")
print("           Header preservation is crucial for structured documents.")

# COMMAND ----------

# DBTITLE 1,Destroy: Cleanup Resources
# Clean up all resources to free memory
print("Cleaning up resources...\n")

try:
    # Delete large objects
    del embedding_model
    del chunking_strategies
    del strategy_embeddings
    del recursive_chunks, sentence_chunks, semantic_chunks, markdown_chunks, fixed_chunks
    del sample_document
    del results, results_df
    
    print("✓ Embedding model released")
    print("✓ All chunking results cleared")
    print("✓ Embeddings cleared")
    print("✓ Sample document cleared")
    print("✓ Analysis results cleared")
    print("\n✓ All resources cleaned up successfully")
    print("\n📌 Note: No persistent storage was used. All computations were in-memory.")
    print("📌 All tools used (LangChain, sentence-transformers, scikit-learn) are free and open-source.")
except Exception as e:
    print(f"⚠ Cleanup warning: {e}")

# COMMAND ----------

# DBTITLE 1,Summary and Next Steps
# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC This notebook successfully demonstrated:
# MAGIC 1. ✅ **Setup**: Installed free chunking tools (LangChain, NLTK, sentence-transformers)
# MAGIC 2. ✅ **Learning**: Implemented 5 different chunking strategies with distinct approaches
# MAGIC 3. ✅ **Measurement**: Evaluated retrieval quality and chunk characteristics
# MAGIC 4. ✅ **Comparison**: Analyzed trade-offs across strategies
# MAGIC 5. ✅ **Cleanup**: Released all resources (no persistent storage used)
# MAGIC
# MAGIC ## Key Takeaways
# MAGIC
# MAGIC ### Strategy Characteristics
# MAGIC
# MAGIC | Strategy | Best For | Pros | Cons |
# MAGIC |----------|----------|------|------|
# MAGIC | **Recursive Character** | General purpose | Respects natural breaks, flexible | May miss semantic boundaries |
# MAGIC | **Sentence-Based** | Q&A systems | Complete thoughts, clean boundaries | Can create very small/large chunks |
# MAGIC | **Semantic** | Topic-based search | Groups related content | Computationally expensive |
# MAGIC | **Markdown-Aware** | Technical docs | Preserves structure | Only for markdown documents |
# MAGIC | **Fixed-Size** | Simple pipelines | Fast, predictable | May break mid-sentence |
# MAGIC
# MAGIC ### Decision Framework
# MAGIC
# MAGIC **Choose your strategy based on:**
# MAGIC 1. **Document type**: Structured (use Markdown-Aware) vs Unstructured (use Recursive/Sentence)
# MAGIC 2. **Use case**: Q&A (Sentence-Based) vs Topic Search (Semantic) vs General (Recursive)
# MAGIC 3. **Performance needs**: Fast ingestion (Fixed-Size) vs Better quality (Semantic)
# MAGIC 4. **Domain**: Technical docs (Markdown-Aware) vs Narratives (Sentence-Based)
# MAGIC
# MAGIC ## Next Steps
# MAGIC
# MAGIC 1. **Test with your own data**: Replace `sample_document` with real documents from your domain
# MAGIC 2. **Tune parameters**: Adjust chunk sizes, overlap, and thresholds for your use case
# MAGIC 3. **Combine strategies**: Use Markdown-Aware for structure, then Semantic for content
# MAGIC 4. **Add reranking**: Use a cross-encoder to rerank retrieved chunks
# MAGIC 5. **Scale to production**: Integrate with Databricks Vector Search or Mosaic AI Agent Framework
# MAGIC 6. **Monitor quality**: Track retrieval metrics and iterate on your chunking strategy
# MAGIC
# MAGIC ## Advanced Techniques (Future Exploration)
# MAGIC
# MAGIC - **Context-aware chunking**: Add surrounding context to each chunk
# MAGIC - **Hierarchical chunking**: Parent-child relationships between chunks
# MAGIC - **Query-specific chunking**: Different strategies for different query types
# MAGIC - **Metadata enrichment**: Add source, section, timestamp to each chunk
# MAGIC - **Hybrid retrieval**: Combine dense (embeddings) + sparse (BM25) search
# MAGIC
# MAGIC ## Resources
# MAGIC
# MAGIC - [Databricks Vector Search](https://docs.databricks.com/en/generative-ai/vector-search.html)
# MAGIC - [Mosaic AI Agent Framework](https://docs.databricks.com/en/generative-ai/agent-framework/index.html)
# MAGIC - [LangChain Text Splitters](https://python.langchain.com/docs/modules/data_connection/document_transformers/)
# MAGIC - [Sentence Transformers](https://www.sbert.net/)
# MAGIC - [Chunking Strategies Research](https://www.pinecone.io/learn/chunking-strategies/)

# COMMAND ----------

