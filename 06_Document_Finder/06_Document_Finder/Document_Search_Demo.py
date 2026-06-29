# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# DBTITLE 1,Introduction
# MAGIC %md
# MAGIC # Document Search Engine - Databricks Demo
# MAGIC
# MAGIC This notebook demonstrates how to use the Document Search Engine in Databricks workspace to search across PDFs, Word documents, and Markdown files.
# MAGIC
# MAGIC ## Features
# MAGIC - Multi-format support (PDF, Word, Markdown)
# MAGIC - TF-IDF based ranking
# MAGIC - Phrase search capability
# MAGIC - Persistent index storage

# COMMAND ----------

# DBTITLE 1,Install Dependencies
# Install required dependencies
%pip install PyPDF2>=3.0.0 python-docx>=1.1.0 markdown>=3.5.0 pyyaml>=6.0.0
dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Setup Paths and Imports
import sys
import os

# Add project to Python path
project_path = '/Workspace/Users/prakhyatkarri@gmail.com/06_Document_Finder'
sys.path.insert(0, project_path)

# Import search engine
from src.search_engine import DocumentSearchEngine

print("✓ Document Search Engine imported successfully!")

# COMMAND ----------

# DBTITLE 1,Create Sample Documents - Header
# MAGIC %md
# MAGIC ## Step 1: Create Sample Documents
# MAGIC
# MAGIC Let's create some sample documents to index and search.

# COMMAND ----------

# DBTITLE 1,Create Sample Documents
# Create directory for sample documents
sample_dir = "/Workspace/Users/prakhyatkarri@gmail.com/06_Document_Finder/demo_docs"
os.makedirs(sample_dir, exist_ok=True)

# Create sample markdown documents
sample_docs = {
    "machine_learning.md": """# Machine Learning Overview

Machine learning is a subset of artificial intelligence that focuses on building systems that can learn from data. It uses algorithms to identify patterns and make predictions without being explicitly programmed.

## Types of Machine Learning
- Supervised Learning
- Unsupervised Learning
- Reinforcement Learning

Deep learning is a specialized form of machine learning using neural networks.
""",
    
    "python_programming.md": """# Python Programming Guide

Python is a high-level, interpreted programming language known for its simplicity and readability. It's widely used in:
- Data Science
- Machine Learning
- Web Development
- Automation

Python's extensive library ecosystem makes it ideal for rapid development.
""",
    
    "data_science.md": """# Introduction to Data Science

Data science combines statistics, mathematics, and programming to extract insights from data. Key skills include:
- Statistical analysis
- Data visualization
- Machine learning algorithms
- Python and R programming

Data scientists use machine learning to build predictive models and uncover patterns in large datasets.
""",
    
    "artificial_intelligence.md": """# Artificial Intelligence

Artificial intelligence (AI) is the simulation of human intelligence by machines. AI includes:
- Natural Language Processing
- Computer Vision
- Machine Learning
- Robotics

AI systems can perform tasks that typically require human intelligence, such as visual perception, speech recognition, and decision-making.
"""
}

# Write sample documents
for filename, content in sample_docs.items():
    filepath = os.path.join(sample_dir, filename)
    with open(filepath, 'w') as f:
        f.write(content)

print(f"✓ Created {len(sample_docs)} sample documents in {sample_dir}")
print("\nDocuments created:")
for filename in sample_docs.keys():
    print(f"  - {filename}")

# COMMAND ----------

# DBTITLE 1,Initialize Search Engine - Header
# MAGIC %md
# MAGIC ## Step 2: Initialize Search Engine
# MAGIC
# MAGIC Initialize the search engine with a workspace path for the index.

# COMMAND ----------

# DBTITLE 1,Initialize Search Engine
# Initialize search engine with workspace path
index_dir = "/Workspace/Users/prakhyatkarri@gmail.com/06_Document_Finder/demo_index"

engine = DocumentSearchEngine(
    index_dir=index_dir,
    log_level="INFO"
)

print("✓ Search engine initialized")
print(f"  Index directory: {index_dir}")

# COMMAND ----------

# DBTITLE 1,Index Documents - Header
# MAGIC %md
# MAGIC ## Step 3: Index Documents
# MAGIC
# MAGIC Index all documents in the sample directory.

# COMMAND ----------

# DBTITLE 1,Index Documents
# Index the sample documents
print("Indexing documents...")
stats = engine.index_directory(sample_dir, recursive=False, save=True)

print("\n" + "="*60)
print("Indexing Results")
print("="*60)
print(f"Total processed: {stats['total_processed']}")
print(f"Successfully indexed: {stats['successful']}")
print(f"Failed: {stats['failed']}")
print()

# Show index statistics
index_stats = engine.get_statistics()
print("Index Statistics:")
print(f"  - Documents: {index_stats['document_count']}")
print(f"  - Unique terms: {index_stats['unique_terms']}")
print(f"  - Index size: {index_stats['index_size_mb']:.3f} MB")
print(f"  - Avg document length: {index_stats['avg_doc_length']:.1f} words")

# COMMAND ----------

# DBTITLE 1,Search Documents - Header
# MAGIC %md
# MAGIC ## Step 4: Search Documents
# MAGIC
# MAGIC Perform various searches to find relevant documents.

# COMMAND ----------

# DBTITLE 1,Perform Searches
# Define search queries
queries = [
    "machine learning",
    "python programming",
    "artificial intelligence"
]

print("="*60)
print("Search Results")
print("="*60)

for query in queries:
    print(f"\n🔍 Query: '{query}'")
    print("-" * 60)
    
    results = engine.search(query, top_k=3)
    
    if results:
        for i, result in enumerate(results, 1):
            print(f"\n{i}. {result['file_name']}")
            print(f"   Score: {result['score']:.3f}")
            print(f"   Type: {result['file_type']}")
            print(f"   Word count: {result['word_count']}")
            print(f"   Matched terms: {', '.join(result['matched_terms'])}")
            
            # Show snippet
            snippet = engine.get_document_snippet(
                result['doc_id'], 
                query, 
                context_length=100
            )
            print(f"   Snippet: {snippet[:150]}...")
    else:
        print("   No results found.")
    
    print()

# COMMAND ----------

# DBTITLE 1,Phrase Search - Header
# MAGIC %md
# MAGIC ## Step 5: Phrase Search
# MAGIC
# MAGIC Demonstrate exact phrase matching.

# COMMAND ----------

# DBTITLE 1,Phrase Search
# Phrase search for exact matches
phrase = "machine learning"

print("="*60)
print(f"Phrase Search: '{phrase}'")
print("="*60)

phrase_results = engine.search(phrase, phrase_search=True, top_k=5)

print(f"\nFound {len(phrase_results)} documents with exact phrase match:\n")

for i, result in enumerate(phrase_results, 1):
    print(f"{i}. {result['file_name']}")
    print(f"   Score: {result['score']:.3f}")
    
    # Get context around the phrase
    snippet = engine.get_document_snippet(result['doc_id'], phrase, context_length=80)
    print(f"   Context: {snippet}")
    print()

# COMMAND ----------

# DBTITLE 1,List Documents - Header
# MAGIC %md
# MAGIC ## Step 6: List All Indexed Documents
# MAGIC
# MAGIC View all documents currently in the index.

# COMMAND ----------

# DBTITLE 1,List All Documents
# List all indexed documents
print("="*60)
print("All Indexed Documents")
print("="*60)

all_docs = engine.list_indexed_documents()

for i, doc in enumerate(all_docs, 1):
    print(f"\n{i}. {doc['file_name']}")
    print(f"   Type: {doc['file_type']}")
    print(f"   Words: {doc['word_count']}")
    print(f"   Size: {doc['size']} bytes")

# COMMAND ----------

# DBTITLE 1,Custom Documents - Header
# MAGIC %md
# MAGIC ## Step 7: Working with Your Own Documents
# MAGIC
# MAGIC Here's how to use the search engine with your own documents.

# COMMAND ----------

# DBTITLE 1,Custom Documents Example
# Example: Index documents from DBFS or Workspace

# Option 1: Index from workspace directory
# engine.index_directory("/Workspace/Users/your_email/your_documents")

# Option 2: Index from DBFS
# engine.index_directory("/dbfs/FileStore/documents")

# Option 3: Index from Unity Catalog Volume
# engine.index_directory("/Volumes/catalog/schema/volume/documents")

# Option 4: Index a single document
# engine.index_document("/path/to/your/document.pdf")

# Save the index for future use
# engine.save_index("my_custom_index")

# Load a previously saved index
# engine.load_index("my_custom_index")

print("💡 Tips:")
print("  - Use workspace paths: /Workspace/Users/...")
print("  - Use DBFS paths: /dbfs/...")
print("  - Use UC Volume paths: /Volumes/catalog/schema/volume/...")
print("  - Save indices to avoid re-indexing")
print("  - Supported formats: PDF, DOCX, DOC, MD, MARKDOWN")

# COMMAND ----------

# DBTITLE 1,Summary
# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC You've successfully learned how to:
# MAGIC - ✓ Install dependencies in Databricks
# MAGIC - ✓ Initialize the Document Search Engine
# MAGIC - ✓ Index documents from workspace
# MAGIC - ✓ Perform keyword searches
# MAGIC - ✓ Use phrase search for exact matches
# MAGIC - ✓ View index statistics
# MAGIC - ✓ Save and load indices
# MAGIC
# MAGIC ## Next Steps
# MAGIC
# MAGIC 1. **Index your own documents**: Point the engine to your document directory
# MAGIC 2. **Customize configuration**: Edit `config/config.yaml` for advanced settings
# MAGIC 3. **Run example scripts**: Check `/examples` directory for more use cases
# MAGIC 4. **Run tests**: Execute `pytest tests/` to verify functionality
# MAGIC
# MAGIC ## Resources
# MAGIC
# MAGIC - README: `/Workspace/Users/prakhyatkarri@gmail.com/06_Document_Finder/README.md`
# MAGIC - Examples: `/Workspace/Users/prakhyatkarri@gmail.com/06_Document_Finder/examples/`
# MAGIC - Configuration: `/Workspace/Users/prakhyatkarri@gmail.com/06_Document_Finder/config/config.yaml`