"""Basic Usage Example for Document Search Engine.

This script demonstrates basic operations:
- Initializing the search engine
- Indexing documents
- Performing searches
- Viewing results
"""

import sys
import os

# Add parent directory to path to import src modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.search_engine import DocumentSearchEngine


def main():
    """Main function demonstrating basic usage."""
    
    print("="*60)
    print("Document Search Engine - Basic Usage Example")
    print("="*60)
    print()
    
    # Step 1: Initialize the search engine
    print("Step 1: Initializing search engine...")
    engine = DocumentSearchEngine(
        index_dir="./example_index",
        log_level="INFO"
    )
    print("✓ Search engine initialized")
    print()
    
    # Step 2: Create sample documents for demo
    print("Step 2: Creating sample documents...")
    os.makedirs("./sample_docs", exist_ok=True)
    
    # Create sample markdown documents
    sample_docs = [
        {
            "name": "machine_learning.md",
            "content": """# Machine Learning

Machine learning is a subset of artificial intelligence that focuses on 
building systems that can learn from data. Deep learning is a specialized 
form of machine learning that uses neural networks.
"""
        },
        {
            "name": "python_programming.md",
            "content": """# Python Programming

Python is a high-level programming language known for its simplicity and 
readability. It's widely used in data science, machine learning, and web 
development.
"""
        },
        {
            "name": "data_science.md",
            "content": """# Data Science

Data science combines statistics, mathematics, and programming to extract 
insights from data. Python and R are popular languages for data science.
Machine learning is an important tool in the data scientist's toolkit.
"""
        }
    ]
    
    for doc in sample_docs:
        file_path = os.path.join("./sample_docs", doc["name"])
        with open(file_path, 'w') as f:
            f.write(doc["content"])
    
    print(f"✓ Created {len(sample_docs)} sample documents")
    print()
    
    # Step 3: Index the documents
    print("Step 3: Indexing documents...")
    stats = engine.index_directory("./sample_docs", recursive=False)
    print(f"✓ Indexed {stats['successful']} documents successfully")
    print(f"  - Total processed: {stats['total_processed']}")
    print(f"  - Failed: {stats['failed']}")
    print()
    
    # Step 4: Get index statistics
    print("Step 4: Index Statistics")
    index_stats = engine.get_statistics()
    print(f"  - Documents: {index_stats['document_count']}")
    print(f"  - Unique terms: {index_stats['unique_terms']}")
    print(f"  - Index size: {index_stats['index_size_mb']:.3f} MB")
    print(f"  - Avg doc length: {index_stats['avg_doc_length']:.1f} words")
    print()
    
    # Step 5: Perform searches
    print("Step 5: Performing searches...")
    print()
    
    queries = [
        "machine learning",
        "python programming",
        "data science"
    ]
    
    for query in queries:
        print(f"Query: '{query}'")
        print("-" * 60)
        
        results = engine.search(query, top_k=3)
        
        if results:
            for i, result in enumerate(results, 1):
                print(f"{i}. {result['file_name']}")
                print(f"   Score: {result['score']:.3f}")
                print(f"   Type: {result['file_type']}")
                print(f"   Matched terms: {', '.join(result['matched_terms'])}")
                
                # Show snippet
                snippet = engine.get_document_snippet(
                    result['doc_id'], 
                    query, 
                    context_length=80
                )
                print(f"   Snippet: {snippet}")
                print()
        else:
            print("   No results found.")
            print()
        
        print()
    
    # Step 6: List all indexed documents
    print("Step 6: All indexed documents")
    print("-" * 60)
    all_docs = engine.list_indexed_documents()
    for doc in all_docs:
        print(f"- {doc['file_name']} ({doc['file_type']})")
        print(f"  Words: {doc['word_count']}, Size: {doc['size']} bytes")
    print()
    
    # Step 7: Save the index
    print("Step 7: Saving index...")
    engine.save_index("demo_index")
    print("✓ Index saved successfully")
    print()
    
    # Step 8: Demonstrate loading
    print("Step 8: Loading saved index...")
    new_engine = DocumentSearchEngine(index_dir="./example_index")
    new_engine.load_index("demo_index")
    loaded_stats = new_engine.get_statistics()
    print(f"✓ Index loaded: {loaded_stats['document_count']} documents")
    print()
    
    print("="*60)
    print("Demo completed successfully!")
    print("="*60)
    print()
    print("Next steps:")
    print("1. Add your own documents to index")
    print("2. Try different search queries")
    print("3. Explore phrase search with phrase_search=True")
    print("4. Check out advanced_usage.py for more features")


if __name__ == "__main__":
    main()