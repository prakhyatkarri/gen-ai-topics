"""Advanced Usage Example for Document Search Engine.

Demonstrates:
- Phrase search
- Custom configuration
- Document management
- Error handling
- Performance monitoring
"""

import sys
import os
import time
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.search_engine import DocumentSearchEngine
from src.document_processor import DocumentProcessor


def demonstrate_phrase_search():
    """Demonstrate exact phrase matching."""
    print("\n" + "="*60)
    print("Phrase Search Demo")
    print("="*60)
    
    engine = DocumentSearchEngine(index_dir="./advanced_index")
    
    # Create documents with specific phrases
    os.makedirs("./phrase_docs", exist_ok=True)
    
    docs = [
        ("ai_trends.md", "Artificial intelligence and machine learning are transforming industries."),
        ("ml_basics.md", "Machine learning is a subset of artificial intelligence."),
        ("dl_intro.md", "Deep learning uses neural networks for artificial intelligence applications.")
    ]
    
    for filename, content in docs:
        with open(f"./phrase_docs/{filename}", 'w') as f:
            f.write(content)
    
    engine.index_directory("./phrase_docs")
    
    # Regular search vs phrase search
    query = "artificial intelligence"
    
    print(f"\nRegular search for: '{query}'")
    results = engine.search(query, phrase_search=False)
    print(f"Found {len(results)} results")
    
    print(f"\nPhrase search for: '{query}'")
    phrase_results = engine.search(query, phrase_search=True)
    print(f"Found {len(phrase_results)} results (exact matches only)")
    
    for result in phrase_results:
        print(f"  - {result['file_name']}")


def demonstrate_custom_config():
    """Demonstrate loading custom configuration."""
    print("\n" + "="*60)
    print("Custom Configuration Demo")
    print("="*60)
    
    # Load configuration
    config_path = os.path.join(
        os.path.dirname(__file__), 
        '..', 
        'config', 
        'config.yaml'
    )
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print("\nLoaded configuration:")
    print(f"  - Index directory: {config['index']['directory']}")
    print(f"  - Default top_k: {config['search']['default_top_k']}")
    print(f"  - Log level: {config['logging']['level']}")
    print(f"  - Max file size: {config['performance']['max_file_size_mb']} MB")
    
    # Initialize with config
    engine = DocumentSearchEngine(
        index_dir=config['index']['directory'],
        log_level=config['logging']['level']
    )
    
    print("\n✓ Engine initialized with custom config")


def demonstrate_document_management():
    """Demonstrate adding and removing documents."""
    print("\n" + "="*60)
    print("Document Management Demo")
    print("="*60)
    
    engine = DocumentSearchEngine(index_dir="./mgmt_index")
    
    os.makedirs("./mgmt_docs", exist_ok=True)
    
    # Create initial documents
    for i in range(5):
        with open(f"./mgmt_docs/doc{i}.md", 'w') as f:
            f.write(f"# Document {i}\n\nContent for document number {i}.")
    
    # Index all
    print("\nIndexing initial documents...")
    stats = engine.index_directory("./mgmt_docs")
    print(f"✓ Indexed {stats['successful']} documents")
    
    # List documents
    print("\nCurrent documents:")
    for doc in engine.list_indexed_documents():
        print(f"  - {doc['file_name']}")
    
    # Remove a document
    print("\nRemoving doc2.md...")
    engine.remove_document("./mgmt_docs/doc2.md")
    
    # Show updated list
    print("\nUpdated documents:")
    for doc in engine.list_indexed_documents():
        print(f"  - {doc['file_name']}")
    
    # Add new document
    print("\nAdding new document...")
    with open("./mgmt_docs/new_doc.md", 'w') as f:
        f.write("# New Document\n\nThis is a newly added document.")
    
    engine.index_document("./mgmt_docs/new_doc.md")
    
    print("\nFinal documents:")
    for doc in engine.list_indexed_documents():
        print(f"  - {doc['file_name']}")


def demonstrate_error_handling():
    """Demonstrate proper error handling."""
    print("\n" + "="*60)
    print("Error Handling Demo")
    print("="*60)
    
    engine = DocumentSearchEngine(index_dir="./error_index")
    
    # Try to index non-existent file
    print("\nAttempting to index non-existent file...")
    try:
        engine.index_document("nonexistent.pdf")
    except FileNotFoundError as e:
        print(f"✓ Caught expected error: {type(e).__name__}")
    
    # Try to index unsupported format
    print("\nAttempting to index unsupported format...")
    os.makedirs("./error_docs", exist_ok=True)
    with open("./error_docs/test.xyz", 'w') as f:
        f.write("test")
    
    try:
        engine.index_document("./error_docs/test.xyz")
    except ValueError as e:
        print(f"✓ Caught expected error: {type(e).__name__}")
    
    # Search on empty index
    print("\nSearching empty index...")
    results = engine.search("test query")
    print(f"✓ Returned {len(results)} results (empty list)")


def demonstrate_performance_monitoring():
    """Demonstrate performance monitoring."""
    print("\n" + "="*60)
    print("Performance Monitoring Demo")
    print("="*60)
    
    engine = DocumentSearchEngine(index_dir="./perf_index")
    
    os.makedirs("./perf_docs", exist_ok=True)
    
    # Create multiple documents
    num_docs = 20
    print(f"\nCreating {num_docs} documents...")
    
    for i in range(num_docs):
        content = f"# Document {i}\n\n"
        content += f"This is document number {i}. "
        content += f"It contains various keywords like machine learning, "
        content += f"artificial intelligence, data science, and programming."
        
        with open(f"./perf_docs/doc{i}.md", 'w') as f:
            f.write(content)
    
    # Time indexing
    print(f"\nIndexing {num_docs} documents...")
    start_time = time.time()
    stats = engine.index_directory("./perf_docs")
    index_time = time.time() - start_time
    
    print(f"✓ Indexed {stats['successful']} documents in {index_time:.3f} seconds")
    print(f"  - Average: {index_time/num_docs:.4f} seconds per document")
    print(f"  - Throughput: {num_docs/index_time:.1f} documents per second")
    
    # Time search
    print("\nPerforming search benchmarks...")
    queries = ["machine learning", "data science", "programming"]
    
    search_times = []
    for query in queries:
        start_time = time.time()
        results = engine.search(query)
        search_time = time.time() - start_time
        search_times.append(search_time)
        
        print(f"  - '{query}': {search_time*1000:.2f}ms ({len(results)} results)")
    
    avg_search_time = sum(search_times) / len(search_times)
    print(f"\nAverage search time: {avg_search_time*1000:.2f}ms")
    
    # Index statistics
    index_stats = engine.get_statistics()
    print(f"\nIndex Statistics:")
    print(f"  - Documents: {index_stats['document_count']}")
    print(f"  - Unique terms: {index_stats['unique_terms']}")
    print(f"  - Index size: {index_stats['index_size_mb']:.3f} MB")
    print(f"  - Avg document length: {index_stats['avg_doc_length']:.1f} words")


def demonstrate_document_processor():
    """Demonstrate direct DocumentProcessor usage."""
    print("\n" + "="*60)
    print("DocumentProcessor Direct Usage Demo")
    print("="*60)
    
    processor = DocumentProcessor()
    
    print("\nSupported formats:")
    for ext, doc_type in processor.SUPPORTED_FORMATS.items():
        print(f"  - {ext}: {doc_type}")
    
    # Create sample files
    os.makedirs("./processor_docs", exist_ok=True)
    
    with open("./processor_docs/test.md", 'w') as f:
        f.write("# Test\n\nThis is a test document.")
    
    # Process single document
    print("\nProcessing single document...")
    doc_data = processor.process_document("./processor_docs/test.md")
    
    print("\nDocument metadata:")
    print(f"  - File: {doc_data['file_name']}")
    print(f"  - Type: {doc_data['file_type']}")
    print(f"  - Size: {doc_data['size']} bytes")
    print(f"  - Words: {doc_data['word_count']}")
    print(f"  - Characters: {doc_data['char_count']}")
    print(f"\nContent preview:")
    print(f"  {doc_data['content'][:100]}...")


def main():
    """Run all advanced demos."""
    print("\n" + "#"*60)
    print("# Document Search Engine - Advanced Usage Examples")
    print("#"*60)
    
    try:
        demonstrate_phrase_search()
        demonstrate_custom_config()
        demonstrate_document_management()
        demonstrate_error_handling()
        demonstrate_performance_monitoring()
        demonstrate_document_processor()
        
        print("\n" + "#"*60)
        print("# All demos completed successfully!")
        print("#"*60)
        print()
        
    except Exception as e:
        print(f"\nError during demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()