# Enterprise Knowledge Assistant (RAG)

## Architecture

```

Documents
     │
     ▼
Ingestion
     │
     ▼
Delta Table
     │
     ▼
Chunking
     │
     ▼
Embeddings
     │
     ▼
Vector Search
     │
     ▼
RAG Pipeline
     │
     ▼
Foundation Model
     │
     ▼
REST API

```

## Repository

```

enterprise-rag-assistant/

README.md

docs/
    architecture.md
    setup.md

datasets/
    sample_docs/

notebooks/
    01_setup.py
    02_ingestion.py
    03_chunking.py
    04_embeddings.py
    05_vector_search.py
    06_rag.py
    07_evaluation.py
    08_serving.py

src/
    ingestion.py
    chunking.py
    retrieval.py
    rag.py
    prompts.py
    evaluation.py

tests/
    test_chunking.py
    test_retrieval.py

screenshots/

demo/

```

## Environment Setup
### Goal

Create the project foundation.

### Deliverables
GitHub repository
Databricks workspace
Unity Catalog catalog/schema
Sample PDFs
Project README
Architecture diagram

Estimated time: 3–4 hours

## Data Ingestion
### Goal

Load documents into Delta.

### Deliverables
Read PDFs and Markdown
Extract text
Create Delta table
Store basic metadata (filename, path, page count)

### Skills demonstrated

Spark
Delta Lake
Unity Catalog

## Chunking & Embeddings
### Goal

Prepare documents for retrieval.

### Deliverables
Split documents into chunks
Generate embeddings
Store chunk metadata

### Skills demonstrated

AI preprocessing
Embedding generation
Data modeling

## Vector Search & RAG
### Goal

Answer questions using retrieved context.

### Deliverables
Create Vector Search index
Retrieve top-k chunks
Build prompt
Generate answers with citations

### Skills demonstrated

Retrieval-Augmented Generation
Prompt construction
Vector Search

## Evaluation
### Goal

Measure answer quality.

### Deliverables
Small evaluation dataset (10–20 questions)
Record experiments with MLflow
Compare prompt variations
Document findings

### Skills demonstrated

MLflow
GenAI evaluation
Experiment tracking

## Model Serving & API
### Goal

Expose the assistant.

### Deliverables
Databricks Model Serving endpoint
Simple REST API
API usage examples

### Skills demonstrated

Deployment
Inference
API integration

## Polish & Portfolio
### Goal

Make the project presentation-ready.

### Deliverables
Clean README
Architecture diagram
Screenshots
Demo video (3–5 minutes)
Future improvements section

### Skills demonstrated

Communication
Documentation
Portfolio presentation

## Deliverables Checklist
### Code
Ingestion notebook
Chunking notebook
Embedding notebook
Vector Search notebook
RAG notebook
Evaluation notebook
Serving notebook

### Documentation
README
Architecture
Setup guide
Demo guide

### AI Components
Chunking
Embeddings
Vector Search
RAG
Prompt templates

### Databricks Features
Unity Catalog
Delta Lake
Vector Search
Foundation Models
MLflow
Model Serving

### Portfolio Assets
GitHub repository
Screenshots
Demo video
Architecture diagram

### Free Resources
Official Databricks documentation
Microsoft Learn for Azure Databricks
MLflow documentation
Delta Lake documentation
Public PDFs from government agencies or open documentation (e.g., FDA guidance, Apache Spark docs)



