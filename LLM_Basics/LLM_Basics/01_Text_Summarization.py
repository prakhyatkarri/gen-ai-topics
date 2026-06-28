# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# DBTITLE 1,Text Summarization with LLMs
# MAGIC %md
# MAGIC # Text Summarization with LLMs
# MAGIC
# MAGIC ## Learning Objectives
# MAGIC In this notebook, you'll learn:
# MAGIC - How to use free LLM models for text summarization
# MAGIC - Different summarization techniques (extractive vs. abstractive)
# MAGIC - Practical applications and best practices
# MAGIC
# MAGIC ## Structure
# MAGIC 1. **Setup**: Install required libraries and load models
# MAGIC 2. **Learning**: Explore summarization techniques
# MAGIC 3. **Cleanup**: Remove temporary resources

# COMMAND ----------

# DBTITLE 1,Setup - Install Dependencies
# MAGIC %md
# MAGIC ## 📦 Setup: Install Dependencies
# MAGIC
# MAGIC We'll use the Hugging Face Transformers library with free, open-source models.

# COMMAND ----------

# DBTITLE 1,Install transformers
# Install required libraries
%pip install transformers torch --quiet
dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Load Summarization Model
# MAGIC %md
# MAGIC ## 🤖 Load a Free Summarization Model
# MAGIC
# MAGIC We'll use **facebook/bart-large-cnn** - a free model optimized for summarization.

# COMMAND ----------

# DBTITLE 1,Initialize model
from transformers import pipeline
import warnings
warnings.filterwarnings('ignore')

# Load the summarization pipeline with a free model
print("Loading summarization model...")
# summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
summarizer = pipeline("text-generation", model="facebook/bart-large-cnn")
print("✅ Model loaded successfully!")

# COMMAND ----------

# DBTITLE 1,Learning - Basic Summarization
# MAGIC %md
# MAGIC ## 📚 Learning: Basic Summarization
# MAGIC
# MAGIC ### Example 1: Summarize a News Article

# COMMAND ----------

# DBTITLE 1,Example 1 - News article
# Sample text to summarize
text = """
Artificial intelligence has made remarkable progress in recent years, transforming industries 
from healthcare to finance. Machine learning algorithms can now diagnose diseases with accuracy 
comparable to human doctors, predict stock market trends, and even create art and music. 
However, this rapid advancement also raises important ethical questions about privacy, bias, 
and the future of work. Experts debate whether AI will augment human capabilities or replace 
human workers entirely. Governments and organizations worldwide are developing frameworks to 
ensure AI is developed and deployed responsibly, balancing innovation with societal concerns.
"""

# Generate summary
summary = summarizer(text, max_length=50, min_length=25, do_sample=False)
print("Original Text Length:", len(text.split()), "words")
print("\nSummary:")
print(summary[0]['generated_text'])
# print(f"Summary: {summary}")
print("\nSummary Length:", len(summary[0]['generated_text'].split()), "words")

# COMMAND ----------

# DBTITLE 1,Advanced Techniques
# MAGIC %md
# MAGIC ### Example 2: Adjusting Summary Length
# MAGIC
# MAGIC You can control the summary length using `max_length` and `min_length` parameters.

# COMMAND ----------

# DBTITLE 1,Example 2 - Length control
# Long technical document
long_text = """
The Transformer architecture, introduced in the paper "Attention is All You Need", revolutionized 
natural language processing. Unlike previous recurrent neural networks, Transformers process all 
tokens in parallel using self-attention mechanisms. This allows them to capture long-range 
dependencies more effectively. The architecture consists of an encoder and decoder, each with 
multiple layers of multi-head attention and feed-forward networks. Positional encoding is added 
to input embeddings to preserve sequence information. BERT, GPT, and other modern language models 
are all based on Transformer architecture. The key innovation is the attention mechanism, which 
computes weighted relationships between all tokens in a sequence, allowing the model to focus on 
relevant context.
"""

# Generate different summary lengths
print("Short Summary (30 words):")
short = summarizer(long_text, max_length=30, min_length=20, do_sample=False)
print(short[0]['generated_text'])

print("\n" + "="*60)
print("\nLonger Summary (60 words):")
long = summarizer(long_text, max_length=60, min_length=40, do_sample=False)
print(long[0]['generated_text'])

# COMMAND ----------

# DBTITLE 1,Batch Processing
# MAGIC %md
# MAGIC ### Example 3: Batch Summarization
# MAGIC
# MAGIC Process multiple documents at once for efficiency.

# COMMAND ----------

# DBTITLE 1,Parameter Tuning
# MAGIC %md
# MAGIC ## 🏛️ Parameter Tuning: Understanding Model Behavior
# MAGIC
# MAGIC ### 1. Temperature Comparison
# MAGIC
# MAGIC Temperature controls randomness in outputs. Lower = more focused, Higher = more creative.

# COMMAND ----------

# DBTITLE 1,Temperature comparison
# Compare different temperature settings
test_text = """
Quantum computing represents a paradigm shift in computational power. Unlike classical computers 
that use bits, quantum computers use qubits that can exist in multiple states simultaneously. 
This allows them to solve certain problems exponentially faster than traditional computers.
"""

temperatures = [0.3, 0.7, 1.0]

print("Temperature Comparison for Summarization:")
print("="*60)

for temp in temperatures:
    summary = summarizer(
        test_text, 
        max_length=40, 
        min_length=20, 
        do_sample=True,  # Enable sampling
        temperature=temp
    )
    print(f"\nTemperature: {temp}")
    print(f"Summary: {summary[0]['generated_text']}")
    print("-"*60)

print("""

💡 Temperature Insights:
- Low (0.3): More deterministic, focused, consistent summaries
- Medium (0.7): Balanced between consistency and variation
- High (1.0): More creative, diverse summaries (may be less focused)
""")

# COMMAND ----------

# DBTITLE 1,Max Tokens
# MAGIC %md
# MAGIC ### 2. Max Tokens Comparison
# MAGIC
# MAGIC Max tokens (max_length) controls the output length.

# COMMAND ----------

# DBTITLE 1,Max tokens comparison
# Compare different max_length settings
long_article = """
Artificial intelligence and machine learning have transformed numerous industries over the past decade. 
From healthcare diagnostics to autonomous vehicles, these technologies are reshaping how we work and live. 
Deep learning models, powered by neural networks, can now perform tasks that were once thought impossible 
for machines. However, challenges remain, including ethical considerations, bias in algorithms, and the 
need for massive computational resources. Researchers continue to push boundaries, developing more efficient 
algorithms and exploring new applications in fields like natural language processing and computer vision.
"""

max_lengths = [30, 60, 100]

print("Max Tokens Comparison:")
print("="*60)

for max_len in max_lengths:
    summary = summarizer(
        long_article, 
        max_length=max_len, 
        min_length=int(max_len * 0.5), 
        do_sample=False
    )
    print(f"\nMax Length: {max_len} tokens")
    print(f"Summary: {summary[0]['generated_text']}")
    print(f"Actual Length: {len(summary[0]['generated_text'].split())} words")
    print("-"*60)

print("""

💡 Max Tokens Insights:
- Short (30): Brief, high-level overview
- Medium (60): Balanced detail and brevity
- Long (100): More comprehensive, detailed summary
""")

# COMMAND ----------

# DBTITLE 1,Top P
# MAGIC %md
# MAGIC ### 3. Top P (Nucleus Sampling) Comparison
# MAGIC
# MAGIC Top P controls diversity by selecting from the smallest set of tokens whose cumulative probability exceeds P.

# COMMAND ----------

# DBTITLE 1,Top P comparison
# Compare different top_p settings
test_article = """
Climate change is one of the most pressing challenges of our time. Rising temperatures, extreme 
weather events, and melting ice caps are clear indicators of global warming. Scientists urge 
immediate action to reduce carbon emissions and transition to renewable energy sources.
"""

top_p_values = [0.5, 0.9, 1.0]

print("Top P Comparison:")
print("="*60)

for top_p in top_p_values:
    summary = summarizer(
        test_article, 
        max_length=40, 
        min_length=20, 
        do_sample=True,
        top_p=top_p,
        temperature=0.8
    )
    print(f"\nTop P: {top_p}")
    print(f"Summary: {summary[0]['summary_text']}")
    print("-"*60)

print("""

💡 Top P Insights:
- Low (0.5): More conservative word choices, focused on high-probability tokens
- Medium (0.9): Balanced diversity (recommended default)
- High (1.0): Maximum diversity, considers all possible tokens

🎯 Best Practice: Use top_p=0.9 with temperature=0.7 for balanced results
""")

# COMMAND ----------

amazon_reviews = [
    "Purchased for my granddaughter at Christmas. She had not quite turned 2. Her legs could not reach the floor. And the whole door thing, she was not interested in using. Preferred to crawl inside on the other side, LOL! It makes her happy. Especially when Daddy pushes her chasing her older brother. Adorable and long lasting. Her daddy had one of the original styles at her age.",
    "O.M.G!!!!! Just put this together..The packages that had the screws and washers and miscellaneous parts were not marked. So I had to keep referring to the parts page. And to make matters worse there was a part I needed NOT in the bags at all. It’s funny because the instructions read that it should take only 50 minutes to put together. 3 hours later, finally. Next day the decals started peeling off. I’m going to use some craft glue and glue them on. Very cute though and I believe my granddaughter will love it. Just a suggestion for parents if you’re going to put together many cozy coupe to give yourself extra time because it never goes the way you plan it to go.",
    "I ordered this for my 21 month old daughter and it's a big hit. She plays in it nearly every day and I also enjoy pushing her around in it and seeing her have fun! There's the nostalgia of the classic Little Tyke's car but this is a fun, unique version - the springy little antennas are super cute. I like that the roof has a handle built into it for easy grip and the car comes with an easily attachable/detachable floor board so kiddo's legs don't drag along the ground if you're doing the pushing. It definitely takes some patience and time to put all together but it's well worth it in my opinion. One thing I found though is putting the spot stickers on you really need to press them on there extra hard. I didn't do it thoroughly enough and they started bubbling and coming off a few days later on ours. A firm smoothing over fixed them all thankfully."
]

review_summaries = summarizer(amazon_reviews, max_length=100, min_length=5, do_sample=False)

for i in review_summaries:
    j = i[0]['generated_text']
    print(f"Summary: {j}")
    print(f"Summary Length: {len(j)}")
    print("-" * 60)

# COMMAND ----------

# DBTITLE 1,Example 3 - Batch processing
# Multiple documents
documents = [
    "Climate change is causing rising sea levels and extreme weather events worldwide. Scientists urge immediate action.",
    "The new smartphone features a revolutionary camera system with AI-enhanced photography and extended battery life.",
    "Recent studies show that regular exercise improves mental health and cognitive function in adults of all ages."
]

# Summarize all documents
summaries = summarizer(documents, max_length=5, min_length=1, do_sample=False)

print(f"Summaries: {summaries}")

# for i, (doc, summary) in enumerate(zip(documents, summaries), 1):
for i, doc in enumerate(zip(documents, summaries), 1):
    # generated_text = summary[0]['generated_text']
    
    print(f"Document {i}:")
    print(f"Original: {doc}")
    # print(f"Summary: {summary}")
    print("-" * 60)

# COMMAND ----------

# DBTITLE 1,Cleanup
# MAGIC %md
# MAGIC ## 🧹 Cleanup
# MAGIC
# MAGIC Free up memory by deleting the model.

# COMMAND ----------

# DBTITLE 1,Delete model
# Clean up resources
import gc

del summarizer
gc.collect()

print("✅ Cleanup complete! Model removed from memory.")

# COMMAND ----------

# DBTITLE 1,Key Takeaways
# MAGIC %md
# MAGIC ## 🎯 Key Takeaways
# MAGIC
# MAGIC 1. **Free models** like BART can perform high-quality summarization
# MAGIC 2. **Control length** using `max_length` and `min_length` parameters
# MAGIC 3. **Batch processing** improves efficiency for multiple documents
# MAGIC 4. **Abstractive summarization** creates new sentences rather than extracting existing ones
# MAGIC
# MAGIC ## 💡 Next Steps
# MAGIC
# MAGIC - Try different models (e.g., `t5-small`, `google/pegasus-xsum`)
# MAGIC - Experiment with domain-specific texts (legal, medical, technical)
# MAGIC - Explore extractive summarization techniques

# COMMAND ----------



# COMMAND ----------

