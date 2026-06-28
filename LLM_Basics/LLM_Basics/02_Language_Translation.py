# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# DBTITLE 1,Language Translation with LLMs
# MAGIC %md
# MAGIC # Language Translation with LLMs
# MAGIC
# MAGIC ## Learning Objectives
# MAGIC In this notebook, you'll learn:
# MAGIC - How to use free translation models
# MAGIC - Translate between multiple languages
# MAGIC - Handle different text formats
# MAGIC - Best practices for translation tasks
# MAGIC
# MAGIC ## Structure
# MAGIC 1. **Setup**: Install libraries and load models
# MAGIC 2. **Learning**: Practice translation techniques
# MAGIC 3. **Cleanup**: Remove resources

# COMMAND ----------

# DBTITLE 1,Setup - Install Dependencies
# MAGIC %md
# MAGIC ## 📦 Setup: Install Dependencies
# MAGIC
# MAGIC We'll use Helsinki-NLP's OPUS-MT models - free, high-quality translation models.

# COMMAND ----------

# DBTITLE 1,Install libraries
# Install required libraries
%pip install transformers==4.56.0 sentencepiece torch --quiet
dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Load Translation Model
# MAGIC %md
# MAGIC ## 🌍 Load Translation Models
# MAGIC
# MAGIC We'll start with English to French translation.

# COMMAND ----------

# DBTITLE 1,Initialize model
from transformers import pipeline
import warnings
warnings.filterwarnings('ignore')

# Load English to French translator
print("Loading translation model (en -> fr)...")
translator_en_fr = pipeline("translation_en_to_fr", model="Helsinki-NLP/opus-mt-en-fr")
# translator_en_fr = pipeline("translation", model="Helsinki-NLP/opus-mt-en-fr")
print("✅ Model loaded successfully!")

# COMMAND ----------

# DBTITLE 1,Basic Translation
# MAGIC %md
# MAGIC ## 📚 Learning: Basic Translation
# MAGIC
# MAGIC ### Example 1: Simple English to French

# COMMAND ----------

# DBTITLE 1,Example 1 - Basic translation
# Translate simple sentences
sentences = [
    "Hello, how are you?",
    "I love learning about artificial intelligence.",
    "The weather is beautiful today."
]

for sentence in sentences:
    translation = translator_en_fr(sentence)
    print(f"English: {sentence}")
    print(f"French: {translation[0]['translation_text']}")
    print("-" * 60)

# COMMAND ----------

# DBTITLE 1,Multiple Language Pairs
# MAGIC %md
# MAGIC ### Example 2: Multiple Language Pairs
# MAGIC
# MAGIC Let's load additional models for different language pairs.

# COMMAND ----------

# DBTITLE 1,Example 2 - Multiple languages
# Load additional translators
print("Loading additional translation models...")

translator_en_es = pipeline("translation_en_to_es", model="Helsinki-NLP/opus-mt-en-es")
print("✅ English -> Spanish loaded")

translator_en_de = pipeline("translation_en_to_de", model="Helsinki-NLP/opus-mt-en-de")
print("✅ English -> German loaded")

# Translate to multiple languages
text = "Machine learning is transforming the world."

print("\nOriginal (English):", text)
print("French:", translator_en_fr(text)[0]['translation_text'])
print("Spanish:", translator_en_es(text)[0]['translation_text'])
print("German:", translator_en_de(text)[0]['translation_text'])

# COMMAND ----------

# DBTITLE 1,Long Text Translation
# MAGIC %md
# MAGIC ### Example 3: Translating Longer Text
# MAGIC
# MAGIC Handle paragraphs and longer documents.

# COMMAND ----------

# DBTITLE 1,Example 3 - Long text
# Longer text example
long_text = """
Artificial intelligence is rapidly advancing. Companies worldwide are investing heavily in AI research. 
These technologies promise to revolutionize healthcare, education, and transportation. However, ethical 
considerations must guide development to ensure AI benefits all of humanity.
"""

print("Original text (English):")
print(long_text)
print("\n" + "="*60 + "\n")

print("Translation (Spanish):")
translation = translator_en_es(long_text, max_length=512)
print(translation[0]['translation_text'])

# COMMAND ----------

# DBTITLE 1,Batch Translation
# MAGIC %md
# MAGIC ### Example 4: Batch Translation
# MAGIC
# MAGIC Translate multiple sentences efficiently.

# COMMAND ----------

# DBTITLE 1,Parameter Tuning
# MAGIC %md
# MAGIC ## 🏛️ Parameter Tuning: Understanding Model Behavior
# MAGIC
# MAGIC ### 1. Temperature Comparison
# MAGIC
# MAGIC Temperature affects translation creativity and variation.

# COMMAND ----------

# DBTITLE 1,Temperature comparison
# Load a model for testing parameters
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model_name = "Helsinki-NLP/opus-mt-en-fr"
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer_test = AutoTokenizer.from_pretrained(model_name)

test_sentence = "The future of technology is incredibly exciting and full of possibilities."

temperatures = [0.3, 0.7, 1.2]

print("Temperature Comparison for Translation:")
print("="*60)
print(f"English: {test_sentence}\n")

for temp in temperatures:
    inputs = tokenizer_test(test_sentence, return_tensors="pt", padding=True)
    outputs = model.generate(
        **inputs,
        max_length=128,
        do_sample=True,
        temperature=temp,
        num_return_sequences=1
    )
    translation = tokenizer_test.decode(outputs[0], skip_special_tokens=True)
    print(f"Temperature: {temp}")
    print(f"French: {translation}")
    print("-"*60)

print("""
💡 Temperature Insights:
- Low (0.3): More literal, consistent translations
- Medium (0.7): Balanced, natural translations
- High (1.2): More creative, varied translations (may be less accurate)
""")

# Cleanup temp model
import gc
del model, tokenizer_test
gc.collect()

# COMMAND ----------

# DBTITLE 1,Max Tokens
# MAGIC %md
# MAGIC ### 2. Max Tokens Comparison
# MAGIC
# MAGIC Max tokens controls the maximum length of translated output.

# COMMAND ----------

# DBTITLE 1,Max tokens comparison
# Test with longer text
long_text = """
The development of artificial intelligence has revolutionized many aspects of modern life, 
from automated translation systems to advanced medical diagnostics and autonomous vehicles.
"""

max_lengths = [30, 60, 128]

print("Max Tokens Comparison:")
print("="*60)
print(f"Original English ({len(long_text.split())} words):\n{long_text}\n")

for max_len in max_lengths:
    translation = translator_en_fr(long_text, max_length=max_len)
    print(f"Max Length: {max_len} tokens")
    print(f"French: {translation[0]['translation_text']}")
    print(f"Length: {len(translation[0]['translation_text'].split())} words")
    print("-"*60)

print("""

💡 Max Tokens Insights:
- Too short: Translation may be truncated
- Appropriate: Complete, natural translation
- Too long: No benefit, just uses more resources
- Rule of thumb: Set max_length to 1.5x source length
""")

# COMMAND ----------

# DBTITLE 1,Top P
# MAGIC %md
# MAGIC ### 3. Top P (Nucleus Sampling) Comparison
# MAGIC
# MAGIC Top P affects word choice diversity in translation.

# COMMAND ----------

# DBTITLE 1,Top P comparison
# Reload model for top_p testing
model_name = "Helsinki-NLP/opus-mt-en-es"
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer_test = AutoTokenizer.from_pretrained(model_name)

test_text = "I really love learning about new technologies and their applications."

top_p_values = [0.5, 0.9, 1.0]

print("Top P Comparison for Translation:")
print("="*60)
print(f"English: {test_text}\n")

for top_p in top_p_values:
    inputs = tokenizer_test(test_text, return_tensors="pt", padding=True)
    outputs = model.generate(
        **inputs,
        max_length=128,
        do_sample=True,
        top_p=top_p,
        temperature=0.8,
        num_return_sequences=1
    )
    translation = tokenizer_test.decode(outputs[0], skip_special_tokens=True)
    print(f"Top P: {top_p}")
    print(f"Spanish: {translation}")
    print("-"*60)

print("""

💡 Top P Insights:
- Low (0.5): More predictable, standard translations
- Medium (0.9): Balanced, natural-sounding translations (recommended)
- High (1.0): Maximum variety, but may be less idiomatic

🎯 Best Practice: For translation, use lower temperature (0.3-0.5) with top_p=0.9
""")

# Cleanup
del model, tokenizer_test
gc.collect()
print("\n✅ Temporary models cleaned up")

# COMMAND ----------

# DBTITLE 1,Example 4 - Batch
# Batch translate multiple phrases
phrases = [
    "Good morning!",
    "Thank you very much.",
    "Where is the library?",
    "I would like some water, please.",
    "Have a great day!"
]

print("English -> German Translations:\n")
translations = translator_en_de(phrases)

for original, translated in zip(phrases, translations):
    print(f"EN: {original}")
    print(f"DE: {translated['translation_text']}")
    print()

# COMMAND ----------

# DBTITLE 1,Cleanup
# MAGIC %md
# MAGIC ## 🧹 Cleanup
# MAGIC
# MAGIC Free memory by removing models.

# COMMAND ----------

# DBTITLE 1,Delete models
# Clean up resources
import gc

del translator_en_fr, translator_en_es, translator_en_de
gc.collect()

print("✅ Cleanup complete! All models removed from memory.")

# COMMAND ----------

# DBTITLE 1,Key Takeaways
# MAGIC %md
# MAGIC ## 🎯 Key Takeaways
# MAGIC
# MAGIC 1. **Helsinki-NLP OPUS-MT models** provide free, quality translations
# MAGIC 2. **Multiple language pairs** available (1000+ translation directions)
# MAGIC 3. **Batch processing** improves efficiency
# MAGIC 4. **Context matters** - translation quality depends on sentence complexity
# MAGIC
# MAGIC ## 💡 Next Steps
# MAGIC
# MAGIC - Explore reverse translation (French->English, etc.)
# MAGIC - Try translating technical or domain-specific content
# MAGIC - Experiment with `max_length` parameter for long texts
# MAGIC - Check available models: `Helsinki-NLP/opus-mt-{src}-{tgt}`

# COMMAND ----------

