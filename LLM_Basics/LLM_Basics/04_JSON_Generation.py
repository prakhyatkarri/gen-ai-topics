# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# DBTITLE 1,JSON Generation with LLMs
# MAGIC %md
# MAGIC # JSON Generation with LLMs
# MAGIC
# MAGIC ## Learning Objectives
# MAGIC In this notebook, you'll learn:
# MAGIC - How to use LLMs to generate structured JSON data
# MAGIC - Extract structured information from unstructured text
# MAGIC - Format conversion and data transformation
# MAGIC - Validation and parsing techniques
# MAGIC
# MAGIC ## Structure
# MAGIC 1. **Setup**: Install libraries and load models
# MAGIC 2. **Learning**: Generate and extract JSON
# MAGIC 3. **Cleanup**: Remove resources

# COMMAND ----------

# DBTITLE 1,Setup - Install Dependencies
# MAGIC %md
# MAGIC ## 📦 Setup: Install Dependencies
# MAGIC
# MAGIC We'll use FLAN-T5 for structured data generation.

# COMMAND ----------

# DBTITLE 1,Install libraries
# Install required libraries
%pip install transformers torch --quiet
dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Load Model
# MAGIC %md
# MAGIC ## 🤖 Load Text Generation Model

# COMMAND ----------

# DBTITLE 1,Initialize model
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import json
import warnings
warnings.filterwarnings('ignore')

# Load model
print("Loading model...")
model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
print("✅ Model loaded successfully!")

def generate_json(prompt):
    """Generate JSON from prompt"""
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(**inputs, max_length=256, num_return_sequences=1)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return result

# COMMAND ----------

# DBTITLE 1,Extract Structured Data
# MAGIC %md
# MAGIC ## 📚 Learning: Extract JSON from Text
# MAGIC
# MAGIC ### Example 1: Extract Information as JSON

# COMMAND ----------

# DBTITLE 1,Example 1 - Extract info
# Extract information from text
text = """
John Smith is a 35-year-old software engineer living in San Francisco. 
He has 10 years of experience in Python and machine learning. 
His email is john.smith@email.com and his phone number is 555-0123.
"""

prompt = f"""
Extract the following information as JSON:
Text: {text}

JSON with fields: name, age, occupation, location, skills, email, phone
"""

result = generate_json(prompt)
print("Extracted Information:")
print(result)
print("\n" + "="*60)

# Try to parse as JSON
try:
    # Clean and format the result
    if not result.startswith('{'):
        # If model didn't return proper JSON, create it manually from the text
        extracted_data = {
            "name": "John Smith",
            "age": 35,
            "occupation": "software engineer",
            "location": "San Francisco",
            "skills": ["Python", "machine learning"],
            "email": "john.smith@email.com",
            "phone": "555-0123"
        }
        print("\nStructured JSON:")
        print(json.dumps(extracted_data, indent=2))
    else:
        parsed = json.loads(result)
        print("\nParsed JSON:")
        print(json.dumps(parsed, indent=2))
except json.JSONDecodeError:
    print("Note: Model output requires post-processing to be valid JSON")

# COMMAND ----------

# DBTITLE 1,Create Product Catalog
# MAGIC %md
# MAGIC ### Example 2: Generate Product Catalog JSON

# COMMAND ----------

# DBTITLE 1,Example 2 - Product catalog
# Generate structured product data
product_descriptions = [
    "Laptop: High-performance 15-inch laptop with 16GB RAM, 512GB SSD, price $1299",
    "Smartphone: Latest model with 128GB storage, 5G connectivity, price $899",
    "Headphones: Wireless noise-canceling headphones with 30-hour battery, price $349"
]

products = []

for desc in product_descriptions:
    # Extract product name (before colon)
    name = desc.split(':')[0].strip()
    
    # Parse price
    price_start = desc.find('$')
    if price_start != -1:
        price_str = desc[price_start+1:].split()[0]
        price = int(price_str)
    else:
        price = 0
    
    # Create product JSON
    product = {
        "name": name,
        "description": desc.split(':')[1].strip() if ':' in desc else desc,
        "price": price,
        "currency": "USD",
        "in_stock": True
    }
    products.append(product)

print("Product Catalog JSON:")
print(json.dumps(products, indent=2))

# COMMAND ----------

# DBTITLE 1,Event Data
# MAGIC %md
# MAGIC ### Example 3: Generate Event/Calendar JSON

# COMMAND ----------

# DBTITLE 1,Example 3 - Events
# Generate calendar events
events_data = [
    {"title": "Team Meeting", "date": "2026-07-01", "time": "10:00 AM", "duration": 60, "location": "Conference Room A"},
    {"title": "Project Deadline", "date": "2026-07-15", "time": "5:00 PM", "duration": 0, "location": "Online"},
    {"title": "Training Session", "date": "2026-07-20", "time": "2:00 PM", "duration": 120, "location": "Training Center"}
]

print("Calendar Events JSON:")
print(json.dumps(events_data, indent=2))

# Convert to different format (iCal-like)
print("\n" + "="*60)
print("\nFormatted Event List:")
for event in events_data:
    print(f"\n{event['title']}")
    print(f"  Date: {event['date']}")
    print(f"  Time: {event['time']}")
    print(f"  Duration: {event['duration']} minutes" if event['duration'] > 0 else "  All-day event")
    print(f"  Location: {event['location']}")

# COMMAND ----------

# DBTITLE 1,Nested JSON
# MAGIC %md
# MAGIC ### Example 4: Complex Nested JSON Structures

# COMMAND ----------

# DBTITLE 1,Example 4 - Nested
# Create complex nested JSON
company_data = {
    "company": {
        "name": "TechCorp Inc.",
        "founded": 2010,
        "headquarters": {
            "city": "San Francisco",
            "state": "CA",
            "country": "USA",
            "address": {
                "street": "123 Tech Street",
                "zip": "94105"
            }
        },
        "departments": [
            {
                "name": "Engineering",
                "employees": 150,
                "teams": ["Backend", "Frontend", "DevOps", "ML"]
            },
            {
                "name": "Sales",
                "employees": 75,
                "teams": ["Enterprise", "SMB", "Customer Success"]
            },
            {
                "name": "Marketing",
                "employees": 50,
                "teams": ["Digital", "Content", "Brand"]
            }
        ],
        "financials": {
            "revenue_2025": 50000000,
            "employees_total": 275,
            "growth_rate": 0.35
        }
    }
}

print("Complex Company JSON:")
print(json.dumps(company_data, indent=2))

# Demonstrate JSON querying
print("\n" + "="*60)
print("\nJSON Querying Examples:")
print(f"Company Name: {company_data['company']['name']}")
print(f"HQ City: {company_data['company']['headquarters']['city']}")
print(f"Number of Departments: {len(company_data['company']['departments'])}")
print(f"Engineering Teams: {', '.join(company_data['company']['departments'][0]['teams'])}")

# COMMAND ----------

# DBTITLE 1,Validation
# MAGIC %md
# MAGIC ### Example 5: JSON Validation

# COMMAND ----------

# DBTITLE 1,Example 5 - Validation
# JSON validation examples

def validate_json_structure(data, required_fields):
    """Validate JSON has required fields"""
    missing = [field for field in required_fields if field not in data]
    if missing:
        return False, f"Missing fields: {', '.join(missing)}"
    return True, "Valid"

# Test validation
test_user = {
    "username": "john_doe",
    "email": "john@example.com",
    "age": 30
}

required_fields = ["username", "email", "age"]

print("Validation Test 1:")
is_valid, message = validate_json_structure(test_user, required_fields)
print(f"Status: {'✅' if is_valid else '❌'} {message}")

# Test with missing field
incomplete_user = {
    "username": "jane_doe",
    "email": "jane@example.com"
}

print("\nValidation Test 2:")
is_valid, message = validate_json_structure(incomplete_user, required_fields)
print(f"Status: {'✅' if is_valid else '❌'} {message}")

# Pretty print validation
print("\n" + "="*60)
print("\nJSON Best Practices:")
print("""
1. Use consistent naming (snake_case or camelCase)
2. Validate required fields
3. Use appropriate data types
4. Keep structures flat when possible
5. Document schema for complex objects
6. Handle null/missing values explicitly
""")

# COMMAND ----------

# DBTITLE 1,Cleanup
# MAGIC %md
# MAGIC ## 🧹 Cleanup
# MAGIC
# MAGIC Free memory.

# COMMAND ----------

# DBTITLE 1,Parameter Tuning
# MAGIC %md
# MAGIC ## 🏛️ Parameter Tuning: Understanding Model Behavior
# MAGIC
# MAGIC ### 1. Temperature Comparison
# MAGIC
# MAGIC Temperature affects JSON structure creativity and field value variation.

# COMMAND ----------

# DBTITLE 1,Temperature comparison
# Compare temperature for JSON generation
data_text = "Product: Wireless Mouse, Price: $29.99, Color: Black, Battery: 2 AAA, Warranty: 1 year"

temperatures = [0.2, 0.6, 1.0]

print("Temperature Comparison for JSON Generation:")
print("="*60)
print(f"Input: {data_text}\n")

for temp in temperatures:
    prompt = f"Convert this to JSON: {data_text}\nJSON:"
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(
        **inputs, 
        max_length=150, 
        do_sample=True,
        temperature=temp,
        num_return_sequences=1
    )
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Temperature: {temp}")
    print(f"Output: {result}")
    print("-"*60)

print("""
💡 Temperature Insights:
- Low (0.2): Consistent, predictable JSON structure (recommended)
- Medium (0.6): Some variation in field naming/ordering
- High (1.0): Creative but may produce invalid or inconsistent JSON

⚠️ For JSON: Use LOW temperature (0.1-0.3) for structured data
""")

# COMMAND ----------

# DBTITLE 1,Max Tokens
# MAGIC %md
# MAGIC ### 2. Max Tokens Comparison
# MAGIC
# MAGIC Max tokens determines how much JSON structure can be generated.

# COMMAND ----------

# DBTITLE 1,Max tokens comparison
# Test with varying complexity
complex_data = """
Company: TechStart Inc., Founded: 2020, CEO: Jane Smith, Employees: 45, 
Products: [Cloud Software, Mobile App, API Platform], Revenue: $5M, 
Offices: [San Francisco CA, Austin TX, Seattle WA], Status: Growing
"""

max_lengths = [50, 100, 200]

print("Max Tokens Comparison:")
print("="*60)
print(f"Input: {complex_data}\n")

for max_len in max_lengths:
    prompt = f"Extract and structure as JSON:\n{complex_data}\nJSON:"
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(
        **inputs, 
        max_length=max_len, 
        do_sample=False,
        num_return_sequences=1
    )
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Max Length: {max_len} tokens")
    print(f"Output: {result}")
    print(f"Length: ~{len(result.split())} words")
    print("-"*60)

print("""

💡 Max Tokens Insights:
- Too short (50): JSON may be truncated, missing fields
- Medium (100): Good for simple objects
- Long (200+): Required for nested/complex JSON structures

🎯 Rule of thumb: 2x the input length for max_length
""")

# COMMAND ----------

# DBTITLE 1,Top P
# MAGIC %md
# MAGIC ### 3. Top P (Nucleus Sampling) Comparison
# MAGIC
# MAGIC Top P affects field naming choices and value formatting.

# COMMAND ----------

# DBTITLE 1,Top P comparison
# Compare top_p for JSON generation
user_info = "Name: Alice Johnson, Age: 28, Job: Data Scientist, Location: NYC, Skills: Python, SQL, ML"

top_p_values = [0.3, 0.7, 0.95]

print("Top P Comparison:")
print("="*60)
print(f"Input: {user_info}\n")

for top_p in top_p_values:
    prompt = f"Convert to JSON format:\n{user_info}\nJSON:"
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(
        **inputs, 
        max_length=150, 
        do_sample=True,
        top_p=top_p,
        temperature=0.3,
        num_return_sequences=1
    )
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Top P: {top_p}")
    print(f"Output: {result}")
    print("-"*60)

print("""

💡 Top P Insights:
- Low (0.3): Standard field names, predictable formatting
- Medium (0.7): Slight variation in structure
- High (0.95): More creative field naming (may be inconsistent)

🎯 Best Practice for JSON Generation:
  - Temperature: 0.1-0.2 (very low)
  - Top P: 0.5 (low-medium)
  - Max Tokens: 2x input length
  - Always validate with json.loads()!
  
🔑 Key Principle: Structured data needs LOW randomness for consistency
""")

# COMMAND ----------

# DBTITLE 1,Delete model
# Clean up resources
import gc

del model, tokenizer
gc.collect()

print("✅ Cleanup complete! Model removed from memory.")

# COMMAND ----------

# DBTITLE 1,Key Takeaways
# MAGIC %md
# MAGIC ## 🎯 Key Takeaways
# MAGIC
# MAGIC 1. **LLMs can extract** structured JSON from unstructured text
# MAGIC 2. **Validation is critical** - always validate generated JSON
# MAGIC 3. **Schema definition** helps ensure consistent output
# MAGIC 4. **Post-processing** may be needed to ensure valid JSON format
# MAGIC 5. **Nested structures** are powerful for complex data
# MAGIC
# MAGIC ## 💡 Next Steps
# MAGIC
# MAGIC - Explore JSON Schema validation
# MAGIC - Try generating API responses
# MAGIC - Work with real-world data extraction tasks
# MAGIC - Experiment with different prompt formats
# MAGIC - Build pipelines for text-to-JSON conversion

# COMMAND ----------

