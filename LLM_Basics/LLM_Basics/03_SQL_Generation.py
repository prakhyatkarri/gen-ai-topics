# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# DBTITLE 1,SQL Generation with LLMs
# MAGIC %md
# MAGIC # SQL Generation with LLMs
# MAGIC
# MAGIC ## Learning Objectives
# MAGIC In this notebook, you'll learn:
# MAGIC - How to use LLMs to generate SQL queries from natural language
# MAGIC - Different prompting strategies for SQL generation
# MAGIC - Validation and safety considerations
# MAGIC - Practical applications for data analysis
# MAGIC
# MAGIC ## Structure
# MAGIC 1. **Setup**: Install libraries and prepare sample database
# MAGIC 2. **Learning**: Generate SQL from natural language
# MAGIC 3. **Cleanup**: Remove resources

# COMMAND ----------

# DBTITLE 1,Setup - Install Dependencies
# MAGIC %md
# MAGIC ## 📦 Setup: Install Dependencies
# MAGIC
# MAGIC We'll use Google's FLAN-T5 model - a free model capable of text-to-SQL generation.

# COMMAND ----------

# DBTITLE 1,Install libraries
# Install required libraries
%pip install transformers torch --quiet
dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Create Sample Database
# MAGIC %md
# MAGIC ## 📊 Create Sample Database Schema
# MAGIC
# MAGIC Let's create a sample schema to work with.

# COMMAND ----------

# DBTITLE 1,Sample schema
# Define a sample database schema
schema = """
Database Schema:

Table: employees
  - employee_id (INT, PRIMARY KEY)
  - first_name (VARCHAR)
  - last_name (VARCHAR)
  - department (VARCHAR)
  - salary (DECIMAL)
  - hire_date (DATE)

Table: departments
  - department_id (INT, PRIMARY KEY)
  - department_name (VARCHAR)
  - manager_id (INT)
  - budget (DECIMAL)

Table: projects
  - project_id (INT, PRIMARY KEY)
  - project_name (VARCHAR)
  - department_id (INT)
  - start_date (DATE)
  - end_date (DATE)
  - budget (DECIMAL)
"""

print(schema)

# COMMAND ----------

# DBTITLE 1,Load Model
# MAGIC %md
# MAGIC ## 🤖 Load Text-to-SQL Model
# MAGIC
# MAGIC We'll use FLAN-T5 which can understand natural language and generate SQL.

# COMMAND ----------

# DBTITLE 1,Initialize model
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import warnings
warnings.filterwarnings('ignore')

# Load a free text generation model
print("Loading model...")
# model_name = "google/flan-t5-base"
model_name = "Alpecevit/flan-t5-base-text2sql"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
print("✅ Model loaded successfully!")

def generate_sql(question, schema):
    """Generate ANSI-SQL query from natural language question"""
    prompt = f"""{schema}

Question: {question}
SQL Query:"""
    
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(**inputs, max_length=150, num_return_sequences=1)
    sql = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return sql

# COMMAND ----------

# DBTITLE 1,Basic SQL Generation
# MAGIC %md
# MAGIC ## 📚 Learning: Generate SQL Queries
# MAGIC
# MAGIC ### Example 1: Simple SELECT Queries

# COMMAND ----------

# DBTITLE 1,Example 1 - Simple queries
# Simple queries
questions = [
    "Show all employees",
    "Get the names and salaries of all employees",
    "List all departments"
]

print("Generating SQL queries...\n")

for question in questions:
    sql = generate_sql(question, schema)
    print(f"Question: {question}")
    print(f"Generated SQL: {sql}")
    print("-" * 60)

# COMMAND ----------

# DBTITLE 1,Complex Queries
# MAGIC %md
# MAGIC ### Example 2: Queries with Filtering and Aggregation

# COMMAND ----------

# DBTITLE 1,Example 2 - Complex queries
# More complex queries
complex_questions = [
    "Find employees with salary greater than 50000",
    "Count the number of employees in each department",
    "Show the average salary by department"
]

print("Complex SQL Generation:\n")

for question in complex_questions:
    sql = generate_sql(question, schema)
    print(f"Question: {question}")
    print(f"Generated SQL: {sql}")
    print("-" * 60)

# COMMAND ----------

# DBTITLE 1,JOIN Queries
# MAGIC %md
# MAGIC ### Example 3: Queries with JOINs

# COMMAND ----------

# DBTITLE 1,Example 3 - JOINs
# Queries requiring joins
join_questions = [
    "Show employee names with their department names",
    "List all projects with their department information",
    "Find employees working in the IT department"
]

print("SQL with JOINs:\n")

for question in join_questions:
    sql = generate_sql(question, schema)
    print(f"Question: {question}")
    print(f"Generated SQL: {sql}")
    print("-" * 60)

# COMMAND ----------

# DBTITLE 1,Best Practices
# MAGIC %md
# MAGIC ### Example 4: Best Practices for SQL Generation
# MAGIC
# MAGIC **Important Safety Considerations:**

# COMMAND ----------

# DBTITLE 1,Example 4 - Safety
# Best practices
print("""
⚠️  SQL Generation Safety Tips:

1. Always validate generated SQL before execution
2. Use read-only database connections for testing
3. Sanitize inputs to prevent SQL injection
4. Review queries for efficiency (avoid SELECT *)
5. Test on small datasets first
6. Consider query timeouts
7. Log and monitor generated queries

✅ Example validation function:
""")

def validate_sql(sql):
    """Basic SQL validation"""
    dangerous_keywords = ['DROP', 'DELETE', 'TRUNCATE', 'ALTER', 'UPDATE']
    sql_upper = sql.upper()
    
    for keyword in dangerous_keywords:
        if keyword in sql_upper:
            return False, f"Dangerous keyword detected: {keyword}"
    
    return True, "SQL appears safe"

# Test validation
test_queries = [
    "SELECT * FROM employees",
    "DROP TABLE employees",
    "SELECT name, salary FROM employees WHERE department = 'IT'"
]

print("\nValidation Results:\n")
for query in test_queries:
    is_safe, message = validate_sql(query)
    status = "✅" if is_safe else "❌"
    print(f"{status} {query}")
    print(f"   {message}\n")

# COMMAND ----------

# DBTITLE 1,Parameter Tuning
# MAGIC %md
# MAGIC ## 🏛️ Parameter Tuning: Understanding Model Behavior
# MAGIC
# MAGIC ### 1. Temperature Comparison
# MAGIC
# MAGIC Temperature affects SQL generation creativity and syntax variety.

# COMMAND ----------

# DBTITLE 1,Temperature comparison
# Compare temperature settings for SQL generation
question = "Find all employees in the IT department with salary over 70000"

temperatures = [0.1, 0.5, 1.0]

print("Temperature Comparison for SQL Generation:")
print("="*60)
print(f"Question: {question}\n")

for temp in temperatures:
    prompt = f"""{schema}\n\nQuestion: {question}\nSQL Query:"""
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(
        **inputs, 
        max_length=150, 
        do_sample=True,
        temperature=temp,
        num_return_sequences=1
    )
    sql = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Temperature: {temp}")
    print(f"SQL: {sql}")
    print("-"*60)

print("""
💡 Temperature Insights:
- Low (0.1): More predictable, standard SQL syntax (recommended for production)
- Medium (0.5): Slight variation in query structure
- High (1.0): Creative but may produce unusual or incorrect SQL

⚠️ For SQL: Use LOW temperature (0.1-0.3) for consistency and accuracy
""")

# COMMAND ----------

# DBTITLE 1,Max Tokens
# MAGIC %md
# MAGIC ### 2. Max Tokens Comparison
# MAGIC
# MAGIC Max tokens controls SQL query complexity and length.

# COMMAND ----------

# DBTITLE 1,Max tokens comparison
# Compare max_length for complex queries
complex_question = "Show employee names, departments, and their project details with budget over 100000"

max_lengths = [50, 100, 200]

print("Max Tokens Comparison:")
print("="*60)
print(f"Question: {complex_question}\n")

for max_len in max_lengths:
    prompt = f"""{schema}\n\nQuestion: {complex_question}\nSQL Query:"""
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(
        **inputs, 
        max_length=max_len, 
        do_sample=False,
        num_return_sequences=1
    )
    sql = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Max Length: {max_len} tokens")
    print(f"SQL: {sql}")
    print(f"Token Count: ~{len(sql.split())} words")
    print("-"*60)

print("""

💡 Max Tokens Insights:
- Too short (50): May truncate complex queries with JOINs
- Medium (100): Good for most queries
- Long (200): Needed for complex multi-table queries

🎯 Recommendation: Start with 150 tokens, adjust based on query complexity
""")

# COMMAND ----------

# DBTITLE 1,Top P
# MAGIC %md
# MAGIC ### 3. Top P (Nucleus Sampling) Comparison
# MAGIC
# MAGIC Top P affects SQL keyword and structure choices.

# COMMAND ----------

# DBTITLE 1,Top P comparison
# Compare top_p for SQL generation
question = "Calculate the average salary by department and show departments with avg > 60000"

top_p_values = [0.3, 0.7, 0.95]

print("Top P Comparison:")
print("="*60)
print(f"Question: {question}\n")

for top_p in top_p_values:
    prompt = f"""{schema}\n\nQuestion: {question}\nSQL Query:"""
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(
        **inputs, 
        max_length=150, 
        do_sample=True,
        top_p=top_p,
        temperature=0.3,
        num_return_sequences=1
    )
    sql = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Top P: {top_p}")
    print(f"SQL: {sql}")
    print("-"*60)

print("""

💡 Top P Insights:
- Low (0.3): Very conservative, standard SQL patterns
- Medium (0.7): Balanced approach
- High (0.95): More variety in syntax choices

🎯 Best Practice for SQL Generation:
  - Temperature: 0.1-0.3 (low)
  - Top P: 0.5-0.7 (medium-low)
  - Max Tokens: 150-200
  - Always validate output before execution!
""")

# COMMAND ----------

# DBTITLE 1,Cleanup
# MAGIC %md
# MAGIC ## 🧹 Cleanup
# MAGIC
# MAGIC Free memory by removing the model.

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
# MAGIC 1. **LLMs can generate SQL** from natural language descriptions
# MAGIC 2. **Provide schema context** for better results
# MAGIC 3. **Always validate** generated SQL before execution
# MAGIC 4. **Safety first** - check for dangerous operations
# MAGIC 5. **Clear questions** produce better SQL
# MAGIC
# MAGIC ## 💡 Next Steps
# MAGIC
# MAGIC - Try with your own database schemas
# MAGIC - Explore specialized text-to-SQL models (e.g., `NumbersStation/nsql-350M`)
# MAGIC - Implement more robust validation
# MAGIC - Create a feedback loop to improve generation quality
# MAGIC - Consider fine-tuning on your specific schema

# COMMAND ----------

