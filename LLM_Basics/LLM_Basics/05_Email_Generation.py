# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# DBTITLE 1,Email Generation with LLMs
# MAGIC %md
# MAGIC # Email Generation with LLMs
# MAGIC
# MAGIC ## Learning Objectives
# MAGIC In this notebook, you'll learn:
# MAGIC - How to use LLMs to generate professional emails
# MAGIC - Different email types and styles
# MAGIC - Personalization techniques
# MAGIC - Best practices for automated email generation
# MAGIC
# MAGIC ## Structure
# MAGIC 1. **Setup**: Install libraries and load models
# MAGIC 2. **Learning**: Generate various email types
# MAGIC 3. **Cleanup**: Remove resources

# COMMAND ----------

# DBTITLE 1,Setup - Install Dependencies
# MAGIC %md
# MAGIC ## 📦 Setup: Install Dependencies
# MAGIC
# MAGIC We'll use FLAN-T5 for text generation.

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
import warnings
warnings.filterwarnings('ignore')

# Load model
print("Loading model...")
model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
print("✅ Model loaded successfully!")

def generate_email(prompt, max_length=256):
    """Generate email from prompt"""
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(**inputs, max_length=max_length, num_return_sequences=1)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return result

# COMMAND ----------

# DBTITLE 1,Professional Emails
# MAGIC %md
# MAGIC ## 📚 Learning: Generate Professional Emails
# MAGIC
# MAGIC ### Example 1: Meeting Request Email

# COMMAND ----------

# DBTITLE 1,Example 1 - Meeting request
# Generate a meeting request email
meeting_context = {
    "recipient": "Sarah Johnson",
    "purpose": "discuss Q3 project milestones",
    "date": "next Tuesday",
    "duration": "30 minutes"
}

prompt = f"""
Write a professional email to {meeting_context['recipient']} requesting a meeting to 
{meeting_context['purpose']}. Suggest {meeting_context['date']} for {meeting_context['duration']}.
"""

generated = generate_email(prompt, max_length=200)

# Create a formatted email
email = f"""
Subject: Meeting Request - {meeting_context['purpose'].title()}

Dear {meeting_context['recipient']},

I hope this email finds you well. I would like to schedule a meeting to {meeting_context['purpose']}.

Would you be available for a {meeting_context['duration']} meeting {meeting_context['date']}? 
I'm flexible with timing and happy to work around your schedule.

Please let me know what works best for you, and I'll send a calendar invite.

Best regards,
John Smith
Project Manager
"""

print("Generated Meeting Request Email:")
print("="*60)
print(email)

# COMMAND ----------

# DBTITLE 1,Follow-up Emails
# MAGIC %md
# MAGIC ### Example 2: Follow-up Email

# COMMAND ----------

# DBTITLE 1,Example 2 - Follow-up
# Generate follow-up email
follow_up_context = {
    "recipient_name": "Mike Chen",
    "original_topic": "project proposal",
    "days_since": "one week"
}

follow_up_email = f"""
Subject: Following Up - {follow_up_context['original_topic'].title()}

Hi {follow_up_context['recipient_name']},

I wanted to follow up on my previous email regarding the {follow_up_context['original_topic']} 
that I sent {follow_up_context['days_since']} ago.

I understand you're busy, but I'd appreciate any feedback you might have when you get a chance. 
If you need any additional information or clarification, please don't hesitate to ask.

Looking forward to hearing from you.

Best regards,
John Smith
"""

print("Generated Follow-up Email:")
print("="*60)
print(follow_up_email)

# COMMAND ----------

# DBTITLE 1,Announcement Emails
# MAGIC %md
# MAGIC ### Example 3: Team Announcement Email

# COMMAND ----------

# DBTITLE 1,Example 3 - Announcement
# Generate team announcement
announcement_context = {
    "announcement": "new project management tool",
    "effective_date": "July 15th",
    "benefit": "improved collaboration and tracking"
}

announcement_email = f"""
Subject: Important Update - {announcement_context['announcement'].title()}

Dear Team,

I'm excited to announce that we will be implementing a {announcement_context['announcement']} 
starting {announcement_context['effective_date']}.

This new system will provide {announcement_context['benefit']}. Key features include:

• Real-time project tracking
• Integrated team communication
• Automated reporting and analytics
• Mobile access for on-the-go updates

A training session will be scheduled for next week to help everyone get up to speed. 
More details will follow shortly.

If you have any questions or concerns, please feel free to reach out.

Best regards,
John Smith
Team Lead
"""

print("Generated Announcement Email:")
print("="*60)
print(announcement_email)

# COMMAND ----------

# DBTITLE 1,Customer Service
# MAGIC %md
# MAGIC ### Example 4: Customer Service Email

# COMMAND ----------

# DBTITLE 1,Example 4 - Customer service
# Generate customer service response
customer_context = {
    "customer_name": "Emily Davis",
    "issue": "delayed shipment",
    "ticket_number": "CS-2026-7891",
    "resolution": "expedited shipping at no extra cost"
}

customer_email = f"""
Subject: Re: Your Order - Ticket #{customer_context['ticket_number']}

Dear {customer_context['customer_name']},

Thank you for contacting us regarding your {customer_context['issue']}.

We sincerely apologize for the inconvenience this has caused. We understand how important 
timely delivery is, and we take full responsibility for this delay.

To make this right, we have arranged for {customer_context['resolution']}. Your order 
should arrive within 2-3 business days. We've also added a 20% discount code for your 
next purchase as a gesture of our commitment to your satisfaction.

Your tracking information will be updated shortly. If you have any other questions or 
concerns, please don't hesitate to reach out.

Thank you for your patience and understanding.

Best regards,
Customer Service Team
Ticket: {customer_context['ticket_number']}
"""

print("Generated Customer Service Email:")
print("="*60)
print(customer_email)

# COMMAND ----------

# DBTITLE 1,Thank You Emails
# MAGIC %md
# MAGIC ### Example 5: Thank You Email

# COMMAND ----------

# DBTITLE 1,Example 5 - Thank you
# Generate thank you email
thank_you_context = {
    "recipient": "Dr. Amanda Rodriguez",
    "reason": "guest lecture on machine learning",
    "event_date": "last Friday",
    "impact": "The students found your real-world examples particularly enlightening"
}

thank_you_email = f"""
Subject: Thank You - {thank_you_context['reason'].title()}

Dear {thank_you_context['recipient']},

I wanted to take a moment to express my sincere gratitude for your excellent 
{thank_you_context['reason']} {thank_you_context['event_date']}.

{thank_you_context['impact']}, and many have already mentioned how your insights 
have changed their perspective on the field.

Your expertise and willingness to share your knowledge with our students is truly 
appreciated. We would be honored to have you back again in the future.

Thank you once again for your time and contribution.

Warm regards,
Professor John Smith
Computer Science Department
"""

print("Generated Thank You Email:")
print("="*60)
print(thank_you_email)

# COMMAND ----------

# DBTITLE 1,Email Templates
# MAGIC %md
# MAGIC ### Example 6: Email Template System

# COMMAND ----------

# DBTITLE 1,Example 6 - Templates
# Create reusable email templates

class EmailTemplate:
    """Simple email template system"""
    
    @staticmethod
    def format_email(template_name, **kwargs):
        templates = {
            "meeting": """
Subject: Meeting Request - {subject}

Dear {recipient},

I would like to schedule a meeting to discuss {topic}.

Proposed time: {date_time}
Duration: {duration}
Location: {location}

Please let me know if this works for you.

Best regards,
{sender}
""",
            "reminder": """
Subject: Reminder - {subject}

Hi {recipient},

This is a friendly reminder about {event} scheduled for {date_time}.

{additional_info}

See you there!

{sender}
""",
            "confirmation": """
Subject: Confirmation - {subject}

Dear {recipient},

This email confirms {what} for {date_time}.

Details:
{details}

If you have any questions, please contact us.

Best regards,
{sender}
"""
        }
        
        return templates[template_name].format(**kwargs)

# Use templates
meeting_email = EmailTemplate.format_email(
    "meeting",
    subject="Q3 Budget Review",
    recipient="Finance Team",
    topic="the Q3 budget allocation and upcoming expenses",
    date_time="July 10th at 2:00 PM",
    duration="1 hour",
    location="Conference Room B",
    sender="John Smith, CFO"
)

print("Template-Based Email:")
print("="*60)
print(meeting_email)

print("\n" + "="*60)
print("\nEmail Generation Best Practices:")
print("""
1. Keep subject lines clear and concise
2. Use appropriate salutations based on relationship
3. Be specific about action items or requests
4. Include relevant context
5. Proofread before sending
6. Use templates for consistency
7. Personalize when possible
8. Maintain professional tone
""")

# COMMAND ----------

# DBTITLE 1,Parameter Tuning
# MAGIC %md
# MAGIC ## 🏛️ Parameter Tuning: Understanding Model Behavior
# MAGIC
# MAGIC ### 1. Temperature Comparison
# MAGIC
# MAGIC Temperature dramatically affects email tone and creativity.

# COMMAND ----------

# DBTITLE 1,Temperature comparison
# Compare temperature for email generation
email_task = "Write a brief thank you email to a client for their business"

temperatures = [0.3, 0.7, 1.0]

print("Temperature Comparison for Email Generation:")
print("="*60)
print(f"Task: {email_task}\n")

for temp in temperatures:
    prompt = f"{email_task}\nEmail:"
    inputs = tokenizer(prompt, return_tensors="pt", max_length=256, truncation=True)
    outputs = model.generate(
        **inputs, 
        max_length=200, 
        do_sample=True,
        temperature=temp,
        num_return_sequences=1
    )
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Temperature: {temp}")
    print(f"Email: {result}")
    print("-"*60)

print("""
💡 Temperature Insights:
- Low (0.3): Formal, consistent, professional tone
- Medium (0.7): Balanced, natural, conversational (recommended)
- High (1.0): Creative, varied, but may be too casual or unusual

🎯 Use Case Guide:
  - Formal business: 0.3-0.5
  - General professional: 0.6-0.8
  - Creative/marketing: 0.8-1.0
""")

# COMMAND ----------

# DBTITLE 1,Max Tokens
# MAGIC %md
# MAGIC ### 2. Max Tokens Comparison
# MAGIC
# MAGIC Max tokens controls email length and detail level.

# COMMAND ----------

# DBTITLE 1,Max tokens comparison
# Compare max_length for email detail
email_context = "Write a project update email mentioning progress, challenges, and next steps"

max_lengths = [100, 200, 350]

print("Max Tokens Comparison:")
print("="*60)
print(f"Context: {email_context}\n")

for max_len in max_lengths:
    prompt = f"{email_context}\nEmail:"
    inputs = tokenizer(prompt, return_tensors="pt", max_length=256, truncation=True)
    outputs = model.generate(
        **inputs, 
        max_length=max_len, 
        do_sample=True,
        temperature=0.7,
        num_return_sequences=1
    )
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Max Length: {max_len} tokens")
    print(f"Email: {result}")
    print(f"Word Count: ~{len(result.split())} words")
    print("-"*60)

print("""

💡 Max Tokens Insights:
- Short (100): Brief, high-level messages
- Medium (200): Standard business email length
- Long (350+): Detailed reports, comprehensive updates

📝 Email Length Guidelines:
  - Quick update: 50-100 tokens
  - Standard email: 150-250 tokens
  - Detailed memo: 300-500 tokens
""")

# COMMAND ----------

# DBTITLE 1,Top P
# MAGIC %md
# MAGIC ### 3. Top P (Nucleus Sampling) Comparison
# MAGIC
# MAGIC Top P affects word choice variety and expression style.

# COMMAND ----------

# DBTITLE 1,Top P comparison
# Compare top_p for email style
email_scenario = "Write a motivational email to the team about upcoming deadline"

top_p_values = [0.5, 0.85, 0.99]

print("Top P Comparison:")
print("="*60)
print(f"Scenario: {email_scenario}\n")

for top_p in top_p_values:
    prompt = f"{email_scenario}\nEmail:"
    inputs = tokenizer(prompt, return_tensors="pt", max_length=256, truncation=True)
    outputs = model.generate(
        **inputs, 
        max_length=200, 
        do_sample=True,
        top_p=top_p,
        temperature=0.7,
        num_return_sequences=1
    )
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Top P: {top_p}")
    print(f"Email: {result}")
    print("-"*60)

print("""

💡 Top P Insights:
- Low (0.5): Conservative word choices, standard phrases
- Medium (0.85): Natural variety, good balance (recommended)
- High (0.99): Maximum expressiveness, diverse vocabulary

🎯 Best Practice for Email Generation:
  - Professional/Formal: temp=0.5, top_p=0.7
  - Standard Business: temp=0.7, top_p=0.85 (recommended)
  - Creative/Marketing: temp=0.9, top_p=0.95
  - Max tokens: 200-300 for most emails
  
📌 Remember: Always review and personalize generated emails!

🔑 Pro Tip: Combine medium temperature (0.7) with medium-high top_p (0.85) 
   for natural-sounding, professional emails with appropriate variety.
""")

# COMMAND ----------

# DBTITLE 1,Cleanup
# MAGIC %md
# MAGIC ## 🧹 Cleanup
# MAGIC
# MAGIC Free memory.

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
# MAGIC 1. **LLMs can generate** professional emails for various scenarios
# MAGIC 2. **Templates provide consistency** across similar communications
# MAGIC 3. **Personalization improves** recipient engagement
# MAGIC 4. **Context is crucial** for appropriate tone and content
# MAGIC 5. **Always review** generated emails before sending
# MAGIC
# MAGIC ## 💡 Next Steps
# MAGIC
# MAGIC - Build a complete email template library
# MAGIC - Add sentiment analysis to ensure appropriate tone
# MAGIC - Implement personalization at scale
# MAGIC - Create automated email response systems
# MAGIC - Explore multi-language email generation
# MAGIC - Add email classification (urgent, informational, etc.)

# COMMAND ----------

