# Databricks notebook source
# MAGIC %run ./config

# COMMAND ----------

from IPython.display import Markdown
from openai import OpenAI
import os
dbutils.widgets.text('catalog_name', 'main')
base_url = f'https://{spark.conf.get("spark.databricks.workspaceUrl")}/serving-endpoints'

# COMMAND ----------

base_url = f'https://{spark.conf.get("spark.databricks.workspaceUrl")}/serving-endpoints'

# COMMAND ----------

# MAGIC %md
# MAGIC #Introduction
# MAGIC
# MAGIC In this notebook, we will cover the following prompting strategies to analyze an article from the WSJ. You can utilize these strategies to improve use cases as simply sending a normal prompt is rarely enough. 
# MAGIC
# MAGIC The prompting strategies we are covering in this notebook: 
# MAGIC 1. Zero Shot Prompting 
# MAGIC 2. Few Shot Prompting 
# MAGIC 3. Self-Consistency (multiple solutions)
# MAGIC 4. ReAct (the basis to function/tool calling)

# COMMAND ----------

# MAGIC %md
# MAGIC #Why Prompt Engineering? 
# MAGIC LLMs have shown an incredible ability to adapt to very different ways of writing thus the entire area of prompt engineering was born. Inherently, we know LLMs are probablistic, trying to predict the next token/word based on the input or the prompt. Prompt Engineering is our efforts to try and increase the probabilty for outputs we would like to see like reducing hallucinations. Retrieval Augmented Generation (RAG) is essentially Prompt Engineering with a lot of existing tools (like vector databases) to help find and augment the prompt with relevant information. 
# MAGIC
# MAGIC Prompt engineering is far less intensive than fine tuning or training models and you can quickly iterate over various different prompts to have LLMs achieve various different tasks. Most use cases can be achieved through simple prompt engineering. 

# COMMAND ----------

# MAGIC %md
# MAGIC #Things to consider
# MAGIC
# MAGIC 1. **LLM Selection**: LLMs aren't made the same. We have foundation models that were trained to do text generation which are often tuned to be instruction/chat tuned models, or models that expect/recognize instruction or chat messages. There are a plethora of other models out there that are being fine-tuned to accept different data, perform at specific tasks better or recognize specific domain knowledge better. For example, Lynx AI just open sourced their LLM as a judge model where they specifically pre-trained this model to be an evaluator. 
# MAGIC     - When you're demoing LLMs, try to use the best available model. Your demo performance will rely heavily on the model you use. 
# MAGIC     - Closed Source, propreitary models current siginficantly outperform open source models out of the box. Fine-tuning is required to approach out of the box closed source model performance 
# MAGIC     - Best Closed Source Models Overall: Claude 3.5 and GPT-4o 
# MAGIC     - Best Open Source Models Overall.: Llama3, Gemma-2, Command-R+ (non commercial), Mixtral 8x22B
# MAGIC
# MAGIC 2. **Latency and Speed**: The larger the LLM, the slower the output, although this is becoming less important as teams improve the throughput of these models. This is more important as you scale up with an LLM. Keep in mind you trade some performance for some speed. The platform you run or host these models on also matters but that's on the provider.
# MAGIC     - Best Closed Source Model for Speed: Claude 3 Haiku, Gemini 1.5 Flash
# MAGIC     - Best Open Source Model for Speed: Llama3-8B, Mistral 7B
# MAGIC
# MAGIC 3. **Multi-Modal Capabilities**: If you need your model to accept multiple different types of data or a different data type entirely, consider finding a model that can accept different file types. The prompt engineering for this changes slightly as you need to put your instructions in different locations. 
# MAGIC     - Best Closed Source Multi-Modal Model: Claude 3.5 and GPT-4o
# MAGIC     - Best Closed Source Multi-Modal Model: Llava, Llama3 (FUTURE)

# COMMAND ----------

# MAGIC %md
# MAGIC # Before we begin
# MAGIC It's important to know that each LLM has different syntax you need to follow for the LLM to work properly or optimally. Additionally, depending on how they're trained, the LLM will recognize some tasks or formatting better than others. This is why some models like Llama3 and Claude can do function calling or instruction following better than DBRX. This performance is an important part to consider and makes **LLM selection** very important. For example, Llama3 follows the ChatML format which looks different from DBRX's ChatML format. I have some examples from different LLMs below:
# MAGIC
# MAGIC **Llama3 Prompt Template**
# MAGIC
# MAGIC *<|begin_of_text|><|start_header_id|>system<|end_header_id|>*
# MAGIC
# MAGIC *You are a helpful AI assistant for travel tips and recommendations<|eot_id|>*
# MAGIC
# MAGIC *<|start_header_id|>user<|end_header_id|>*
# MAGIC
# MAGIC *What can you help me with?<|eot_id|>*
# MAGIC
# MAGIC *<|start_header_id|>assistant<|end_header_id|>*
# MAGIC
# MAGIC **DBRX Prompt Template**
# MAGIC
# MAGIC *<|im_start|>system prompt goes here*
# MAGIC *<|im_end|>*
# MAGIC
# MAGIC *<|im_start|>user prompt goes here*
# MAGIC *<|im_end|>*
# MAGIC
# MAGIC *<|im_start|>assistant or answer goes here*
# MAGIC *<|im_end|>*
# MAGIC
# MAGIC **Claude Prompt Template (there are a lot more optional Claude specific things)** 
# MAGIC
# MAGIC *Human:*
# MAGIC
# MAGIC *Assistant:*
# MAGIC
# MAGIC These templates are a direct result of how the provider trained or how someone fine-tunes said model plus the tokenizer used. Each LLM also recognizes different text better depending on they're trained. For example, Claude knows what XML tags are and is commonly used to explicitly organize information for Claude.
# MAGIC
# MAGIC Databrick's Foundation Model API and many other provdiers like Bedrock, Together.ai and Groq do this formatting for you in the API call. 
# MAGIC
# MAGIC **Important**: **If your customer wants to host a custom model, fine-tune a model or pre-train a model, you must optimize your prompt to follow the prompt template, which is usually provided in the model card.**
# MAGIC
# MAGIC ### *We do not need to worry about this in this workshop*
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC #Get started immediately with your Data with AI Functions
# MAGIC
# MAGIC We have a number of AI Functions designed as SQL functions that you can use in a SQL cell or SQL editor and use LLMs directly on your data immediately
# MAGIC
# MAGIC 1. ai_analyze_sentiment
# MAGIC 2. ai_classify
# MAGIC 3. ai_extract
# MAGIC 4. ai_fix_grammar
# MAGIC 5. ai_gen
# MAGIC 6. ai_mask
# MAGIC 7. ai_similarity
# MAGIC 8. ai_summarize
# MAGIC 9. ai_translate
# MAGIC 10. ai_query
# MAGIC
# MAGIC We will run a demo each of these functions below. 
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ### ai_fix_grammar
# MAGIC The ai_fix_grammar() function allows you to invoke a state-of-the-art generative AI model to correct grammatical errors in a given text using SQL. This function uses a chat model serving endpoint made available by Databricks Foundation Model APIs.
# MAGIC
# MAGIC Documentation: https://docs.databricks.com/en/sql/language-manual/functions/ai_fix_grammar.html

# COMMAND ----------

# MAGIC %sql
# MAGIC -- verify that we're running on a SQL Warehouse
# MAGIC SELECT assert_true(current_version().dbsql_version is not null, 'YOU MUST USE A SQL WAREHOUSE, not a cluster');
# MAGIC
# MAGIC SELECT ai_fix_grammar('This sentence have some mistake');

# COMMAND ----------

# MAGIC %md
# MAGIC ### ai_similarity
# MAGIC The ai_similarity() function invokes a state-of-the-art generative AI model from Databricks Foundation Model APIs to compare two strings and computes the semantic similarity score using SQL.
# MAGIC
# MAGIC Documentation: https://docs.databricks.com/en/sql/language-manual/functions/ai_similarity.html

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT ai_similarity('Databricks', 'Apache Spark'),  ai_similarity('Apache Spark', 'The Apache Spark Engine');

# COMMAND ----------

# MAGIC %md
# MAGIC ### ai_gen
# MAGIC The ai_gen() function invokes a state-of-the-art generative AI model to answer the user-provided prompt using SQL. This function uses a chat model serving endpoint made available by Databricks Foundation Model APIs.
# MAGIC
# MAGIC Documentation: https://docs.databricks.com/en/sql/language-manual/functions/ai_gen.html

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT review, ai_gen('Generate a response to the following review: ' || review) as answer
# MAGIC from austin_choi_demo_catalog.demo_data.customer_demo_data_reviews
# MAGIC Limit 5;

# COMMAND ----------

# MAGIC %md
# MAGIC ### ai_extract
# MAGIC The ai_extract() function allows you to invoke a state-of-the-art generative AI model to extract entities specified by labels from a given text using SQL.
# MAGIC
# MAGIC Documentation: https://docs.databricks.com/en/sql/language-manual/functions/ai_extract.html

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT review, ai_extract(review, array("store", "product")) as Keywords
# MAGIC from austin_choi_demo_catalog.demo_data.customer_demo_data_reviews
# MAGIC Limit 3;

# COMMAND ----------

# MAGIC %md
# MAGIC ### ai_analyze_sentiment
# MAGIC The ai_analyze_sentiment() function allows you to invoke a state-of-the-art generative AI model to perform sentiment analysis on input text using SQL.
# MAGIC
# MAGIC Documentation: https://docs.databricks.com/en/sql/language-manual/functions/ai_analyze_sentiment.html

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT review, ai_analyze_sentiment(review) as Sentiment
# MAGIC from austin_choi_demo_catalog.demo_data.customer_demo_data_reviews
# MAGIC Limit 5;

# COMMAND ----------

# MAGIC %md
# MAGIC ### ai_classify
# MAGIC The ai_classify() function allows you to invoke a state-of-the-art generative AI model to classify input text according to labels you provide using SQL.
# MAGIC
# MAGIC Documentation: https://docs.databricks.com/en/sql/language-manual/functions/ai_classify.html

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT country, ai_classify(country, ARRAY("APAC", "AMER", "EU")) as Region
# MAGIC from austin_choi_demo_catalog.demo_data.customer_demo_data_franchises
# MAGIC limit 5;

# COMMAND ----------

# MAGIC %md
# MAGIC ### ai_translate
# MAGIC The ai_translate() function allows you to invoke a state-of-the-art generative AI model to translate text to a specified target language using SQL. During the preview, the function supports translation between English (en) and Spanish (es) only.
# MAGIC
# MAGIC Documentation: https://docs.databricks.com/en/sql/language-manual/functions/ai_translate.html
# MAGIC

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT review, ai_translate(review, "es")
# MAGIC from austin_choi_demo_catalog.demo_data.customer_demo_data_reviews
# MAGIC limit 3;

# COMMAND ----------

# MAGIC %md
# MAGIC ### ai_mask
# MAGIC The ai_mask() function allows you to invoke a state-of-the-art generative AI model to mask specified entities in a given text using SQL. 
# MAGIC
# MAGIC Documentation: https://docs.databricks.com/en/sql/language-manual/functions/ai_mask.html

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT first_name, last_name, (first_name || " " || last_name || " lives at " || address) as unmasked_output, ai_mask(first_name || "" || last_name || " lives at " || address, array("person", "address")) as Masked_Output
# MAGIC from austin_choi_demo_catalog.demo_data.customer_demo_data_customers
# MAGIC limit 5

# COMMAND ----------

# MAGIC %md
# MAGIC ### ai_query
# MAGIC The ai_query() function allows you to query machine learning models and large language models served using Mosaic AI Model Serving. To do so, this function invokes an existing Mosaic AI Model Serving endpoint and parses and returns its response. Databricks recommends using ai_query with Model Serving for batch inference
# MAGIC
# MAGIC Documentation: https://docs.databricks.com/en/large-language-models/ai-functions.html#ai_query

# COMMAND ----------

# MAGIC %sql 
# MAGIC SELECT
# MAGIC   `Misspelled Make`,   -- Placeholder for the input column
# MAGIC   ai_query(
# MAGIC     'databricks-meta-llama-3-3-70b-instruct',
# MAGIC     CONCAT(format_string('You will always receive a make of a car. Check to see if it is misspelled and a real car. Correct the mistake. Only provide the corrected make. Never add additional details'), `Misspelled Make`)    -- Placeholder for the prompt and input
# MAGIC   ) AS ai_guess  -- Placeholder for the output column
# MAGIC FROM austin_choi_demo_catalog.demo_data.synthetic_car_data
# MAGIC Limit 3;  -- Placeholder for the table name
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ###Below is the WSJ article we will use for the example exercises
# MAGIC
# MAGIC Run the cell

# COMMAND ----------

#The Article

content = """
The last Tax Report delved into when it’s smart for seniors to convert taxable traditional IRAs to tax-free Roth IRAs. It prompted a flood of questions from Journal readers asking for more specifics, so we’ll address some of them.

Why does it make sense to do a Roth IRA conversion if that means paying taxes before they’re due and shrinking the investment amount? I was always taught to defer taxes as long as possible.
Roth IRA conversions aren’t appropriate for all savers. But they often make sense for people with large traditional IRAs, which many have due to rollovers of traditional 401(k)s. 

The key factor is the tax rate on the Roth IRA conversion amount vs. the tax rate on the funds at the time of withdrawal—tax arbitrage, if you will. If the rate on the conversion will be lower than the rate at withdrawal, conversion is often a smart move.

What about “opportunity cost,” the idea that dollars used to pay conversion tax lose out on investment growth? This isn’t an issue if tax rates are the same at contribution and withdrawal. While the conversion tax is paid earlier, it’s also smaller—and the deferred tax on the traditional IRA grows as the funds do.

Ed Slott, a Roth IRA advocate and CPA, offers a simplified example: Jane and Fred each have traditional IRAs with $100,000. These are invested identically and double over 10 years. The tax rate is 30%, so Jane pays $30,000 of tax to convert to a Roth IRA and has $70,000 to invest. At the end, she has $140,000 to withdraw tax-free.

Newsletter Sign-up

Markets A.M.

A pre-markets primer packed with news, trends and ideas. Plus, up-to-the-minute market data.


Preview

Subscribe
Meanwhile, Fred doesn’t do a conversion, and his $100,000 pot grows to $200,000. But when he withdraws it, he owes $60,000 of tax. That leaves him with $140,000 after tax, the same as Jane.

What makes a Roth conversion smart or not is if the tax rates vary. If Jane’s tax rate on a Roth conversion is 20% but her tax rate at withdrawal is 30%, she’s likely a gainer. And if Fred’s tax rate on a conversion is 30% but his tax rate at withdrawal is 15%, he’s not. 

Of course, evaluating a Roth conversion forces savers to guess the future. But sometimes that’s not hard—say, if a saver will be moving from high-tax state to a low- or no-tax state. The death of a spouse can also subject the survivor to a “widow’s penalty,” leaving him or her with a higher top rate as a single filer.  

But savers using outside funds to pay the conversion tax in order to keep more dollars in the Roth IRA should consider their source. If the tax money is sitting in a low-interest account, using them to pay conversion tax could be smart. But if the saver will be selling stock and paying 15% on the capital gains, think more. Could selling a loser offset that tax?

Investment returns also matter, and that involves more guessing. The higher the returns are, the greater the Roth IRA’s advantage. In general, investors have done better by betting on long-term asset growth than against it.   

I inherited a traditional IRA. Can I convert it to a Roth IRA? 
You can, if you are the surviving spouse of the owner. This involves rolling funds from the inherited IRA into an IRA of your own and then doing a Roth conversion.  

But most other heirs of traditional IRAs can’t do such conversions. And whether it’s a traditional or a Roth IRA, most heirs must drain the accounts within 10 years of the original owner’s death. 

How does the five-year rule affect my Roth IRA withdrawals?   
The five-year waiting period for tax-free Roth IRA withdrawals confuses many savers evaluating conversions. There are actually two five-year rules, but only one applies to people ages 59½ or older. The good news is that most older IRA owners won’t have trouble with it. Here’s a summary. 

If a saver who is 59½ or older has at least one Roth IRA open for five years or more, there’s no five-year waiting period for tax-free withdrawals after a conversion. 
"""

# COMMAND ----------

# MAGIC %md
# MAGIC #No Shot Prompting
# MAGIC This is prompting without any additional formatting. You are simply asking a question or instruction to the LLM just like you would with ChatGPT
# MAGIC
# MAGIC ### Task
# MAGIC Let's try to extract all the people's names in the article and have it placed in a jsonl format with each person as a separate entry. That way we can immediately use the jsonl structure for immediate use.

# COMMAND ----------

# MAGIC %md
# MAGIC ### DBRX No Shot Prompting
# MAGIC Try running this same cell a few times and see how the outputs change

# COMMAND ----------

# How to get your Databricks token: https://docs.databricks.com/en/dev-tools/auth/pat.html
# DATABRICKS_TOKEN = os.environ.get('DATABRICKS_TOKEN')
# Alternatively in a Databricks notebook you can use this:
DATABRICKS_TOKEN = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

# this is OpenAI's Client library that we can use to set up our API calls. It's simply a way to structure our API call to Databrick's Foundation Model API and does not hit OpenAI's servers

client = OpenAI(
  api_key=DATABRICKS_TOKEN,
  base_url=base_url
)

chat_completion = client.chat.completions.create(
  messages=[
  {
    "role": "system",
    "content": "You'll receive a WSJ Article. Extract the names from the article you receive and put them into a jsonl format."
  },
  {
    "role": "user",
    "content": content
  }
  ],
  model="databricks-dbrx-instruct",
  max_tokens=1000,
  top_p=0.1,
  temperature=0.1,
  n=1,
)
#there's a lot of other arguements that you can put in to the chat.completions.create request, but we're not going to go into them here.
print(f"The LLM Output:\n\n {chat_completion.choices[0].message.content}")

# COMMAND ----------

# MAGIC %md
# MAGIC ###Notice a few things: 
# MAGIC 1. It did not make a jsonl file per person. It made a single dictionary with the names in an array. 
# MAGIC 2. It did not capture all the people in the content
# MAGIC 3. It captured the names of the entities too but not consistent on what exactly consitutes a name. I personally would not pull Chief US Economist
# MAGIC 4. It added Joe to Biden even though the WSJ article never says Joe Biden explicitly. It does say Donald Trump explicitly
# MAGIC 5. The vagueness of the prompt forces DBRX is to make `assumptions`
# MAGIC 6. There are repeats here and there

# COMMAND ----------

# MAGIC %md
# MAGIC ### Llama3 No Shot Prompting

# COMMAND ----------

# How to get your Databricks token: https://docs.databricks.com/en/dev-tools/auth/pat.html
# DATABRICKS_TOKEN = os.environ.get('DATABRICKS_TOKEN')
# Alternatively in a Databricks notebook you can use this:
DATABRICKS_TOKEN = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

client = OpenAI(
  api_key=DATABRICKS_TOKEN,
  base_url=base_url
)

chat_completion = client.chat.completions.create(
  messages=[
  {
    "role": "system",
    "content": "You'll receive a WSJ Article. Extract the names from the article you receive and put them into a jsonl format."
  },
  {
    "role": "user",
    "content": content
  }
  ],
  model="databricks-meta-llama-3-3-70b-instruct",
  max_tokens=1000,
  top_p=0.1,
  temperature=0.1,
  n=1,
)

print(f"The LLM Output:\n\n {chat_completion.choices[0].message.content}")

# COMMAND ----------

# MAGIC %md
# MAGIC ###Notice a few things: 
# MAGIC 1. My output is in jsonl like I requested 
# MAGIC 2. Only the names of PEOPLE were pulled 
# MAGIC 3. It ran slower than DBRX due to its size and different model architecture
# MAGIC 4. It did not capture people's full names, Biden for Example 
# MAGIC 5. It did not fill in names like for Biden, it did not add Joe. 
# MAGIC
# MAGIC Llama3 did significantly better than DBRX in following my instructions but that is expected as Llama3 is a more performant LLM.

# COMMAND ----------

# MAGIC %md
# MAGIC #Note
# MAGIC This is not to bash DBRX. It's a reality that newer models WILL beat past models as research comes out with better ways to train models and so forth. It is inevitable. DBRX did not handle ambiguity as well as Llama3 but this is expected, as we learned, they are simply trying to predict the most likely word that comes next given the input. 
# MAGIC
# MAGIC **We are promoting our expertise and platform's ability to create state of the art models**
# MAGIC
# MAGIC The Mosaic Team has switched their pitch from having a leading open source model to offering LLM Training expertise and experience to help the customer optimally train their models. We provide the framework (Mosaic Composer), the hardware and the platform to do this training. This is how they were utilized for Shutterstock. 

# COMMAND ----------

# MAGIC %md
# MAGIC #Few Shot Prompting
# MAGIC We can improve the performance of Llama3 and significantly improve DBRX by using few-shot prompting as it removes some ambiguity.
# MAGIC
# MAGIC Now, let's also say we want a jsonl file with first and last name split up, plus a brief description about the person in the dictionary. Additionally, we need it to follow a very specific format so that it matches our columns in our table so that it's easy to append this information to a table. Let's pretend this table has columns first_name, last_name, description. So we need something consistently structured in a certain way.
# MAGIC
# MAGIC By using few shot prompting, we can "teach", "enforce", "tune" (whatever you want to call it) the model to what you expect to see, which greatly increases the consistency of the output to what you want. 
# MAGIC
# MAGIC A good rule of thumb is to provide 3 dummy examples or shots of what you want to see
# MAGIC
# MAGIC For example: 
# MAGIC {"first_name": "Austin", "last_name": "Choi", "description": "Austin works at Databricks and lives in LA"}
# MAGIC
# MAGIC By putting an example like the one above and specifically calling out that these are examples the LLM must follow, the LLM will try to match its output to those examples. 

# COMMAND ----------

# MAGIC %md
# MAGIC ###DBRX Few Shot Prompting

# COMMAND ----------

# How to get your Databricks token: https://docs.databricks.com/en/dev-tools/auth/pat.html
# DATABRICKS_TOKEN = os.environ.get('DATABRICKS_TOKEN')
# Alternatively in a Databricks notebook you can use this:
DATABRICKS_TOKEN = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

client = OpenAI(
  api_key=DATABRICKS_TOKEN,
  base_url=base_url
)

#Note in the prompt we kept the same prompt as we did for zero shot prompting. We just added the examples at the bottom. 
prompt = """You'll receive a WSJ Article. Extract the names and a brief one sentence description about the person based on the article you receive and put them into a jsonl format. Your answer must follow the examples below:

{"first_name": "Jane", "last_name": "Doe", "description": "Jane doe is a senior leader of consulting who travels the world to accomplish private equity tasks"},

{"first_name": "Alice", "last_name": "Smith", "description": "Alice Smith is a senior engineer who is passionate about AI and machine learning and evangelizes people on Linkedin"},

{"first_name": "Bob", "last_name": "Smith", "Bob Smith is the manager of a professional lacrosse team with significant influence in the lacrosse sports field"}

Take your time, think about your answer and review your answer to make sure the first and last names are correct. 
""" 
# You can add comments like "take your time" which does help the LLM perform better


chat_completion = client.chat.completions.create(
  messages=[
  {
    "role": "system",
    "content": prompt
  },
  {
    "role": "user",
    "content": content
  }
  ],
  model="databricks-dbrx-instruct",
  max_tokens=1000,
  top_p=0.1,
  temperature=0.1,
  n=1,
)

print(f"The LLM Output:\n\n {chat_completion.choices[0].message.content}")


# COMMAND ----------

# MAGIC %md
# MAGIC ###Notice a few things
# MAGIC 1. DBRX is capturing even less people now
# MAGIC 2. The information is correctly described for the people DBRX did capture
# MAGIC 3. DBRX now provides the information in a jsonl format thanks to few shot prompting and following my formatting 
# MAGIC
# MAGIC This output from DBRX is usable but still doesn't completely complete our requested ask. However, it is a substantial improvement in what we want.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Llama3 Few Shot Prompting

# COMMAND ----------

# How to get your Databricks token: https://docs.databricks.com/en/dev-tools/auth/pat.html
# DATABRICKS_TOKEN = os.environ.get('DATABRICKS_TOKEN')
# Alternatively in a Databricks notebook you can use this:
DATABRICKS_TOKEN = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

client = OpenAI(
  api_key=DATABRICKS_TOKEN,
  base_url=base_url
)
prompt = """You'll receive a WSJ Article. Extract the names and a brief one sentence description about the person based on the article you receive and put them into a jsonl format. Your answer must follow the examples below:

{"first_name": "Jane", "last_name": "Doe", "description": "Jane doe is a senior leader of consulting who travels the world to accomplish private equity tasks"},

{"first_name": "Alice", "last_name": "Smith", "description": "Alice Smith is a senior engineer who is passionate about AI and machine learning and evangelizes people on Linkedin"},

{"first_name": "Bob", "last_name": "Smith", "Bob Smith is the manager of a professional lacrosse team with significant influence in the lacrosse sports field"}

Take your time, think about your answer and review your answer to make sure the first and last names are correct. 
""" 

chat_completion = client.chat.completions.create(
  messages=[
  {
    "role": "system",
    "content": prompt
  },
  {
    "role": "user",
    "content": content
  }
  ],
  model="databricks-meta-llama-3-3-70b-instruct",
  max_tokens=1000,
  stop=["<|eot_id|>"],
  top_p=0.1,
  temperature=0.1,
  n=1,
)

print(f"The LLM Output:\n\n {chat_completion.choices[0].message.content}")

# COMMAND ----------

# MAGIC %md
# MAGIC ###Notice a few things
# MAGIC 1. More people are captured by Llama3 
# MAGIC 2. The descriptions generated by Llama3 are accurate and, for me personally, preferred as it gives a better description of the person. 
# MAGIC 3. Llama3 was slower than DBRX 
# MAGIC
# MAGIC At this point, I would still pick Llama3 because I like the descriptions more than what DBRX wrote. However, extraction is much more reliable now for both models

# COMMAND ----------

# MAGIC %md
# MAGIC # Self-Consistency
# MAGIC
# MAGIC This is another method to improve the outputs of the model and improve reasoning capabilities

# COMMAND ----------

# MAGIC %md
# MAGIC ###DBRX No consistency 
# MAGIC
# MAGIC Notice how DBRX cannot get this reasoning question correct

# COMMAND ----------

# How to get your Databricks token: https://docs.databricks.com/en/dev-tools/auth/pat.html
# DATABRICKS_TOKEN = os.environ.get('DATABRICKS_TOKEN')
# Alternatively in a Databricks notebook you can use this:
DATABRICKS_TOKEN = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

client = OpenAI(
  api_key=DATABRICKS_TOKEN,
  base_url=base_url
)

content = """When I was 6 my sister was half my age. Now I’m 70 how old is my sister?"""

chat_completion = client.chat.completions.create(
  messages=[
  {
    "role": "system",
    "content": ""
  },
  {
    "role": "user",
    "content": content
  }
  ],
  model="databricks-dbrx-instruct",
  max_tokens=1000,
  top_p=0.1,
  temperature=0.1,
  n=1,
)
Markdown(f"The LLM Output:\n\n {chat_completion.choices[0].message.content}")

# COMMAND ----------

# MAGIC %md
# MAGIC In this reasoning question, the LLM just does not do the math correctly. It provides its reasoning which is great for transparency as we see where in the reasoning the LLM is wrong. 
# MAGIC
# MAGIC Try different ages and see what answers you get and see how the reasoning is wrong. Here were some of the results I received when changing the input age of 6: 
# MAGIC
# MAGIC 1. Input Age: 24; Answer: 42; Right Answer: 58 
# MAGIC 2. Input Age: 12; Answer: 68; Right Answer: 64
# MAGIC 3. Input Age: 31; Answer: 58; Right Answer: 54.5 

# COMMAND ----------

# MAGIC %md
# MAGIC ###DBRX with Self-Consistency

# COMMAND ----------

# How to get your Databricks token: https://docs.databricks.com/en/dev-tools/auth/pat.html
# DATABRICKS_TOKEN = os.environ.get('DATABRICKS_TOKEN')
# Alternatively in a Databricks notebook you can use this:
DATABRICKS_TOKEN = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

client = OpenAI(
  api_key=DATABRICKS_TOKEN,
  base_url=base_url
)
prompt = """
Use the logic examples below to help answer the user's questions.

Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done,
there will be 21 trees. How many trees did the grove workers plant today?
A: We start with 15 trees. Later we have 21 trees. The difference must be the number of trees they planted.
So, they must have planted 21 - 15 = 6 trees. The answer is 6.

Q: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
A: There are 3 cars in the parking lot already. 2 more arrive. Now there are 3 + 2 = 5 cars. The answer is 5.

Q: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
A: Leah had 32 chocolates and Leah’s sister had 42. That means there were originally 32 + 42 = 74
chocolates. 35 have been eaten. So in total they still have 74 - 35 = 39 chocolates. The answer is 39.

Q: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops
did Jason give to Denny?
A: Jason had 20 lollipops. Since he only has 12 now, he must have given the rest to Denny. The number of
lollipops he has given to Denny must have been 20 - 12 = 8 lollipops. The answer is 8.

Q: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does
he have now?
A: He has 5 toys. He got 2 from mom, so after that he has 5 + 2 = 7 toys. Then he got 2 more from dad, so
in total he has 7 + 2 = 9 toys. The answer is 9.

Q: There were nine computers in the server room. Five more computers were installed each day, from
monday to thursday. How many computers are now in the server room?
A: There are 4 days from monday to thursday. 5 computers were added each day. That means in total 4 * 5 =
20 computers were added. There were 9 computers in the beginning, so now there are 9 + 20 = 29 computers.
The answer is 29.

Q: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many
golf balls did he have at the end of wednesday?
A: Michael initially had 58 balls. He lost 23 on Tuesday, so after that he has 58 - 23 = 35 balls. On
Wednesday he lost 2 more so now he has 35 - 2 = 33 balls. The answer is 33.

Q: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?
A: She bought 5 bagels for $3 each. This means she spent $15. She has $8 left:"""

content = """Q: When I was 6 my sister was half my age. Now I’m 70 how old is my sister?"""

chat_completion = client.chat.completions.create(
  messages=[
  {
    "role": "system",
    "content": prompt
  },
  {
    "role": "user",
    "content": content
  }
  ],
  model="databricks-dbrx-instruct",
  max_tokens=1000,
  top_p=0.1,
  temperature=0.1,
  n=1,
)

Markdown(f"The LLM Output:\n\n {chat_completion.choices[0].message.content}")

# COMMAND ----------

# MAGIC %md
# MAGIC Now we can see improvements to the answers coming from DBRX. Try the cell again with the same numbers: 24, 12 and 31. However, it's still not consistently correctly. 
# MAGIC
# MAGIC While the self-consistency prompt helps, you can see it's a lot of text which means more tokens thus high costs. This also means we need to take more of the precious context length when we could be using it for more relevant information. The performance gain with DBRX may be worth it.

# COMMAND ----------

# MAGIC %md
# MAGIC ###Llama3
# MAGIC Try the same question without Self-Consistency with Llama3. Llama3's superior performance to DBRX shows we don't really need Self-Consistency to improve the performance of Llama3

# COMMAND ----------

# How to get your Databricks token: https://docs.databricks.com/en/dev-tools/auth/pat.html
# DATABRICKS_TOKEN = os.environ.get('DATABRICKS_TOKEN')
# Alternatively in a Databricks notebook you can use this:
DATABRICKS_TOKEN = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

client = OpenAI(
  api_key=DATABRICKS_TOKEN,
  base_url=base_url
)

content = """When I was 6 my sister was half my age. Now I’m 70 how old is my sister?"""

chat_completion = client.chat.completions.create(
  messages=[
  {
    "role": "system",
    "content": ""
  },
  {
    "role": "user",
    "content": content
  }
  ],
  model="databricks-meta-llama-3-3-70b-instruct",
  max_tokens=1000,
  stop=["<|eot_id|>"],
  top_p=0.1,
  temperature=0.1,
  n=1,
)
#suffix, top_k, echo and error_behavior don't work
Markdown(f"The LLM Output:\n\n {chat_completion.choices[0].message.content}")
# print(chat_completion)


# COMMAND ----------

# MAGIC %md
# MAGIC See how Llama3 is generally correct without adding self-consistency. The benefits of this are: 
# MAGIC
# MAGIC 1. Less tokens means more cost effective messages
# MAGIC 2. Less complicated prompt 
# MAGIC 3. Room in the context window to augment the prompt with other information like RAG
# MAGIC
# MAGIC You can add self-consistency to improve performance but it's a much smaller gain compared to DBRX. It may be worth adding to ensure consistency for more complex reasoning questions

# COMMAND ----------

# MAGIC %md
# MAGIC # Augmenting Information
# MAGIC You can start to see how augmenting information in the prompt is critical to a successful use case. The LLM is heavily influenced by how and what you put into the prompt and we can see drastic improvements just by improving the prompts. 
# MAGIC
# MAGIC This is the basis of many compound AI solutions that are coming out today. We call various LLMs multiple times depending on what an LLM excels at and use prompt engineering to design a solution. Each prompt could potentially be drastically different depending on what step you are at in the solution. 
# MAGIC
# MAGIC <img width="1000px" src="https://blog.langchain.dev/content/images/2024/01/simple_multi_agent_diagram--1-.png">
# MAGIC
# MAGIC The picture above is from Langgraph, a branch of Langchain that helps build Agentic LLM solutions by connecting them together like a graph. Each "node" or LLM call (the researcher and router) have their own Prompt specifying what task they need to complete. They send outputs from each other across the edges or lines and use that information in their prompts to accomplish their tasks before sending a final output to the user. If the LLM determines a function or tool is necessary, then one is called. 

# COMMAND ----------

# MAGIC %md
# MAGIC #ReAct: The Basis for Function Calling
# MAGIC
# MAGIC We will wrap up this prompt engineering examples with ReAct, the idea behind function calling/tool calling. 
# MAGIC
# MAGIC When you need to pull in specific information but are unsure when it will be needed, you can use ReAct. 
# MAGIC
# MAGIC ReAct stands for Reasoning and Action where we use the reasoning capabilities of an LLM to determine what the next actions should be. This reasoning will determine if we need to 
# MAGIC 1. Use a function/tool to pull information 
# MAGIC 2. Route to a different LLM or a block of code 
# MAGIC 3. Handle exception or ask the question again 
# MAGIC 4. Ask for more information or follow up questions if there is not enough information in the initial question 
# MAGIC
# MAGIC If you use a multi-agent framework like AutoGen, Langgraph or CrewAI, there is a built in orchestrator that enables the LLM to "talk" to multiple agents until a satisfactory output is created. It will call functions and LLMs multiple times to achieve this
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC # ReAct (Function Calling/Tool Use) Code Activity
# MAGIC For this section, we will use the notebook to add python functions as tools for the LLM to use. We will use a pokeAPI to lookup up to date information about pokemon. 
# MAGIC
# MAGIC New Pokemon keep coming out that the LLM may not recognize a pokemon depending on when it was trained. We can call the pokeAPI for the most up to date information about a pokemon
# MAGIC
# MAGIC First, let's see what DBRX and Llama3 have to say about Sinistcha (pictured below) without any prompt engineering. 

# COMMAND ----------

# MAGIC %md
# MAGIC <img width="250px" align="center" src="https://archives.bulbagarden.net/media/upload/thumb/4/4c/1013Sinistcha.png/1200px-1013Sinistcha.png"/>

# COMMAND ----------

# MAGIC %md
# MAGIC ### DBRX No Tools
# MAGIC We will start this by having DBRX **roleplay** as a **pokemon master** that can answer any question about pokemon

# COMMAND ----------

# How to get your Databricks token: https://docs.databricks.com/en/dev-tools/auth/pat.html
# DATABRICKS_TOKEN = os.environ.get('DATABRICKS_TOKEN')
# Alternatively in a Databricks notebook you can use this:
DATABRICKS_TOKEN = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

client = OpenAI(
  api_key=DATABRICKS_TOKEN,
  base_url=base_url
)

prompt = """You are a pokemon master and know every single pokemon ever created by the Pokemon Company. You will be helping people answer questions about pokemon"""

content = """Tell me about Sinistcha"""

chat_completion = client.chat.completions.create(
  messages=[
  {
    "role": "system",
    "content": prompt
  },
  {
    "role": "user",
    "content": content
  }
  ],
  model="databricks-dbrx-instruct",
  max_tokens=1000,
  top_p=0.1,
  temperature=0.1,
  n=1,
)

Markdown(f"The LLM Output:\n\n {chat_completion.choices[0].message.content}")

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Things to note
# MAGIC DBRX believes Sinistcha is not a pokemon but it is from the most recent update to the pokemon games. No matter how many times you ask for Sinistcha, you will not get an answer. 
# MAGIC
# MAGIC DBRX also provided information about other pokemon I did not ask for but that can be fixed in the prompt
# MAGIC
# MAGIC When DBRX thinks Sinistcha is a pokemon, it hallucinates information

# COMMAND ----------

# MAGIC %md
# MAGIC ### Can Llama3 answer this question?

# COMMAND ----------

# How to get your Databricks token: https://docs.databricks.com/en/dev-tools/auth/pat.html
# DATABRICKS_TOKEN = os.environ.get('DATABRICKS_TOKEN')
# Alternatively in a Databricks notebook you can use this:
DATABRICKS_TOKEN = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

client = OpenAI(
  api_key=DATABRICKS_TOKEN,
  base_url=base_url
)

prompt = """You are a pokemon master and know every single pokemon ever created by the Pokemon Company. You will be helping people answer questions about pokemon"""

content = """Tell me about Sinistcha"""

chat_completion = client.chat.completions.create(
  messages=[
  {
    "role": "system",
    "content": prompt
  },
  {
    "role": "user",
    "content": content
  }
  ],
  model="databricks-meta-llama-3-3-70b-instruct",
  max_tokens=1000,
  top_p=0.1,
  temperature=0.1,
  n=1,
)

Markdown(f"The LLM Output:\n\n {chat_completion.choices[0].message.content}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Things to note 
# MAGIC
# MAGIC It's completely wrong and completely hallucinated with similar problems to DBRX.

# COMMAND ----------

# MAGIC %md
# MAGIC # Adding the Pokemon Look Up Tool
# MAGIC First, define the pokemon look up tool

# COMMAND ----------

import requests
def pokemon_lookup(pokemon_name):
    url = f"https://pokeapi.co/api/v2/pokemon/{pokemon_name.lower()}"
    response = requests.get(url)
    if response.status_code == 200:
        pokemon_data = response.json()
        pokemon_info = {
            "name": pokemon_data["name"],
            "height": pokemon_data["height"],
            "weight": pokemon_data["weight"],
            "abilities": [ability["ability"]["name"] for ability in pokemon_data["abilities"]],
            "types": [type_data["type"]["name"] for type_data in pokemon_data["types"]],
            "stats_name": [stat['stat']['name'] for stat in pokemon_data["stats"]],
            "stats_no": [stat['base_stat'] for stat in pokemon_data["stats"]]
        }
        results = str(pokemon_info)
        return results
    else:
        return None

# COMMAND ----------

# MAGIC %md
# MAGIC ###Test that the function works

# COMMAND ----------

pokemon_result = pokemon_lookup('Sinistcha')
print(pokemon_result)
print(type(pokemon_result))

# COMMAND ----------

# MAGIC %md
# MAGIC ###Define the code to add the tool and run the conversation
# MAGIC We first call the LLM to determine if a tool is necessary to answer the question. If determined yes, we call the function, get the results from the function, augment the information into a 2nd prompt and then send that over to the LLM for a final output. 
# MAGIC
# MAGIC **Don't worry about reviewing the code below.** It's to set up the payload we send to our Foundation Model API with what tools the LLM can access. Just run the cell and come back to it if you're curious on how we set this up. 

# COMMAND ----------

import json
from openai import RateLimitError

# A token and the workspace's base FMAPI URL are needed to talk to endpoints
fmapi_token = (
    dbutils.notebook.entry_point.getDbutils()
    .notebook()
    .getContext()
    .apiToken()
    .getOrElse(None)
)
fmapi_base_url = (
    base_url
)

openai_client = OpenAI(api_key=fmapi_token, base_url=fmapi_base_url)
MODEL_ENDPOINT_ID = "databricks-meta-llama-3-3-70b-instruct"

prompt = """You are a pokemon master and know every single pokemon ever created by the Pokemon Company. You will be helping people answer questions about pokemon. Stick strictly to the information provided to you to answer the question"""

def run_conversation(input):
    # Step 1: send the conversation and available functions to the model
    messages = [{"role": "system", "content": prompt},
                {"role": "user", "content": input}]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "pokemon_lookup",
                "description": "Get information about a pokemon. This tool should be used to check to see if the pokemon is real or not as well.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pokemon": {
                            "type": "string",
                            "description": "The pokemon the user is asking information for e.g bulbasaur",
                        },
                    },
                    "required": ["pokemon"],
                },
            },
        }
    ]
    #We've seen this response package in the past cells
    response = openai_client.chat.completions.create(
        model=MODEL_ENDPOINT_ID,
        messages=messages,
        tools=tools,
        tool_choice="auto",  # auto is default, but we'll be explicit
    )
    response_message = response.choices[0].message
    print(f"## Call #1 The Reasoning from the llm determining to use the function call:\n\n {response_message}\n")
    tool_calls = response_message.tool_calls
    # Step 2: check if the model wanted to call a function
    if tool_calls:
        # Step 3: call the function
        # Note: the JSON response may not always be valid; be sure to handle errors
        available_functions = {
            "pokemon_lookup": pokemon_lookup,
        }  # only one function in this example, but you can have multiple
        messages.append(response_message)  # extend conversation with assistant's reply
        # Step 4: send the info for each function call and function response to the model
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_to_call = available_functions[function_name]
            function_args = json.loads(tool_call.function.arguments)
            function_response = function_to_call(
                pokemon_name=function_args.get("pokemon")
            )
            messages.append(
                {
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "content": function_response,
                }
            )  # extend conversation with function response
        print(f"## Call #2 Prompt sent to LLM with function call results giving us the answer:\n\n {messages}\n")
        second_response = openai_client.chat.completions.create(
            model=MODEL_ENDPOINT_ID,
            messages=messages,
        )  # get a new response from the model where it can see the function response
        return second_response


# COMMAND ----------

# MAGIC %md
# MAGIC ###Ask the LLM a question about a pokemon
# MAGIC You'll notice in this output there will be two LLM calls. The first call is the LLM looking at the user's question and reasoning that a function/tool is required and specifies which function to call. We can use that to automatically call a function, especially if we have multiple. 
# MAGIC
# MAGIC The second call contains the output of the LLM after it receives a prompt augmented with the results of the function. This provides the LLM with relevant and accurate domain knowledge to answer the question correctly.

# COMMAND ----------

input1 = "Tell me about Sinistcha"
results1 = run_conversation(input1)
Markdown(f"**The LLM Answer:**\n\n{results1.choices[0].message.content}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Things to Note
# MAGIC 1. Llama3 now answers the pokemon question correctly
# MAGIC 2. Llama3 does not hallucinate as we asked it to stick to the information provided from the function call in the system prompt
# MAGIC 3. We called Llama3 twice here. This will not work for DBRX as DBRX does not support multi-turn conversations

# COMMAND ----------

# MAGIC %md
# MAGIC #Best Practices and Guidelines
# MAGIC We've run through some a lot of different prompt examples today and highlighting how much your LLM selection affects your outputs. You can probably tell at this point that the better your model is, the less prompt engineering you *theoretically* have to do (you should still do your due diligence though). This is why LLM selection is by far the most important part when determining how to tackle GenAI use cases. 
# MAGIC
# MAGIC That said, some customers prefer to use DBRX because we can confidently tell them what data we used to train DBRX, something that cannot be said about many different LLMs, including Llama3. Others may need a significantly smaller model (thus lower performance) due to budget constraints or other needs. You may just not be allowed to use the best LLM out there. It's important to be upfront to the customer that some things, like function calling, will suffer by choosing to use a worse LLM and a lot more work will be required. 
# MAGIC
# MAGIC That said, here are key tips and tricks to incorporate into the language of your prompts that can provide improvements: 
# MAGIC 1. **Be very clear and concise**: If your colleagues can understand your prompt, then you're off to a good start
# MAGIC 2. **Instruction placement**: Some LLMs struggle with instructions if the prompt gets too long. This becomes a big issue with RAG and known as the "needle in the haystack" problem. The placement of your prompt instructions can help with this. 
# MAGIC 3. **Organization**: Using deliminators or markers to make it clear where things are to the LLM. Some models like Claude have specific ways to deliminate information
# MAGIC 4. **Personas/Role Playing**: LLMs can easily incorprate writing styles, personas and much more if you specify it. This can help with clarity, conciseness or match the audience you're work for 
# MAGIC 5. **Explicitly tell the LLM to say "I don't know", "take your time", "think about your answer", "write your reasoning out and double check it"**: Because LLMs are trained to generate the next most likely token we want to see, LLMs will hallucinate to try and generate something you want to read as fast as possible. Stating that we will accept I don't know allows the LLM an out essentially and reduces hallucinations. This is less of a problem with more powerful models but are strategies used by them to improve outputs.
# MAGIC 6. **Experiment with prompts**: LLMs are probablistic. You need to try different prompts to see how the LLM will generate outputs compared to other LLMs. This will give you understanding of how these LLMs behave 
# MAGIC 7. **Provide relevant information and explicitly tell the LLM to use it**: The more relevant information you provide an LLM, the better it can respond in the domain you're looking for 
# MAGIC 8. **Just Ask**: You'll be surprised at what the LLM will do for you. Just try it! For example, if you want it order things a certain way, just ask! Do you want HTML tags around specific words? Just ask! 

# COMMAND ----------

# MAGIC %md
# MAGIC #Hands-On Exercise
# MAGIC A customer has reached out asking if Databricks had a way to translate highly technical chemistry passages into a language a high schooler could understand it. The customer wants a delta table with scientific terms and a description of what the term is so that they can use it for academic purposes. Additionally, they want to see if any preliminary analysis can be done on the passages to quickly understand the material 
# MAGIC
# MAGIC Your goal is to complete as many of the tasks below within a single prompt. You are making five different prompts for each of the tasks. Your prompts should not allow irrelevant questions to be answered and avoid hallucinating as much as possible. You must use Llama3 through our Foundation Model API. How you choose to interact with the API is up to you. You may not use our RAG capabilities like vector search/index or other external tools like ChromaDB. 
# MAGIC
# MAGIC You can use the test questions below to see how well you designed your prompts. The test questions will testing the following: 
# MAGIC 1. Hallucination 
# MAGIC 2. Accuracy
# MAGIC 3. Security 
# MAGIC
# MAGIC Remember that simple adjustments to the prompt language and statements like "I don't know" can drastically change and/or improve outputs
# MAGIC
# MAGIC ##Tasks 
# MAGIC 1. Translate the content below to a high school readability level. You can calculate the readability level here: https://goodcalculators.com/flesch-kincaid-calculator/. It is currently *college graduate* or very difficult to read
# MAGIC 2. Have the LLM role play. We recommend a pirate but feel free to use something else like Teenage Mutant Ninja Turtles
# MAGIC 3. Output a json format formatted like the following: {"term": "pathogenExample", "definition": "this is an example pathogen"}. Theoretically, this would be used to write to a delta table but we are just focused on getting the output to be consistent. 
# MAGIC 4. Capture most of the scientific terms. Hint: You will want to provide examples of scientific terms so that the LLM knows what you count as said terms. 
# MAGIC 5. Calculate the age of the rats when each recorded event happened: Body weight, adrenals weight, adrenal protein count. So for the 2 day old rat, the rat was 202, 122 and 32 days old for each event respectively. 
# MAGIC 6. (Optional) Use one prompt to accomplish all the tasks above
# MAGIC 7. (Optional) Ask the LLM to format your output in a specific way like markdown or use bold or use HTML 
# MAGIC
# MAGIC ##Starter Prompt 
# MAGIC prompt = 
# MAGIC
# MAGIC "
# MAGIC
# MAGIC (your instructions here)
# MAGIC
# MAGIC (content seen below here)
# MAGIC
# MAGIC "

# COMMAND ----------

content = """
Male Wistar specific-pathogen-free rats aged 2, 7, 17, 30, 60, 120, 200, 360 and 
600 days, all killed in experiment on the same day, were examined. The body 
weight significantly increased until the 200th day, the weight of adrenals until 
the 120th day and the adrenal protein content until the 30th day of life. The 
adrenaline content of the adrenals increased continuously during the 600 days 
studied. Adrenal noradrenaline content increased rapidly over the first 17 days, 
remained at a stable level until the 120th day, and rose to a higher level after 
200 days. The activity of adrenal catecholamine-synthesizing enzymes also 
increased with age: tyrosine hydroxylase gradually increased until the 360th 
day, dopamine-beta-hydroxylase and phenylethanolamine-N-methyl transferase until 
the 200th day. Our results demonstrate that, in the rat, during development 
there is a gradual increase of adrenal weight, adrenaline content, tyrosine 
hydroxylase and phenylethanolamine-N-methyl transferase activity until 
maturation (120th day), whereas the adrenal noradrenaline content reaches the 
adult values earlier, around the 17th day. During aging, adrenal catecholamines 
significantly increase when compared to young-adult rats (120-day-old), probably 
due to the elevated activity of the adrenal catecholamine-synthesizing enzymes. 
The increased adrenal catecholamine levels in old animals might be connected 
with a higher incidence of cardiovascular diseases in aged.
"""

# COMMAND ----------

# MAGIC %md
# MAGIC ###Working area.
# MAGIC 1. Translate the content below to a high school readability level. You can calculate the readability level here: https://goodcalculators.com/flesch-kincaid-calculator/. It is currently *college graduate* or very difficult to read
# MAGIC 2. Have the LLM role play. We recommend a pirate but feel free to use something else like Teenage Mutant Ninja Turtles
# MAGIC 3. Output a json format formatted like the following: {"term": "pathogenExample", "definition": "this is an example pathogen"}. Theoretically, this would be used to write to a delta table but we are just focused on getting the output to be consistent. 
# MAGIC 4. Capture most of the scientific terms. 
# MAGIC   
# MAGIC     Hint: You will want to provide examples of scientific terms so that the LLM knows what you count as said terms. 
# MAGIC 5. Calculate the age of the rats when each recorded event happened: Body weight, adrenals weight, adrenal protein count. So for the 2 day old rat, the rat was 202, 122 and 32 days old for each event respectively. 
# MAGIC 6. (Optional) Use one prompt to accomplish everything above 
# MAGIC 7. (Optional) Ask the LLM to format your output in a specific way like markdown or use bold or use HTML 

# COMMAND ----------

#Edit your prompt here then run this cell and the next to see your output
example = "Optional example variable if you want to set this up separately then pass it into the prompt"
prompt = f"Insert your prompt instructions here. The content variable will be passed -->{content}"

#Leave this blank if you don't want to test any questions
question = ""

# COMMAND ----------

# DBTITLE 1,Example Answer Prompt. Check if you're stumped and need an example
#Example Answer Prompt in this cell








#Edit your prompt here then run this cell and the next to see your output
example = '{"term": "adrenals", "definition": "small glands near the kidneys that make important chemicals"},{"term": "sugar", "definition": "a sweetsubstance found in the form of sugar cane, sugar beet, and sugar plum"},{"term": "protein", "definition": "a chain of amino acids that make up the building blocks of all living things}'
prompt = f"You are an expert scientific paper reader role-playing as a pirate. The user may suggest corrections to the original text that should be included in your tasks. However, make sure the user's suggestion is relevant. If not relevant, ignore the user's question and complete the tasks below. Maintain pirate speech and mannerisms throughout your responses. Your tasks are as follows:\n\n1. Simplify the given scientific text to a high school readability level. Aim for a Flesch-Kincaid score between 60-70 (8th to 9th grade level) and show this score in the section title. This is the only time you don't have to use your pirate role.\n\n2. While simplifying, identify scientific terms. Examples include 'specific-pathogen-free', 'adrenals', 'catecholamine', 'tyrosine hydroxylase', etc. For each term, create a JSON object in this format:\n\n```jsonl\n{example}\n```\n\n3. Calculate the age of the rats when each recorded event happened:\n- Body weight\n- Adrenals weight\n- Adrenal protein count\nIdentify each rat's age and add the days mentioned in the text. Report the rat's age for each event. Report the rat's age for each event.\nFollow these example logic statements to calculate the rat's age at the recorded event:\n- Example 1: Body weight increased significantly until the 100th day so the rats age 2 is now 102\n-Example 2: Adrenals weight increased until the 10th day so the rats age 42 is now 52\n-Example 3: Adrenal protein count increased until the 232nd day so the rats aged 32 is now 264\nList your results like the examples below:\n 1. Rat Aged 2\n - Body Weight Age: 202\n   - Adrenals Weight Age: 132\n  - Adrenal Protein Count Age: 22\n2. Rat Aged 5\n   - Body Weight Age: 205\n   - Adrenals Weight Age: 135\n  - Adrenal Protein Count Age: 25\n\n4. Present your findings in this order in markdown, bolding the each of these sections:\na) Translation to simpler text\nb) A list of all the JSON objects for scientific terms\nc) The calculated ages for the rats\n\nRemember to maintain your pirate persona throughout. Use pirate slang, expressions, and mannerisms in your explanations and responses. However, ensure that your pirate speak doesn't interfere with the clarity of the simplified scientific information.\n\nAvoid hallucinating. Stick to the facts provided, just express them in simpler terms and with a pirate's flair.\nHere's the scientific text to process:\n\n{content}"

#Leave this blank if you don't want to test any questions
question = ""

# COMMAND ----------

# How to get your Databricks token: https://docs.databricks.com/en/dev-tools/auth/pat.html
# DATABRICKS_TOKEN = os.environ.get('DATABRICKS_TOKEN')
# Alternatively in a Databricks notebook you can use this:
DATABRICKS_TOKEN = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

client = OpenAI(
  api_key=DATABRICKS_TOKEN,
  base_url=base_url
)

chat_completion = client.chat.completions.create(
  messages=[
  {
    "role": "system",
    "content": prompt
  },
  {
    "role": "user",
    "content": question
  }
  ],
  model="databricks-meta-llama-3-3-70b-instruct",
  max_tokens=2000,
  top_p=0.1,
  temperature=0.1,
  n=1,
)
print(chat_completion.choices[0].message.content)
Markdown(f"**The LLM Output:**\n\n {chat_completion.choices[0].message.content}")

# COMMAND ----------

# MAGIC %md
# MAGIC #Next Step: Retrieval Augmented Generation 
# MAGIC You already saw the substantial improvement function calling brought to Llama3, where we augmented external data from the pokemon API to the final prompt we sent to Llama3. By doing so, Llama3 had the latest information about domain-specific knowledge and could answer accurately. 
# MAGIC
# MAGIC We can do a lot more if we could find relevant information based on the task or question brought up by a user. These models have enough of a context window to add a lot more information into the prompt. 
# MAGIC
# MAGIC Thus, RAG was created and continues to be a powerful method to augment and design GenAI applications
