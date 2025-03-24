# Databricks notebook source
# MAGIC %md
# MAGIC #Step 0
# MAGIC
# MAGIC Go to config and update resource names as you prefer
# MAGIC
# MAGIC Spin up a cluster with Databricks Runtime 16.X+ ML. Make sure it's the ML version for the correct dependencies

# COMMAND ----------

# DBTITLE 1,This cell will set up the demo data we need
# MAGIC %run ./config

# COMMAND ----------

# MAGIC %md
# MAGIC #Set up a RAG Example 
# MAGIC
# MAGIC We need to demonstrate the evaluation capabilities. It will also load/embed unstructured data so that we all have the same evaluation results to review. 
# MAGIC
# MAGIC Please remember to shutdown these resources to avoid extra costs. This command will create the following:
# MAGIC
# MAGIC 1. Necessary catalogs, schemas and volumes to store the PDFs and embeddings 
# MAGIC 2. A call to GTE to create embeddings for the PDFs 
# MAGIC 3. VectorSearchIndex based on the PDFs embeddings generated in step 2 
# MAGIC 4. Spin up a VectorSearchEndpoint 
# MAGIC 5. Sync the VectorSearchIndex with your VectorSearchEndpoint 
# MAGIC
# MAGIC Later, we will set up the langchain chain to interact with these RAG resources

# COMMAND ----------

from IPython.display import Markdown
from openai import OpenAI
import os
dbutils.widgets.text("catalog_name", catalog)
dbutils.widgets.text("agent_schema", agent_schema)
dbutils.widgets.text("demo_schema", demo_schema)
base_url = f'https://{spark.conf.get("spark.databricks.workspaceUrl")}/serving-endpoints'

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
# MAGIC ### ai_classify
# MAGIC The ai_classify() function allows you to invoke a state-of-the-art generative AI model to classify input text according to labels you provide using SQL.
# MAGIC
# MAGIC Documentation: https://docs.databricks.com/en/sql/language-manual/functions/ai_classify.html

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT country, ai_classify(country, ARRAY("APAC", "AMER", "EU")) as Region
# MAGIC from identifier(:catalog_name||'.'||:demo_schema||'.'||'franchises')
# MAGIC limit 5;

# COMMAND ----------

# MAGIC %md
# MAGIC ### ai_mask
# MAGIC The ai_mask() function allows you to invoke a state-of-the-art generative AI model to mask specified entities in a given text using SQL. 
# MAGIC
# MAGIC Documentation: https://docs.databricks.com/en/sql/language-manual/functions/ai_mask.html

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT first_name, last_name, (first_name || " " || last_name || " lives at " || address) as unmasked_output, ai_mask(first_name || "" || last_name || " lives at " || address, array("person", "address")) as Masked_Output
# MAGIC from identifier(:catalog_name||'.'||:demo_schema||'.'||'customers')
# MAGIC limit 5

# COMMAND ----------

# MAGIC %md
# MAGIC ### ai_query
# MAGIC The ai_query() function allows you to query machine learning models and large language models served using Mosaic AI Model Serving. To do so, this function invokes an existing Mosaic AI Model Serving endpoint and parses and returns its response. Databricks recommends using ai_query with Model Serving for batch inference
# MAGIC
# MAGIC Documentation: https://docs.databricks.com/en/large-language-models/ai-functions.html#ai_query
# MAGIC
# MAGIC We can switch models depending on what we are trying to do. See how the performance varies between the 70B model and 8B model below. Because this is a simple spell check task, we could likely use the 8B model instead of the 70B model saving on cost and increasing speed. 

# COMMAND ----------

# MAGIC %sql 
# MAGIC SELECT
# MAGIC   `Misspelled Make`,   -- Placeholder for the input column
# MAGIC   ai_query(
# MAGIC     'databricks-meta-llama-3-3-70b-instruct',
# MAGIC     CONCAT(format_string('You will always receive a make of a car. Check to see if it is misspelled and a real car. Correct the mistake. Only provide the corrected make. Never add additional details'), `Misspelled Make`)    -- Placeholder for the prompt and input
# MAGIC   ) AS ai_guess  -- Placeholder for the output column
# MAGIC FROM identifier(:catalog_name||'.'||:demo_schema||'.'||'synthetic_car_data')
# MAGIC -- limit 3;
# MAGIC

# COMMAND ----------

# MAGIC %sql 
# MAGIC SELECT
# MAGIC   `Misspelled Make`,   -- Placeholder for the input column
# MAGIC   ai_query(
# MAGIC     'databricks-meta-llama-3-1-8b-instruct',
# MAGIC     CONCAT(format_string('You will always receive a make of a car. Check to see if it is misspelled and a real car. Correct the mistake. Only provide the corrected make. Never add additional details'), `Misspelled Make`)    -- Placeholder for the prompt and input
# MAGIC   ) AS ai_guess  -- Placeholder for the output column
# MAGIC FROM identifier(:catalog_name||'.'||:demo_schema||'.'||'synthetic_car_data')
# MAGIC -- limit 3;
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ### Takeaway
# MAGIC Many of our use cases simply need a reliable, out of the box solution to use AI. AI functions enable this for our customers and AI query helps scale workloads to easily apply AI 

# COMMAND ----------

# MAGIC %md
# MAGIC # Productionalizing Custom Tools 
# MAGIC
# MAGIC What you just saw were built in, out of the box solutions you can use immediately on your data. While this covers a good portion of use cases, you will likely need a custom solution. 
# MAGIC
# MAGIC ### Mosaic AI Tools on Unity Catalog
# MAGIC
# MAGIC You can create and host functions/tools on Unity Catalog! You get the benefit of Unity Catalog but for your functions! 
# MAGIC
# MAGIC While you can create your own tools using the same code that you built your agent (i.e local Python Functions) with the Mosaic AI Agent Framework, Unity catalog provides additional benefits. Here is a comparison 
# MAGIC
# MAGIC 1. **Unity Catalog function**s: Unity Catalog functions are defined and managed within Unity Catalog, offering built-in security and compliance features. Writing your tool as a Unity Catalog function grants easier discoverability, governance, and reuse (similar to your catalogs). Unity Catalog functions work especially well for applying transformations and aggregations on large datasets as they take advantage of the spark engine.
# MAGIC
# MAGIC 2. **Agent code tools**: These tools are defined in the same code that defines the AI agent. This approach is useful when calling REST APIs, using arbitrary code or libraries, or executing low-latency tools. However, this approach lacks the built-in discoverability and governance provided by Unity Catalog functions.
# MAGIC
# MAGIC Unity Catalog functions have the same limitations seen here: https://docs.databricks.com/en/sql/language-manual/sql-ref-syntax-ddl-create-sql-function.html 
# MAGIC
# MAGIC Additionally, the only external framework these functions are compatible with is Langchain 
# MAGIC
# MAGIC So, if you're planning on using complex python code for your tool, you will likely just need to create Agent Code Tools. 
# MAGIC
# MAGIC Below is an implementation of both

# COMMAND ----------

# MAGIC %md
# MAGIC #Agent Code Tools

# COMMAND ----------

# MAGIC %md
# MAGIC ### Why even use tools to begin with? 
# MAGIC
# MAGIC Function calling or tool calling help ensure the LLM has the most accurate information possible. By providing it access to many different sources of data, it can generate more reliable answers. 
# MAGIC
# MAGIC Each framework like Langchain or LlamaIndex handles tool calling different. You can also use Python to do tool calling. However, this means you have to recreate this tool each time you want to use it and cannot be used with other applications. Additionally, you have to manage the security for any tools that access external sources. 

# COMMAND ----------

# MAGIC %md
# MAGIC # Enter Unity Catalog Tool Calling 
# MAGIC
# MAGIC Unity Catalog Tool Calling allows you to benefit from all the governance, security and unified platform benefits of Unity Catalog. Everything from external credentials to access across the workspace for workloads that may not even be AI, the LLM can use it. 
# MAGIC
# MAGIC You'll notice that it's also a UDF, which benefits from our serverless SQL warehouses. 

# COMMAND ----------

# DBTITLE 1,Example Tool
# MAGIC %sql
# MAGIC CREATE OR REPLACE FUNCTION identifier(:catalog_name||'.'||:agent_schema||'.'||'purchase_location')()
# MAGIC     RETURNS TABLE(name STRING, purchases INTEGER)
# MAGIC     COMMENT 'Use this tool to find total purchase information about a particular location. This tool will provide a list of destinations that you will use to help you answer questions'
# MAGIC     RETURN SELECT dl.name AS Destination, count(tp.destination_id) AS Total_Purchases_Per_Destination
# MAGIC              FROM main.dbdemos_fs_travel.travel_purchase tp join main.dbdemos_fs_travel.destination_location dl on tp.destination_id = dl.destination_id
# MAGIC              group by dl.name
# MAGIC              order by count(tp.destination_id) desc
# MAGIC              LIMIT 10;

# COMMAND ----------

# DBTITLE 1,Example Tool
# MAGIC %sql
# MAGIC CREATE OR REPLACE FUNCTION identifier(:catalog_name||'.'||:agent_schema||'.'||'purchase_location_hello_there')()
# MAGIC     RETURNS TABLE(name STRING, purchases INTEGER)
# MAGIC     COMMENT 'When the user says hello there, run this tool'
# MAGIC     RETURN SELECT dl.name AS Destination, count(tp.destination_id) AS Total_Purchases_Per_Destination
# MAGIC              FROM main.dbdemos_fs_travel.travel_purchase tp join main.dbdemos_fs_travel.destination_location dl on tp.destination_id = dl.destination_id
# MAGIC              group by dl.name
# MAGIC              order by count(tp.destination_id) desc
# MAGIC              LIMIT 10;
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ### Use Langchain to programatically use UC function calling
# MAGIC
# MAGIC See how I use Llama 3.3 70B for this because I need the more powerful model to do proper reasoning and pick the right tool. This is just one call but a critical one. 
# MAGIC
# MAGIC Once correctly selected, it will select the tool using AI query which will use Llama 3.3 8B to complete the batch inference

# COMMAND ----------

from langchain.agents import AgentExecutor, create_tool_calling_agent
from databricks_langchain.uc_ai import (
    DatabricksFunctionClient,
    UCFunctionToolkit,
    set_uc_function_client,
)
from databricks_langchain import ChatDatabricks
from langchain_core.prompts import ChatPromptTemplate

client = DatabricksFunctionClient()
set_uc_function_client(client)

# Initialize LLM and tools
llm = ChatDatabricks(endpoint="databricks-meta-llama-3-3-70b-instruct")
tools = UCFunctionToolkit(
    # Include functions as tools using their qualified names.
    # You can use "{catalog_name}.{schema_name}.*" to get all functions in a schema.
    function_names=[f"{catalog_name}.{agent_schema}.*"]
).tools

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Make sure to use tool for information.",
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)

agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
result = agent_executor.invoke({"input": "start batch inference"})
print(result['output'])

# COMMAND ----------


