# Databricks notebook source
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

# MAGIC %md ##First, set up environment variables

# COMMAND ----------

!pip install transformers datasets mlflow
dbutils.library.restartPython()

# COMMAND ----------

from databricks.sdk import WorkspaceClient

w = WorkspaceClient()
user_email = w.current_user.me().display_name
username = user_email.split("@")[0]
default_schema_name = username.replace(" ", "_").lower()

# COMMAND ----------

dbutils.widgets.text("catalog_name", "austin_choi_demo_catalog", "Data UC Catalog") #change this to a catalog of your choice
dbutils.widgets.text("schema_name", "demo_data", "Data UC Schema") #change this to a schema of your choice
dbutils.widgets.text("table_name", "batch_sentiment_data", "Data UC Table") #change this to a table name of your choice 

catalog_name = dbutils.widgets.get("catalog_name")
schema_name = dbutils.widgets.get("schema_name")
table_name = dbutils.widgets.get("table_name")

# COMMAND ----------

spark.sql(
f"""
    CREATE SCHEMA IF NOT EXISTS {catalog_name}.{schema_name}
"""
)

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

# MAGIC %sql
# MAGIC CREATE OR REPLACE FUNCTION identifier(:catalog_name||'.'||:schema_name||'.'||'purchase_location')()
# MAGIC     RETURNS TABLE(name STRING, purchases INTEGER)
# MAGIC     COMMENT 'Use this tool to find total purchase information about a particular location. This tool will provide a list of destinations that you will use to help you answer questions'
# MAGIC     RETURN SELECT dl.name AS Destination, count(tp.destination_id) AS Total_Purchases_Per_Destination
# MAGIC              FROM main.dbdemos_fs_travel.travel_purchase tp join main.dbdemos_fs_travel.destination_location dl on tp.destination_id = dl.destination_id
# MAGIC              group by dl.name
# MAGIC              order by count(tp.destination_id) desc
# MAGIC              LIMIT 10;

# COMMAND ----------

# MAGIC %md 
# MAGIC #Batch Inference Sentiment Demo
# MAGIC
# MAGIC In this notebook, we'll walk through an end-to-end batch inference solution to extract sentiment from financial Twitter data. The notebook covers:
# MAGIC - Setting up an environment, relevant variables, and underlying UC data
# MAGIC - Executing a prompt-only batch inference query with ai_query
# MAGIC - Combining structured outputs with batch inference to generate more reliable outputs
# MAGIC - Adding your new AI Query as a UC Function

# COMMAND ----------

# MAGIC %md ## Set up data
# MAGIC
# MAGIC We'll use an opensource Huggingface finance news dataset to classify company sentiment

# COMMAND ----------

from datasets import load_dataset

dataset = load_dataset("zeroshot/twitter-financial-news-sentiment", cache_dir=None) 

# COMMAND ----------

train = dataset['train'].to_pandas()
validation = dataset['validation'].to_pandas()
validation

# COMMAND ----------

train_spark = spark.createDataFrame(train)
validation_spark = spark.createDataFrame(validation)
train_spark.write.mode('overwrite').saveAsTable(".".join([catalog_name, schema_name, f"{table_name}_train"]))
validation_spark.write.mode('overwrite').saveAsTable(".".join([catalog_name, schema_name, f"{table_name}_val"]))

# COMMAND ----------

display(
    spark.sql(
        f"SELECT * FROM {catalog_name}.{schema_name}.{table_name}_train LIMIT 10"
    )
)

# COMMAND ----------

# MAGIC %md ## Use batch inference with `ai_query`, using only a prompt
# MAGIC
# MAGIC The predictions should only be 0, 1, or 2. Can prompting guarantee the outputs?

# COMMAND ----------

# DBTITLE 1,Validation Set 2.4K Rows
import time

# endpoint_name = "databricks-meta-llama-3-1-8b-instruct" #12 seconds average 
endpoint_name = "databricks-meta-llama-3-3-70b-instruct" #180 seconds average
start_time = time.time()

command = f"""
    SELECT text,  
    ai_query(
        \'{endpoint_name}\', --endpoint name
        CONCAT('Classify the financial news-related Tweet sentiment as 0 for bearish, 1 for bullish, or 2 for neutral. Give just the number.', text)
    ) AS sentiment_pred,
    label as sentiment_gt 
    FROM {catalog_name}.{schema_name}.{table_name}_val
"""

result = spark.sql(command)

display(result)

end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time} seconds")

# COMMAND ----------

result_baseline_pd_llama_70b = result.toPandas()

result_baseline_pd_llama_70b.sentiment_pred.value_counts()

# COMMAND ----------

result_baseline_pd_llama_70b.sentiment_gt.value_counts()

# COMMAND ----------

confusion_matrix = result_baseline_pd_llama_70b.pivot_table(index='sentiment_gt', columns='sentiment_pred', aggfunc='size', fill_value=0)
display(confusion_matrix)

# COMMAND ----------

# DBTITLE 1,Validation Set (2.4K rows)
import time

endpoint_name = "databricks-meta-llama-3-1-8b-instruct" #20 seconds average 
# endpoint_name = "databricks-meta-llama-3-3-70b-instruct" #180 seconds average
start_time = time.time()

command = f"""
    SELECT text,  
    ai_query(
        \'{endpoint_name}\', --endpoint name
        CONCAT('Classify the financial news-related Tweet sentiment as 0 for bearish, 1 for bullish, or 2 for neutral. Give just the number.', text)
    ) AS sentiment_pred,
    label as sentiment_gt 
    FROM {catalog_name}.{schema_name}.{table_name}_val
"""

result = spark.sql(command)

display(result)

end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time} seconds")

# COMMAND ----------

# DBTITLE 1,Did predictions get generated in correct format?
result_baseline_pd = result.toPandas()

result_baseline_pd.sentiment_pred.value_counts()

# COMMAND ----------

result_baseline_pd.sentiment_gt.value_counts()

# COMMAND ----------

# DBTITLE 1,How accurate are the predictions compared to ground truth?
confusion_matrix = result_baseline_pd.pivot_table(index='sentiment_gt', columns='sentiment_pred', aggfunc='size', fill_value=0)
display(confusion_matrix)

# COMMAND ----------

# MAGIC %md There may be a handful of improperly formatted responses

# COMMAND ----------

# MAGIC %md ## Using Structured Output with Batch Inference
# MAGIC
# MAGIC We can enforce formatting constraints rather than relying on prompts. This is particularly useful for smaller models that struggle in accuracy. If we know what outputs we are looking for, we can enforce it.
# MAGIC
# MAGIC Documentation: https://docs.databricks.com/aws/en/sql/language-manual/functions/ai_query#enforce-output-schema-with-structured-output

# COMMAND ----------

response_schema = """
{
    "type": "json_schema",
    "json_schema":
        {
        "name": "sentiment_score",
        "schema":
            {
            "type": "object",
            "properties":
            {
            "sentiment": { "type": "string" ,
                        "enum": ["0", "1", "2"]}
            }
            },
        "strict": true
        }
}
"""

# COMMAND ----------

import time
endpoint_name = "databricks-meta-llama-3-1-8b-instruct" #14 seconds average 

start_time = time.time()

result_structured = spark.sql(
f"""
    SELECT text,  
    ai_query(
        \'{endpoint_name}\', --endpoint name
        CONCAT('Classify the financial news-related Tweet sentiment as 0 for bearish, 1 for bullish, or 2 for neutral. Give just the number', text),
        responseFormat => '{response_schema}'
    ) AS sentiment_pred,
    label as sentiment_gt,
    CAST(get_json_object(sentiment_pred, '$.sentiment') AS LONG) AS sentiment_pred_value
    FROM {catalog_name}.{schema_name}.{table_name}_val
""")

display(result_structured)

end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time} seconds")

# COMMAND ----------

# DBTITLE 1,Using structured outputs, did predictions get generated in correct format?
result_structured_pd = result_structured.toPandas()

result_structured_pd.sentiment_pred_value.value_counts()

# COMMAND ----------

result_structured_pd.sentiment_gt.value_counts()

# COMMAND ----------

# DBTITLE 1,How accurate are the predictions compared to ground truth?
confusion_matrix_structured = result_structured_pd.pivot_table(index='sentiment_gt', columns='sentiment_pred_value', aggfunc='size', fill_value=0)
display(confusion_matrix_structured)

# COMMAND ----------

# MAGIC %md
# MAGIC Structured outputs ensures compliance to the allowed results. The run time is only marginally longer!

# COMMAND ----------

# MAGIC %md
# MAGIC #AI QUERY as a UC Function or UDF
# MAGIC
# MAGIC You can leverage the governance and spark capabilities of a UDF by defining a sql function and registering it on Unity Catalog! This gives you more flexibility in using a customized AI Query with all your prompts across workflows and share them in your workspace.  

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE FUNCTION identifier(CONCAT(:catalog_name||'.'||:schema_name||'.'||'batch_inference'))(text STRING)
# MAGIC     RETURNS STRING
# MAGIC     COMMENT 'When user says, start batch inference, Use this tool to run a batch inference job to review and correct the spelling of make of a car.'
# MAGIC     RETURN SELECT   -- Placeholder for the input column
# MAGIC           ai_query(
# MAGIC             'databricks-meta-llama-3-1-8b-instruct',
# MAGIC             CONCAT(format_string('You will always receive a make of a car. Check to see if it is misspelled and a real car. Correct the mistake. Only provide the corrected make. Never add additional details'), text)    -- Placeholder for the prompt and input
# MAGIC           ) AS ai_guess  -- Placeholder for the output column 

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT `Misspelled Make`, identifier(CONCAT(:catalog_name||'.'||:schema_name||'.'||'batch_inference'))(`Misspelled Make`) AS ai_guess from identifier(CONCAT(:catalog_name||'.'||:schema_name||'.'||'synthetic_car_data'));

# COMMAND ----------

# MAGIC %md
# MAGIC #Access UC Functions via Code
# MAGIC
# MAGIC You can use Langchain to access UC functions through code! 

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
    function_names=[f"{catalog_name}.{schema_name}.*"]
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

# MAGIC %md
# MAGIC #Workshop: Make your own Response Structure! 
