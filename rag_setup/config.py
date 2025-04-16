# Databricks notebook source
# %pip install --upgrade --quiet databricks-sdk lxml langchain databricks-vectorsearch cloudpickle openai pypdf llama_index langgraph==0.3.4 sqlalchemy transformers openai mlflow mlflow[databricks] langchain_community databricks-agents databricks-langchain uv torch databricks-connect==16.1.*

# COMMAND ----------

# MAGIC %pip install --upgrade --quiet databricks-sdk lxml langchain databricks-vectorsearch cloudpickle openai pypdf llama_index langgraph==0.3.4 sqlalchemy transformers openai mlflow mlflow[databricks] langchain_community databricks-agents databricks-langchain uv markdownify

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Edit this cell with your resource names
catalog = "genai_in_production_demo_catalog"
agent_schema = "agents"
demo_schema = "demo_data"
volumeName = "rag_volume"
folderName = "sample_pdf_folder"
vectorSearchIndexName = "pdf_content_embeddings_index"
# vectorSearchIndexName = "databricks_documentation_index"
chunk_size = 1000
chunk_overlap = 50
embeddings_endpoint = "databricks-gte-large-en"
VECTOR_SEARCH_ENDPOINT_NAME = "one-env-shared-endpoint-4" 
chatBotModel = "databricks-meta-llama-3-3-70b-instruct"
max_tokens = 2000
finalchatBotModelName = "rag_bot"
yourEmailAddress = "austin.choi@databricks.com"

# COMMAND ----------

dbutils.widgets.text("catalog_name", catalog)
dbutils.widgets.text("agent_schema", agent_schema)

# COMMAND ----------

catalog_name = dbutils.widgets.get("catalog_name")

# COMMAND ----------

spark.sql(f"CREATE CATALOG IF NOT EXISTS {catalog}")

# COMMAND ----------

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{demo_schema}")

# COMMAND ----------

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{agent_schema}")

# COMMAND ----------

import pandas as pd
df = pd.read_csv('./data/customers.csv')
spark_df = spark.createDataFrame(df)
spark_df.write.format("delta").mode("overwrite").saveAsTable(f"{catalog}.{demo_schema}.customers")

# COMMAND ----------

df = pd.read_csv('./data/franchises.csv')
spark_df = spark.createDataFrame(df)
spark_df.write.format("delta").mode("overwrite").saveAsTable(f"{catalog}.{demo_schema}.franchises")

# COMMAND ----------

df = pd.read_csv('./data/reviews.csv')
spark_df = spark.createDataFrame(df)
spark_df.write.format("delta").mode("overwrite").saveAsTable(f"{catalog}.{demo_schema}.reviews")

# COMMAND ----------

df = pd.read_csv('./data/synthetic_car_data.csv')
spark_df = spark.createDataFrame(df)
spark_df.write.format("delta").mode("overwrite").saveAsTable(f"{catalog}.{demo_schema}.synthetic_car_data")

# COMMAND ----------

df = pd.read_csv('./data/transactions.csv')
spark_df = spark.createDataFrame(df)
spark_df.write.format("delta").mode("overwrite").saveAsTable(f"{catalog}.{demo_schema}.transactions")

# COMMAND ----------

df = pd.read_csv('./data/fs_travel.csv')
spark_df = spark.createDataFrame(df)
spark_df.write.format("delta").mode("overwrite").saveAsTable(f"{catalog}.{demo_schema}.fs_travel")

# COMMAND ----------

df = pd.read_csv('./data/destinations.csv')
spark_df = spark.createDataFrame(df)
spark_df.write.format("delta").mode("overwrite").saveAsTable(f"{catalog}.{demo_schema}.destinations")
