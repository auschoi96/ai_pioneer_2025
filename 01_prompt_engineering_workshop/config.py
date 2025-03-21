# Databricks notebook source
# MAGIC %pip install lxml==4.9.3 langchain==0.0.344 databricks-vectorsearch==0.22 cloudpickle==2.2.1 databricks-sdk==0.12.0 cloudpickle==2.2.1 pydantic==2.5.2 openai databricks_genai_inference
# MAGIC %pip install --upgrade sqlalchemy openai mlflow

# COMMAND ----------

dbutils.widgets.text("catalog_name","main")

# COMMAND ----------

catalog_name = dbutils.widgets.get("catalog_name")

# COMMAND ----------

print(catalog_name)

# COMMAND ----------

# current_user = spark.sql("SELECT current_user() as username").collect()[0].username
# schema_name = f'genai_workshop_{current_user.split("@")[0].split(".")[0]}'

# print(f"\nUsing catalog + schema: {catalog_name}.{schema_name}")
# spark.sql(f"CREATE SCHEMA IF NOT EXISTS {schema_name}")

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

