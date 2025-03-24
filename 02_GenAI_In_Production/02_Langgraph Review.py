# Databricks notebook source
# MAGIC %md
# MAGIC #Step 0: Depedencies and Demo Data
# MAGIC
# MAGIC We need to install the necessary libraries and some demo data for the demo to run properly

# COMMAND ----------

# DBTITLE 1,Install Dependencies and Sample Data
# MAGIC %run ./config

# COMMAND ----------

# DBTITLE 1,Set up Vector Search Index (If needed)
# MAGIC %run ./rag_setup/rag_setup

# COMMAND ----------

# MAGIC %md
# MAGIC # Disclaimer 
# MAGIC
# MAGIC In order for Langgraph to run, we need to write the code into a separate file called **agent.py**. We will go through each step of setting up Langgraph in pieces to exaplin what is happening, then combine all of it in one cell to create the file.

# COMMAND ----------

# MAGIC %md
# MAGIC #What is Langgraph? 
# MAGIC
# MAGIC Langgraph is a stateful, agentic framework from Langchain that simplifies agent development. It allows developers to create complex agent workflows by organizing language model calls, tool use, and memory into structured, composable components that can be deployed as robust applications. LangGraph's key innovation is enabling dynamic, multi-step interactions between different "agents" while maintaining contextual awareness and providing clear visualization of the application's flow.
# MAGIC
# MAGIC We will break each portion of Langgraph out to demonstrate this structure and explain what is happening.

# COMMAND ----------

# MAGIC %md
# MAGIC #Step 0: Import the necessary libraries

# COMMAND ----------

from typing import Any, Generator, Optional, Sequence, Union

from config_import import demo_schema, catalog, chatBotModel, vectorSearchIndexName, embeddings_endpoint, agent_schema #we have some default configurations that we will import. You are welcome to change these. 

import mlflow
from databricks_langchain import (
    ChatDatabricks,
    UCFunctionToolkit,
    VectorSearchRetrieverTool,
    DatabricksEmbeddings
)
from unitycatalog.ai.core.databricks import DatabricksFunctionClient
from unitycatalog.ai.core.base import set_uc_function_client

from langchain_core.language_models import LanguageModelLike
from langchain_core.runnables import RunnableConfig, RunnableLambda
from langchain_core.tools import BaseTool
from langgraph.graph import END, StateGraph
from langgraph.graph.graph import CompiledGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt.tool_node import ToolNode
from mlflow.langchain.chat_agent_langgraph import ChatAgentState, ChatAgentToolNode
from mlflow.pyfunc import ChatAgent
from mlflow.types.agent import (
    ChatAgentChunk,
    ChatAgentMessage,
    ChatAgentResponse,
    ChatContext,
)

# COMMAND ----------

# MAGIC %md
# MAGIC #Step 1: Define your LLM Endpoint and System Prompt
# MAGIC
# MAGIC We need to specify what LLM we plan to use for this application. On Databricks, any LLM that you host on Model serving, be it an open source model or external model, can be used. Model Serving handles compatibility between different providers allowing you to use any model you bring to Databricks 
# MAGIC
# MAGIC

# COMMAND ----------


# chatBotModel = "databricks-meta-llama-3-3-70b-instruct"
LLM_ENDPOINT_NAME = chatBotModel
llm = ChatDatabricks(endpoint=LLM_ENDPOINT_NAME) #Langgraph's implementation on what model to use

# TODO: Update with your system prompt
system_prompt = f"""## Instructions for Testing the Databricks Documentation Assistant chatbot

Your inputs are invaluable for the development team. By providing detailed feedback and corrections, you help us fix issues and improve the overall quality of the application. We rely on your expertise to identify any gaps or areas needing enhancement."""

# COMMAND ----------

# MAGIC %md
# MAGIC #Step 2: Define our Tools
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
# MAGIC ###Langgraph Implementation
# MAGIC For Langgraph, we need to define and list our tools out before hand so that we can pass it into the application. 

# COMMAND ----------

# MAGIC %md
# MAGIC ###First, let's define our own UDFs or Unity Catalog Functions

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE FUNCTION identifier(CONCAT(:catalog_name||'.'||:agent_schema||'.','purchase_location'))()
# MAGIC     RETURNS Table(name STRING, purchases INTEGER)
# MAGIC     COMMENT 'Use this tool to find total purchase information about a particular location. This tool will provide a list of destinations that you will use to help you answer questions. Only use this if the user asks about locations.'
# MAGIC     RETURN SELECT dl.name AS Destination, count(tp.destination_id) AS Total_Purchases_Per_Destination
# MAGIC              FROM main.dbdemos_fs_travel.travel_purchase tp join main.dbdemos_fs_travel.destination_location dl on tp.destination_id = dl.destination_id
# MAGIC              group by dl.name
# MAGIC              order by count(tp.destination_id) desc
# MAGIC              LIMIT 10

# COMMAND ----------

# MAGIC %md
# MAGIC ###Next, let's add to as one of the tools that Langgraph Agent Can call
# MAGIC
# MAGIC We need to specific the location of the function using catalog.schema.tool_name, which we set in the configuration file. The * simply means that we want the agent to be able to use all UDFs located in that schema. If you have multiple functions, it will considerr all the functions for use

# COMMAND ----------

###############################################################################
## To create and see usage examples of more tools, see https://docs.databricks.com/en/generative-ai/agent-framework/agent-tool.html
###############################################################################
tools = []

# Add additional tools
client = DatabricksFunctionClient()
set_uc_function_client(client)
uc_tool_names = [f"{catalog}.{agent_schema}.*"] # you can specify individual tools as list but we are going to select all for this demo
uc_toolkit = UCFunctionToolkit(function_names=uc_tool_names)
tools.extend(uc_toolkit.tools)

# Use Databricks vector search indexes as tools
# See https://docs.databricks.com/en/generative-ai/agent-framework/unstructured-retrieval-tools.html for details

# Add vector search indexes, we set this up using rag_setup. 
vector_search_tools = [
        VectorSearchRetrieverTool(
          index_name=f"{catalog}.{demo_schema}.{vectorSearchIndexName}",
          tool_name="databricks_docs_retriever",
          tool_description="Retrieves information about Databricks products from official Databricks documentation.",
          columns=["id", "url", "content"],
          embedding=DatabricksEmbeddings(endpoint=embeddings_endpoint),
          text_column="content",
        )
]
tools.extend(vector_search_tools)

# COMMAND ----------

# MAGIC %md
# MAGIC #Step 3: Define the Agent's Logic
# MAGIC
# MAGIC Now we have to set up the code that dictates the logic the Agent will go through. Langgraph has a concept called Nodes and Edges that help dictate and enforce what step an Agent should take next which could be a new Agent or a new tool. We compile this Langgraph workflow at the end to represent the state of the graph. Between each node, the ChatAgentState is passed so that the Agent remembers what has happened previously 

# COMMAND ----------

def create_tool_calling_agent(
    model: LanguageModelLike,
    tools: Union[ToolNode, Sequence[BaseTool]],
    system_prompt: Optional[str] = None,
) -> CompiledGraph:
    model = model.bind_tools(tools)

    # Define the function that determines which node to go to
    def should_continue(state: ChatAgentState):
        messages = state["messages"]
        last_message = messages[-1]
        # If there are function calls, continue. else, end
        if last_message.get("tool_calls"):
            return "continue"
        else:
            return "end"

    if system_prompt:
        preprocessor = RunnableLambda(
            lambda state: [{"role": "system", "content": system_prompt}]
            + state["messages"]
        )
    else:
        preprocessor = RunnableLambda(lambda state: state["messages"])
    model_runnable = preprocessor | model

    def call_model(
        state: ChatAgentState,
        config: RunnableConfig,
    ):
        response = model_runnable.invoke(state, config)

        return {"messages": [response]}

    workflow = StateGraph(ChatAgentState)

    workflow.add_node("agent", RunnableLambda(call_model))
    workflow.add_node("tools", ChatAgentToolNode(tools))

    workflow.set_entry_point("agent")
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "continue": "tools",
            "end": END,
        },
    )
    workflow.add_edge("tools", "agent")

    return workflow.compile()

# COMMAND ----------

# MAGIC %md
# MAGIC #Step 4: Create a ChatAgent Model
# MAGIC
# MAGIC ChatAgent is a new MLflow Interface that enforces a chat schema to author conversational agents. It is an important step in reviewing our Agent's performance, confirm tools calls and support multi-agent scenarios. We will see this in action with Databrick's Review App. 
# MAGIC
# MAGIC Check out MLflow's documentaton here: https://mlflow.org/docs/latest/api_reference/python_api/mlflow.pyfunc.html#mlflow.pyfunc.ChatAgent
# MAGIC
# MAGIC We complete the setup by telling MLflow where the Agent configuration is located and enabling mlflow.langchain.autolog which include Langgraph. This enables MLflow traces, experiments and other logging capabilities critical in managing our Agent in production. 

# COMMAND ----------

class LangGraphChatAgent(ChatAgent):
    def __init__(self, agent: CompiledStateGraph):
        self.agent = agent

    def predict(
        self,
        messages: list[ChatAgentMessage],
        context: Optional[ChatContext] = None,
        custom_inputs: Optional[dict[str, Any]] = None,
    ) -> ChatAgentResponse:
        request = {"messages": self._convert_messages_to_dict(messages)}

        messages = []
        for event in self.agent.stream(request, stream_mode="updates"):
            for node_data in event.values():
                messages.extend(
                    ChatAgentMessage(**msg) for msg in node_data.get("messages", [])
                )
        return ChatAgentResponse(messages=messages)

    def predict_stream(
        self,
        messages: list[ChatAgentMessage],
        context: Optional[ChatContext] = None,
        custom_inputs: Optional[dict[str, Any]] = None,
    ) -> Generator[ChatAgentChunk, None, None]:
        request = {"messages": self._convert_messages_to_dict(messages)}
        for event in self.agent.stream(request, stream_mode="updates"):
            for node_data in event.values():
                yield from (
                    ChatAgentChunk(**{"delta": msg}) for msg in node_data["messages"]
                )

mlflow.langchain.autolog()
agent = create_tool_calling_agent(llm, tools, system_prompt)
AGENT = LangGraphChatAgent(agent)
mlflow.models.set_model(AGENT)

# COMMAND ----------

# MAGIC %md
# MAGIC #Completed Code
# MAGIC
# MAGIC Now we run all our code to create an agent.py file that we will use to run the Langgraph agent

# COMMAND ----------

# MAGIC %%writefile agent.py
# MAGIC from typing import Any, Generator, Optional, Sequence, Union
# MAGIC
# MAGIC from config_import import demo_schema, catalog, chatBotModel, vectorSearchIndexName, embeddings_endpoint, agent_schema
# MAGIC import mlflow
# MAGIC from databricks_langchain import (
# MAGIC     ChatDatabricks,
# MAGIC     UCFunctionToolkit,
# MAGIC     VectorSearchRetrieverTool,
# MAGIC     DatabricksEmbeddings
# MAGIC )
# MAGIC from unitycatalog.ai.core.databricks import DatabricksFunctionClient
# MAGIC from unitycatalog.ai.core.base import set_uc_function_client
# MAGIC from langchain_core.language_models import LanguageModelLike
# MAGIC from langchain_core.runnables import RunnableConfig, RunnableLambda
# MAGIC from langchain_core.tools import BaseTool
# MAGIC from langgraph.graph import END, StateGraph
# MAGIC from langgraph.graph.graph import CompiledGraph
# MAGIC from langgraph.graph.state import CompiledStateGraph
# MAGIC from langgraph.prebuilt.tool_node import ToolNode
# MAGIC from mlflow.langchain.chat_agent_langgraph import ChatAgentState, ChatAgentToolNode
# MAGIC from mlflow.pyfunc import ChatAgent
# MAGIC from mlflow.types.agent import (
# MAGIC     ChatAgentChunk,
# MAGIC     ChatAgentMessage,
# MAGIC     ChatAgentResponse,
# MAGIC     ChatContext,
# MAGIC )
# MAGIC ############################################
# MAGIC # Define your LLM endpoint and system prompt
# MAGIC ############################################
# MAGIC # TODO: Replace with your model serving endpoint
# MAGIC # LLM_ENDPOINT_NAME = "databricks-meta-llama-3-3-70b-instruct"
# MAGIC LLM_ENDPOINT_NAME = chatBotModel
# MAGIC llm = ChatDatabricks(endpoint=LLM_ENDPOINT_NAME)
# MAGIC
# MAGIC # TODO: Update with your system prompt
# MAGIC system_prompt = f"""## Instructions for Testing the Databricks Documentation Assistant chatbot
# MAGIC
# MAGIC Your inputs are invaluable for the development team. By providing detailed feedback and corrections, you help us fix issues and improve the overall quality of the application. We rely on your expertise to identify any gaps or areas needing enhancement."""
# MAGIC
# MAGIC ###############################################################################
# MAGIC ## Define tools for your agent, enabling it to retrieve data or take actions
# MAGIC ## beyond text generation
# MAGIC ## To create and see usage examples of more tools, see
# MAGIC ## https://docs.databricks.com/en/generative-ai/agent-framework/agent-tool.html
# MAGIC ###############################################################################
# MAGIC tools = []
# MAGIC
# MAGIC # You can use UDFs in Unity Catalog as agent tools
# MAGIC # Below, we add the `system.ai.python_exec` UDF, which provides
# MAGIC # a python code interpreter tool to our agent
# MAGIC # You can also add local LangChain python tools. See https://python.langchain.com/docs/concepts/tools
# MAGIC
# MAGIC # Add additional tools
# MAGIC client = DatabricksFunctionClient()
# MAGIC set_uc_function_client(client)
# MAGIC uc_tool_names = [f"{catalog}.{agent_schema}.*"]
# MAGIC uc_toolkit = UCFunctionToolkit(function_names=uc_tool_names)
# MAGIC tools.extend(uc_toolkit.tools)
# MAGIC
# MAGIC # Use Databricks vector search indexes as tools
# MAGIC # See https://docs.databricks.com/en/generative-ai/agent-framework/unstructured-retrieval-tools.html
# MAGIC # for details
# MAGIC
# MAGIC # Add vector search indexes
# MAGIC vector_search_tools = [
# MAGIC         VectorSearchRetrieverTool(
# MAGIC           index_name=f"{catalog}.{demo_schema}.{vectorSearchIndexName}",
# MAGIC           tool_name="databricks_docs_retriever",
# MAGIC           tool_description="Retrieves information about Databricks products from official Databricks documentation. This must be used",
# MAGIC           columns=["id", "url", "content"],
# MAGIC           embedding=DatabricksEmbeddings(endpoint=embeddings_endpoint),
# MAGIC           text_column="content",
# MAGIC         )
# MAGIC ]
# MAGIC tools.extend(vector_search_tools)
# MAGIC
# MAGIC #####################
# MAGIC ## Define agent logic
# MAGIC #####################
# MAGIC
# MAGIC
# MAGIC def create_tool_calling_agent(
# MAGIC     model: LanguageModelLike,
# MAGIC     tools: Union[ToolNode, Sequence[BaseTool]],
# MAGIC     system_prompt: Optional[str] = None,
# MAGIC ) -> CompiledGraph:
# MAGIC     model = model.bind_tools(tools)
# MAGIC
# MAGIC     # Define the function that determines which node to go to
# MAGIC     def should_continue(state: ChatAgentState):
# MAGIC         messages = state["messages"]
# MAGIC         last_message = messages[-1]
# MAGIC         # If there are function calls, continue. else, end
# MAGIC         if last_message.get("tool_calls"):
# MAGIC             return "continue"
# MAGIC         else:
# MAGIC             return "end"
# MAGIC
# MAGIC     if system_prompt:
# MAGIC         preprocessor = RunnableLambda(
# MAGIC             lambda state: [{"role": "system", "content": system_prompt}]
# MAGIC             + state["messages"]
# MAGIC         )
# MAGIC     else:
# MAGIC         preprocessor = RunnableLambda(lambda state: state["messages"])
# MAGIC     model_runnable = preprocessor | model
# MAGIC
# MAGIC     def call_model(
# MAGIC         state: ChatAgentState,
# MAGIC         config: RunnableConfig,
# MAGIC     ):
# MAGIC         response = model_runnable.invoke(state, config)
# MAGIC
# MAGIC         return {"messages": [response]}
# MAGIC
# MAGIC     workflow = StateGraph(ChatAgentState)
# MAGIC
# MAGIC     workflow.add_node("agent", RunnableLambda(call_model))
# MAGIC     workflow.add_node("tools", ChatAgentToolNode(tools))
# MAGIC
# MAGIC     workflow.set_entry_point("agent")
# MAGIC     workflow.add_conditional_edges(
# MAGIC         "agent",
# MAGIC         should_continue,
# MAGIC         {
# MAGIC             "continue": "tools",
# MAGIC             "end": END,
# MAGIC         },
# MAGIC     )
# MAGIC     workflow.add_edge("tools", "agent")
# MAGIC
# MAGIC     return workflow.compile()
# MAGIC
# MAGIC
# MAGIC class LangGraphChatAgent(ChatAgent):
# MAGIC     def __init__(self, agent: CompiledStateGraph):
# MAGIC         self.agent = agent
# MAGIC
# MAGIC     def predict(
# MAGIC         self,
# MAGIC         messages: list[ChatAgentMessage],
# MAGIC         context: Optional[ChatContext] = None,
# MAGIC         custom_inputs: Optional[dict[str, Any]] = None,
# MAGIC     ) -> ChatAgentResponse:
# MAGIC         request = {"messages": self._convert_messages_to_dict(messages)}
# MAGIC
# MAGIC         messages = []
# MAGIC         for event in self.agent.stream(request, stream_mode="updates"):
# MAGIC             for node_data in event.values():
# MAGIC                 messages.extend(
# MAGIC                     ChatAgentMessage(**msg) for msg in node_data.get("messages", [])
# MAGIC                 )
# MAGIC         return ChatAgentResponse(messages=messages)
# MAGIC
# MAGIC     def predict_stream(
# MAGIC         self,
# MAGIC         messages: list[ChatAgentMessage],
# MAGIC         context: Optional[ChatContext] = None,
# MAGIC         custom_inputs: Optional[dict[str, Any]] = None,
# MAGIC     ) -> Generator[ChatAgentChunk, None, None]:
# MAGIC         request = {"messages": self._convert_messages_to_dict(messages)}
# MAGIC         for event in self.agent.stream(request, stream_mode="updates"):
# MAGIC             for node_data in event.values():
# MAGIC                 yield from (
# MAGIC                     ChatAgentChunk(**{"delta": msg}) for msg in node_data["messages"]
# MAGIC                 )
# MAGIC
# MAGIC
# MAGIC # Create the agent object, and specify it as the agent object to use when
# MAGIC # loading the agent back for inference via mlflow.models.set_model()
# MAGIC mlflow.langchain.autolog()
# MAGIC agent = create_tool_calling_agent(llm, tools, system_prompt)
# MAGIC AGENT = LangGraphChatAgent(agent)
# MAGIC mlflow.models.set_model(AGENT)

# COMMAND ----------

# MAGIC %md
# MAGIC #Test the Langgraph Agent
# MAGIC
# MAGIC Now that we have a theoretically working Agent. Let's make sure it works! 

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

from agent import AGENT

result = AGENT.predict({"messages": [{"role": "user", "content": "Hello! what is databricks?"}]})
print(f"Full Payload which shows the Agent determining what tool to use and the tool it called before providing answer: {result}.\n\n**Actual Answer**: {result.messages[-1].content}")

# COMMAND ----------

# MAGIC %md
# MAGIC #Log the Langgraph Agent
# MAGIC
# MAGIC Now that we have a confirmed working Langgraph Agent, we need to log the model to MLflow. This will capture all the dependencies and necessary credentials we need to deploy the Langgraph Agent.
# MAGIC
# MAGIC If you plan on using other Databricks features that need credentials, you can use automatic authetication passthrough. This is what our _resources_ variable is for: https://docs.databricks.com/aws/en/generative-ai/agent-framework/deploy-agent#authentication-for-dependent-resources

# COMMAND ----------

import mlflow
from agent import tools, LLM_ENDPOINT_NAME
from databricks_langchain import VectorSearchRetrieverTool
from mlflow.models.resources import DatabricksFunction, DatabricksServingEndpoint
from unitycatalog.ai.langchain.toolkit import UnityCatalogTool

mlflow.set_registry_uri("databricks-uc")

resources = [DatabricksServingEndpoint(endpoint_name=LLM_ENDPOINT_NAME)]
for tool in tools:
    if isinstance(tool, VectorSearchRetrieverTool):
        resources.extend(tool.resources)
    elif isinstance(tool, UnityCatalogTool):
        resources.append(DatabricksFunction(function_name=tool.uc_function_name))


with mlflow.start_run():
    logged_agent_info = mlflow.pyfunc.log_model(
        artifact_path="agent",
        python_model="agent.py",
        pip_requirements=[
            "mlflow",
            "langgraph==0.3.4",
            "databricks-langchain",
        ],
        resources=resources,
        code_paths=['config_import.py']
    )

# COMMAND ----------

# MAGIC %md
# MAGIC #Test the MLflow Model! 
# MAGIC
# MAGIC Make sure it was properly logged! By running this test, we can confirm that we have logged everything we need to use this Agent

# COMMAND ----------

mlflow.models.predict(
    model_uri=f"runs:/{logged_agent_info.run_id}/agent",
    input_data={"messages": [{"role": "user", "content": "Hello!"}]},
    env_manager="uv"
)

# COMMAND ----------

# MAGIC %md
# MAGIC #Register the MLflow Model to Unity Catalog
# MAGIC
# MAGIC To prepare our model for serving, we must make sure the model is registered to Unity Catalog

# COMMAND ----------

from config_import import demo_schema, catalog, finalchatBotModelName

# TODO: define the catalog, schema, and model name for your UC model
catalog = catalog
schema = demo_schema
model_name = finalchatBotModelName
UC_MODEL_NAME = f"{catalog}.{schema}.{model_name}"

# register the model to UC
uc_registered_model_info = mlflow.register_model(
    model_uri=logged_agent_info.model_uri, name=UC_MODEL_NAME
)

# COMMAND ----------

# MAGIC %md
# MAGIC # RAG in Production
# MAGIC
# MAGIC This workshop is not to show you how to set up RAG on Databricks. Please check out our self paced learning here: <insert link here> 
# MAGIC
# MAGIC You can follow the notebooks in the folder called RAG to set one up. However, this workshop we will demonstrate what it looks like to prepare and monitor your RAG application in Production. 

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC
# MAGIC ### Evaluate your bot's quality with Mosaic AI Agent Evaluation specialized LLM judge models
# MAGIC
# MAGIC Evaluation is a key part of deploying a RAG application. Databricks simplify this tasks with specialized LLM models tuned to evaluate your bot's quality/cost/latency, even if ground truth is not available.
# MAGIC
# MAGIC This Agent Evaluation's specialized AI evaluator is integrated into integrated into `mlflow.evaluate(...)`, all you need to do is pass `model_type="databricks-agent"`.
# MAGIC
# MAGIC Mosaic AI Agent Evaluation evaluates:
# MAGIC 1. Answer correctness - requires ground truth
# MAGIC 2. Hallucination / groundness - no ground truth required
# MAGIC 3. Answer relevance - no ground truth required
# MAGIC 4. Retrieval precision - no ground truth required
# MAGIC 5. (Lack of) Toxicity - no ground truth required
# MAGIC
# MAGIC In this example, we'll use an evaluation set that we curated based on our internal experts using the Mosaic AI Agent Evaluation review app interface.  This proper Eval Dataset is saved as a Delta Table.

# COMMAND ----------

# DBTITLE 1,Synthetic Data Evaluation
from databricks.agents.evals import generate_evals_df
import mlflow

agent_description = "A chatbot that answers questions about Databricks Documentation."
question_guidelines = """
Questions must strictly be about Databricks and its documentation. 
# User personas
- A developer new to the Databricks platform and documentation
# Example questions
- What API lets me parallelize operations over rows of a delta table?
"""
# TODO: Spark/Pandas DataFrame with "content" and "doc_uri" columns.
docs = spark.table(f"{catalog}.{demo_schema}.databricks_documentation")
docs = docs.withColumnRenamed("url", "doc_uri")
evals = generate_evals_df(
    docs=docs,
    num_evals=10,
    agent_description=agent_description,
    question_guidelines=question_guidelines,
)
print(evals)
eval_result = mlflow.evaluate(data=evals, model=logged_agent_info.model_uri, model_type="databricks-agent")

# COMMAND ----------

eval_dataset = spark.table(f"{catalog}.{demo_schema}.eval_set_databricks_documentation").sample(fraction=0.1, seed=2).limit(10).toPandas()
display(eval_dataset)

# COMMAND ----------

with mlflow.start_run():
    # Evaluate the logged model
    eval_results = mlflow.evaluate(
        data=eval_dataset, # Your evaluation set
        model=logged_agent_info.model_uri,
        model_type="databricks-agent", # active Mosaic AI Agent Evaluation
    )

# COMMAND ----------

# MAGIC %md
# MAGIC #Deploy your Agent/Model
# MAGIC
# MAGIC Databricks provides an easy function called deploy() from the Mosaic AI Agent Framework. This deploys a Model Serving Endpoint on a CPU resource, a Feedback Model and a Review App UI with a shareable link. This is a critical step in review your application as you can send the Review App to subject matter experts for review. They can then provide feedback directly on the app that you can later review. 

# COMMAND ----------

from databricks import agents
agents.deploy(UC_MODEL_NAME, uc_registered_model_info.version, tags = {"endpointSource": "docs"})

# COMMAND ----------

# MAGIC %md
# MAGIC # Mosaic AI Model Training for Fine Tuning LLMs 
# MAGIC
# MAGIC We do not expect you for this workshop to fine tune an LLM. However, we will be demonstrating the performance impact of fine-tuning a Llama-1B model through the playground! 
# MAGIC
# MAGIC We trained this model on a dataset containing medical terms. While larger models can handle these words well, the smaller models struggle with them since they are rarely used. 

# COMMAND ----------


