import json
import os
from typing import List, Optional, Dict, Any

from pydantic import BaseModel
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from couchbase.cluster import Cluster
from couchbase.options import ClusterOptions
from couchbase.auth import PasswordAuthenticator
from datetime import timedelta
from functools import lru_cache
import streamlit as st

load_dotenv()
llm = ChatOpenAI(model="gpt-4o", temperature=0)


class State(BaseModel):
    """State class for the agent workflow"""

    user_question: str = ""
    collections: List[str] = []
    collections_schema: str = ""
    sqlpp_query: str = ""
    output_data: List[Dict[str, Any]] = []
    results_summary: str = ""
    error: Optional[str] = None
    revision_count: int = 0
    max_revisions: int = 2


@lru_cache
def get_db_connection():
    """Get the Couchbase bucket and scope objects."""
    options = ClusterOptions(
        PasswordAuthenticator(os.getenv("CB_USERNAME"), os.getenv("CB_PASSWORD"))
    )

    options.apply_profile("wan_development")
    cluster = Cluster(os.getenv("CB_CONNECTION_STRING"), options)
    cluster.wait_until_ready(timedelta(seconds=5))
    bucket = cluster.bucket(os.getenv("CB_BUCKET_NAME"))
    scope = bucket.scope(os.getenv("CB_SCOPE_NAME"))
    return bucket, scope


def list_of_collections(state: State) -> Dict[str, Any]:
    """
    Get a list of all collections in the specified bucket.
    Returns a list of collections.
    """
    bucket, _ = get_db_connection()
    collections: List = []

    print("Getting collections in the scope...")

    # Get a list of all the collections in the scope
    for scope in bucket.collections().get_all_scopes():
        if scope.name != os.getenv("CB_SCOPE_NAME"):
            continue
        # Get a list of all the collections in the scope
        for collection in scope.collections:
            collections.append(collection.name)

    return state.model_copy(update={"collections": collections}, deep=True)


def get_schema_for_collections(state: State):
    """Get the schema for all the collections in the scope in bucket.
    Returns the schema for the collections in the scope in the bucket."""
    _, scope = get_db_connection()
    schema = {}

    print("Getting schema for collections...")
    for collection_name in state.collections:
        try:
            results = scope.query(f"INFER {collection_name}")
            for row in results:
                schema[collection_name] = generate_schema_from_infer(row[-1], state)

        except Exception as e:
            print(f"Failed to get schema for {collection_name}", e)
            continue

    return state.model_copy(
        update={"collections_schema": json.dumps(schema)}, deep=True
    )


def generate_schema_from_infer(infer_output: str, state: State):
    """Generate a schema for a collection based on the INFER statement
    Returns a JSON object with the schema for the collection."""
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert at generating the schema for a collection from a Couchbase INFER statement output. Generate the schema in JSON format along with examples and brief description for the properties based on the following INFER statement output. Return only the JSON and nothing else.",
            ),
            ("human", "{infer_output}"),
        ]
    )

    chain = prompt | llm

    response = chain.invoke({"infer_output": infer_output})
    clean_response = response.content.replace("```json", "").replace("```", "")
    return clean_response


def generate_query(state: State):
    """Generate a SQL++ query to retrieve data from a collection in a scope based on the user query and the schema."""
    query_gen_prompt = """You are an expert at generating SQL++ queries to retrieve data from a collection in a scope.

    Given an input question, output a syntactically correct SQL++ query that retrieves the relevant data from the collection. Then look at the resutlts of the query and return the answer.
    
    When generating the query:

    Output the SQL++ query that answers the input question without a tool call.

    Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most 5 results.
    You can order the results by a relevant column to return the most interesting examples in the database.
    Never query for all the columns from a specific table, only ask for the relevant columns given the question.

    If you get an error while executing a query, rewrite the query and try again.

    If you get an empty result set, you should try to rewrite the query to get a non-empty result set. 

    NEVER make stuff up if you don't have enough information to answer the query... just say you don't have enough information.

    If you have enough information to answer the input question, simply invoke the appropriate tool to submit the final answer to the user.

    DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.
    
    The schema for the collections in the database is as follows: {collections_schema}

    Only return the SQL++ query and nothing else.

    """
    print("Generating SQL++ query...")

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                query_gen_prompt,
            ),
            ("human", "{user_question}"),
        ]
    )

    chain = prompt | llm
    response = chain.invoke(
        {
            "user_question": state.user_question,
            "collections_schema": state.collections_schema,
        }
    )

    # Remove the SQL tag from the response
    clean_sqlpp_query = (
        response.content.replace("```sql", "").replace("```", "").strip()
    )

    revision_count = state.revision_count + 1
    return state.model_copy(
        update={"sqlpp_query": clean_sqlpp_query, "revision_count": revision_count},
        deep=True,
    )


def run_query(state: State):
    """
    Execute a SQL++ query on a collection in a scope and return the results.
    If the query is not correct, an error message will be returned.
    If an error is returned, rewrite the query, check the query, and try again.
    """
    _, scope = get_db_connection()
    output = []
    # print(f"{state.sqlpp_query=}")
    print("Running SQL++ query...")
    try:
        results = scope.query(state.sqlpp_query)
        output = [row for row in results]
    except Exception as e:
        state.error = f"Error: Query failed. Please rewrite your query and try again. Error details:{e}"
    return state.model_copy(
        update={"output_data": output, "error": state.error}, deep=True
    )


def should_continue(state: State):
    """Check if the workflow should continue or end based on the state of the workflow."""
    if state.revision_count > state.max_revisions or len(state.output_data) > 0:
        return END
    elif state.error:
        return "generate_query"


if __name__ == "__main__":
    # os.environ.setdefault("LANGSMITH_TRACING_ENABLED", "true")
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    # Define the workflow
    workflow = StateGraph(State)

    # Add the nodes to the workflow
    workflow.add_node("get_collections", list_of_collections)
    workflow.add_node("generate_schema", get_schema_for_collections)
    workflow.add_node("generate_query", generate_query)
    workflow.add_node("run_query", run_query)

    # Add the edges to the workflow
    workflow.add_edge("get_collections", "generate_schema")
    workflow.add_edge("generate_schema", "generate_query")
    workflow.add_edge("generate_query", "run_query")
    workflow.add_conditional_edges(
        "run_query",
        should_continue,
        {"generate_query": "generate_query", END: END},
    )
    workflow.set_entry_point("get_collections")

    graph = workflow.compile()

    st.set_page_config(
        page_title="Generate SQL++ Query from natural language question",
        page_icon="📊",
        layout="centered",
        initial_sidebar_state="auto",
        menu_items=None,
    )
    thread = {"configurable": {"thread_id": "1"}}
    st.title("Text to SQL++ Generator")
    user_question = st.text_input("Enter your question")
    submit = st.button("Submit")
    show_graph = st.checkbox("Show Graph", value=False)
    if show_graph:
        st.image(graph.get_graph().draw_mermaid_png(), use_column_width=True)
    st.divider()

    if submit:
        result = graph.invoke(
            {
                "user_question": user_question,
                "max_revisions": 2,
                "revision_number": 0,
            },
            config=thread,
        )

        st.header("Results")
        st.subheader("SQL++ Query")
        st.write(result["sqlpp_query"])
        st.subheader("Output Data")
        st.write(result["output_data"])
