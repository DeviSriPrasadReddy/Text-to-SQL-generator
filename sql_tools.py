# tools/sql_tools.py
# -----------------
# Defines the tools for interacting with the Databricks SQL warehouse.

import databricks.sql
from typing import List
from langchain_core.tools import tool

# Import configuration from our config file
import config

def get_db_connection():
    """Helper function to get a Databricks SQL connection."""
    return databricks.sql.connect(
        server_hostname=config.SERVER_HOSTNAME,
        http_path=f"/sql/1.0/warehouses/{config.WAREHOUSE_ID}",
        auth_type="databricks-oauth" # Automatically uses notebook/cluster auth
    )

@tool
def get_table_schema(tables: List[str]) -> str:
    """
    Retrieves the DDL (Data Definition Language) schema for a specific list of tables
    from the Databricks catalog '{config.CATALOG}.{config.SCHEMA}'.
    
    Args:
        tables: A list of table names to get the schema for (e.g., ['my_table_1', 'my_table_2']).
    """
    print(f"--- Calling Tool: get_table_schema for tables {tables} ---")
    
    schemas = []
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(f"USE CATALOG {config.CATALOG}")
                cursor.execute(f"USE SCHEMA {config.SCHEMA}")
                
                for table in tables:
                    cursor.execute(f"DESCRIBE EXTENDED {table}")
                    rows = cursor.fetchall()
                    
                    schema_str = f"Schema for {config.CATALOG}.{config.SCHEMA}.{table}:\n"
                    schema_str += "Column | Type | Comment\n"
                    schema_str += "--- | --- | ---\n"
                    for row in rows:
                        if not row[0] or row[0].startswith('#'):
                            continue
                        schema_str += f"{row[0]} | {row[1]} | {row[2] or ''}\n"
                    schemas.append(schema_str)
                    
        return "\n\n".join(schemas)
    except Exception as e:
        return f"Error: Unable to get schema. {str(e)}"

@tool
def execute_sql_query(query: str) -> str:
    """
    Executes a SQL query against the Databricks SQL warehouse in the 
    '{config.CATALOG}.{config.SCHEMA}' context.
    
    Args:
        query: The SQL query to be executed.
    """
    print(f"--- Calling Tool: execute_sql_query with query: {query} ---")
    
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(f"USE CATALOG {config.CATALOG}")
                cursor.execute(f"USE SCHEMA {config.SCHEMA}")
                cursor.execute(query)
                result = cursor.fetchall()
        
        return f"Query executed successfully. Result:\n{str(result)}"
    except Exception as e:
        return f"Error: Query execution failed. {str(e)}"

# A list of all tools to be easily imported by the graph
all_tools = [get_table_schema, execute_sql_query]
