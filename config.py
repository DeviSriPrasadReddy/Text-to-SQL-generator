# config.py
# -----------------
# Store all your credentials and endpoint configurations here.

# --- Databricks SQL Warehouse ---
# Find this in your SQL Warehouse's "Connection Details" tab
WAREHOUSE_ID = "YOUR_SQL_WAREHOUSE_ID"
SERVER_HOSTNAME = "YOUR_SERVER_HOSTNAME" # e.g., "dbc-a1b234c5.cloud.databricks.com"

# --- Database Schema ---
CATALOG = "main" # Or your desired catalog
SCHEMA = "default" # Or your desired schema
# --- Databricks Model Endpoint ---
# Use the endpoint name for your Databricks Foundation Model or Serverless Endpoint
DBRX_INSTRUCT_ENDPOINT = "databricks-dbrx-instruct"
