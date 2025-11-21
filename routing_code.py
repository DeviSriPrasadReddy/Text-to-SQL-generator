import requests
import os
import json

# Configuration for your spaces
SPACE_CONFIG = {
    "FINANCE_SPACE": {
        "id": os.environ.get("FINANCE_SPACE_ID"), # Set in env vars
        "description": "Questions about revenue, sales, profit, quarterly reports, and taxes."
    },
    "HR_SPACE": {
        "id": os.environ.get("HR_SPACE_ID"), # Set in env vars
        "description": "Questions about employee headcount, hiring, retention, and payroll."
    }
}

# Databricks Serving Endpoint for LLM (e.g., databricks-dbrx-instruct or azure-openai)
LLM_ENDPOINT_URL = f"https://{os.environ.get('DATABRICKS_HOST')}/serving-endpoints/databricks-meta-llama-3-70b-instruct/invocations"
LLM_TOKEN = os.environ.get("DATABRICKS_TOKEN") # Or generate via TokenMinter

def route_question(user_question: str, history: list = []) -> str:
    """
    Determines which Space ID to use based on the question context.
    """
    
    # Construct prompt with definitions
    space_defs = "\n".join([f"- {k}: {v['description']}" for k, v in SPACE_CONFIG.items()])
    
    prompt = f"""
    You are a routing assistant. 
    Classify the following user question into one of these categories: {list(SPACE_CONFIG.keys())}.
    
    Definitions:
    {space_defs}
    
    User Question: "{user_question}"
    
    Return ONLY the category name (e.g., FINANCE_SPACE). If unsure, default to FINANCE_SPACE.
    """

    payload = {
        "messages": [{"role": "user", "content": prompt}], 
        "temperature": 0.1, 
        "max_tokens": 10
    }
    
    headers = {"Authorization": f"Bearer {LLM_TOKEN}", "Content-Type": "application/json"}
    
    try:
        # Determine space
        response = requests.post(LLM_ENDPOINT_URL, json=payload, headers=headers)
        response.raise_for_status()
        prediction = response.json()['choices'][0]['message']['content'].strip()
        
        # Clean up response just in case
        for key in SPACE_CONFIG.keys():
            if key in prediction:
                return SPACE_CONFIG[key]["id"]
        
        # Fallback default
        return SPACE_CONFIG["FINANCE_SPACE"]["id"]

    except Exception as e:
        print(f"Router Error: {e}. Defaulting to primary space.")
        return SPACE_CONFIG["FINANCE_SPACE"]["id"]
