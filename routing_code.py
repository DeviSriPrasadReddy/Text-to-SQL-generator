import os
import requests
import json
import logging
from typing import Dict, List, Optional
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- CONFIGURATION ---
# This is the Single Source of Truth for your Spaces
SPACE_CONFIG = {
    "FINANCE": {
        "id": os.environ.get("FINANCE_SPACE_ID"), 
        "label": "Finance Dept",
        "description": "Revenue, sales, profit, taxes, quarterly reports, and budget variance."
    },
    "HR": {
        "id": os.environ.get("HR_SPACE_ID"), 
        "label": "Human Resources",
        "description": "Employee headcount, hiring, retention, payroll, and attrition."
    }
}

# System Token (Service Principal) for the Router LLM
# This allows routing to happen without user-specific context permissions
SYSTEM_LLM_TOKEN = os.environ.get("DATABRICKS_TOKEN") 
LLM_HOST = os.environ.get("DATABRICKS_HOST")
LLM_ENDPOINT_URL = f"https://{LLM_HOST}/serving-endpoints/databricks-meta-llama-3-70b-instruct/invocations"

def orchestrate_routing(user_question: str, active_sessions: Dict[str, Dict]) -> Dict[str, str]:
    """
    Decides if the question is a follow-up to an existing session OR a new topic.
    
    Args:
        user_question: The current input.
        active_sessions: Dictionary from App State:
            {
                "SPACE_ID_1": {"last_topic": "What is Q3 Revenue?", "conv_id": "123..."},
                "SPACE_ID_2": {"last_topic": "How many engineers?", "conv_id": "456..."}
            }
    """
    
    # 1. Format Active Contexts for the Prompt
    context_str = "NO ACTIVE CONVERSATIONS."
    if active_sessions:
        context_str = "ACTIVE CONVERSATIONS (The user might be following up on these):\n"
        for space_id, data in active_sessions.items():
            # Find human readable label
            label = next((v['label'] for k, v in SPACE_CONFIG.items() if v['id'] == space_id), "Unknown Space")
            context_str += f"- Space: {label} (ID: {space_id}) | Last User Query: '{data.get('last_topic', 'N/A')}' | ConversationID: {data.get('conv_id')}\n"

    # 2. Format Space Definitions
    space_defs = "\n".join([f"- {v['label']} (ID: {v['id']}): {v['description']}" for k, v in SPACE_CONFIG.items()])

    # 3. Construct Prompt
    prompt = f"""
    You are a smart conversational router. You manage multiple conversation threads across different data spaces.
    
    AVAILABLE SPACES:
    {space_defs}
    
    {context_str}
    
    USER INPUT: "{user_question}"
    
    YOUR TASK:
    1. Analyze the USER INPUT.
    2. Check if it is a follow-up question related to one of the "ACTIVE CONVERSATIONS".
       - Example: If Active Context is "Revenue" and Input is "What about Q4?", it is a follow-up.
       - If it is a follow-up, you MUST return that Space ID and the existing Conversation ID.
    3. If it is NOT a follow-up, classify the input into one of the "AVAILABLE SPACES".
       - In this case, return the new Space ID and set Conversation ID to null.
    
    OUTPUT JSON FORMAT ONLY:
    {{
        "target_space_id": "string", 
        "target_conversation_id": "string or null",
        "reasoning": "short explanation"
    }}
    """

    payload = {
        "messages": [{"role": "user", "content": prompt}], 
        "temperature": 0.1, 
        "max_tokens": 150
    }
    
    headers = {"Authorization": f"Bearer {SYSTEM_LLM_TOKEN}", "Content-Type": "application/json"}
    
    try:
        response = requests.post(LLM_ENDPOINT_URL, json=payload, headers=headers)
        response.raise_for_status()
        content = response.json()['choices'][0]['message']['content']
        
        # Clean potential markdown wrapping
        content = content.replace("```json", "").replace("```", "").strip()
        result = json.loads(content)
        
        logger.info(f"Router Decision: {result['reasoning']}")
        return result

    except Exception as e:
        logger.error(f"Router Error: {e}")
        # Fallback: Default to first space in config
        default_key = list(SPACE_CONFIG.keys())[0]
        return {
            "target_space_id": SPACE_CONFIG[default_key]["id"], 
            "target_conversation_id": None, 
            "reasoning": "Error fallback"
        }
