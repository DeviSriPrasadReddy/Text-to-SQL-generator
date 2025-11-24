import pandas as pd
import time
import requests
import os
from dotenv import load_dotenv
from typing import Dict, Any, Optional, List, Union, Tuple
import logging
import backoff

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# --- CONFIGURATION ---
DATABRICKS_HOST = os.environ.get("DATABRICKS_HOST")
DATABRICKS_TOKEN = os.environ.get("DATABRICKS_TOKEN")

# Define Space Configs (Load from env or hardcode)
SPACE_CONFIG = {
    "FINANCE": {"id": os.environ.get("SPACE_ID_FINANCE", "default_finance_id"), "label": "Finance"},
    "HR": {"id": os.environ.get("SPACE_ID_HR", "default_hr_id"), "label": "Human Resources"}
}

if not DATABRICKS_TOKEN:
    raise ValueError("DATABRICKS_TOKEN is missing from environment variables.")

# --- CLIENT CLASS ---
class GenieClient:
    def __init__(self, host: str, space_id: str, user_token: str):
        self.host = host
        self.space_id = space_id
        # STRICTLY use the user token provided
        self.user_token = user_token
        self.base_url = f"https://{host}/api/2.0/genie/spaces/{space_id}"
        self.headers = {
            "Authorization": f"Bearer {self.user_token}",
            "Content-Type": "application/json"
        }

    @backoff.on_exception(backoff.expo, Exception, max_tries=3, factor=2)
    def start_conversation(self, question: str) -> Dict[str, Any]:
        url = f"{self.base_url}/start-conversation"
        payload = {"content": question}
        response = requests.post(url, headers=self.headers, json=payload)
        response.raise_for_status()
        return response.json()

    @backoff.on_exception(backoff.expo, Exception, max_tries=3, factor=2)
    def send_message(self, conversation_id: str, message: str) -> Dict[str, Any]:
        url = f"{self.base_url}/conversations/{conversation_id}/messages"
        payload = {"content": message}
        response = requests.post(url, headers=self.headers, json=payload)
        response.raise_for_status()
        return response.json()

    @backoff.on_exception(backoff.expo, Exception, max_tries=3, factor=2)
    def get_message(self, conversation_id: str, message_id: str) -> Dict[str, Any]:
        url = f"{self.base_url}/conversations/{conversation_id}/messages/{message_id}"
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        return response.json()

    @backoff.on_exception(backoff.expo, Exception, max_tries=3, factor=2)
    def get_query_result(self, conversation_id: str, message_id: str, attachment_id: str) -> Dict[str, Any]:
        url = f"{self.base_url}/conversations/{conversation_id}/messages/{message_id}/attachments/{attachment_id}/query-result"
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        result = response.json()
        
        data_array = result.get('statement_response', {}).get('result', {}).get('data_array', [])
        schema = result.get('statement_response', {}).get('manifest', {}).get('schema', {})
        return {'data_array': data_array, 'schema': schema}

    def wait_for_message_completion(self, conversation_id: str, message_id: str, timeout: int = 300) -> Dict[str, Any]:
        start_time = time.time()
        poll_interval = 2
        
        while time.time() - start_time < timeout:
            try:
                message = self.get_message(conversation_id, message_id)
                status = message.get("status")
                if status in ["COMPLETED", "ERROR", "FAILED"]:
                    return message
            except Exception as e:
                logger.warning(f"Transient polling error: {e}")
            
            time.sleep(poll_interval)
            
        raise TimeoutError(f"Genie did not respond within {timeout} seconds.")

# --- CORE LOGIC ---

def process_genie_response(client: GenieClient, conversation_id: str, message_id: str, complete_message: Dict[str, Any]) -> Tuple[Union[str, pd.DataFrame], Optional[str]]:
    attachments = complete_message.get("attachments", [])
    for attachment in attachments:
        attachment_id = attachment.get("attachment_id")
        
        # Handle Text Response
        if "text" in attachment and "content" in attachment["text"]:
            return attachment["text"]["content"], None
            
        # Handle SQL/Data Response
        elif "query" in attachment:
            query_text = attachment.get("query", {}).get("query", "")
            try:
                query_result = client.get_query_result(conversation_id, message_id, attachment_id)
                data_array = query_result.get('data_array', [])
                schema = query_result.get('schema', {})
                columns = [col.get('name') for col in schema.get('columns', [])]
                
                if data_array:
                    if not columns:
                        columns = [f"col_{i}" for i in range(len(data_array[0]))]
                    df = pd.DataFrame(data_array, columns=columns)
                    return df, query_text
            except Exception as e:
                logger.error(f"Error fetching query result: {e}")
                return "Error retrieving data.", query_text
    
    if 'content' in complete_message:
        return complete_message.get('content', ''), None
    return "No response available", None

def genie_query_router_aware(question: str, target_space_id: str, conversation_id: Optional[str] = None) -> Tuple[Optional[str], Optional[str], Union[str, pd.DataFrame], Optional[str]]:
    # Initialize client with USER TOKEN only
    client = GenieClient(host=DATABRICKS_HOST, space_id=target_space_id, user_token=DATABRICKS_TOKEN)

    try:
        if conversation_id is None:
            logger.info(f"Starting NEW conversation in Space: {target_space_id}...")
            resp = client.start_conversation(question)
            conv_id = resp.get("conversation_id")
            msg_id = resp.get("message_id")
        else:
            logger.info(f"Continuing conversation {conversation_id}...")
            try:
                resp = client.send_message(conversation_id, question)
                conv_id = conversation_id
                msg_id = resp.get("message_id")
            except Exception as e:
                logger.warning("Conversation expired or invalid. Starting new one.")
                resp = client.start_conversation(question)
                conv_id = resp.get("conversation_id")
                msg_id = resp.get("message_id")

        complete_msg = client.wait_for_message_completion(conv_id, msg_id)
        result, query_text = process_genie_response(client, conv_id, msg_id, complete_msg)
        
        return conv_id, msg_id, result, query_text

    except Exception as e:
        logger.error(f"Genie Query Error: {e}")
        return conversation_id, None, f"Error: {str(e)}", None

# --- APP INTERFACE FUNCTIONS ---

def route_question(text: str) -> str:
    """Simple keyword routing."""
    text_lower = text.lower()
    if any(x in text_lower for x in ['salary', 'employee', 'hiring', 'hr']):
        return SPACE_CONFIG["HR"]["id"]
    # Default to Finance
    return SPACE_CONFIG["FINANCE"]["id"]

def execute_genie_query(user_query: str, space_id: str, current_conv_id: Optional[str]):
    """Wrapper used by the Dash App."""
    conv_id, _, result, sql = genie_query_router_aware(
        question=user_query,
        target_space_id=space_id,
        conversation_id=current_conv_id
    )
    return conv_id, result, sql

def generate_insights(df: pd.DataFrame) -> str:
    """Basic insight generator (Placeholder for LLM summary)."""
    if df.empty:
        return "The result set is empty."
    
    desc = df.describe().to_markdown()
    return f"### Data Summary\n\n**Rows:** {len(df)}\n**Columns:** {', '.join(df.columns)}\n\n{desc}"
