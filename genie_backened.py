import requests
import pandas as pd
import time
import logging
import backoff
from typing import Dict, Any, Optional, Tuple, Union

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- CLIENT CLASS ---
class GenieClient:
    def __init__(self, host: str, space_id: str, user_token: str):
        self.host = host
        self.space_id = space_id
        # Token is injected from app.py logic
        self.user_token = user_token
        self.base_url = f"https://{host}/api/2.0/genie/spaces/{space_id}"
        self.headers = {
            "Authorization": f"Bearer {self.user_token}",
            "Content-Type": "application/json"
        }

    @backoff.on_exception(backoff.expo, Exception, max_tries=3)
    def start_conversation(self, question: str) -> Dict[str, Any]:
        url = f"{self.base_url}/start-conversation"
        payload = {"content": question}
        response = requests.post(url, headers=self.headers, json=payload)
        response.raise_for_status()
        return response.json()

    @backoff.on_exception(backoff.expo, Exception, max_tries=3)
    def send_message(self, conversation_id: str, message: str) -> Dict[str, Any]:
        url = f"{self.base_url}/conversations/{conversation_id}/messages"
        payload = {"content": message}
        response = requests.post(url, headers=self.headers, json=payload)
        response.raise_for_status()
        return response.json()

    def get_message(self, conversation_id: str, message_id: str) -> Dict[str, Any]:
        url = f"{self.base_url}/conversations/{conversation_id}/messages/{message_id}"
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        return response.json()

    def get_query_result(self, conversation_id: str, message_id: str, attachment_id: str) -> Dict[str, Any]:
        url = f"{self.base_url}/conversations/{conversation_id}/messages/{message_id}/attachments/{attachment_id}/query-result"
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        return response.json()

    def wait_for_completion(self, conversation_id: str, message_id: str, timeout: int = 120) -> Dict[str, Any]:
        start = time.time()
        while time.time() - start < timeout:
            msg = self.get_message(conversation_id, message_id)
            if msg.get("status") in ["COMPLETED", "FAILED", "ERROR"]:
                return msg
            time.sleep(2)
        raise TimeoutError("Genie request timed out.")

# --- PARSING LOGIC ---
def extract_genie_data(client: GenieClient, conv_id: str, msg_id: str, message: Dict) -> Tuple[Any, Optional[str]]:
    """
    Extracts DataFrame and SQL from the Genie response.
    """
    attachments = message.get("attachments", [])
    
    for att in attachments:
        # 1. Handle SQL & Data
        if "query" in att:
            # Extract SQL text immediately
            sql_query = att.get("query", {}).get("query", "-- No SQL text provided")
            try:
                # Fetch Result Data
                raw = client.get_query_result(conv_id, msg_id, att["attachment_id"])
                
                # Parse Schema and Data
                schema = raw.get('statement_response', {}).get('manifest', {}).get('schema', {})
                cols = [c['name'] for c in schema.get('columns', [])]
                data = raw.get('statement_response', {}).get('result', {}).get('data_array', [])
                
                df = pd.DataFrame(data, columns=cols) if cols else pd.DataFrame(data)
                return df, sql_query
            except Exception as e:
                return f"Error executing query: {str(e)}", sql_query
        
        # 2. Handle Text Response (No SQL)
        if "text" in att:
            return att["text"]["content"], None

    return message.get("content", "No content returned."), None

# --- MAIN EXECUTION ENTRY POINT ---
def execute_genie_query(user_query: str, space_id: str, current_conv_id: Optional[str], user_token: str, host: str):
    """
    Orchestrates Genie call. 
    Crucial: It accepts 'user_token' as an argument, passed from app.py.
    """
    if not user_token:
        return current_conv_id, "Error: Token missing.", None

    client = GenieClient(host, space_id, user_token)
    
    try:
        # Start New OR Continue based on what the Router/App passed in
        if not current_conv_id:
            logger.info(f"Starting NEW conversation in {space_id}")
            resp = client.start_conversation(user_query)
            conv_id = resp["conversation_id"]
            msg_id = resp["message_id"]
        else:
            logger.info(f"Continuing conversation {current_conv_id}")
            try:
                resp = client.send_message(current_conv_id, user_query)
                conv_id = current_conv_id
                msg_id = resp["message_id"]
            except Exception:
                # If ID is invalid/expired, auto-restart
                logger.warning("Conversation ID invalid/expired. Starting new.")
                resp = client.start_conversation(user_query)
                conv_id = resp["conversation_id"]
                msg_id = resp["message_id"]

        final_msg = client.wait_for_completion(conv_id, msg_id)
        result, sql = extract_genie_data(client, conv_id, msg_id, final_msg)
        
        return conv_id, result, sql

    except Exception as e:
        logger.error(f"Genie Execution Failed: {e}")
        return current_conv_id, f"System Error: {str(e)}", None
