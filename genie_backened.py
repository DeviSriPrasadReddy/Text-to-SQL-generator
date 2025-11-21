import pandas as pd
import time
import requests
import os
from dotenv import load_dotenv
from typing import Dict, Any, Optional, List, Union, Tuple
import logging
import backoff
from token_minter import TokenMinter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# Load Generic Environment Variables
DATABRICKS_HOST = os.environ.get("DATABRICKS_HOST")
CLIENT_ID = os.environ.get("DATABRICKS_CLIENT_ID")
CLIENT_SECRET = os.environ.get("DATABRICKS_CLIENT_SECRET")

# Initialize Token Minter (Fallback)
sp_token_minter = TokenMinter(
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET,
    host=DATABRICKS_HOST
)

class GenieClient:
    def __init__(self, host: str, space_id: str, user_token: Optional[str] = None):
        self.host = host
        self.space_id = space_id
        self.user_token = user_token
        # Base URL is dynamic based on the space_id passed to __init__
        self.base_url = f"https://{host}/api/2.0/genie/spaces/{space_id}"
        self.update_headers()

    def update_headers(self) -> None:
        token = ""
        if self.user_token:
            token = self.user_token
        else:
            # Fallback to SP token
            token = sp_token_minter.get_token()

        self.headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }

    @backoff.on_exception(backoff.expo, Exception, max_tries=5, factor=2)
    def start_conversation(self, question: str) -> Dict[str, Any]:
        self.update_headers()
        url = f"{self.base_url}/start-conversation"
        payload = {"content": question}
        response = requests.post(url, headers=self.headers, json=payload)
        response.raise_for_status()
        return response.json()

    @backoff.on_exception(backoff.expo, Exception, max_tries=5, factor=2)
    def send_message(self, conversation_id: str, message: str) -> Dict[str, Any]:
        self.update_headers()
        url = f"{self.base_url}/conversations/{conversation_id}/messages"
        payload = {"content": message}
        response = requests.post(url, headers=self.headers, json=payload)
        response.raise_for_status()
        return response.json()

    @backoff.on_exception(backoff.expo, Exception, max_tries=5, factor=2)
    def get_message(self, conversation_id: str, message_id: str) -> Dict[str, Any]:
        self.update_headers()
        url = f"{self.base_url}/conversations/{conversation_id}/messages/{message_id}"
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        return response.json()

    @backoff.on_exception(backoff.expo, Exception, max_tries=5, factor=2)
    def get_query_result(self, conversation_id: str, message_id: str, attachment_id: str) -> Dict[str, Any]:
        self.update_headers()
        url = f"{self.base_url}/conversations/{conversation_id}/messages/{message_id}/attachments/{attachment_id}/query-result"
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        result = response.json()
        
        data_array = result.get('statement_response', {}).get('result', {}).get('data_array', [])
        return {
            'data_array': data_array,
            'schema': result.get('statement_response', {}).get('manifest', {}).get('schema', {})
        }

    def wait_for_message_completion(self, conversation_id: str, message_id: str, timeout: int = 300) -> Dict[str, Any]:
        """
        Waits for message completion.
        Timeout defaults to 300 seconds (5 minutes) as requested.
        """
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
            
        raise TimeoutError(f"Genie did not respond within {timeout/60} minutes. Please try again.")

# --- DYNAMIC HELPER FUNCTIONS ---

def process_genie_response(client: GenieClient, conversation_id: str, message_id: str, complete_message: Dict[str, Any]) -> Tuple[Union[str, pd.DataFrame], Optional[str]]:
    # Logic remains the same, but uses the passed 'client' instance
    attachments = complete_message.get("attachments", [])
    for attachment in attachments:
        attachment_id = attachment.get("attachment_id")
        if "text" in attachment and "content" in attachment["text"]:
            return attachment["text"]["content"], None
        elif "query" in attachment:
            query_text = attachment.get("query", {}).get("query", "")
            query_result = client.get_query_result(conversation_id, message_id, attachment_id)
            data_array = query_result.get('data_array', [])
            schema = query_result.get('schema', {})
            columns = [col.get('name') for col in schema.get('columns', [])]
            if data_array:
                if not columns and data_array:
                    columns = [f"column_{i}" for i in range(len(data_array[0]))]
                df = pd.DataFrame(data_array, columns=columns)
                return df, query_text
    
    if 'content' in complete_message:
        return complete_message.get('content', ''), None
    return "No response available", None

def genie_query_router_aware(
    question: str, 
    target_space_id: str, 
    conversation_id: Optional[str] = None, 
    user_token: Optional[str] = None
) -> Tuple[Optional[str], Optional[str], Union[str, pd.DataFrame], Optional[str]]:
    """
    Dynamic entry point. Requires target_space_id to be passed in.
    """
    client = GenieClient(host=DATABRICKS_HOST, space_id=target_space_id, user_token=user_token)

    try:
        # START NEW
        if conversation_id is None:
            logger.info(f"Starting NEW conversation in Space: {target_space_id}...")
            resp = client.start_conversation(question)
            conv_id = resp.get("conversation_id")
            msg_id = resp.get("message_id")
            
        # CONTINUE EXISTING
        else:
            logger.info(f"Continuing conversation {conversation_id} in Space: {target_space_id}...")
            try:
                resp = client.send_message(conversation_id, question)
                conv_id = conversation_id # ID stays the same
                msg_id = resp.get("message_id")
            except Exception as e:
                if "Conversation not found" in str(e):
                    logger.warning("Conversation expired. Starting new one.")
                    resp = client.start_conversation(question)
                    conv_id = resp.get("conversation_id")
                    msg_id = resp.get("message_id")
                else:
                    raise e

        # WAIT FOR COMPLETION (Timeout handled in wait_for_message_completion)
        complete_msg = client.wait_for_message_completion(conv_id, msg_id, timeout=300)
        result, query_text = process_genie_response(client, conv_id, msg_id, complete_msg)
        
        return conv_id, msg_id, result, query_text

    except TimeoutError as te:
        return conversation_id, None, str(te), None
    except Exception as e:
        logger.error(f"Genie Query Error: {e}")
        return conversation_id, None, f"An error occurred: {str(e)}", None
