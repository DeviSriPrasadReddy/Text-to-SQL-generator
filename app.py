import os
from genie_backend import genie_query_router_aware
from router import route_question, SPACE_CONFIG

# --- SESSION STATE MOCKUP ---
# In a real app (Streamlit/Flask), this would be session_state or a database
# This dict maps:  Space_ID -> Conversation_ID
user_session_state = {} 

def handle_user_input(user_question: str, user_token: str = None):
    """
    Main handler function.
    1. Routes the question.
    2. Retrieves existing conversation ID for that specific space.
    3. Calls Genie.
    4. Updates state.
    """
    
    print(f"User asks: {user_question}")
    
    # Step 1: Route the question
    # (You can pass conversation history to the router here if you want smarter follow-ups)
    target_space_id = route_question(user_question)
    
    # Identify which space name this ID belongs to (for logging)
    space_name = next((k for k, v in SPACE_CONFIG.items() if v["id"] == target_space_id), "Unknown Space")
    print(f"--> Routing to: {space_name} ({target_space_id})")
    
    # Step 2: Check if we already have a conversation open for THIS space
    existing_conv_id = user_session_state.get(target_space_id)
    
    if existing_conv_id:
        print(f"--> Resuming conversation ID: {existing_conv_id}")
    else:
        print("--> Starting new conversation for this space.")

    # Step 3: Execute Query with Timeout logic (handled inside backend)
    new_conv_id, msg_id, result, sql_query = genie_query_router_aware(
        question=user_question,
        target_space_id=target_space_id,
        conversation_id=existing_conv_id,
        user_token=user_token
    )
    
    # Step 4: Update State
    # If the query was successful (we got a valid ID back), save it
    if new_conv_id:
        user_session_state[target_space_id] = new_conv_id
        
    return result, sql_query

# --- TEST RUN ---
if __name__ == "__main__":
    # Mock Environment Setup
    os.environ["FINANCE_SPACE_ID"] = "01ef..." 
    os.environ["HR_SPACE_ID"] = "01ed..."
    
    # 1. User asks a finance question
    response1, _ = handle_user_input("What was our revenue last quarter?")
    print(f"Genie: {response1}\n")
    
    # 2. User switches context to HR (Should start new conv in HR space)
    response2, _ = handle_user_input("How many engineers did we hire?")
    print(f"Genie: {response2}\n")
    
    # 3. User asks follow up regarding HR (Should reuse HR conv ID)
    response3, _ = handle_user_input("List their names.")
    print(f"Genie: {response3}\n")
    
    # 4. User switches back to Finance (Should reuse Finance conv ID)
    response4, _ = handle_user_input("Break down that revenue by region.")
    print(f"Genie: {response4}\n")
