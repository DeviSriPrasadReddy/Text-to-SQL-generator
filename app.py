import dash
from dash import html, dcc, Input, Output, State, MATCH, ALL, no_update, callback_context, dash_table
import dash_bootstrap_components as dbc
import pandas as pd
import json
import logging
import uuid
import os
import datetime
import concurrent.futures
import flask
import sqlparse
import numpy as np
from dotenv import load_dotenv

# --- DATABRICKS SDK ---
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import ChatMessage, ChatMessageRole

load_dotenv()

# --- CONFIGURATION ---
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

DATABRICKS_HOST = os.environ.get("DATABRICKS_HOST")
# Endpoint names from .env
SERVING_ENDPOINT_NAME = os.environ.get("SERVING_ENDPOINT_NAME", "databricks-meta-llama-3-70b-instruct")

# --- APP SETUP ---
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], title="Genie AI")
server = app.server

# ==============================================================================
# 🎨 CSS & LAYOUT CONFIGURATION (The "Fixed" UI)
# ==============================================================================
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            /* 1. MAIN CONTAINER: Uses Flexbox to separate Header, Chat, and Input */
            .main-container {
                display: flex;
                flex-direction: column;
                height: 100vh; /* Full Viewport Height */
                overflow: hidden; /* No scroll on the body */
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            }

            /* 2. TOP NAV */
            .top-nav {
                flex-shrink: 0; /* Don't shrink */
                height: 60px;
                background: #1b1e23;
                color: white;
                display: flex;
                align-items: center;
                padding: 0 20px;
                justify-content: space-between;
            }

            /* 3. CONTENT WRAPPER (Sidebar + Chat) */
            .content-wrapper {
                display: flex;
                flex-grow: 1; /* Fills remaining height */
                overflow: hidden; 
            }

            /* 4. SIDEBAR */
            .sidebar {
                width: 260px;
                background: #f8f9fa;
                border-right: 1px solid #dee2e6;
                display: flex;
                flex-direction: column;
                padding: 15px;
                transition: margin-left 0.3s ease;
                flex-shrink: 0;
            }
            .sidebar.closed { margin-left: -260px; }
            .chat-item { padding: 10px; cursor: pointer; border-radius: 5px; margin-bottom: 5px; font-size: 0.9rem; color: #333; }
            .chat-item:hover { background: #e9ecef; }
            .chat-item.active { background: #e2e6ea; font-weight: 600; border-left: 3px solid #007bff; }

            /* 5. CHAT AREA (The Scrolling Part) */
            .chat-area {
                flex-grow: 1;
                display: flex;
                flex-direction: column;
                background: white;
                position: relative;
            }

            .chat-messages {
                flex-grow: 1;
                overflow-y: auto; /* SCROLL HERE */
                padding: 20px 10% 20px 10%; /* Center content slightly */
                scroll-behavior: smooth;
            }

            /* 6. FIXED INPUT AREA (Pinned to Bottom) */
            .input-area {
                flex-shrink: 0;
                background: white;
                padding: 20px 10%;
                border-top: 1px solid #eee;
            }

            /* MESSAGES STYLING */
            .message { margin-bottom: 20px; max-width: 100%; }
            .user-message { text-align: right; }
            .user-message .message-content { background: #007bff; color: white; display: inline-block; padding: 10px 15px; border-radius: 15px 15px 0 15px; text-align: left;}
            .bot-message { text-align: left; }
            .bot-message .message-content { background: #f1f3f4; color: black; display: inline-block; padding: 15px; border-radius: 15px 15px 15px 0; max-width: 100%; width: fit-content; }
            
            /* TABLE & INSIGHTS */
            .insight-section { margin-top: 15px; border-top: 1px solid #ddd; padding-top: 10px; }
            .insight-content { background: #e8f4ff; padding: 15px; border-radius: 8px; border: 1px solid #b6d4fe; margin-top: 10px; font-size: 0.9rem; }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>{%config%}{%scripts%}{%renderer%}</footer>
    </body>
</html>
'''

# ==============================================================================
# ⏱️ HELPERS: TIMEOUT & BACKEND
# ==============================================================================

def run_with_timeout(func, args=(), kwargs=None, timeout_seconds=300):
    """Executes a function with a strict 5-minute time limit."""
    if kwargs is None: kwargs = {}
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(func, *args, **kwargs)
        try:
            return future.result(timeout=timeout_seconds)
        except concurrent.futures.TimeoutError:
            return "⚠️ **Timeout Error:** The process took longer than 5 minutes and was stopped."
        except Exception as e:
            return f"⚠️ **Error:** {str(e)}"

def get_user_info_from_header():
    """Extracts user info from Flask headers (standard for Databricks Apps)."""
    try:
        # Standard Databricks Proxy Headers
        token = flask.request.headers.get('X-Forwarded-Access-Token')
        email = flask.request.headers.get('X-Forwarded-User', '')
        initial = email[0].upper() if email else "U"
        return token, initial
    except:
        return None, "U"

def mock_genie_query(user_input):
    """
    MOCK Backend for demonstration. 
    Replace this with your actual 'genie_query' function.
    """
    import time
    time.sleep(1) # Simulate network
    
    if "sales" in user_input.lower() or "data" in user_input.lower():
        # Return a DataFrame
        df = pd.DataFrame({
            "Date": pd.date_range(start="2023-01-01", periods=10),
            "Region": ["North", "South", "East", "West", "North", "South", "East", "West", "North", "South"],
            "Sales": [100, 150, 200, 130, 120, 160, 210, 140, 110, 155],
            "Profit": [20, 30, 40, 25, 22, 35, 45, 28, 21, 32]
        })
        sql = "SELECT * FROM sales_data WHERE date >= '2023-01-01' LIMIT 10"
        return "conv_123", "msg_456", df.to_dict('records'), sql
    else:
        # Return Text
        return "conv_123", "msg_456", f"I received your query: '{user_input}'. Try asking for 'sales data' to see a table.", None

def call_llm_for_insights(df_json):
    """Uses Databricks SDK to analyze the dataframe."""
    try:
        df = pd.read_json(df_json, orient='split')
        preview = df.head(30).to_csv(index=False)
        
        prompt = f"""
        Analyze the following data (top 30 rows):
        {preview}
        
        Provide 3 concise, bulleted business insights.
        """
        
        w = WorkspaceClient() # Uses env vars
        response = w.serving_endpoints.query(
            name=SERVING_ENDPOINT_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating insights: {str(e)}"

def get_visual_spec_mock(df):
    """Mock visualization generator."""
    return {
        "data": [{"type": "bar", "x": df[df.columns[0]], "y": df[df.columns[2]]}],
        "layout": {"title": "Auto-Generated Chart", "height": 300}
    }

# ==============================================================================
# 🖥️ LAYOUT
# ==============================================================================
app.layout = html.Div([
    # 1. Top Navigation
    html.Div([
        html.Div([
            html.Button("☰", id="sidebar-toggle", className="btn btn-outline-light btn-sm me-2"),
            html.Span("🧞 Genie AI Space", style={"fontWeight": "bold", "fontSize": "1.2rem"})
        ]),
        html.Div(id="user-avatar", className="rounded-circle bg-primary text-white p-2", style={"width": "40px", "height": "40px", "textAlign": "center"})
    ], className="top-nav"),

    # 2. Content Wrapper
    html.Div([
        # Sidebar
        html.Div([
            dbc.Button("+ New Chat", id="new-chat-btn", color="primary", className="w-100 mb-3"),
            html.Div(id="chat-list", className="flex-grow-1"),
        ], id="sidebar", className="sidebar"),

        # Chat Area
        html.Div([
            # Messages Scroll Area
            html.Div(id="chat-messages", className="chat-messages"),
            
            # Fixed Input Area
            html.Div([
                dbc.InputGroup([
                    dbc.Input(id="user-input", placeholder="Ask a question about your data...", autocomplete="off"),
                    dbc.Button("Send", id="send-btn", color="primary")
                ]),
                html.Div(id="typing-indicator", className="text-muted small mt-1")
            ], className="input-area")
            
        ], className="chat-area")

    ], className="content-wrapper"),

    # 3. Stores & Invisible Components
    dcc.Store(id="chat-history-store", data=[]),
    dcc.Store(id="session-store", data={"current_index": 0}),
    dcc.Store(id="trigger-process", data=None),
    html.Div(id="dummy-scroll"),
    dcc.Store(id="initial-load", data=True)

], className="main-container")


# ==============================================================================
# 🔄 CALLBACKS
# ==============================================================================

# 1. SIDEBAR TOGGLE
@app.callback(
    Output("sidebar", "className"),
    Input("sidebar-toggle", "n_clicks"),
    State("sidebar", "className"),
    prevent_initial_call=True
)
def toggle_sidebar(n, current_class):
    if "closed" in current_class:
        return "sidebar"
    return "sidebar closed"

# 2. HANDLE INPUT & ROUTING
@app.callback(
    [Output("chat-messages", "children", allow_duplicate=True),
     Output("user-input", "value"),
     Output("trigger-process", "data"),
     Output("chat-list", "children"),
     Output("chat-history-store", "data", allow_duplicate=True),
     Output("session-store", "data", allow_duplicate=True)],
    [Input("send-btn", "n_clicks"), Input("user-input", "n_submit"), Input("new-chat-btn", "n_clicks")],
    [State("user-input", "value"),
     State("chat-messages", "children"),
     State("chat-history-store", "data"),
     State("session-store", "data")],
    prevent_initial_call=True
)
def handle_input(n_send, n_submit, n_new, text, current_msgs, history, session):
    ctx = callback_context
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
    
    if not history: history = []
    if current_msgs is None: current_msgs = []
    
    # CASE: NEW CHAT
    if trigger_id == "new-chat-btn":
        new_session_idx = 0
        history.insert(0, {"query": "New Chat", "msgs": [], "dfs": {}})
        # Rebuild Chat List
        chat_list_ui = [html.Div(h.get("query", "Chat"), className="chat-item active" if i==0 else "chat-item") for i, h in enumerate(history)]
        return [], "", None, chat_list_ui, history, {"current_index": 0}

    # CASE: USER SUBMIT
    if not text: return no_update
    
    # 1. Append User Message
    user_msg_ui = html.Div(html.Div(text, className="message-content"), className="message user-message")
    current_msgs.append(user_msg_ui)
    
    # 2. Append Loading Spinner
    loading_ui = html.Div(html.Div(dbc.Spinner(size="sm"), className="message-content"), className="message bot-message", id="loading-placeholder")
    current_msgs.append(loading_ui)
    
    # Update History
    curr_idx = session.get("current_index", 0)
    if curr_idx < len(history):
        history[curr_idx]["msgs"] = current_msgs
        if history[curr_idx]["query"] == "New Chat": history[curr_idx]["query"] = text[:20]
    else:
        history.insert(0, {"query": text[:20], "msgs": current_msgs, "dfs": {}})
        curr_idx = 0

    # Update UI List
    chat_list_ui = [html.Div(h.get("query", "Chat"), className="chat-item active" if i==curr_idx else "chat-item") for i, h in enumerate(history)]

    return current_msgs, "", {"text": text, "session_idx": curr_idx}, chat_list_ui, history, {"current_index": curr_idx}


# 3. BACKEND EXECUTION (WITH TIMEOUT)
@app.callback(
    [Output("chat-messages", "children", allow_duplicate=True),
     Output("chat-history-store", "data", allow_duplicate=True)],
    [Input("trigger-process", "data")],
    [State("chat-messages", "children"),
     State("chat-history-store", "data")],
    prevent_initial_call=True
)
def execute_backend(trigger, current_msgs, history):
    if not trigger: return no_update
    
    user_text = trigger["text"]
    session_idx = trigger["session_idx"]
    
    # Remove loading spinner (last item)
    if current_msgs: current_msgs.pop()
    
    # --- WRAPPED TIMEOUT EXECUTION ---
    # We wrap the external call
    def run_query():
        # Replace 'mock_genie_query' with 'genie_query' or your real function
        return mock_genie_query(user_text)

    try:
        # EXECUTE WITH 5 MINUTE LIMIT
        result = run_with_timeout(run_query, timeout_seconds=300)
        
        # Check for timeout string return
        if isinstance(result, str) and result.startswith("⚠️"):
            bot_content = dcc.Markdown(result)
        else:
            # Unpack successful result
            conv_id, msg_id, data, sql = result
            
            if isinstance(data, list): # It's a Table/DataFrame
                df = pd.DataFrame(data)
                df_id = f"df_{uuid.uuid4().hex[:8]}"
                
                # Save DF to history
                history[session_idx]["dfs"][df_id] = df.to_json(orient='split')
                
                # 1. Render Table
                table_comp = dash_table.DataTable(
                    data=df.to_dict('records'),
                    columns=[{"name": i, "id": i} for i in df.columns],
                    style_table={'overflowX': 'auto'},
                    page_size=5,
                    style_header={'backgroundColor': '#f8f9fa', 'fontWeight': 'bold'},
                    style_cell={'textAlign': 'left', 'padding': '10px'}
                )
                
                # 2. Render Graph (Simple Mock)
                fig = get_visual_spec_mock(df)
                graph_comp = dcc.Graph(figure=fig, config={'displayModeBar': False})
                
                # 3. INSIGHT BUTTON (Placed Below)
                insight_btn = dbc.Button(
                    "✨ Generate Insights", 
                    id={"type": "insight-btn", "index": df_id}, 
                    color="warning", outline=True, size="sm", 
                    className="mt-3"
                )
                
                # 4. Insight Output Container
                insight_div = html.Div(
                    id={"type": "insight-out", "index": df_id},
                    className="insight-section"
                )
                
                bot_content = html.Div([
                    dcc.Markdown(f"Found {len(df)} rows matching your query."),
                    table_comp,
                    html.Hr(),
                    graph_comp,
                    html.Div([insight_btn, insight_div]) # Grouped together
                ])
                
            else: # Text Response
                bot_content = dcc.Markdown(str(data))
        
        # Create Message Bubble
        bot_msg_ui = html.Div(html.Div(bot_content, className="message-content"), className="message bot-message")
        current_msgs.append(bot_msg_ui)
        
        # Save Messages to History
        history[session_idx]["msgs"] = current_msgs
        
        return current_msgs, history

    except Exception as e:
        err_ui = html.Div(html.Div(f"Error: {str(e)}", className="message-content text-danger"), className="message bot-message")
        current_msgs.append(err_ui)
        return current_msgs, history


# 4. GENERATE INSIGHTS (WITH TIMEOUT)
@app.callback(
    Output({"type": "insight-out", "index": MATCH}, "children"),
    Input({"type": "insight-btn", "index": MATCH}, "n_clicks"),
    State({"type": "insight-btn", "index": MATCH}, "id"),
    State("chat-history-store", "data"),
    State("session-store", "data"),
    prevent_initial_call=True
)
def generate_insights_click(n, btn_id, history, session):
    if not n: return no_update
    
    df_id = btn_id["index"]
    curr_idx = session.get("current_index", 0)
    
    # 1. Get DataFrame from Store
    try:
        df_json = history[curr_idx]["dfs"].get(df_id)
        if not df_json: return "Error: Data expired."
    except:
        return "Error: Session context lost."
        
    # 2. Call LLM with Timeout
    def run_llm():
        return call_llm_for_insights(df_json)
        
    loading = html.Div([dbc.Spinner(size="sm"), " Analyzing data..."], className="text-muted")
    
    # We return loading first? Dash doesn't support generators easily in one callback.
    # So we block (using run_with_timeout) and return result.
    
    insights = run_with_timeout(run_llm, timeout_seconds=300)
    
    return html.Div([
        html.H6("📊 AI Analysis", className="mb-2"),
        dcc.Markdown(insights)
    ], className="insight-content")


# 5. AUTO SCROLL
app.clientside_callback(
    """
    function(children) {
        var chat_div = document.getElementById('chat-messages');
        if(chat_div) {
            chat_div.scrollTop = chat_div.scrollHeight;
        }
        return null;
    }
    """,
    Output("dummy-scroll", "children"),
    Input("chat-messages", "children")
)

# 6. LOAD USER INITIAL
@app.callback(
    Output("user-avatar", "children"),
    Input("initial-load", "data")
)
def load_user(d):
    _, initial = get_user_info_from_header()
    return initial

if __name__ == "__main__":
    app.run_server(debug=True, port=8050)
