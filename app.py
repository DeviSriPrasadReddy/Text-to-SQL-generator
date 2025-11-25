import dash
from dash import html, dcc, Input, Output, State, callback_context, no_update, dash_table
import dash_bootstrap_components as dbc
import pandas as pd
import logging
import os
import traceback
import datetime
import uuid
import json
from dotenv import load_dotenv

# --- IMPORTS ---
from routing import SPACE_CONFIG, orchestrate_routing
from genie_backend import execute_genie_query

load_dotenv()

DATABRICKS_HOST = os.environ.get("DATABRICKS_HOST")
GENIE_USER_TOKEN = os.environ.get("DATABRICKS_TOKEN")

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], title="Genie Debugger")
server = app.server

# --- LAYOUT ---
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H3("🧞 Genie Robust Router"), width=8, className="mt-3"),
        dbc.Col(html.Div(id="status-header", children="System Ready", className="text-muted mt-3 text-end"), width=4)
    ]),
    html.Hr(),
    
    # --- DEBUG WATCHER (Displays Raw Trigger Data) ---
    dbc.Alert(id="trigger-watcher", color="danger", style={"fontSize": "10px", "fontFamily": "monospace", "overflowWrap": "break-word"}, children="Waiting for trigger..."),

    dbc.Row([
        dbc.Col([
            dbc.Card([dbc.CardHeader("Active Contexts"), dbc.CardBody(id="active-sessions-list")], className="mb-3"),
            dbc.Button("Reset", id="reset-btn", color="outline-danger", size="sm", className="w-100")
        ], width=3),

        dbc.Col([
            html.Div(id="chat-window", style={"height": "500px", "overflowY": "auto", "padding": "20px", "border": "1px solid #ccc"}),
            dbc.Row([
                dbc.Col(dbc.Input(id="user-input", placeholder="Type here...", autocomplete="off"), width=10),
                dbc.Col(dbc.Button("Send", id="send-btn", color="primary", className="w-100"), width=2)
            ], className="mt-3"),
            html.Div(id="typing-indicator", className="text-muted small mt-1")
        ], width=9)
    ], className="chat-container mb-3"),

    # --- TERMINAL ---
    dbc.Collapse(
        dbc.Card([dbc.CardHeader("🐞 Live Execution Log"), dbc.CardBody(html.Div(id="debug-console", style={"background": "#000", "color": "#0f0", "height": "200px", "overflowY": "scroll", "whiteSpace": "pre-wrap"}))]),
        id="debug-collapse", is_open=True
    ),

    dcc.Store(id="chat-history", data=[]),
    dcc.Store(id="session-store", data={}), 
    
    # THIS IS THE CRITICAL COMPONENT
    dcc.Store(id="backend-trigger", data=None),
    
    html.Div(id="dummy-scroll-target")
], fluid=True, style={"padding": "20px"})

# --- LOGGING HELPER ---
def format_log(current_logs, new_entry):
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    if not isinstance(current_logs, list): current_logs = []
    current_logs.append(html.Div(f"[{ts}] {new_entry}"))
    return current_logs

# --- MESSAGE RENDERER ---
def render_message(msg):
    is_user = msg["role"] == "user"
    css_class = "ml-auto bg-primary text-white" if is_user else "mr-auto bg-light text-dark border"
    content = msg["content"]
    
    children = [html.Small(msg.get('space_label', ''), style={"fontWeight":"bold", "color":"#ddd" if is_user else "#555"})]
    
    if isinstance(content, str) and content.startswith('{') and "columns" in content:
        try:
            df = pd.read_json(content, orient='split')
            children.append(dash_table.DataTable(data=df.to_dict('records'), columns=[{"name": i, "id": i} for i in df.columns], style_table={'overflowX': 'auto'}))
        except: children.append(html.Div("Error parsing table"))
    else:
        children.append(dcc.Markdown(str(content)))
        
    return html.Div(children, className=f"p-3 rounded {css_class}", style={"width": "fit-content", "maxWidth": "85%", "marginBottom": "10px"})


# ==============================================================================
# 🔄 CALLBACK 1: USER INPUT & ROUTING
# ==============================================================================
@app.callback(
    [Output("chat-history", "data", allow_duplicate=True),
     Output("chat-window", "children", allow_duplicate=True),
     Output("user-input", "value"),
     Output("backend-trigger", "data", allow_duplicate=True), # Writes to Store
     Output("debug-console", "children", allow_duplicate=True),
     Output("typing-indicator", "children", allow_duplicate=True)],
    [Input("send-btn", "n_clicks"),
     Input("user-input", "n_submit")],
    [State("user-input", "value"),
     State("chat-history", "data"),
     State("session-store", "data"),
     State("debug-console", "children")],
    prevent_initial_call=True
)
def handle_user_routing(n_c, n_s, user_text, history, session_store, current_logs):
    if not user_text: return no_update
    if history is None: history = []
    
    # 1. Update UI immediately
    history.append({"role": "user", "content": user_text})
    ui_messages = [render_message(m) for m in history]
    current_logs = format_log(current_logs, f"STEP 1 START: User typed '{user_text}'")
    
    try:
        # 2. Run Router
        route_decision = orchestrate_routing(user_text, session_store)
        current_logs = format_log(current_logs, f"Router Result: {route_decision}")
        
        target_space_id = route_decision.get("target_space_id")
        target_conv_id = route_decision.get("target_conversation_id")
        space_label = next((v['label'] for k,v in SPACE_CONFIG.items() if v['id'] == target_space_id), "Unknown")

        # 3. Payload
        trigger_payload = {
            "text": user_text,
            "space_id": target_space_id,
            "conv_id": target_conv_id,
            "space_label": space_label,
            "uuid": str(uuid.uuid4()) # Unique ID
        }
        
        current_logs = format_log(current_logs, f"WRITING TO TRIGGER STORE: {space_label}")
        return history, ui_messages, "", trigger_payload, current_logs, f"Routing to {space_label}..."
        
    except Exception as e:
        current_logs = format_log(current_logs, f"ROUTING ERROR: {e}")
        return history, ui_messages, "", None, current_logs, "Error"


# ==============================================================================
# 👁️ WATCHER CALLBACK (Debugs the Hand-off)
# ==============================================================================
@app.callback(
    Output("trigger-watcher", "children"),
    Input("backend-trigger", "data")
)
def show_trigger_data(data):
    # This proves if the data actually reached the browser
    if not data: return "Trigger Store is Empty (None)"
    return f"TRIGGER STORE UPDATED: {json.dumps(data)}"


# ==============================================================================
# ⚙️ CALLBACK 2: BACKEND EXECUTION
# Triggered by TIMESTAMP, not Data (100% Reliable)
# ==============================================================================
@app.callback(
    [Output("chat-history", "data", allow_duplicate=True),
     Output("chat-window", "children", allow_duplicate=True),
     Output("session-store", "data"),
     Output("active-sessions-list", "children"),
     Output("debug-console", "children", allow_duplicate=True),
     Output("typing-indicator", "children", allow_duplicate=True)],
    [Input("backend-trigger", "modified_timestamp")], # <--- CHANGED INPUT
    [State("backend-trigger", "data"),                # <--- READ DATA AS STATE
     State("chat-history", "data"),
     State("session-store", "data"),
     State("debug-console", "children")],
    prevent_initial_call=True
)
def execute_backend_logic(ts, trigger_data, history, session_store, current_logs):
    # 1. Validation
    if not ts or not trigger_data:
        return no_update

    if history is None: history = []
    if session_store is None: session_store = {}
    
    current_logs = format_log(current_logs, "STEP 2 START: Timestamp detected change")

    try:
        # 2. Extract & Execute
        space_id = trigger_data["space_id"]
        current_logs = format_log(current_logs, f"Executing Genie in {space_id}...")
        
        final_conv_id, result, sql = execute_genie_query(
            user_query=trigger_data["text"],
            space_id=space_id,
            current_conv_id=trigger_data["conv_id"], 
            user_token=GENIE_USER_TOKEN,
            host=DATABRICKS_HOST
        )
        
        current_logs = format_log(current_logs, "Genie returned results.")

        # 3. Update History
        content = result
        if isinstance(result, pd.DataFrame):
            content = result.to_json(orient='split')

        history.append({
            "role": "assistant",
            "content": content,
            "space_label": trigger_data["space_label"],
            "sql": sql
        })
        ui_messages = [render_message(m) for m in history]

        # 4. Update Session
        session_store[space_id] = {"conv_id": final_conv_id, "last_topic": trigger_data["text"]}
        active_ui = [html.Div(f"● {sid[:5]}... : {data['last_topic'][:15]}...") for sid, data in session_store.items()]

        return history, ui_messages, session_store, active_ui, current_logs, ""

    except Exception as e:
        err_msg = f"BACKEND ERROR: {str(e)}"
        current_logs = format_log(current_logs, err_msg)
        history.append({"role": "system", "content": f"❌ {err_msg}"})
        return history, [render_message(m) for m in history], session_store, no_update, current_logs, ""

# --- RESET ---
@app.callback(
    [Output("chat-history", "data", allow_duplicate=True),
     Output("chat-window", "children", allow_duplicate=True),
     Output("session-store", "data", allow_duplicate=True),
     Output("active-sessions-list", "children", allow_duplicate=True),
     Output("debug-console", "children", allow_duplicate=True),
     Output("backend-trigger", "data", allow_duplicate=True)],
    [Input("reset-btn", "n_clicks")],
    [State("debug-console", "children")],
    prevent_initial_call=True
)
def reset_app(n, logs):
    logs = format_log(logs, "--- SYSTEM RESET ---")
    return [], [], {}, "No active threads.", logs, None

# Scroll Script
app.clientside_callback(
    """function(c){ var w = document.getElementById('chat-window'); if(w){ w.scrollTop = w.scrollHeight; } return null; }""",
    Output("dummy-scroll-target", "children"), Input("chat-window", "children")
)

if __name__ == "__main__":
    app.run_server(debug=True, port=8050)
