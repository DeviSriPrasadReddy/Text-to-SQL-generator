import dash
from dash import html, dcc, Input, Output, State, MATCH, callback_context, no_update, dash_table
import dash_bootstrap_components as dbc
import pandas as pd
import logging
import os
import traceback
import datetime
from dotenv import load_dotenv

# --- IMPORTS ---
from routing import SPACE_CONFIG, orchestrate_routing
from genie_backend import execute_genie_query

load_dotenv()

# --- CONFIG ---
DATABRICKS_HOST = os.environ.get("DATABRICKS_HOST")
GENIE_USER_TOKEN = os.environ.get("DATABRICKS_TOKEN")

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], title="Genie Debugger")
server = app.server

# --- CUSTOM CSS (Terminal Style) ---
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            .chat-container { height: 75vh; display: flex; flex-direction: column; }
            .chat-window { flex-grow: 1; overflow-y: auto; padding: 20px; background: #f8f9fa; border: 1px solid #dee2e6; border-radius: 5px;}
            
            /* The Debug Terminal */
            .debug-terminal {
                background-color: #1e1e1e;
                color: #00ff00;
                font-family: 'Courier New', monospace;
                font-size: 0.8rem;
                padding: 15px;
                height: 200px;
                overflow-y: auto;
                border-radius: 5px;
                white-space: pre-wrap;
            }
            .log-timestamp { color: #888; margin-right: 10px; }
            .log-error { color: #ff5555; font-weight: bold; }
            .log-info { color: #66d9ef; }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>{%config%}{%scripts%}{%renderer%}</footer>
    </body>
</html>
'''

# --- LAYOUT ---
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H3("🧞 Genie Router + Debugger"), width=8, className="mt-3"),
        dbc.Col(dbc.Switch(id="debug-mode-toggle", label="Show Debug Log", value=True), width=4, className="mt-3 text-end")
    ]),
    html.Hr(),
    
    # --- CHAT SECTION ---
    dbc.Row([
        dbc.Col([
            dbc.Card([dbc.CardHeader("Active Contexts"), dbc.CardBody(id="active-sessions-list")], className="mb-3"),
            dbc.Button("Reset", id="reset-btn", color="outline-danger", size="sm", className="w-100")
        ], width=3),

        dbc.Col([
            html.Div(id="chat-window", className="chat-window"),
            dbc.Row([
                dbc.Col(dbc.Input(id="user-input", placeholder="Type here...", autocomplete="off"), width=10),
                dbc.Col(dbc.Button("Send", id="send-btn", color="primary", className="w-100"), width=2)
            ], className="mt-3"),
            html.Div(id="typing-indicator", className="text-muted small mt-1")
        ], width=9)
    ], className="chat-container mb-3"),

    # --- DEBUG CONSOLE SECTION ---
    dbc.Collapse(
        dbc.Card([
            dbc.CardHeader("🐞 Live Execution Log"),
            dbc.CardBody(
                html.Div(id="debug-console", className="debug-terminal", children="Waiting for input...")
            )
        ]),
        id="debug-collapse", is_open=True
    ),

    # --- STORES ---
    dcc.Store(id="chat-history", data=[]),
    dcc.Store(id="session-store", data={}), 
    dcc.Store(id="backend-trigger", data=None),
    html.Div(id="dummy-scroll-target")
], fluid=True, style={"padding": "20px"})

# --- HELPER: LOGGING ---
def format_log(current_logs, new_entry, level="INFO"):
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    css = "log-error" if level == "ERROR" else "log-info" if level == "SYSTEM" else ""
    
    # We return a list of HTML components to keep styling
    if not isinstance(current_logs, list): current_logs = []
    
    log_line = html.Div([
        html.Span(f"[{ts}] ", className="log-timestamp"),
        html.Span(f"[{level}] ", className=css),
        html.Span(new_entry)
    ])
    current_logs.append(log_line)
    return current_logs

# --- HELPER: RENDER MESSAGE ---
def render_message(msg):
    is_user = msg["role"] == "user"
    css_class = "user-message" if is_user else "bot-message"
    children = [html.Span(f"{msg.get('space_label', '')}", className="route-badge")] if not is_user else []
    
    content = msg["content"]
    if isinstance(content, str) and content.startswith('{') and "columns" in content:
        df = pd.read_json(content, orient='split')
        children.append(dash_table.DataTable(data=df.to_dict('records'), columns=[{"name": i, "id": i} for i in df.columns], style_table={'overflowX': 'auto'}))
    else:
        children.append(dcc.Markdown(str(content)))
        
    return html.Div(children, className=f"message {css_class}", style={"textAlign": "right" if is_user else "left", "marginBottom": "10px"})

# --- MAIN CALLBACK ---
@app.callback(
    [Output("chat-window", "children"),
     Output("user-input", "value"),
     Output("chat-history", "data"),
     Output("backend-trigger", "data"),
     Output("typing-indicator", "children"),
     Output("session-store", "data"),
     Output("active-sessions-list", "children"),
     Output("debug-console", "children"),  # <--- UPDATING THE CONSOLE
     Output("debug-collapse", "is_open")],
    [Input("send-btn", "n_clicks"),
     Input("user-input", "n_submit"),
     Input("reset-btn", "n_clicks"),
     Input("backend-trigger", "data"),
     Input("debug-mode-toggle", "value")],
    [State("user-input", "value"),
     State("chat-history", "data"),
     State("session-store", "data"),
     State("debug-console", "children")], # <--- GET OLD LOGS
    prevent_initial_call=True
)
def manage_chat(n_clicks, n_submit, n_reset, trigger_data, debug_toggle,
                user_text, history, session_store, current_logs):
    
    ctx = callback_context
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if history is None: history = []
    if session_store is None: session_store = {}
    
    # Initialize Log if empty
    if not isinstance(current_logs, list): current_logs = []

    # --- RESET ---
    if trigger_id == "reset-btn":
        current_logs = format_log(current_logs, "--- RESET INITIATED ---", "SYSTEM")
        return [], "", [], None, "", {}, "No active threads.", current_logs, debug_toggle

    # --- STEP 1: ROUTING ---
    if trigger_id in ["send-btn", "user-input"] and user_text:
        current_logs = format_log(current_logs, f"User Input: '{user_text}'")
        history.append({"role": "user", "content": user_text})
        
        try:
            current_logs = format_log(current_logs, "FUNCTION CALL: orchestrate_routing()", "SYSTEM")
            
            # CALL ROUTER
            route_decision = orchestrate_routing(user_text, session_store)
            
            # Log the JSON result from LLM
            current_logs = format_log(current_logs, f"Router Result: {route_decision}")
            
            target_space_id = route_decision.get("target_space_id")
            target_conv_id = route_decision.get("target_conversation_id")
            space_label = "Unknown"
            
            for k,v in SPACE_CONFIG.items():
                if v['id'] == target_space_id: space_label = v['label']

            next_trigger = {
                "text": user_text,
                "space_id": target_space_id,
                "conv_id": target_conv_id, 
                "space_label": space_label
            }
            
            current_logs = format_log(current_logs, f"Step 1 Success. Routing to: {space_label}")
            
            return [render_message(m) for m in history], "", history, next_trigger, f"Routing to {space_label}...", session_store, no_update, current_logs, debug_toggle

        except Exception as e:
            err = traceback.format_exc()
            current_logs = format_log(current_logs, f"ROUTING CRASH: {str(e)}", "ERROR")
            current_logs = format_log(current_logs, err, "ERROR") # Show full traceback in UI
            return [render_message(m) for m in history], "", history, None, "Error", session_store, no_update, current_logs, debug_toggle

    # --- STEP 2: BACKEND EXECUTION ---
    if trigger_id == "backend-trigger" and trigger_data:
        try:
            current_logs = format_log(current_logs, "--- STEP 2: GENIE EXECUTION ---", "SYSTEM")
            
            user_query = trigger_data["text"]
            space_id = trigger_data["space_id"]
            conv_id = trigger_data["conv_id"]
            
            current_logs = format_log(current_logs, f"Target Space ID: {space_id}")
            current_logs = format_log(current_logs, f"Conversation ID: {conv_id} (None = New)")
            
            current_logs = format_log(current_logs, "FUNCTION CALL: execute_genie_query()", "SYSTEM")
            
            # CALL GENIE
            final_conv_id, result, sql = execute_genie_query(
                user_query=user_query,
                space_id=space_id,
                current_conv_id=conv_id, 
                user_token=GENIE_USER_TOKEN,
                host=DATABRICKS_HOST
            )
            
            current_logs = format_log(current_logs, f"Genie Returned. SQL Length: {len(str(sql))}")
            
            # Format Result
            content = result
            if isinstance(result, pd.DataFrame):
                content = result.to_json(orient='split')
                current_logs = format_log(current_logs, f"Data Received: {len(result)} rows.")

            history.append({
                "role": "assistant",
                "content": content,
                "space_label": trigger_data["space_label"],
                "sql": sql
            })

            # Update Session Store
            session_store[space_id] = {
                "conv_id": final_conv_id,
                "last_topic": user_query 
            }
            
            # Update Active List UI
            active_ui = []
            for sid, data in session_store.items():
                active_ui.append(html.Div(f"● ID: {sid[:5]}... Topic: {data['last_topic'][:15]}..."))

            current_logs = format_log(current_logs, "--- CYCLE COMPLETE ---", "SYSTEM")
            
            return [render_message(m) for m in history], no_update, history, None, "", session_store, active_ui, current_logs, debug_toggle

        except Exception as e:
            err = traceback.format_exc()
            current_logs = format_log(current_logs, f"BACKEND CRASH: {str(e)}", "ERROR")
            current_logs = format_log(current_logs, err, "ERROR")
            return [render_message(m) for m in history], no_update, history, None, "Error", session_store, no_update, current_logs, debug_toggle

    return no_update

# Scroll Script
app.clientside_callback(
    """function(c){ var w = document.getElementById('chat-window'); if(w){ w.scrollTop = w.scrollHeight; } return null; }""",
    Output("dummy-scroll-target", "children"), Input("chat-window", "children")
)

if __name__ == "__main__":
    app.run_server(debug=True, port=8050)
