import dash
from dash import html, dcc, Input, Output, State, callback_context, no_update, dash_table
import dash_bootstrap_components as dbc
import pandas as pd
import datetime
import uuid
import os
import traceback
import json
from dotenv import load_dotenv

# --- IMPORTS ---
# Ensure routing.py and genie_backend.py are in the same folder
from routing import SPACE_CONFIG, orchestrate_routing
from genie_backend import execute_genie_query

load_dotenv()

# --- CONFIGURATION ---
DATABRICKS_HOST = os.environ.get("DATABRICKS_HOST")
GENIE_USER_TOKEN = os.environ.get("DATABRICKS_TOKEN")

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], title="Genie Router")
server = app.server

# --- CSS STYLING ---
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            /* 1. Main Layout: Fixed height */
            .chat-container { height: 70vh; }
            
            /* 2. Column Wrapper */
            .chat-col-wrapper {
                height: 100%;
                display: flex;
                flex-direction: column;
            }

            /* 3. Chat Window: Grows and Scrolls */
            .chat-window { 
                flex-grow: 1; 
                overflow-y: auto; 
                padding: 20px; 
                background: #f8f9fa; 
                border: 1px solid #dee2e6; 
                border-radius: 5px;
            }

            /* Debug Terminal */
            .debug-terminal { background-color: #1e1e1e; color: #00ff00; font-family: 'Courier New', monospace; font-size: 0.8rem; padding: 15px; height: 200px; overflow-y: auto; border-radius: 5px; white-space: pre-wrap; }
            .log-timestamp { color: #888; margin-right: 10px; }
            .log-error { color: #ff5555; font-weight: bold; }
            .log-info { color: #66d9ef; }
            
            /* SQL Toggle Styling */
            details > summary { cursor: pointer; color: #007bff; font-size: 0.8rem; margin-top: 8px; outline: none; list-style: none; }
            details > summary::-webkit-details-marker { display: none; }
            details > summary::after { content: " ▼ Show Generated SQL"; }
            details[open] > summary::after { content: " ▲ Hide SQL"; }
            details > pre { background: #2d2d2d; color: #f8f8f2; padding: 10px; border-radius: 5px; margin-top: 5px; font-size: 0.75rem; overflow-x: auto; white-space: pre-wrap; }
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
        dbc.Col(html.H3("🧞 Genie Multi-Space Router"), width=8, className="mt-3"),
        dbc.Col(html.Div(id="status-indicator", className="mt-4"), width=4)
    ]),
    html.Hr(),
    
    dbc.Row([
        # Sidebar
        dbc.Col([
            dbc.Card([dbc.CardHeader("Active Contexts"), dbc.CardBody(id="active-sessions-list")], className="mb-3"),
            dbc.Button("Reset Chat", id="reset-btn", color="outline-danger", size="sm", className="w-100")
        ], width=3, style={"height": "100%"}),

        # Chat Area
        dbc.Col([
            html.Div([
                html.Div(id="chat-window", className="chat-window"),
                html.Div([
                    dbc.Row([
                        dbc.Col(dbc.Input(id="user-input", placeholder="Type here...", autocomplete="off"), width=10),
                        dbc.Col(dbc.Button("Send", id="send-btn", color="primary", className="w-100"), width=2)
                    ], className="mt-3"),
                    html.Div(id="typing-indicator", className="text-muted small mt-1")
                ])
            ], className="chat-col-wrapper")
        ], width=9, style={"height": "100%"})

    ], className="chat-container mb-3"),

    # Debugger
    dbc.Collapse(
        dbc.Card([dbc.CardHeader("🐞 Logic Log"), dbc.CardBody(html.Div(id="debug-console", className="debug-terminal"))]),
        id="debug-collapse", is_open=True
    ),

    # Stores
    dcc.Store(id="chat-history", data=[]),
    dcc.Store(id="session-store", data={}), 
    dcc.Store(id="backend-trigger", data=None),
    html.Div(id="dummy-scroll-target")
], fluid=True, style={"padding": "20px"})

# --- LOGGING HELPERS ---
def create_log_element(text, level="INFO"):
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    css = "log-error" if level == "ERROR" else "log-info" if level == "SYSTEM" else ""
    return html.Div([html.Span(f"[{ts}] ", className="log-timestamp"), html.Span(f"[{level}] ", className=css), html.Span(str(text))])

def format_log(current_logs, new_entry):
    if not isinstance(current_logs, list): current_logs = []
    current_logs.append(create_log_element(new_entry))
    return current_logs

# --- MESSAGE RENDERER ---
def render_message(msg):
    is_user = msg["role"] == "user"
    css_class = "user-message" if is_user else "bot-message"
    align = "right" if is_user else "left"
    bg = "#007bff" if is_user else "#e9ecef"
    color = "white" if is_user else "black"
    
    # Badge (Bot Only)
    children = [html.Small(msg.get('space_label', ''), style={"display":"block", "marginBottom":"5px", "color":"#ccc" if is_user else "#666"})]
    
    content = msg["content"]
    
    # 1. Table Handling
    if isinstance(content, str) and content.startswith('{') and "columns" in content:
        try:
            df = pd.read_json(content, orient='split')
            children.append(dash_table.DataTable(
                data=df.to_dict('records'), 
                columns=[{"name": i, "id": i} for i in df.columns], 
                style_table={'overflowX': 'auto'}, 
                style_cell={'textAlign': 'left', 'color': 'black', 'fontFamily': 'sans-serif'},
                page_size=10
            ))
        except: 
            children.append(html.Div(str(content)))
    else:
        # 2. Markdown Handling
        children.append(dcc.Markdown(str(content)))
    
    # 3. SQL Toggle
    if not is_user and msg.get("sql"):
        children.append(html.Details([html.Summary(""), html.Pre(msg["sql"])]))
        
    return html.Div(children, style={"textAlign": align, "backgroundColor": bg, "color": color, "padding": "10px", "borderRadius": "10px", "marginBottom": "10px", "marginLeft": "auto" if is_user else "0", "marginRight": "0" if is_user else "auto", "width": "fit-content", "maxWidth": "85%"})


# ==============================================================================
# 🔄 STEP 1: ROUTING (Determines Space & ID)
# ==============================================================================
@app.callback(
    [Output("chat-window", "children", allow_duplicate=True),
     Output("user-input", "value"),
     Output("backend-trigger", "data", allow_duplicate=True),
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
def step_1_routing(n_c, n_s, user_text, history, session_store, current_ui_logs):
    if not user_text: return no_update
    if history is None: history = []
    
    # Render User Message Visuals
    temp_history = history + [{"role": "user", "content": user_text}]
    ui_messages = [render_message(m) for m in temp_history]
    
    # 📝 COLLECT LOGS (Don't render yet, pass to Step 2)
    step_logs = [f"Step 1: Routing '{user_text}'"]
    
    try:
        route_decision = orchestrate_routing(user_text, session_store)
        target_space_id = route_decision.get("target_space_id")
        
        target_conv_id = None
        if target_space_id in session_store:
            existing_id = session_store[target_space_id].get("conv_id")
            if existing_id:
                target_conv_id = existing_id
                step_logs.append(f"♻️ Reusing Context ID: {existing_id}")
            else:
                step_logs.append("ℹ️ Context found, but ID missing. New Thread.")
        else:
            step_logs.append("🆕 New Space detected.")

        space_label = next((v['label'] for k,v in SPACE_CONFIG.items() if v['id'] == target_space_id), "Unknown")
        
        trigger_payload = {
            "text": user_text,
            "space_id": target_space_id,
            "conv_id": target_conv_id,
            "space_label": space_label,
            "uuid": str(uuid.uuid4()),
            "logs": step_logs # <--- PASS LOGS HERE
        }
        
        return ui_messages, "", trigger_payload, no_update, f"Routing to {space_label}..."

    except Exception as e:
        # If Step 1 fails, we DO update the logs immediately
        current_ui_logs = format_log(current_ui_logs, f"ROUTING ERROR: {e}")
        return ui_messages, "", None, current_ui_logs, "Error"


# ==============================================================================
# ⚙️ STEP 2: EXECUTION (Safe Serialization)
# ==============================================================================
@app.callback(
    [Output("chat-history", "data", allow_duplicate=True),
     Output("chat-window", "children", allow_duplicate=True),
     Output("session-store", "data"),
     Output("active-sessions-list", "children"),
     Output("debug-console", "children", allow_duplicate=True),
     Output("typing-indicator", "children", allow_duplicate=True)],
    [Input("backend-trigger", "modified_timestamp")],
    [State("backend-trigger", "data"),
     State("chat-history", "data"), 
     State("session-store", "data"),
     State("debug-console", "children")],
    prevent_initial_call=True
)
def step_2_execution(ts, trigger_data, history, session_store, current_ui_logs):
    if not ts or not trigger_data: return no_update
    if history is None: history = []
    if session_store is None: session_store = {}
    if not isinstance(current_ui_logs, list): current_ui_logs = []

    # 1. UNPACK LOGS FROM STEP 1
    passed_logs = trigger_data.get("logs", [])
    for log_text in passed_logs:
        current_ui_logs.append(create_log_element(log_text))

    user_text = trigger_data["text"]
    current_ui_logs.append(create_log_element(f"Step 2: Calling Genie API for {trigger_data['space_label']}...", "SYSTEM"))

    # Save User Input to History Store
    history.append({"role": "user", "content": user_text})

    try:
        final_conv_id, result, sql = execute_genie_query(
            user_query=user_text,
            space_id=trigger_data["space_id"],
            current_conv_id=trigger_data["conv_id"], 
            user_token=GENIE_USER_TOKEN,
            host=DATABRICKS_HOST
        )
        
        # ---------------------------------------------------------------------
        # 🛡️ SERIALIZATION SAFETY BLOCK
        # ---------------------------------------------------------------------
        content_to_store = "No content returned."

        if isinstance(result, pd.DataFrame):
            if result.empty:
                content_to_store = "**No data found matching your query.**"
                current_ui_logs.append(create_log_element("⚠️ Query returned 0 rows.", "SYSTEM"))
            else:
                # Convert DF to JSON String
                content_to_store = result.to_json(orient='split', date_format='iso')
                current_ui_logs.append(create_log_element(f"✅ Data received: {len(result)} rows.", "SYSTEM"))
        elif isinstance(result, str):
            content_to_store = result
            current_ui_logs.append(create_log_element("✅ Text response received.", "SYSTEM"))
        else:
            # Fallback for unexpected types (Lists, Dicts, etc)
            content_to_store = str(result)
            current_ui_logs.append(create_log_element(f"⚠️ Unexpected data type: {type(result)}", "SYSTEM"))
        # ---------------------------------------------------------------------

        history.append({
            "role": "assistant",
            "content": content_to_store,
            "space_label": trigger_data["space_label"],
            "sql": sql
        })

        session_store[trigger_data["space_id"]] = {"conv_id": final_conv_id, "last_topic": user_text}

        ui_messages = [render_message(m) for m in history]
        
        active_ui = []
        for sid, data in session_store.items():
            lbl = next((v['label'] for k,v in SPACE_CONFIG.items() if v['id'] == sid), sid[:5])
            active_ui.append(html.Div(f"● {lbl}: ...{data['last_topic'][-15:]}", style={"fontSize":"10px"}))

        return history, ui_messages, session_store, active_ui, current_ui_logs, ""

    except Exception as e:
        current_ui_logs.append(create_log_element(f"BACKEND ERROR: {e}", "ERROR"))
        history.append({"role": "system", "content": f"Error: {str(e)}"})
        return history, [render_message(m) for m in history], session_store, no_update, current_ui_logs, ""

# --- RESET ---
@app.callback(
    [Output("chat-history", "data", allow_duplicate=True),
     Output("chat-window", "children", allow_duplicate=True),
     Output("session-store", "data", allow_duplicate=True),
     Output("active-sessions-list", "children", allow_duplicate=True),
     Output("backend-trigger", "data", allow_duplicate=True)],
    [Input("reset-btn", "n_clicks")],
    prevent_initial_call=True
)
def reset_app(n):
    return [], [], {}, "No active contexts.", None

# --- SCROLL (Delayed) ---
app.clientside_callback(
    """
    function(children) {
        var chat_window = document.getElementById('chat-window');
        if(chat_window) {
            setTimeout(function() { chat_window.scrollTop = chat_window.scrollHeight; }, 100);
        }
        return null;
    }
    """,
    Output("dummy-scroll-target", "children"),
    Input("chat-window", "children")
)

if __name__ == "__main__":
    app.run_server(debug=True, port=8050)
