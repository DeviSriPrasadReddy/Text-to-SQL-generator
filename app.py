import dash
from dash import html, dcc, Input, Output, State, MATCH, callback_context, no_update, dash_table
import dash_bootstrap_components as dbc
import pandas as pd
import datetime
import uuid
import os
import traceback
import json
import flask

# --- DATABRICKS IMPORTS ---
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import ChatMessage, ChatMessageRole

# --- LOCAL MODULE IMPORTS ---
# We wrap this in try/except to prevent the app from crashing if these files are missing
try:
    from route import SPACE_CONFIG, orchestrate_routing
    from genie_backend import execute_genie_query
except ImportError as e:
    print(f"⚠️ WARNING: Local modules not found ({e}). Using MOCK mode.")
    SPACE_CONFIG = {"1": {"id": "1", "label": "Finance"}, "2": {"id": "2", "label": "HR"}}
    def orchestrate_routing(text, store): return {"target_space_id": "1"}
    def execute_genie_query(**kwargs): return "123", pd.DataFrame({"Mock Column": ["Data A", "Data B"]}), "SELECT * FROM mock_table"

# --- CONFIGURATION ---
DATABRICKS_HOST = os.environ.get("DATABRICKS_HOST")
GENIE_USER_TOKEN = os.environ.get("DATABRICKS_TOKEN")
LLM_ENDPOINT_URL = os.environ.get("SERVING_ENDPOINT_NAME")

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.BOOTSTRAP], title="Genie Router")
server = app.server

# --- CSS ---
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            .chat-container { height: 70vh; }
            .chat-col-wrapper { height: 100%; display: flex; flex-direction: column; }
            .chat-window { flex-grow: 1; overflow-y: auto; padding: 20px; background: #f8f9fa; border: 1px solid #dee2e6; border-radius: 5px; }
            .debug-terminal { background-color: #1e1e1e; color: #00ff00; font-family: 'Courier New', monospace; font-size: 0.8rem; padding: 15px; height: 200px; overflow-y: auto; border-radius: 5px; white-space: pre-wrap; }
            .user-message { background-color: #007bff; color: white; align-self: flex-end; margin-left: auto; border-radius: 10px 10px 0 10px; }
            .bot-message { background-color: #ffffff; color: #333; align-self: flex-start; margin-right: auto; border: 1px solid #e5e7eb; border-radius: 10px 10px 10px 0; box-shadow: 0 1px 2px rgba(0,0,0,0.05); }
            details > summary { cursor: pointer; color: #6c757d; font-size: 0.75rem; margin-top: 8px; outline: none; }
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
        dbc.Col(html.H3("🧞 Genie Super-Router"), width=8, className="mt-3"),
        dbc.Col(html.Div(id="status-indicator", className="mt-4 text-muted"), width=4)
    ]),
    html.Hr(),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([dbc.CardHeader("Active Contexts"), dbc.CardBody(id="active-sessions-list")], className="mb-3"),
            dbc.Button("Reset Chat", id="reset-btn", color="outline-danger", size="sm", className="w-100")
        ], width=3, style={"height": "100%"}),

        dbc.Col([
            html.Div([
                html.Div(id="chat-window", className="chat-window"),
                html.Div([
                    dbc.Row([
                        dbc.Col(dbc.Input(id="user-input", placeholder="Ask a question...", n_submit=0, autocomplete="off"), width=10),
                        dbc.Col(dbc.Button("Send", id="send-btn", color="primary", className="w-100", n_clicks=0), width=2)
                    ], className="mt-3"),
                    html.Div(id="typing-indicator", className="text-muted small mt-1", style={"minHeight": "20px"})
                ])
            ], className="chat-col-wrapper")
        ], width=9, style={"height": "100%"})
    ], className="chat-container mb-3"),

    dbc.Collapse(
        dbc.Card([dbc.CardHeader("🐞 Logic Log"), dbc.CardBody(html.Div(id="debug-console", className="debug-terminal"))]),
        id="debug-collapse", is_open=True
    ),

    # Storage
    dcc.Store(id="chat-history", data=[]),
    dcc.Store(id="session-store", data={}), 
    dcc.Store(id="backend-trigger", data=None),
    dcc.Download(id="download-dataframe-csv"), 
    html.Div(id="dummy-scroll-target")
], fluid=True, style={"padding": "20px"})


# --- HELPER FUNCTIONS ---
def create_log_element(text, level="INFO"):
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    css = "text-danger" if level == "ERROR" else "text-info" if level == "SYSTEM" else ""
    return html.Div([html.Span(f"[{ts}] ", className="text-muted mr-2"), html.Span(f"[{level}] ", className=css), html.Span(str(text))])

def render_message(msg):
    is_user = msg["role"] == "user"
    msg_id = msg.get("msg_id", str(uuid.uuid4()))
    css_class = "user-message" if is_user else "bot-message"
    align = "right" if is_user else "left"
    
    children = [html.Small(msg.get('space_label', ''), style={"display":"block", "marginBottom":"5px", "color":"#ddd" if is_user else "#666", "fontWeight":"bold"})]
    content = msg["content"]
    
    if isinstance(content, str) and content.startswith('{') and "columns" in content:
        try:
            df = pd.read_json(content, orient='split')
            children.append(html.Div([dbc.Button("⬇️ CSV", id={'type': 'download-btn', 'index': msg_id}, size="sm", color="light", className="mb-2", style={"fontSize": "0.7rem"})], style={"textAlign": "right"}))
            children.append(dash_table.DataTable(data=df.to_dict('records'), columns=[{"name": i, "id": i} for i in df.columns], style_table={'overflowX': 'auto'}, page_size=5))
            children.append(html.Div([dbc.Button("✨ Insights", id={'type': 'insight-btn', 'index': msg_id}, size="sm", color="success", outline=True, className="mt-2"), html.Div(id={'type': 'insight-output', 'index': msg_id})]))
        except: children.append(html.Div(str(content)))
    else:
        children.append(dcc.Markdown(str(content)))
    
    if not is_user and msg.get("sql"):
        children.append(html.Details([html.Summary("View SQL"), html.Pre(msg["sql"])]))
        
    return html.Div(children, className=f"p-3 {css_class}", style={"textAlign": align, "marginBottom": "15px", "maxWidth": "85%", "width": "fit-content", "marginLeft": "auto" if is_user else "0"})

# ==============================================================================
# 🔄 STEP 1: ROUTING (UI TRIGGER)
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
def step_1_routing(n_c, n_s, user_text, history, session_store, logs):
    print(f"DEBUG: Step 1 triggered. Text: {user_text}") # CHECK YOUR TERMINAL FOR THIS
    
    if not user_text: return no_update
    if history is None: history = []
    if not isinstance(logs, list): logs = []

    # 1. Update UI immediately
    new_msg_id = str(uuid.uuid4())
    temp_history = history + [{"role": "user", "content": user_text, "msg_id": new_msg_id}]
    ui_messages = [render_message(m) for m in temp_history]
    
    logs.append(create_log_element(f"Step 1: Routing '{user_text}'..."))

    # 2. Logic
    try:
        route_decision = orchestrate_routing(user_text, session_store)
        target_space_id = route_decision.get("target_space_id")
        target_conv_id = session_store.get(target_space_id, {}).get("conv_id")
        
        space_label = SPACE_CONFIG.get(target_space_id, {}).get('label', target_space_id)
        
        trigger_payload = {
            "text": user_text,
            "space_id": target_space_id,
            "conv_id": target_conv_id,
            "space_label": space_label,
            "uuid": str(uuid.uuid4()),
            "log_msg": f"Routing to {space_label}..." 
        }
        
        return ui_messages, "", trigger_payload, logs, f"Thinking in {space_label}..."
    except Exception as e:
        traceback.print_exc()
        logs.append(create_log_element(f"ROUTING ERROR: {e}", "ERROR"))
        return ui_messages, "", None, logs, "Error."

# ==============================================================================
# ⚙️ STEP 2: EXECUTION (BACKEND TRIGGER)
# ==============================================================================
@app.callback(
    [Output("chat-history", "data", allow_duplicate=True),
     Output("chat-window", "children", allow_duplicate=True),
     Output("session-store", "data", allow_duplicate=True),
     Output("active-sessions-list", "children"),
     Output("debug-console", "children", allow_duplicate=True),
     Output("typing-indicator", "children", allow_duplicate=True)],
    [Input("backend-trigger", "data")],
    [State("chat-history", "data"), 
     State("session-store", "data"),
     State("debug-console", "children")],
    prevent_initial_call=True
)
def step_2_execution(trigger_data, history, session_store, logs):
    if not trigger_data: return no_update
    print(f"DEBUG: Step 2 triggered for {trigger_data.get('space_label')}")
    
    if history is None: history = []
    if session_store is None: session_store = {}
    if not isinstance(logs, list): logs = []

    user_text = trigger_data["text"]
    logs.append(create_log_element(trigger_data.get("log_msg", "Processing..."), "SYSTEM"))

    # Add User Msg to History (Source of Truth)
    history.append({"role": "user", "content": user_text, "msg_id": str(uuid.uuid4())})
    
    # Exec Backend
    try:
        user_token = flask.request.headers.get('X-Forwarded-Access-Token') or GENIE_USER_TOKEN
        
        final_conv_id, result, sql = execute_genie_query(
            user_query=user_text,
            space_id=trigger_data["space_id"],
            current_conv_id=trigger_data["conv_id"], 
            user_token=user_token,
            host=DATABRICKS_HOST
        )

        # Process Result
        content = result.to_json(orient='split') if isinstance(result, pd.DataFrame) else str(result)
        logs.append(create_log_element("✅ Response received.", "SYSTEM"))

        history.append({
            "role": "assistant",
            "content": content,
            "space_label": trigger_data["space_label"],
            "sql": sql,
            "msg_id": str(uuid.uuid4())
        })

        # Update Session
        session_store[trigger_data["space_id"]] = {"conv_id": final_conv_id, "last_topic": user_text}

        # Render
        ui_messages = [render_message(m) for m in history]
        active_ui = [html.Div(f"● {k}: Active", style={"fontSize":"12px"}) for k in session_store.keys()]

        return history, ui_messages, session_store, active_ui, logs, ""

    except Exception as e:
        traceback.print_exc()
        logs.append(create_log_element(f"BACKEND ERROR: {e}", "ERROR"))
        history.append({"role": "system", "content": f"Error: {e}"})
        return history, [render_message(m) for m in history], session_store, no_update, logs, ""

# ==============================================================================
# 🧹 RESET & CSV & INSIGHTS
# ==============================================================================
@app.callback(
    [Output("chat-history", "data", allow_duplicate=True),
     Output("chat-window", "children", allow_duplicate=True),
     Output("session-store", "data", allow_duplicate=True),
     Output("backend-trigger", "data", allow_duplicate=True)],
    [Input("reset-btn", "n_clicks")],
    prevent_initial_call=True
)
def reset_app(n):
    if not n: return no_update
    return [], [], {}, None

@app.callback(
    Output("download-dataframe-csv", "data"),
    Input({'type': 'download-btn', 'index': MATCH}, "n_clicks"),
    State("chat-history", "data"),
    prevent_initial_call=True
)
def download_csv(n, history):
    if not n: return no_update
    ctx = callback_context
    target = ctx.triggered_id['index']
    for m in history:
        if m.get("msg_id") == target:
            return dcc.send_data_frame(pd.read_json(m["content"], orient='split').to_csv, "data.csv")
    return no_update

@app.callback(
    Output({'type': 'insight-output', 'index': MATCH}, "children"),
    Input({'type': 'insight-btn', 'index': MATCH}, "n_clicks"),
    State("chat-history", "data"),
    prevent_initial_call=True
)
def insight_action(n, history):
    if not n: return no_update
    return html.Div("Insight generation triggered (add logic here).", className="text-success small")

# Auto-scroll
app.clientside_callback(
    """function(children) { var w = document.getElementById('chat-window'); if(w){ setTimeout(function(){ w.scrollTop = w.scrollHeight; }, 100); } return null; }""",
    Output("dummy-scroll-target", "children"),
    Input("chat-window", "children")
)

if __name__ == "__main__":
    app.run_server(debug=True, port=8050)
