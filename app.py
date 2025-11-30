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
import concurrent.futures
from dotenv import load_dotenv

# --- IMPORTS FROM YOUR LOCAL FILES ---
# These must exist in the same directory
from route import SPACE_CONFIG, orchestrate_routing
from genie_backend import execute_genie_query
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import ChatMessage, ChatMessageRole

load_dotenv()

# --- CONFIGURATION ---
DATABRICKS_HOST = os.environ.get("DATABRICKS_HOST")
# Fallback to local env var if header is missing (for local testing)
GENIE_USER_TOKEN = os.environ.get("DATABRICKS_TOKEN") 
LLM_ENDPOINT_URL = os.environ.get("SERVING_ENDPOINT_NAME")
TIMEOUT_SECONDS = 180  # 3 Minutes

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.BOOTSTRAP], title="Genie Router")
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
            .chat-container { height: 80vh; display: flex; flex-direction: column; }
            .chat-window { flex-grow: 1; overflow-y: auto; padding: 20px; background: #f8f9fa; border: 1px solid #dee2e6; border-radius: 5px; }
            .debug-terminal { background-color: #1e1e1e; color: #00ff00; font-family: 'Courier New', monospace; font-size: 0.8rem; padding: 15px; height: 150px; overflow-y: auto; border-radius: 5px; white-space: pre-wrap; }
            
            /* Message Bubbles */
            .user-message { background-color: #007bff; color: white; align-self: flex-end; margin-left: auto; border-radius: 10px 10px 0 10px; max-width: 85%; width: fit-content; }
            .bot-message { background-color: #ffffff; color: #333; align-self: flex-start; margin-right: auto; border: 1px solid #e5e7eb; border-radius: 10px 10px 10px 0; box-shadow: 0 1px 2px rgba(0,0,0,0.05); max-width: 90%; width: fit-content; }
            .error-message { background-color: #fee2e2; color: #991b1b; border: 1px solid #f87171; border-radius: 8px; padding: 10px; margin-bottom: 10px; width: fit-content; }
            
            /* SQL Toggle */
            details > summary { cursor: pointer; color: #6c757d; font-size: 0.75rem; margin-top: 8px; outline: none; list-style: none; }
            details > summary::-webkit-details-marker { display: none; }
            details > summary::after { content: " ▼ Show SQL"; }
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
        dbc.Col(html.H3("🧞 Genie Super-Router"), width=8, className="mt-3"),
        dbc.Col(html.Div(id="status-indicator", className="mt-4 text-end text-muted small"), width=4)
    ]),
    html.Hr(),
    
    dbc.Row([
        # Sidebar
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Active Contexts"), 
                dbc.CardBody(id="active-sessions-list", className="p-2 small")
            ], className="mb-3"),
            dbc.Button("🗑️ Reset Chat", id="reset-btn", color="outline-danger", size="sm", className="w-100 mb-2"),
            dbc.Collapse(
                dbc.Card([dbc.CardHeader("🐞 Log"), dbc.CardBody(html.Div(id="debug-console", className="debug-terminal"))]),
                id="debug-collapse", is_open=True
            ),
        ], width=3),

        # Chat Area
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
            ], className="chat-container")
        ], width=9)
    ], className="mb-3"),

    # State Management
    dcc.Store(id="chat-history", data=[]),
    dcc.Store(id="session-store", data={}), 
    dcc.Store(id="backend-trigger", data=None),
    dcc.Download(id="download-dataframe-csv"), 
    html.Div(id="dummy-scroll-target")
], fluid=True, style={"padding": "20px"})


# --- HELPER FUNCTIONS ---
def create_log_element(text, level="INFO"):
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    return f"[{ts}] [{level}] {text}"

def render_message(msg):
    role = msg.get("role")
    msg_id = msg.get("msg_id", str(uuid.uuid4()))
    
    if role == "user":
        return html.Div([
            html.Small("You", className="fw-bold text-light mb-1 d-block"),
            html.Div(msg["content"])
        ], className="p-3 user-message mb-3")
        
    elif role == "error":
        return html.Div([
            html.Strong("⚠️ System Error"),
            html.Div(msg["content"])
        ], className="error-message mb-3")
        
    else: # Assistant/Bot
        children = [html.Small(f"Genie ({msg.get('space_label', 'Bot')})", className="fw-bold text-muted mb-1 d-block")]
        content = msg["content"]
        
        # DataFrame Rendering
        if isinstance(content, str) and content.startswith('{') and "columns" in content:
            try:
                df = pd.read_json(content, orient='split')
                children.append(html.Div([
                    dbc.Button("⬇️ CSV", id={'type': 'download-btn', 'index': msg_id}, size="sm", color="light", className="mb-2", style={"fontSize": "0.7rem"}),
                    dash_table.DataTable(
                        data=df.to_dict('records'), 
                        columns=[{"name": i, "id": i} for i in df.columns],
                        style_table={'overflowX': 'auto'},
                        style_cell={'textAlign': 'left', 'fontFamily': 'sans-serif', 'padding': '5px'},
                        page_size=5
                    ),
                    dbc.Button("✨ AI Insights", id={'type': 'insight-btn', 'index': msg_id}, size="sm", color="success", outline=True, className="mt-2"),
                    html.Div(id={'type': 'insight-output', 'index': msg_id})
                ]))
            except:
                children.append(html.Div(str(content)))
        else:
            children.append(dcc.Markdown(str(content)))
            
        if msg.get("sql"):
            children.append(html.Details([html.Summary("View SQL"), html.Pre(msg["sql"])]))
            
        return html.Div(children, className="p-3 bot-message mb-3")

# --- INSIGHTS GENERATION (Uses Databricks LLM) ---
def generate_data_insights(df: pd.DataFrame) -> str:
    if df.empty: return "No data available."
    try:
        stats = df.describe().to_markdown()
        sample = df.head(3).to_markdown(index=False)
        prompt = f"Analyze this data snippet:\nMETADATA: {list(df.columns)}\nSTATS:\n{stats}\nSAMPLE:\n{sample}\nProvide 3 concise insights."
        
        client = WorkspaceClient()
        response = client.serving_endpoints.query(
            name=LLM_ENDPOINT_URL,
            messages=[ChatMessage(content=prompt, role=ChatMessageRole.USER)],
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Insight generation failed: {str(e)}"

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
    if not user_text: return no_update
    if history is None: history = []
    if not isinstance(logs, list): logs = []
    
    # 1. Update UI immediately with User Message
    msg_id = str(uuid.uuid4())
    history.append({"role": "user", "content": user_text, "msg_id": msg_id})
    ui_messages = [render_message(m) for m in history]
    
    logs.append(create_log_element(f"User: {user_text}"))
    
    # 2. Routing Logic
    try:
        route_decision = orchestrate_routing(user_text, session_store)
        target_space_id = route_decision.get("target_space_id")
        
        # Retrieve existing conversation ID if available
        target_conv_id = None
        if target_space_id in session_store:
            target_conv_id = session_store[target_space_id].get("conv_id")

        # Get readable label
        space_label = "Unknown"
        if SPACE_CONFIG and target_space_id in SPACE_CONFIG:
             space_label = SPACE_CONFIG[target_space_id].get('label', target_space_id)
        
        logs.append(create_log_element(f"Routed to: {space_label} (ID: {target_space_id})"))
        
        # Prepare trigger for Step 2
        trigger_payload = {
            "text": user_text,
            "space_id": target_space_id,
            "conv_id": target_conv_id,
            "space_label": space_label,
            "uuid": msg_id
        }
        
        return ui_messages, "", trigger_payload, "\n".join(logs), f"Genie is thinking in {space_label}..."

    except Exception as e:
        traceback.print_exc()
        logs.append(create_log_element(f"Routing Error: {e}", "ERROR"))
        return ui_messages, "", None, "\n".join(logs), "Routing Error."


# ==============================================================================
# ⚙️ STEP 2: EXECUTION (WITH TIMEOUT)
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
def step_2_execution(trigger_data, history, session_store, logs_str):
    if not trigger_data: return no_update
    
    # Rehydrate logs
    logs = [logs_str] if logs_str else []
    logs.append(create_log_element(f"Executing Query (Timeout: {TIMEOUT_SECONDS}s)..."))

    # Determine Token (Headers for Prod, Env for Local)
    user_token = flask.request.headers.get('X-Forwarded-Access-Token') or GENIE_USER_TOKEN

    # --- WRAPPER FOR TIMEOUT ---
    def run_genie_process():
        return execute_genie_query(
            user_query=trigger_data["text"],
            space_id=trigger_data["space_id"],
            current_conv_id=trigger_data["conv_id"], 
            user_token=user_token,
            host=DATABRICKS_HOST
        )

    try:
        # Execute with ThreadPool to enable Timeout
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(run_genie_process)
            
            try:
                # WAIT FOR RESULT OR TIMEOUT
                final_conv_id, result, sql = future.result(timeout=TIMEOUT_SECONDS)
                
                # --- SUCCESS PATH ---
                logs.append(create_log_element("✅ Backend returned data."))
                
                # Serialize Data
                content_to_store = "No content."
                if isinstance(result, pd.DataFrame):
                    content_to_store = result.to_json(orient='split', date_format='iso') if not result.empty else "**No data found.**"
                else:
                    content_to_store = str(result)

                history.append({
                    "role": "assistant",
                    "content": content_to_store,
                    "space_label": trigger_data["space_label"],
                    "sql": sql,
                    "msg_id": str(uuid.uuid4())
                })

                # Update Session
                session_store[trigger_data["space_id"]] = {"conv_id": final_conv_id, "last_topic": trigger_data["text"]}

            except concurrent.futures.TimeoutError:
                # --- TIMEOUT PATH ---
                logs.append(create_log_element("❌ TIMEOUT REACHED.", "ERROR"))
                history.append({
                    "role": "error",
                    "content": f"⏱️ **Request Timed Out**\n\nThe query took longer than {TIMEOUT_SECONDS} seconds. This usually happens if the Genie space is cold or the query is too complex.\n\nPlease try again in a moment.",
                    "msg_id": str(uuid.uuid4())
                })

        # Render UI
        ui_messages = [render_message(m) for m in history]
        
        # Update Sidebar
        active_ui = []
        for sid, data in session_store.items():
            lbl = SPACE_CONFIG.get(sid, {}).get('label', sid) if SPACE_CONFIG else sid
            active_ui.append(html.Div(f"● {lbl}", style={"fontSize":"12px", "color":"green"}))

        return history, ui_messages, session_store, active_ui, "\n".join(logs), ""

    except Exception as e:
        traceback.print_exc()
        logs.append(create_log_element(f"CRITICAL ERROR: {e}", "ERROR"))
        history.append({"role": "error", "content": f"System Error: {str(e)}"})
        return history, [render_message(m) for m in history], session_store, no_update, "\n".join(logs), ""

# ==============================================================================
# 🧹 UTILITIES (Reset, Download, Insights)
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
    target_id = ctx.triggered_id['index']
    for msg in history:
        if msg.get("msg_id") == target_id:
            try:
                df = pd.read_json(msg["content"], orient='split')
                return dcc.send_data_frame(df.to_csv, "genie_data.csv")
            except: pass
    return no_update

@app.callback(
    Output({'type': 'insight-output', 'index': MATCH}, "children"),
    Input({'type': 'insight-btn', 'index': MATCH}, "n_clicks"),
    State("chat-history", "data"),
    prevent_initial_call=True
)
def insights_action(n, history):
    if not n: return no_update
    ctx = callback_context
    target_id = ctx.triggered_id['index']
    for msg in history:
        if msg.get("msg_id") == target_id:
            try:
                df = pd.read_json(msg["content"], orient='split')
                insights = generate_data_insights(df)
                return html.Div([html.Strong("✨ AI Analysis:"), dcc.Markdown(insights)], className="alert alert-success mt-2 small")
            except Exception as e:
                return html.Div(f"Error: {e}", className="text-danger small")
    return no_update

# Auto-scroll Logic
app.clientside_callback(
    """function(children) { 
        var chat_window = document.getElementById('chat-window'); 
        if(chat_window) { 
            setTimeout(function() { chat_window.scrollTop = chat_window.scrollHeight; }, 100); 
        } 
        return null; 
    }""",
    Output("dummy-scroll-target", "children"),
    Input("chat-window", "children")
)

if __name__ == "__main__":
    app.run_server(debug=True, port=8050)
