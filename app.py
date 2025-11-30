import dash
from dash import html, dcc, Input, Output, State, MATCH, callback_context, no_update, dash_table
import dash_bootstrap_components as dbc
import pandas as pd
import datetime
import uuid
import os
import traceback
import json
import requests
from dotenv import load_dotenv
import flask

# --- DATABRICKS IMPORTS ---
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import ChatMessage, ChatMessageRole

# --- LOCAL MODULE IMPORTS ---
# Ensure 'route.py' and 'genie_backend.py' are in the same folder
try:
    from route import SPACE_CONFIG, orchestrate_routing
    from genie_backend import execute_genie_query
except ImportError as e:
    print(f"CRITICAL WARNING: Could not import local modules: {e}")
    # Mocking for standalone testing if files are missing
    SPACE_CONFIG = {"1": {"id": "1", "label": "Finance"}}
    def orchestrate_routing(text, store): return {"target_space_id": "1"}
    def execute_genie_query(**kwargs): return "123", pd.DataFrame({"col": ["mock"]}), "SELECT * FROM mock"

load_dotenv()

# --- CONFIGURATION ---
DATABRICKS_HOST = os.environ.get("DATABRICKS_HOST")
# Fallback token for local testing if headers aren't present
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
            
            /* Insights Box */
            .insight-box { background: #f0fdf4; border: 1px solid #bbf7d0; padding: 15px; border-radius: 8px; margin-top: 10px; font-size: 0.9rem; color: #166534; }
            
            /* Message Styles */
            .user-message { background-color: #007bff; color: white; align-self: flex-end; margin-left: auto; border-radius: 10px 10px 0 10px; }
            .bot-message { background-color: #ffffff; color: #333; align-self: flex-start; margin-right: auto; border: 1px solid #e5e7eb; border-radius: 10px 10px 10px 0; box-shadow: 0 1px 2px rgba(0,0,0,0.05); }
            
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
        dbc.Col(html.Div(id="status-indicator", className="mt-4"), width=4)
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
                        dbc.Col(dbc.Input(id="user-input", placeholder="Ask about Finance or HR...", autocomplete="off"), width=10),
                        dbc.Col(dbc.Button("Send", id="send-btn", color="primary", className="w-100"), width=2)
                    ], className="mt-3"),
                    html.Div(id="typing-indicator", className="text-muted small mt-1")
                ])
            ], className="chat-col-wrapper")
        ], width=9, style={"height": "100%"})
    ], className="chat-container mb-3"),

    dbc.Collapse(
        dbc.Card([dbc.CardHeader("🐞 Logic Log"), dbc.CardBody(html.Div(id="debug-console", className="debug-terminal"))]),
        id="debug-collapse", is_open=True
    ),

    dcc.Store(id="chat-history", data=[]),
    dcc.Store(id="session-store", data={}), 
    dcc.Store(id="backend-trigger", data=None),
    dcc.Download(id="download-dataframe-csv"), 
    html.Div(id="dummy-scroll-target")
], fluid=True, style={"padding": "20px"})


# --- LOGGING HELPERS ---
def create_log_element(text, level="INFO"):
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    css = "text-danger" if level == "ERROR" else "text-info" if level == "SYSTEM" else ""
    return html.Div([html.Span(f"[{ts}] ", className="text-muted mr-2"), html.Span(f"[{level}] ", className=css), html.Span(str(text))])

def format_log(current_logs, new_entry):
    if not isinstance(current_logs, list): current_logs = []
    current_logs.append(create_log_element(new_entry))
    return current_logs

# --- INSIGHTS LOGIC ---
def generate_data_insights(df: pd.DataFrame) -> str:
    """Sends dataframe stats to Databricks LLM for summary."""
    if df.empty: return "No data available."

    # 1. Prepare Prompt
    stats = df.describe().to_markdown() if not df.empty else "No numeric stats"
    sample = df.head(5).to_markdown(index=False)
    
    prompt = f"""
    You are a Data Analyst. Analyze this dataset and provide 3 key insights.
    METADATA: Rows: {len(df)}, Cols: {list(df.columns)}
    SAMPLE:
    {sample}
    STATS:
    {stats}
    Provide insights in a concise bulleted list.
    """

    # 2. Call LLM
    try:
        client = WorkspaceClient()
        endpoint_name = os.environ.get("SERVING_ENDPOINT_NAME")
        if not endpoint_name:
            return "Error: SERVING_ENDPOINT_NAME env var is missing."
            
        # FIX: Corrected Syntax for ChatMessage and object access
        response = client.serving_endpoints.query(
            name=endpoint_name,
            messages=[ChatMessage(content=prompt, role=ChatMessageRole.USER)],
        )
        # FIX: Corrected response access syntax (dot notation)
        return response.choices[0].message.content
    except Exception as e:
        traceback.print_exc()
        return f"Error generating insights: {str(e)}"

# --- RENDER HELPER ---
def render_message(msg):
    is_user = msg["role"] == "user"
    msg_id = msg.get("msg_id", str(uuid.uuid4()))
    
    css_class = "user-message" if is_user else "bot-message"
    align = "right" if is_user else "left"
    
    children = [html.Small(msg.get('space_label', ''), style={"display":"block", "marginBottom":"5px", "color":"#ddd" if is_user else "#666", "fontWeight":"bold"})]
    content = msg["content"]
    
    # 1. TABLE RENDERER
    if isinstance(content, str) and content.startswith('{') and "columns" in content:
        try:
            df = pd.read_json(content, orient='split')
            
            # A. Export Button
            children.append(html.Div([
                dbc.Button("⬇️ CSV", id={'type': 'download-btn', 'index': msg_id}, size="sm", color="light", className="mb-2", style={"fontSize": "0.7rem"})
            ], style={"textAlign": "right"}))
            
            # B. Table
            children.append(dash_table.DataTable(
                data=df.to_dict('records'), 
                columns=[{"name": i, "id": i} for i in df.columns], 
                style_table={'overflowX': 'auto'}, 
                style_cell={'textAlign': 'left', 'color': 'black', 'fontFamily': 'sans-serif', 'padding': '5px'},
                page_size=5
            ))
            
            # C. Insights Button
            children.append(html.Div([
                dbc.Button("✨ Generate Insights", id={'type': 'insight-btn', 'index': msg_id}, size="sm", color="success", outline=True, className="mt-2"),
                html.Div(id={'type': 'insight-output', 'index': msg_id}) 
            ]))
        except: 
            children.append(html.Div(str(content)))
    else:
        children.append(dcc.Markdown(str(content)))
    
    # 2. SQL Toggle
    if not is_user and msg.get("sql"):
        children.append(html.Details([html.Summary(""), html.Pre(msg["sql"])]))
        
    return html.Div(children, className=f"p-3 {css_class}", style={"textAlign": align, "marginBottom": "15px", "maxWidth": "85%", "width": "fit-content", "marginLeft": "auto" if is_user else "0"})


# ==============================================================================
# 📥 CALLBACK: EXPORT CSV
# ==============================================================================
@app.callback(
    Output("download-dataframe-csv", "data"),
    Input({'type': 'download-btn', 'index': MATCH}, "n_clicks"),
    State("chat-history", "data"),
    prevent_initial_call=True
)
def download_csv(n_clicks, history):
    if not n_clicks: return no_update
    ctx = callback_context
    target_id = ctx.triggered_id['index']
    
    for msg in history:
        if msg.get("msg_id") == target_id:
            try:
                df = pd.read_json(msg["content"], orient='split')
                return dcc.send_data_frame(df.to_csv, f"genie_export_{target_id[:5]}.csv")
            except: return no_update
    return no_update

# ==============================================================================
# ✨ CALLBACK: GENERATE INSIGHTS
# ==============================================================================
@app.callback(
    Output({'type': 'insight-output', 'index': MATCH}, "children"),
    Input({'type': 'insight-btn', 'index': MATCH}, "n_clicks"),
    State("chat-history", "data"),
    prevent_initial_call=True
)
def generate_insights_action(n_clicks, history):
    if not n_clicks: return no_update
    ctx = callback_context
    target_id = ctx.triggered_id['index']
    
    for msg in history:
        if msg.get("msg_id") == target_id:
            try:
                df = pd.read_json(msg["content"], orient='split')
                insights = generate_data_insights(df)
                return html.Div([html.Strong("AI Analysis:"), dcc.Markdown(insights)], className="insight-box")
            except Exception as e:
                return html.Div(f"Error: {e}", className="text-danger small")
    return html.Div("Context lost.", className="text-danger small")


# ==============================================================================
# 🔄 STEP 1: ROUTING 
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
    # Print to server console to confirm event firing
    print(f"[DEBUG] Step 1 fired. Text: {user_text}")

    if not user_text: 
        return no_update
    
    if history is None: history = []
    
    try:
        # Generate ID for User Message
        temp_history = history + [{"role": "user", "content": user_text, "msg_id": str(uuid.uuid4())}]
        ui_messages = [render_message(m) for m in temp_history]
        
        logs = format_log(logs, f"Step 1: Routing '{user_text}'")
        
        # Call Routing Logic
        route_decision = orchestrate_routing(user_text, session_store)
        target_space_id = route_decision.get("target_space_id")
        
        target_conv_id = None
        if target_space_id in session_store:
            target_conv_id = session_store[target_space_id].get("conv_id")
            if target_conv_id:
                logs = format_log(logs, f"♻️ Reusing Context ID: {target_conv_id}")

        # Label lookup
        space_label = "Unknown"
        if SPACE_CONFIG and target_space_id in SPACE_CONFIG:
             space_label = SPACE_CONFIG[target_space_id].get('label', target_space_id)
        
        trigger_payload = {
            "text": user_text,
            "space_id": target_space_id,
            "conv_id": target_conv_id,
            "space_label": space_label,
            "uuid": str(uuid.uuid4()),
            "logs": [f"Routing to {space_label}..."] # Pass simplistic logs to avoid serializing complex HTML
        }
        
        return ui_messages, "", trigger_payload, logs, f"Routing to {space_label}..."

    except Exception as e:
        # Catch errors so the app doesn't freeze
        err_msg = traceback.format_exc()
        print(err_msg)
        logs = format_log(logs, f"ROUTING ERROR: {e}")
        return no_update, no_update, None, logs, "Error in routing."


# ==============================================================================
# ⚙️ STEP 2: EXECUTION 
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
    
    # Re-hydrate logs
    if not isinstance(current_ui_logs, list): current_ui_logs = []
    
    user_text = trigger_data["text"]
    current_ui_logs.append(create_log_element(f"Step 2: Calling Genie ({trigger_data['space_label']})...", "SYSTEM"))

    # Update history with User Message (Source of Truth)
    history.append({"role": "user", "content": user_text, "msg_id": str(uuid.uuid4())})
    
    # Token Logic: Check Flask headers (Prod) or Env Var (Local)
    user_token = flask.request.headers.get('X-Forwarded-Access-Token') or GENIE_USER_TOKEN
    
    try:
        final_conv_id, result, sql = execute_genie_query(
            user_query=user_text,
            space_id=trigger_data["space_id"],
            current_conv_id=trigger_data["conv_id"], 
            user_token=user_token,
            host=DATABRICKS_HOST
        )
        
        # Serialize Result
        content_to_store = "No content."
        if isinstance(result, pd.DataFrame):
            content_to_store = result.to_json(orient='split', date_format='iso') if not result.empty else "**No data found.**"
            current_ui_logs.append(create_log_element(f"✅ Data received: {len(result) if not result.empty else 0} rows.", "SYSTEM"))
        else:
            content_to_store = str(result)
            current_ui_logs.append(create_log_element("✅ Text response.", "SYSTEM"))

        history.append({
            "role": "assistant",
            "content": content_to_store,
            "space_label": trigger_data["space_label"],
            "sql": sql,
            "msg_id": str(uuid.uuid4())
        })

        session_store[trigger_data["space_id"]] = {"conv_id": final_conv_id, "last_topic": user_text}

        ui_messages = [render_message(m) for m in history]
        
        active_ui = []
        for sid, data in session_store.items():
            # Safe label lookup
            lbl = sid
            if SPACE_CONFIG and sid in SPACE_CONFIG:
                lbl = SPACE_CONFIG[sid].get('label', sid)
            
            short_topic = data['last_topic'][:20] + "..." if len(data['last_topic']) > 20 else data['last_topic']
            active_ui.append(html.Div(f"● {lbl}: {short_topic}", style={"fontSize":"12px", "padding":"2px"}))

        return history, ui_messages, session_store, active_ui, current_ui_logs, ""

    except Exception as e:
        print(traceback.format_exc())
        current_ui_logs.append(create_log_element(f"BACKEND ERROR: {e}", "ERROR"))
        
        history.append({"role": "system", "content": f"Error: {str(e)}", "msg_id": str(uuid.uuid4())})
        ui_messages = [render_message(m) for m in history]
        
        return history, ui_messages, session_store, no_update, current_ui_logs, ""

# --- RESET & SCROLL ---
@app.callback([Output("chat-history", "data", allow_duplicate=True), Output("chat-window", "children", allow_duplicate=True), Output("session-store", "data", allow_duplicate=True), Output("backend-trigger", "data", allow_duplicate=True)], [Input("reset-btn", "n_clicks")], prevent_initial_call=True)
def reset_app(n): return [], [], {}, None

app.clientside_callback("""function(children) { var chat_window = document.getElementById('chat-window'); if(chat_window) { setTimeout(function() { chat_window.scrollTop = chat_window.scrollHeight; }, 100); } return null; }""", Output("dummy-scroll-target", "children"), Input("chat-window", "children"))

if __name__ == "__main__":
    app.run_server(debug=True, port=8050)
