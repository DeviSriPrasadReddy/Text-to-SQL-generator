import dash
from dash import html, dcc, Input, Output, State, callback_context, no_update, dash_table
import dash_bootstrap_components as dbc
import pandas as pd
import datetime
import uuid
import os
import json
import traceback
import concurrent.futures
from dotenv import load_dotenv

# --- IMPORTS ---
from routing import SPACE_CONFIG, orchestrate_routing
from genie_backend import execute_genie_query
from databricks.sdk import WorkspaceClient  # <--- NEW IMPORT

load_dotenv()

DATABRICKS_HOST = os.environ.get("DATABRICKS_HOST")
GENIE_USER_TOKEN = os.environ.get("DATABRICKS_TOKEN")
# Define your LLM Endpoint Name here (e.g., "databricks-meta-llama-3-70b-instruct")
LLM_ENDPOINT_NAME = os.environ.get("LLM_ENDPOINT_NAME", "databricks-meta-llama-3-70b-instruct")

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], title="Genie Router")
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
            .chat-container { height: 75vh; }
            .chat-col-wrapper { height: 100%; display: flex; flex-direction: column; }
            .chat-window { flex-grow: 1; overflow-y: auto; padding: 20px; background: #f8f9fa; border: 1px solid #dee2e6; border-radius: 5px; }
            
            /* Debug Terminal */
            .debug-terminal { background-color: #1e1e1e; color: #00ff00; font-family: 'Courier New', monospace; font-size: 0.8rem; padding: 15px; height: 200px; overflow-y: auto; border-radius: 5px; white-space: pre-wrap; }
            
            /* SQL Toggle Styling */
            details > summary { cursor: pointer; color: #007bff; font-size: 0.8rem; margin-top: 8px; outline: none; list-style: none; }
            details > summary::-webkit-details-marker { display: none; }
            details > summary::after { content: " ▼ Show Generated SQL"; }
            details[open] > summary::after { content: " ▲ Hide SQL"; }
            details > pre { background: #2d2d2d; color: #f8f8f2; padding: 10px; border-radius: 5px; margin-top: 5px; font-size: 0.75rem; overflow-x: auto; white-space: pre-wrap; }
            
            /* Insight Button */
            .insight-btn-wrapper { display: flex; justify-content: flex-end; margin-top: 5px; }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>{%config%}{%scripts%}{%renderer%}</footer>
    </body>
</html>
'''

# --- TIMEOUT HELPER ---
def run_with_timeout(func, args=(), kwargs=None, timeout_seconds=300):
    """Executes a function with a strict time limit."""
    if kwargs is None: kwargs = {}
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(func, *args, **kwargs)
        try:
            return future.result(timeout=timeout_seconds)
        except concurrent.futures.TimeoutError:
            raise TimeoutError(f"Process exceeded the {timeout_seconds/60} minute time limit.")

# --- LLM INSIGHTS HELPER ---
def generate_llm_insights(df_json):
    """Sends DF to Databricks Model Serving for analysis."""
    try:
        w = WorkspaceClient(host=DATABRICKS_HOST, token=GENIE_USER_TOKEN)
        
        # Parse and truncate DF to avoid token limits (top 50 rows)
        df = pd.read_json(df_json, orient='split')
        preview_csv = df.head(50).to_csv(index=False)
        
        prompt = f"""
        Analyze the following dataset (showing top 50 rows):
        
        {preview_csv}
        
        Provide 3-5 concise, high-value business insights based on this data. 
        Format the output as a bulleted list.
        """
        
        response = w.serving_endpoints.query(
            name=LLM_ENDPOINT_NAME,
            messages=[
                {"role": "system", "content": "You are a data analyst helper."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Failed to generate insights: {str(e)}"

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
                        dbc.Col(dbc.Input(id="user-input", placeholder="Ask a question...", autocomplete="off"), width=8),
                        dbc.Col(dbc.Button("Send", id="send-btn", color="primary", className="w-100"), width=2),
                        # NEW BUTTON
                        dbc.Col(dbc.Button("✨ Insights", id="insight-btn", color="warning", outline=True, className="w-100", title="Analyze last result with LLM"), width=2)
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

# --- LOGGING HELPER ---
def format_log(current_logs, new_entry):
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    if not isinstance(current_logs, list): current_logs = []
    current_logs.append(html.Div(f"[{ts}] {new_entry}"))
    return current_logs

# --- MESSAGE RENDERER ---
def render_message(msg):
    is_user = msg["role"] == "user"
    css_class = "user-message" if is_user else "bot-message"
    align = "right" if is_user else "left"
    bg = "#007bff" if is_user else "#e9ecef"
    color = "white" if is_user else "black"
    
    children = [html.Small(msg.get('space_label', ''), style={"display":"block", "marginBottom":"5px", "color":"#ccc" if is_user else "#666"})]
    content = msg["content"]
    
    # Check for DataFrame JSON
    if isinstance(content, str) and content.startswith('{') and "columns" in content:
        try:
            df = pd.read_json(content, orient='split')
            children.append(dash_table.DataTable(
                data=df.to_dict('records'), 
                columns=[{"name": i, "id": i} for i in df.columns], 
                style_table={'overflowX': 'auto'}, 
                style_cell={'textAlign': 'left', 'color': 'black'},
                page_size=10
            ))
        except: children.append(html.Div(str(content)))
    else:
        children.append(dcc.Markdown(str(content)))
    
    # SQL Toggle
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
def step_1_routing(n_c, n_s, user_text, history, session_store, logs):
    if not user_text: return no_update
    if history is None: history = []
    
    # Update UI immediately
    temp_history = history + [{"role": "user", "content": user_text}]
    ui_messages = [render_message(m) for m in temp_history]
    logs = format_log(logs, f"Step 1: Routing '{user_text}'")
    
    try:
        # Run Router
        route_decision = orchestrate_routing(user_text, session_store)
        target_space_id = route_decision.get("target_space_id")
        
        target_conv_id = None
        if target_space_id in session_store:
            existing_id = session_store[target_space_id].get("conv_id")
            if existing_id:
                target_conv_id = existing_id
                logs = format_log(logs, f"♻️ Reusing ID: {existing_id}")
            else:
                logs = format_log(logs, "ℹ️ New ID needed.")
        else:
            logs = format_log(logs, "🆕 New Space.")

        space_label = next((v['label'] for k,v in SPACE_CONFIG.items() if v['id'] == target_space_id), "Unknown")
        
        trigger_payload = {
            "text": user_text,
            "space_id": target_space_id,
            "conv_id": target_conv_id,
            "space_label": space_label,
            "uuid": str(uuid.uuid4())
        }
        
        return ui_messages, "", trigger_payload, logs, f"Routing to {space_label}..."

    except Exception as e:
        logs = format_log(logs, f"ROUTING ERROR: {e}")
        return ui_messages, "", None, logs, "Error"


# ==============================================================================
# ⚙️ STEP 2: EXECUTION (Genie Query with Timeout)
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
def step_2_execution(ts, trigger_data, history, session_store, logs):
    if not ts or not trigger_data: return no_update
    if history is None: history = []
    if session_store is None: session_store = {}

    user_text = trigger_data["text"]
    logs = format_log(logs, f"Step 2: Executing in {trigger_data['space_id']}...")

    history.append({"role": "user", "content": user_text})

    try:
        # --- WRAPPED EXECUTION WITH TIMEOUT ---
        # We wrap the external call in our timeout helper
        def run_genie():
            return execute_genie_query(
                user_query=user_text,
                space_id=trigger_data["space_id"],
                current_conv_id=trigger_data["conv_id"], 
                user_token=GENIE_USER_TOKEN,
                host=DATABRICKS_HOST
            )

        final_conv_id, result, sql = run_with_timeout(run_genie, timeout_seconds=300)
        # --------------------------------------
        
        logs = format_log(logs, f"Genie Success. ID: {final_conv_id}")

        content = result
        if isinstance(result, pd.DataFrame):
            content = result.to_json(orient='split')

        history.append({
            "role": "assistant",
            "content": content,
            "space_label": trigger_data["space_label"],
            "sql": sql
        })

        session_store[trigger_data["space_id"]] = {"conv_id": final_conv_id, "last_topic": user_text}

        ui_messages = [render_message(m) for m in history]
        
        active_ui = []
        for sid, data in session_store.items():
            lbl = next((v['label'] for k,v in SPACE_CONFIG.items() if v['id'] == sid), sid[:5])
            active_ui.append(html.Div(f"● {lbl}: ...{data['last_topic'][-15:]}", style={"fontSize":"10px"}))

        return history, ui_messages, session_store, active_ui, logs, ""

    except TimeoutError:
        logs = format_log(logs, "❌ ERROR: Operation timed out (5 min limit).")
        history.append({"role": "system", "content": "Error: The request timed out after 5 minutes."})
        return history, [render_message(m) for m in history], session_store, no_update, logs, ""
        
    except Exception as e:
        logs = format_log(logs, f"BACKEND ERROR: {e}")
        history.append({"role": "system", "content": f"Error: {str(e)}"})
        return history, [render_message(m) for m in history], session_store, no_update, logs, ""


# ==============================================================================
# ✨ STEP 3: INSIGHT GENERATION (LLM via SDK)
# ==============================================================================
@app.callback(
    [Output("chat-history", "data", allow_duplicate=True),
     Output("chat-window", "children", allow_duplicate=True),
     Output("debug-console", "children", allow_duplicate=True),
     Output("typing-indicator", "children", allow_duplicate=True)],
    [Input("insight-btn", "n_clicks")],
    [State("chat-history", "data"),
     State("debug-console", "children")],
    prevent_initial_call=True
)
def step_3_generate_insights(n_clicks, history, logs):
    if not n_clicks or not history: return no_update
    
    logs = format_log(logs, "✨ Insight generation triggered...")
    
    # 1. Find the last data response
    last_assistant_msg = None
    for msg in reversed(history):
        if msg["role"] == "assistant":
            last_assistant_msg = msg
            break
            
    if not last_assistant_msg:
        logs = format_log(logs, "⚠️ No data found to analyze.")
        return no_update, no_update, logs, ""

    content = last_assistant_msg.get("content")
    
    # 2. Check if it is a DataFrame JSON
    if not (isinstance(content, str) and content.startswith('{') and "columns" in content):
        logs = format_log(logs, "⚠️ Last message was not a dataset.")
        history.append({"role": "system", "content": "⚠️ Can only generate insights on Data Tables."})
        ui_messages = [render_message(m) for m in history]
        return history, ui_messages, logs, ""

    # 3. Call LLM with Timeout
    try:
        logs = format_log(logs, "Sending data to Databricks LLM...")
        
        # We wrap this in timeout as well
        insights = run_with_timeout(generate_llm_insights, args=(content,), timeout_seconds=300)
        
        history.append({
            "role": "assistant", 
            "content": f"**✨ AI Insights:**\n\n{insights}",
            "space_label": "LLM Analysis"
        })
        
        ui_messages = [render_message(m) for m in history]
        logs = format_log(logs, "Insights generated successfully.")
        
        return history, ui_messages, logs, ""
        
    except Exception as e:
        logs = format_log(logs, f"LLM ERROR: {e}")
        history.append({"role": "system", "content": f"Error generating insights: {str(e)}"})
        return history, [render_message(m) for m in history], logs, ""


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
