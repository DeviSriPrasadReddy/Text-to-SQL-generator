import dash
from dash import html, dcc, Input, Output, State, ALL, MATCH, callback_context, no_update, dash_table
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
# Ensure these files exist in your directory
from routing import SPACE_CONFIG, orchestrate_routing
from genie_backend import execute_genie_query
from databricks.sdk import WorkspaceClient

load_dotenv()

# --- CONFIGURATION ---
DATABRICKS_HOST = os.environ.get("DATABRICKS_HOST")
GENIE_USER_TOKEN = os.environ.get("DATABRICKS_TOKEN")
LLM_ENDPOINT_NAME = os.environ.get("LLM_ENDPOINT_NAME", "databricks-meta-llama-3-70b-instruct")

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
            .chat-container { height: 85vh; }
            .chat-col-wrapper { height: 100%; display: flex; flex-direction: column; }
            .chat-window { flex-grow: 1; overflow-y: auto; padding: 20px; background: #f8f9fa; border: 1px solid #dee2e6; border-radius: 5px; }
            
            /* Debug Terminal */
            .debug-terminal { background-color: #1e1e1e; color: #00ff00; font-family: 'Courier New', monospace; font-size: 0.8rem; padding: 15px; height: 150px; overflow-y: auto; border-radius: 5px; white-space: pre-wrap; }
            
            /* SQL Toggle Styling */
            details > summary { cursor: pointer; color: #007bff; font-size: 0.8rem; margin-top: 8px; outline: none; list-style: none; }
            details > summary::-webkit-details-marker { display: none; }
            details > summary::after { content: " ▼ Show Generated SQL"; }
            details[open] > summary::after { content: " ▲ Hide SQL"; }
            
            /* Insight Bubble Style */
            .insight-bubble { background-color: #fff3cd; color: #856404; border: 1px solid #ffeeba; padding: 15px; border-radius: 10px; margin-top: 10px; animation: fadeIn 0.5s; }
            @keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>{%config%}{%scripts%}{%renderer%}</footer>
    </body>
</html>
'''

# --- HELPER: TIMEOUT WRAPPER ---
def run_with_timeout(func, args=(), kwargs=None, timeout_seconds=300):
    if kwargs is None: kwargs = {}
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(func, *args, **kwargs)
        try:
            return future.result(timeout=timeout_seconds)
        except concurrent.futures.TimeoutError:
            raise TimeoutError(f"Process exceeded the {timeout_seconds/60} minute time limit.")

# --- HELPER: LLM INSIGHTS ---
def generate_llm_insights(df):
    """
    Accepts a DataFrame object directly.
    Sends top 50 rows to Databricks Model Serving.
    """
    try:
        w = WorkspaceClient(host=DATABRICKS_HOST, token=GENIE_USER_TOKEN)
        
        # Convert DF to CSV string for the prompt (Limit to top 50 rows to save tokens)
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
            max_tokens=600
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Failed to generate insights: {str(e)}"

# --- HELPER: LOGGING ---
def format_log(current_logs, new_entry):
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    if not isinstance(current_logs, list): current_logs = []
    current_logs.append(html.Div(f"[{ts}] {new_entry}"))
    return current_logs[-20:] # Keep last 20 lines

# --- RENDERER (The Core UI Logic) ---
def render_message_bubble(msg):
    """
    Renders a single message dictionary into a Dash component.
    """
    try:
        is_user = msg["role"] == "user"
        align = "right" if is_user else "left"
        bg = "#007bff" if is_user else "#ffffff"
        color = "white" if is_user else "black"
        border = "none" if is_user else "1px solid #dee2e6"
        
        # Content Container
        children = []
        
        # 1. Label
        if msg.get('space_label'):
            children.append(html.Small(msg['space_label'], style={"display":"block", "marginBottom":"5px", "color":"#ccc" if is_user else "#888"}))
        
        content = msg["content"]
        msg_type = msg.get("type", "text")
        msg_id = msg.get("id", str(uuid.uuid4())) # Ensure ID exists

        # 2. Render Content based on Type
        if msg_type == "table":
            try:
                # Deserialize
                df = pd.read_json(content, orient='split')
                
                # --- FIX: UNIQUE ID FOR EVERY TABLE ---
                # This guarantees Dash re-renders the table even if data looks similar
                unique_table_id = f"table-{msg_id}"
                
                children.append(dash_table.DataTable(
                    id=unique_table_id,
                    data=df.to_dict('records'), 
                    columns=[{"name": i, "id": i} for i in df.columns], 
                    style_table={'overflowX': 'auto', 'minWidth': '100%'}, 
                    style_cell={'textAlign': 'left', 'color': 'black', 'minWidth': '100px'},
                    page_size=10
                ))
                
                # --- FEATURE: PER-TABLE INSIGHT BUTTON ---
                # Pattern Matching ID: {'type': 'insight-btn', 'index': msg_id}
                if not is_user:
                    children.append(html.Div([
                        dbc.Button(
                            "✨ Analyze this", 
                            id={'type': 'insight-btn', 'index': msg_id}, 
                            color="warning", 
                            outline=True, 
                            size="sm", 
                            className="mt-2"
                        )
                    ], className="d-flex justify-content-end"))

            except Exception as e:
                # Fallback if table breaks
                children.append(html.Div([
                    html.Strong("⚠️ Data Display Error:"),
                    html.Pre(str(content)[:500])
                ], style={"color": "red"}))
        
        elif msg_type == "insight":
            # Special styling for insights
            bg = "#fff3cd"
            color = "#856404"
            border = "1px solid #ffeeba"
            children.append(dcc.Markdown(str(content)))
            
        else:
            # Standard Text
            children.append(dcc.Markdown(str(content)))

        # 3. SQL Toggle (if present)
        if not is_user and msg.get("sql"):
            children.append(html.Details([html.Summary(""), html.Pre(msg["sql"], style={"background": "#333", "color": "#fff", "padding": "10px", "borderRadius": "5px"})]))

        # Return the bubble
        return html.Div(
            children, 
            style={
                "textAlign": align, 
                "backgroundColor": bg, 
                "color": color, 
                "border": border,
                "padding": "15px", 
                "borderRadius": "15px", 
                "marginBottom": "15px", 
                "marginLeft": "auto" if is_user else "0", 
                "marginRight": "0" if is_user else "auto", 
                "width": "fit-content", 
                "maxWidth": "90%",
                "boxShadow": "0 2px 4px rgba(0,0,0,0.05)"
            }
        )
    except Exception as e:
        return html.Div(f"Render Error: {e}", style={"color": "red"})

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
                        dbc.Col(dbc.Input(id="user-input", placeholder="Ask a question...", autocomplete="off"), width=10),
                        dbc.Col(dbc.Button("Send", id="send-btn", color="primary", className="w-100"), width=2),
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
    dcc.Store(id="chat-history", data=[]), # Main source of truth
    dcc.Store(id="session-store", data={}), 
    dcc.Store(id="backend-trigger", data=None),
    html.Div(id="dummy-scroll-target")
], fluid=True, style={"padding": "20px"})


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
    
    # 1. Immediate UI Feedback
    # We create a temp message for UI but don't save to store yet (Step 2 does that)
    temp_msg = {"role": "user", "content": user_text, "type": "text", "id": str(uuid.uuid4())}
    temp_history = history + [temp_msg]
    ui_messages = [render_message_bubble(m) for m in temp_history]
    
    logs = format_log(logs, f"Step 1: Routing '{user_text}'")
    
    try:
        # 2. Run Router
        route_decision = orchestrate_routing(user_text, session_store)
        target_space_id = route_decision.get("target_space_id")
        
        target_conv_id = None
        if target_space_id in session_store:
            target_conv_id = session_store[target_space_id].get("conv_id")

        space_label = next((v['label'] for k,v in SPACE_CONFIG.items() if v['id'] == target_space_id), "Unknown")
        
        # 3. Payload for Step 2
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
# ⚙️ STEP 2: EXECUTION (Genie Query)
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
    
    # Use list copy to avoid mutation issues
    current_history = list(history)
    
    user_text = trigger_data["text"]
    logs = format_log(logs, f"Step 2: Executing in {trigger_data['space_id']}...")

    # 1. Commit User Message to History
    current_history.append({
        "role": "user", 
        "content": user_text, 
        "type": "text", 
        "id": str(uuid.uuid4())
    })

    try:
        def run_genie():
            return execute_genie_query(
                user_query=user_text,
                space_id=trigger_data["space_id"],
                current_conv_id=trigger_data["conv_id"], 
                user_token=GENIE_USER_TOKEN,
                host=DATABRICKS_HOST
            )

        final_conv_id, result, sql = run_with_timeout(run_genie, timeout_seconds=300)
        logs = format_log(logs, f"Genie Success. ID: {final_conv_id}")

        # 2. Process Result
        content = result
        msg_type = "text"
        
        if isinstance(result, pd.DataFrame):
            # --- FIX: SAFE JSON SERIALIZATION ---
            # date_format='iso' prevents format errors with Timestamps in Dash
            content = result.to_json(orient='split', date_format='iso')
            msg_type = "table"
        elif isinstance(result, str):
            msg_type = "text"

        # 3. Commit Assistant Message to History
        # We add a unique ID here which will be used by the Insight button
        msg_uuid = str(uuid.uuid4())
        
        current_history.append({
            "role": "assistant",
            "content": content,
            "type": msg_type,
            "space_label": trigger_data["space_label"],
            "sql": sql,
            "id": msg_uuid 
        })

        session_store[trigger_data["space_id"]] = {"conv_id": final_conv_id, "last_topic": user_text}

        # 4. Render
        ui_messages = [render_message_bubble(m) for m in current_history]
        
        active_ui = []
        for sid, data in session_store.items():
            lbl = next((v['label'] for k,v in SPACE_CONFIG.items() if v['id'] == sid), sid[:5])
            active_ui.append(html.Div(f"● {lbl}: ...{data['last_topic'][-15:]}", style={"fontSize":"10px"}))

        return current_history, ui_messages, session_store, active_ui, logs, ""

    except Exception as e:
        logs = format_log(logs, f"BACKEND ERROR: {e}")
        current_history.append({"role": "system", "content": f"Error: {str(e)}", "type": "text", "id": str(uuid.uuid4())})
        return current_history, [render_message_bubble(m) for m in current_history], session_store, no_update, logs, ""


# ==============================================================================
# ✨ STEP 3: INSIGHT GENERATION (PATTERN MATCHING)
# ==============================================================================
@app.callback(
    [Output("chat-history", "data", allow_duplicate=True),
     Output("chat-window", "children", allow_duplicate=True),
     Output("debug-console", "children", allow_duplicate=True)],
    [Input({'type': 'insight-btn', 'index': ALL}, 'n_clicks')],
    [State("chat-history", "data"),
     State("debug-console", "children")],
    prevent_initial_call=True
)
def step_3_generate_specific_insights(n_clicks_list, history, logs):
    """
    This callback triggers when ANY 'Analyze this' button is clicked.
    It identifies WHICH button was clicked, finds the matching table data,
    and inserts the insight immediately after that table.
    """
    ctx = callback_context
    if not ctx.triggered or not history:
        return no_update, no_update, no_update

    # 1. Identify which button was clicked
    clicked_prop_id = ctx.triggered[0]['prop_id'] # e.g., '{"index":"uuid-123","type":"insight-btn"}.n_clicks'
    if not ".n_clicks" in clicked_prop_id: return no_update
    
    try:
        # Extract the dictionary from the string
        import ast
        button_id_dict = ast.literal_eval(clicked_prop_id.split('.')[0])
        target_msg_id = button_id_dict['index'] # This is the UUID of the message containing the table
        
        logs = format_log(logs, f"✨ Analyzing table ID: {target_msg_id}...")

        # 2. Find the message in history
        target_msg_index = -1
        target_content = None
        
        for i, msg in enumerate(history):
            if msg.get('id') == target_msg_id:
                target_msg_index = i
                target_content = msg.get('content')
                break
        
        if target_msg_index == -1 or not target_content:
            logs = format_log(logs, "❌ Error: Could not find original table data.")
            return no_update, no_update, logs

        # 3. Generate Insights
        df_obj = pd.read_json(target_content, orient='split')
        insights_text = run_with_timeout(generate_llm_insights, args=(df_obj,), timeout_seconds=300)
        
        # 4. Insert Insight Message into History
        # We insert it right after the table message
        insight_msg = {
            "role": "assistant",
            "content": f"**✨ AI Insights:**\n\n{insights_text}",
            "type": "insight",
            "space_label": "Analysis",
            "id": str(uuid.uuid4())
        }
        
        new_history = list(history)
        new_history.insert(target_msg_index + 1, insight_msg)
        
        # 5. Re-render
        ui_messages = [render_message_bubble(m) for m in new_history]
        logs = format_log(logs, "Insights generated and added.")
        
        return new_history, ui_messages, logs

    except Exception as e:
        logs = format_log(logs, f"INSIGHT ERROR: {e}")
        return no_update, no_update, logs


# --- SCROLL HELPER ---
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

if __name__ == "__main__":
    app.run_server(debug=True, port=8050)
