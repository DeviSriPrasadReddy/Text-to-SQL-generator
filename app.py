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
import re
import ast  # Needed for parsing dictionary IDs safely
from dotenv import load_dotenv

# --- IMPORTS (Assumed to exist in your project) ---
# Ensure routing.py and genie_backend.py are in the same directory
try:
    from routing import SPACE_CONFIG, orchestrate_routing
    from genie_backend import execute_genie_query
except ImportError:
    print("WARNING: 'routing' or 'genie_backend' modules not found. Ensure they exist locally.")
    SPACE_CONFIG = {} # Dummy fallback to prevent immediate crash during copy-paste testing

from databricks.sdk import WorkspaceClient

load_dotenv()

# --- CONFIGURATION ---
DATABRICKS_HOST = os.environ.get("DATABRICKS_HOST")
GENIE_USER_TOKEN = os.environ.get("DATABRICKS_TOKEN")
LLM_ENDPOINT_NAME = os.environ.get("LLM_ENDPOINT_NAME", "databricks-meta-llama-3-70b-instruct")
VISUAL_ENDPOINT_NAME = os.environ.get("VISUAL_ENDPOINT_NAME", LLM_ENDPOINT_NAME)

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], title="Genie Router")
server = app.server

# --- CSS STYLING (FIXED SCROLLING) ---
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            /* Ensures the container takes up specific viewport height */
            .chat-container { 
                height: 75vh; 
                overflow: hidden; 
            }

            /* Forces Bootstrap columns to fill the height */
            .h-100 { height: 100%; }

            /* Wrapper for the right side (Chat + Input) */
            .chat-col-wrapper { 
                height: 100%; 
                display: flex; 
                flex-direction: column; 
            }

            /* The actual scrolling area - FIXED */
            .chat-window { 
                flex: 1;              /* Fill remaining space */
                overflow-y: auto;     /* Scroll vertically */
                min-height: 0;        /* CRITICAL: Allows flex child to shrink/scroll */
                padding: 20px; 
                background: #f8f9fa; 
                border: 1px solid #dee2e6; 
                border-radius: 5px; 
                margin-bottom: 10px;
            }

            /* Debug Terminal */
            .debug-terminal { background-color: #1e1e1e; color: #00ff00; font-family: 'Courier New', monospace; font-size: 0.8rem; padding: 15px; height: 150px; overflow-y: auto; border-radius: 5px; white-space: pre-wrap; }

            /* SQL Toggle Styling */
            details > summary { cursor: pointer; color: #007bff; font-size: 0.8rem; margin-top: 8px; outline: none; list-style: none; }
            details > summary::-webkit-details-marker { display: none; }
            details > summary::after { content: " ▼ Show Generated SQL"; }
            details[open] > summary::after { content: " ▲ Hide SQL"; }

            /* Graph Container */
            .graph-container { margin-top: 15px; border: 1px solid #ddd; padding: 5px; background: white; border-radius: 8px; }
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
    try:
        w = WorkspaceClient(host=DATABRICKS_HOST, token=GENIE_USER_TOKEN)
        preview_csv = df.head(50).to_csv(index=False)
        prompt = f"Analyze this data (top 50 rows):\n{preview_csv}\nProvide 3-5 concise, high-value business insights as a bulleted list."
        response = w.serving_endpoints.query(
            name=LLM_ENDPOINT_NAME,
            messages=[{"role": "system", "content": "You are a data analyst helper."}, {"role": "user", "content": prompt}],
            temperature=0.7, max_tokens=600
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Failed to generate insights: {str(e)}"

# --- HELPER: VISUAL SPEC GENERATOR ---
def get_visual_spec(df: pd.DataFrame, user_query: str) -> str:
    if not VISUAL_ENDPOINT_NAME:
        return None
    prompt = f"""
    You are a data visualization expert. Your task is to generate a Plotly JSON specification.
    User's Request: "{user_query}"
    Data Schema: {df.dtypes.to_string()}
    Data (first 5 rows): {df.head().to_string()}
    CRITICAL RULES:
    1. Response MUST be a single valid JSON object.
    2. Format: {{"data": [...], "layout": {{...}}}}
    """
    try:
        w = WorkspaceClient(host=DATABRICKS_HOST, token=GENIE_USER_TOKEN)
        response = w.serving_endpoints.query(
            name=VISUAL_ENDPOINT_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1, max_tokens=1000
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error getting visual spec: {str(e)}"

# --- HELPER: INTENT DETECTION ---
def check_visual_intent(text):
    if not text: return False
    keywords = ['chart', 'plot', 'graph', 'visual', 'visualize', 'pie', 'bar', 'line', 'scatter', 'heatmap', 'histogram']
    return any(k in text.lower() for k in keywords)

# --- RENDERER ---
def render_message_bubble(msg):
    try:
        is_user = msg["role"] == "user"
        align = "right" if is_user else "left"
        bg = "#007bff" if is_user else "#ffffff"
        color = "white" if is_user else "black"
        border = "none" if is_user else "1px solid #dee2e6"

        children = []

        if msg.get('space_label'):
            children.append(html.Small(msg['space_label'], style={"display":"block", "marginBottom":"5px", "color":"#ccc" if is_user else "#888"}))

        content = msg["content"]
        msg_type = msg.get("type", "text")
        msg_id = msg.get("id", str(uuid.uuid4()))

        # 1. TABLE RENDER
        if msg_type == "table":
            try:
                df = pd.read_json(content, orient='split')
                unique_table_id = f"table-{msg_id}"
                children.append(dash_table.DataTable(
                    id=unique_table_id,
                    data=df.to_dict('records'),
                    columns=[{"name": i, "id": i} for i in df.columns],
                    style_table={'overflowX': 'auto', 'minWidth': '100%'},
                    style_cell={'textAlign': 'left', 'color': 'black', 'minWidth': '100px'},
                    page_size=10
                ))
                if not is_user:
                    children.append(html.Div([
                        dbc.Button("✨ Analyze", id={'type': 'insight-btn', 'index': msg_id}, color="warning", outline=True, size="sm", className="mt-2")
                    ], className="d-flex justify-content-end"))
            except Exception:
                children.append(html.Pre(str(content)[:500], style={"color": "red"}))
        
        # 2. INSIGHT RENDER
        elif msg_type == "insight":
            bg = "#fff3cd"; color = "#856404"; border = "1px solid #ffeeba"
            children.append(dcc.Markdown(str(content)))
        
        # 3. TEXT RENDER
        else:
            children.append(dcc.Markdown(str(content)))

        # 4. VISUALIZATION RENDER
        if not is_user and msg.get('graph_spec'):
            try:
                figure_json = msg['graph_spec']
                if isinstance(figure_json, str): figure_json = json.loads(figure_json)
                children.append(html.Div([
                    dcc.Graph(figure=figure_json, config={'displayModeBar': False}, style={'height': '400px'})
                ], className="graph-container"))
            except Exception as e:
                children.append(html.Small(f"Failed to render chart: {e}", style={"color": "red"}))

        if not is_user and msg.get("sql"):
            children.append(html.Details([html.Summary(""), html.Pre(msg["sql"], style={"background": "#333", "color": "#fff", "padding": "10px", "borderRadius": "5px"})]))

        return html.Div(
            children,
            style={"textAlign": align, "backgroundColor": bg, "color": color, "border": border, "padding": "15px", "borderRadius": "15px", "marginBottom": "15px", "marginLeft": "auto" if is_user else "0", "marginRight": "0" if is_user else "auto", "width": "fit-content", "maxWidth": "90%", "boxShadow": "0 2px 4px rgba(0,0,0,0.05)"}
        )
    except Exception as e:
        return html.Div(f"Render Error: {e}", style={"color": "red"})

# --- LAYOUT (FIXED STRUCTURE) ---
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H3("🧞 Genie Multi-Space Router"), width=8, className="mt-3"),
        dbc.Col(html.Div(id="status-indicator", className="mt-4"), width=4)
    ]),
    html.Hr(),
    
    # Main Chat Area
    dbc.Row([
        # LEFT COLUMN (Sidebar)
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Active Contexts"), 
                dbc.CardBody(id="active-sessions-list", style={"height": "100%"})
            ], className="mb-3 h-100"), 
            dbc.Button("Reset Chat", id="reset-btn", color="outline-danger", size="sm", className="w-100")
        ], width=3, className="h-100"),
        
        # RIGHT COLUMN (Chat Window)
        dbc.Col([
            html.Div([
                # 1. The Scrollable Chat History
                html.Div(id="chat-window", className="chat-window"),
                
                # 2. The Input Area (Fixed at bottom)
                html.Div([
                    dbc.Row([
                        dbc.Col(dbc.Input(id="user-input", placeholder="Ask a question...", autocomplete="off"), width=10),
                        dbc.Col(dbc.Button("Send", id="send-btn", color="primary", className="w-100"), width=2),
                    ], className="mt-0"),
                    html.Div(id="typing-indicator", className="text-muted small mt-1")
                ])
            ], className="chat-col-wrapper")
        ], width=9, className="h-100")
    ], className="chat-container mb-3"),
    
    # Debug/Footer Area
    dbc.Collapse(
        dbc.Card([
            dbc.CardHeader("🐞 Logic Log"), 
            dbc.CardBody(html.Div(id="debug-console", className="debug-terminal"))
        ]), 
        id="debug-collapse", 
        is_open=True
    ),
    
    dcc.Store(id="chat-history", data=[]), 
    dcc.Store(id="session-store", data={}), 
    dcc.Store(id="backend-trigger", data=None),
    html.Div(id="dummy-scroll-target")
], fluid=True, style={"padding": "20px", "height": "100vh", "display": "flex", "flexDirection": "column"})

# --- LOGGING HELPER ---
def format_log(current_logs, new_entry):
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    if not isinstance(current_logs, list): current_logs = []
    current_logs.append(html.Div(f"[{ts}] {new_entry}"))
    return current_logs[-20:]

# --- STEP 1: ROUTING ---
@app.callback(
    [Output("chat-window", "children", allow_duplicate=True),
     Output("user-input", "value"),
     Output("backend-trigger", "data", allow_duplicate=True),
     Output("debug-console", "children", allow_duplicate=True),
     Output("typing-indicator", "children", allow_duplicate=True)],
    [Input("send-btn", "n_clicks"), Input("user-input", "n_submit")],
    [State("user-input", "value"), State("chat-history", "data"), State("session-store", "data"), State("debug-console", "children")],
    prevent_initial_call=True
)
def step_1_routing(n_c, n_s, user_text, history, session_store, logs):
    if not user_text: return no_update
    if history is None: history = []

    temp_msg = {"role": "user", "content": user_text, "type": "text", "id": str(uuid.uuid4())}
    ui_messages = [render_message_bubble(m) for m in history + [temp_msg]]
    logs = format_log(logs, f"Step 1: Routing '{user_text}'")

    try:
        route_decision = orchestrate_routing(user_text, session_store)
        target_space_id = route_decision.get("target_space_id")
        target_conv_id = session_store.get(target_space_id, {}).get("conv_id")
        space_label = next((v['label'] for k,v in SPACE_CONFIG.items() if v['id'] == target_space_id), "Unknown")

        trigger_payload = {
            "text": user_text, "space_id": target_space_id, "conv_id": target_conv_id, "space_label": space_label, "uuid": str(uuid.uuid4())
        }
        return ui_messages, "", trigger_payload, logs, f"Routing to {space_label}..."
    except Exception as e:
        logs = format_log(logs, f"ROUTING ERROR: {e}")
        return ui_messages, "", None, logs, "Error"

# --- STEP 2: EXECUTION & VISUALIZATION ---
@app.callback(
    [Output("chat-history", "data", allow_duplicate=True),
     Output("chat-window", "children", allow_duplicate=True),
     Output("session-store", "data"),
     Output("active-sessions-list", "children"),
     Output("debug-console", "children", allow_duplicate=True),
     Output("typing-indicator", "children", allow_duplicate=True)],
    [Input("backend-trigger", "modified_timestamp")],
    [State("backend-trigger", "data"), State("chat-history", "data"), State("session-store", "data"), State("debug-console", "children")],
    prevent_initial_call=True
)
def step_2_execution(ts, trigger_data, history, session_store, logs):
    if not ts or not trigger_data: return no_update
    if history is None: history = []
    current_history = list(history)
    user_text = trigger_data["text"]

    logs = format_log(logs, f"Step 2: Executing in {trigger_data['space_id']}...")
    current_history.append({"role": "user", "content": user_text, "type": "text", "id": str(uuid.uuid4())})

    try:
        def run_genie():
            return execute_genie_query(
                user_query=user_text, space_id=trigger_data["space_id"],
                current_conv_id=trigger_data["conv_id"], user_token=GENIE_USER_TOKEN, host=DATABRICKS_HOST
            )

        final_conv_id, result, sql = run_with_timeout(run_genie, timeout_seconds=300)
        logs = format_log(logs, f"Genie Success. ID: {final_conv_id}")

        content = result
        msg_type = "text"
        graph_spec = None
        df_for_visual = None

        if isinstance(result, pd.DataFrame):
            content = result.to_json(orient='split', date_format='iso')
            msg_type = "table"
            df_for_visual = result
        elif isinstance(result, str) and check_visual_intent(user_text):
            last_assistant_msg = None
            for m in reversed(history):
                if m['role'] == 'assistant' and m.get('type') == 'table':
                    last_assistant_msg = m
                    break
            if last_assistant_msg:
                logs = format_log(logs, "Use Previous Table for Visual...")
                try:
                    df_for_visual = pd.read_json(last_assistant_msg['content'], orient='split')
                    msg_type = "text"
                except:
                    pass

        if df_for_visual is not None and check_visual_intent(user_text):
            logs = format_log(logs, "🎨 Generating Visual Spec...")
            try:
                raw_spec = run_with_timeout(get_visual_spec, args=(df_for_visual, user_text), timeout_seconds=60)
                cleaned_spec = raw_spec.replace('`json', '').replace('`', '').strip()
                if cleaned_spec.startswith('{') and cleaned_spec.endswith('}'):
                    graph_spec = json.loads(cleaned_spec)
                    logs = format_log(logs, "Visual Spec Generated.")
                else:
                    logs = format_log(logs, "❌ Invalid JSON from LLM")
            except Exception as e:
                logs = format_log(logs, f"Visual Error: {e}")

        current_history.append({
            "role": "assistant",
            "content": content,
            "type": msg_type,
            "space_label": trigger_data["space_label"],
            "sql": sql,
            "id": str(uuid.uuid4()),
            "graph_spec": graph_spec
        })

        session_store[trigger_data["space_id"]] = {"conv_id": final_conv_id, "last_topic": user_text}
        ui_messages = [render_message_bubble(m) for m in current_history]
        active_ui = [html.Div(f"● {next((v['label'] for k,v in SPACE_CONFIG.items() if v['id'] == k), k[:5])}: ...{v['last_topic'][-15:]}", style={"fontSize":"10px"}) for k,v in session_store.items()]

        return current_history, ui_messages, session_store, active_ui, logs, ""

    except Exception as e:
        logs = format_log(logs, f"BACKEND ERROR: {e}")
        current_history.append({"role": "system", "content": f"Error: {str(e)}", "type": "text", "id": str(uuid.uuid4())})
        return current_history, [render_message_bubble(m) for m in current_history], session_store, no_update, logs, ""

# --- STEP 3: SPECIFIC INSIGHTS (FIXED AUTO-TRIGGER) ---
@app.callback(
    [Output("chat-history", "data", allow_duplicate=True), Output("chat-window", "children", allow_duplicate=True), Output("debug-console", "children", allow_duplicate=True)],
    [Input({'type': 'insight-btn', 'index': ALL}, 'n_clicks')],
    [State("chat-history", "data"), State("debug-console", "children")],
    prevent_initial_call=True
)
def step_3_insights(n_clicks_list, history, logs):
    ctx = callback_context
    if not ctx.triggered or not history: return no_update, no_update, no_update

    # FIX: Check if the triggered value is actually a click (not None/0 from creation)
    triggered_value = ctx.triggered[0]['value']
    if not triggered_value or triggered_value == 0:
        return no_update, no_update, no_update

    try:
        # Extract the button ID
        triggered_prop_id = ctx.triggered[0]['prop_id']
        button_id_str = triggered_prop_id.split('.')[0]
        button_id = ast.literal_eval(button_id_str)
        target_id = button_id['index']

        target_msg = next((m for m in history if m.get('id') == target_id), None)
        if not target_msg: return no_update, no_update, logs

        df_obj = pd.read_json(target_msg['content'], orient='split')
        logs = format_log(logs, f"Generating insights for {target_id}...")
        
        insights = run_with_timeout(generate_llm_insights, args=(df_obj,), timeout_seconds=300)

        insight_msg = {"role": "assistant", "content": f"**✨ AI Insights:**\n\n{insights}", "type": "insight", "space_label": "Analysis", "id": str(uuid.uuid4())}

        idx = history.index(target_msg)
        new_history = list(history)
        new_history.insert(idx + 1, insight_msg)

        return new_history, [render_message_bubble(m) for m in new_history], format_log(logs, "Insights delivered.")
    except Exception as e:
        return no_update, no_update, format_log(logs, f"Insight Error: {e}")

# --- SCROLL ---
app.clientside_callback(
    """function(c){var w=document.getElementById('chat-window'); if(w){setTimeout(function(){w.scrollTop=w.scrollHeight;},100);} return null;}""",
    Output("dummy-scroll-target", "children"), Input("chat-window", "children")
)

# --- RESET ---
@app.callback([Output("chat-history", "data", allow_duplicate=True), Output("chat-window", "children", allow_duplicate=True), Output("session-store", "data", allow_duplicate=True), Output("active-sessions-list", "children", allow_duplicate=True), Output("backend-trigger", "data", allow_duplicate=True)], [Input("reset-btn", "n_clicks")], prevent_initial_call=True)
def reset(n): return [], [], {}, "No contexts.", None

if __name__ == "__main__":
    app.run_server(debug=True, port=8050)
