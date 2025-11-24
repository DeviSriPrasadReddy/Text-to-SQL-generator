import dash
from dash import html, dcc, Input, Output, State, MATCH, callback_context, no_update, dash_table
import dash_bootstrap_components as dbc
import pandas as pd
import logging
import uuid
from dotenv import load_dotenv

# --- IMPORT BACKEND LOGIC ---
from genie_backend import execute_genie_query, route_question, generate_insights, SPACE_CONFIG

# --- SETUP ---
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    title="Genie Enterprise Router"
)
server = app.server

# --- CUSTOM STYLES ---
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            .chat-container { height: 90vh; display: flex; flex-direction: column; }
            .chat-window { flex-grow: 1; overflow-y: auto; padding: 20px; background: #f8f9fa; border-radius: 8px; margin-bottom: 15px; border: 1px solid #dee2e6; }
            .message { margin-bottom: 15px; padding: 10px 15px; border-radius: 15px; max-width: 80%; position: relative; }
            .user-message { background-color: #007bff; color: white; align-self: flex-end; margin-left: auto; border-bottom-right-radius: 2px; }
            .bot-message { background-color: #ffffff; color: #333; align-self: flex-start; margin-right: auto; border: 1px solid #e9ecef; border-bottom-left-radius: 2px; box-shadow: 0 1px 2px rgba(0,0,0,0.05); }
            .route-badge { font-size: 0.7rem; font-weight: bold; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 4px; display: block; color: #6c757d; }
            .insight-box { background: #f0f7ff; border-left: 4px solid #007bff; padding: 10px; margin-top: 10px; font-size: 0.9rem; }
            .typing-indicator { font-style: italic; color: #888; font-size: 0.8rem; margin-left: 10px; min-height: 20px;}
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# --- LAYOUT ---
app.layout = dbc.Container([
    # Header
    dbc.Row([
        dbc.Col(html.H3("🧞 Genie Multi-Space Router"), width=10, className="mt-3"),
        dbc.Col(html.Div(id="user-badge"), width=2, className="mt-3 text-end")
    ]),
    
    html.Hr(),

    # Main Chat Area
    dbc.Row([
        # Sidebar (Active Sessions)
        dbc.Col([
            html.H6("Active Contexts", className="text-muted"),
            html.Div(id="active-sessions-list"),
            dbc.Button("Reset All", id="reset-btn", color="outline-danger", size="sm", className="mt-3 w-100")
        ], width=3, style={"borderRight": "1px solid #eee"}),

        # Chat Window
        dbc.Col([
            html.Div(id="chat-window", className="chat-window"),
            
            # Input Area
            dbc.Row([
                dbc.Col(
                    dbc.Input(id="user-input", placeholder="Ask about Finance or HR...", type="text", autocomplete="off"),
                    width=10
                ),
                dbc.Col(
                    dbc.Button("Send", id="send-btn", color="primary", className="w-100"),
                    width=2
                )
            ]),
            html.Div(id="typing-indicator", className="typing-indicator")
        ], width=9)
    ], className="chat-container"),

    # --- STORES ---
    dcc.Store(id="chat-history", data=[]),
    dcc.Store(id="session-map", data={}),
    dcc.Store(id="backend-trigger", data=None),
    html.Div(id="dummy-div", style={"display": "none"})

], fluid=True, style={"height": "100vh", "padding": "20px"})


# --- HELPER: Render Message ---
def render_chat_message(msg):
    """Converts a message dictionary into a Dash HTML component."""
    is_user = msg["role"] == "user"
    css_class = "user-message" if is_user else "bot-message"
    
    children = []
    
    # 1. Add Routing Badge (Bot only)
    if not is_user and msg.get("space_label"):
        children.append(html.Span(f"Source: {msg['space_label']}", className="route-badge"))

    # 2. Add Content (Text or Table)
    content = msg["content"]
    if isinstance(content, str):
        # If it looks like a JSON table (from backend), parse it
        if content.strip().startswith('{') and "columns" in content:
             try:
                 df = pd.read_json(content, orient='split')
                 # Render DataTable
                 children.append(dash_table.DataTable(
                     data=df.to_dict('records'),
                     columns=[{"name": i, "id": i} for i in df.columns],
                     style_table={'overflowX': 'auto'},
                     page_size=5,
                     style_cell={'textAlign': 'left', 'fontSize': '12px'}
                 ))
                 # Add "Generate Insights" Button
                 btn_id = {"type": "insight-btn", "index": str(uuid.uuid4())}
                 children.append(dbc.Button("✨ Generate Insights", id=btn_id, size="sm", color="info", outline=True, className="mt-2"))
                 children.append(html.Div(id={"type": "insight-target", "index": btn_id["index"]}))
             except:
                 children.append(dcc.Markdown(content))
        else:
            children.append(dcc.Markdown(content))
    
    # 3. Add SQL (Bot only)
    if not is_user and msg.get("sql"):
        children.append(html.Details([
            html.Summary("View SQL"),
            html.Pre(msg["sql"], style={"fontSize": "0.7rem", "background": "#eee", "padding": "5px"})
        ]))

    return html.Div(children, className=f"message {css_class}")


# --- CALLBACK 1: Handle User Input & UI Updates ---
@app.callback(
    [Output("chat-window", "children"),
     Output("user-input", "value"),
     Output("chat-history", "data"),
     Output("backend-trigger", "data"),
     Output("typing-indicator", "children"),
     Output("session-map", "data"),
     Output("active-sessions-list", "children")],
    [Input("send-btn", "n_clicks"),
     Input("user-input", "n_submit"),
     Input("reset-btn", "n_clicks"),
     Input("backend-trigger", "data")], 
    [State("user-input", "value"),
     State("chat-history", "data"),
     State("session-map", "data")],
    prevent_initial_call=True
)
def manage_chat(n_clicks, n_submit, n_reset, trigger_data, 
                user_text, history, session_map):
    
    ctx = callback_context
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
    
    # INITIALIZE
    if history is None: history = []
    if session_map is None: session_map = {}

    # CASE A: RESET
    if trigger_id == "reset-btn":
        return [], "", [], None, "", {}, html.Div("No active contexts.")

    # CASE B: USER SUBMITS TEXT (Step 1)
    if trigger_id in ["send-btn", "user-input"] and user_text:
        history.append({"role": "user", "content": user_text})
        
        # Determine Routing
        space_id = route_question(user_text)
        
        # Trigger Backend
        space_config_entry = next((v for k, v in SPACE_CONFIG.items() if v["id"] == space_id), {"label": "Unknown"})
        
        next_trigger = {
            "text": user_text,
            "space_id": space_id,
            "space_label": space_config_entry["label"]
        }
        
        ui_messages = [render_chat_message(m) for m in history]
        
        return (ui_messages, "", history, next_trigger, 
                f"Routing to {next_trigger['space_label']}...", session_map, no_update)

    # CASE C: BACKEND RESPONSE (Step 2)
    if trigger_id == "backend-trigger" and trigger_data:
        user_query = trigger_data["text"]
        space_id = trigger_data["space_id"]
        space_label = trigger_data["space_label"]
        
        current_conv_id = session_map.get(space_id)
        
        # EXECUTE QUERY
        new_conv_id, result_raw, sql_query = execute_genie_query(user_query, space_id, current_conv_id)
        
        # Process Result
        content_to_store = result_raw
        if isinstance(result_raw, pd.DataFrame):
            content_to_store = result_raw.to_json(orient='split')
            
        history.append({
            "role": "assistant",
            "content": content_to_store,
            "space_label": space_label,
            "sql": sql_query
        })
        
        if new_conv_id:
            session_map[space_id] = new_conv_id
            
        ui_messages = [render_chat_message(m) for m in history]
        
        active_sessions = []
        for sid, cid in session_map.items():
            label = next((v["label"] for k, v in SPACE_CONFIG.items() if v["id"] == sid), "Genie Space")
            active_sessions.append(html.Div([
                html.Span("● ", style={"color": "green"}),
                html.Span(f"{label}: "),
                html.Span(f"{cid[:6]}...", style={"fontSize": "0.8em", "fontFamily": "monospace"})
            ]))

        return (ui_messages, no_update, history, None, "", session_map, active_sessions)

    return no_update


# --- CALLBACK 2: GENERATE INSIGHTS ---
@app.callback(
    Output({"type": "insight-target", "index": MATCH}, "children"),
    Input({"type": "insight-btn", "index": MATCH}, "n_clicks"),
    State("chat-history", "data"),
    prevent_initial_call=True
)
def show_insights(n_clicks, history):
    if not n_clicks: return no_update
    
    last_df_content = None
    for msg in reversed(history):
        if msg["role"] == "assistant" and isinstance(msg["content"], str) and "columns" in msg["content"]:
            last_df_content = msg["content"]
            break
            
    if last_df_content:
        df = pd.read_json(last_df_content, orient='split')
        insight_text = generate_insights(df)
        return html.Div(dcc.Markdown(insight_text), className="insight-box")
    
    return html.Div("No data found to analyze.", className="text-danger")


# --- CLIENTSIDE: AUTO-SCROLL ---
app.clientside_callback(
    """
    function(children) {
        var chat_window = document.getElementById('chat-window');
        if(chat_window){
            chat_window.scrollTop = chat_window.scrollHeight;
        }
        return null;
    }
    """,
    Output("dummy-div", "children"),
    Input("chat-window", "children")
)

if __name__ == "__main__":
    app.run_server(debug=True, port=8050)
