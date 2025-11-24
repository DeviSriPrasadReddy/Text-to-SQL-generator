import dash
from dash import html, dcc, Input, Output, State, MATCH, callback_context, no_update, dash_table
import dash_bootstrap_components as dbc
import pandas as pd
import logging
import os
from dotenv import load_dotenv

# --- IMPORTS ---
from routing import SPACE_CONFIG, orchestrate_routing
from genie_backend import execute_genie_query

load_dotenv()
logging.basicConfig(level=logging.INFO)

# --- AUTH CONFIGURATION ---
# App manages the User Token (e.g., Service Account or OBO Token)
DATABRICKS_HOST = os.environ.get("DATABRICKS_HOST")
GENIE_USER_TOKEN = os.environ.get("DATABRICKS_TOKEN") 

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], title="Genie Omni-Router")
server = app.server

# --- CUSTOM CSS ---
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            .chat-container { height: 95vh; display: flex; flex-direction: column; }
            .chat-window { flex-grow: 1; overflow-y: auto; padding: 20px; background: #f8f9fa; border: 1px solid #dee2e6; border-radius: 5px;}
            
            /* Messages */
            .message { margin-bottom: 15px; padding: 10px 15px; border-radius: 10px; max-width: 85%; }
            .user-message { background-color: #007bff; color: white; align-self: flex-end; margin-left: auto; }
            .bot-message { background-color: white; color: #333; align-self: flex-start; margin-right: auto; border: 1px solid #ddd; box-shadow: 0 1px 2px rgba(0,0,0,0.1); }
            
            /* Badges */
            .route-badge { font-size: 0.7rem; font-weight: bold; color: #6c757d; margin-bottom: 5px; display: block; text-transform: uppercase; }
            
            /* SQL Toggle */
            details > summary { cursor: pointer; color: #007bff; font-size: 0.85rem; margin-top: 5px; outline: none; list-style: none; }
            details > summary::-webkit-details-marker { display: none; }
            details > summary::after { content: " ▼ View Generated SQL"; font-weight: bold; }
            details[open] > summary::after { content: " ▲ Hide SQL"; }
            details > pre { background: #2d2d2d; color: #a6e22e; padding: 10px; border-radius: 5px; margin-top: 5px; font-size: 0.75rem; white-space: pre-wrap; font-family: 'Courier New', monospace; }
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
        dbc.Col(html.H3("🧞 Genie Omni-Router"), width=8, className="mt-3"),
        dbc.Col(html.Div(id="status-indicator"), width=4, className="mt-3 text-end")
    ]),
    html.Hr(),
    dbc.Row([
        # --- SIDEBAR ---
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Active Contexts"),
                dbc.CardBody(id="active-sessions-list", className="p-2")
            ], className="mb-3"),
            dbc.Button("🗑️ Reset All Threads", id="reset-btn", color="outline-danger", size="sm", className="w-100")
        ], width=3),

        # --- CHAT WINDOW ---
        dbc.Col([
            html.Div(id="chat-window", className="chat-window"),
            dbc.Row([
                dbc.Col(dbc.Input(id="user-input", placeholder="Ask a question...", autocomplete="off"), width=10),
                dbc.Col(dbc.Button("Send", id="send-btn", color="primary", className="w-100"), width=2)
            ], className="mt-3"),
            html.Div(id="typing-indicator", className="text-muted small mt-1")
        ], width=9)
    ], className="chat-container"),

    # --- STORES ---
    dcc.Store(id="chat-history", data=[]),
    # session-store: { "SPACE_ID": { "conv_id": "...", "last_topic": "..." } }
    dcc.Store(id="session-store", data={}), 
    dcc.Store(id="backend-trigger", data=None),
    html.Div(id="dummy-scroll-target")
], fluid=True, style={"height": "100vh", "padding": "20px"})

# --- MESSAGE RENDERER ---
def render_message(msg):
    is_user = msg["role"] == "user"
    css_class = "user-message" if is_user else "bot-message"
    children = []

    # 1. Source Badge
    if not is_user and msg.get("space_label"):
        children.append(html.Span(f"{msg['space_label']}", className="route-badge"))

    content = msg["content"]
    
    # 2. Render Table or Text
    if isinstance(content, str) and content.startswith('{') and "columns" in content:
        try:
            df = pd.read_json(content, orient='split')
            children.append(dash_table.DataTable(
                data=df.to_dict('records'),
                columns=[{"name": i, "id": i} for i in df.columns],
                style_table={'overflowX': 'auto'},
                page_size=5,
                style_cell={'textAlign': 'left', 'fontSize': '12px', 'padding': '5px'}
            ))
        except:
            children.append(dcc.Markdown("Error parsing data table."))
    else:
        children.append(dcc.Markdown(str(content)))

    # 3. SQL Toggle (Hidden by default)
    if not is_user and msg.get("sql"):
        children.append(html.Details([
            html.Summary(""), 
            html.Pre(msg["sql"])
        ]))

    return html.Div(children, className=f"message {css_class}")

# --- MAIN CALLBACK ---
@app.callback(
    [Output("chat-window", "children"),
     Output("user-input", "value"),
     Output("chat-history", "data"),
     Output("backend-trigger", "data"),
     Output("typing-indicator", "children"),
     Output("session-store", "data"),
     Output("active-sessions-list", "children")],
    [Input("send-btn", "n_clicks"),
     Input("user-input", "n_submit"),
     Input("reset-btn", "n_clicks"),
     Input("backend-trigger", "data")],
    [State("user-input", "value"),
     State("chat-history", "data"),
     State("session-store", "data")],
    prevent_initial_call=True
)
def manage_chat(n_clicks, n_submit, n_reset, trigger_data, user_text, history, session_store):
    ctx = callback_context
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if history is None: history = []
    if session_store is None: session_store = {}

    # RESET ALL
    if trigger_id == "reset-btn":
        return [], "", [], None, "", {}, html.Div("No active threads.", className="text-muted small")

    # STEP 1: ROUTING ORCHESTRATION
    if trigger_id in ["send-btn", "user-input"] and user_text:
        history.append({"role": "user", "content": user_text})
        
        # --- CALL ROUTER ---
        # We pass the ENTIRE session store so the Router LLM knows what we talked about previously
        route_decision = orchestrate_routing(user_text, session_store)
        
        target_space_id = route_decision.get("target_space_id")
        # Router decides if we use an existing Conv ID or None (New)
        target_conv_id = route_decision.get("target_conversation_id") 
        
        # Get Readable Label
        space_label = "Unknown"
        for k,v in SPACE_CONFIG.items():
            if v['id'] == target_space_id: space_label = v['label']
            
        indicator_text = f"Routing to {space_label}..." 
        if target_conv_id:
             indicator_text = f"Following up in {space_label}..."

        # Prepare Step 2 Trigger
        next_trigger = {
            "text": user_text,
            "space_id": target_space_id,
            "conv_id": target_conv_id, 
            "space_label": space_label
        }
        
        return [render_message(m) for m in history], "", history, next_trigger, indicator_text, session_store, no_update

    # STEP 2: GENIE EXECUTION
    if trigger_id == "backend-trigger" and trigger_data:
        user_query = trigger_data["text"]
        space_id = trigger_data["space_id"]
        conv_id_request = trigger_data["conv_id"]
        space_label = trigger_data["space_label"]

        # Call Genie Backend (using User Token from App Env)
        final_conv_id, result, sql = execute_genie_query(
            user_query=user_query,
            space_id=space_id,
            current_conv_id=conv_id_request, 
            user_token=GENIE_USER_TOKEN,
            host=DATABRICKS_HOST
        )

        # Format Content for Store
        content = result
        if isinstance(result, pd.DataFrame):
            content = result.to_json(orient='split')

        history.append({
            "role": "assistant",
            "content": content,
            "space_label": space_label,
            "sql": sql
        })

        # UPDATE SESSION STORE
        # We save the 'final_conv_id' and the 'user_query' as the 'last_topic'
        # This is crucial for the Router to make smart decisions next time
        session_store[space_id] = {
            "conv_id": final_conv_id,
            "last_topic": user_query 
        }

        # Render Active List Sidebar
        active_ui = []
        for sid, data in session_store.items():
            s_label = next((v['label'] for k,v in SPACE_CONFIG.items() if v['id'] == sid), sid)
            active_ui.append(html.Div([
                html.Div(f"● {s_label}", style={"color":"green", "fontWeight":"bold", "fontSize":"0.9rem"}),
                html.Div(f"Topic: {data['last_topic'][:25]}...", style={"fontSize":"0.75rem", "color":"#666", "paddingLeft":"12px"}),
                html.Hr(style={"margin":"5px 0"})
            ]))

        return [render_message(m) for m in history], no_update, history, None, "", session_store, active_ui

    return no_update

# --- SCROLL SCRIPT ---
app.clientside_callback(
    """function(c){ var w = document.getElementById('chat-window'); if(w){ w.scrollTop = w.scrollHeight; } return null; }""",
    Output("dummy-scroll-target", "children"),
    Input("chat-window", "children")
)

if __name__ == "__main__":
    app.run_server(debug=True, port=8050)
