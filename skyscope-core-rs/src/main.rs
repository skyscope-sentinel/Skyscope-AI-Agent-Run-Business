use axum::{
    extract::State,
    http::StatusCode,
    response::Json,
    routing::{get, post},
    Router,
};
use serde_json::{json, Value};
use std::net::SocketAddr;
use std::sync::Arc;

mod communication;
mod database;
mod integrations;

use integrations::n8n::N8nTool;

// The application state, which will hold our tool instances.
// We use Arc to allow the state to be shared safely across threads.
#[derive(Clone)]
struct AppState {
    n8n_tool: Arc<N8nTool>,
}

pub struct McpLauncher {
    state: AppState,
}

impl McpLauncher {
    pub fn new() -> Self {
        let state = AppState {
            n8n_tool: Arc::new(N8nTool::new()),
        };
        Self { state }
    }

    pub async fn run(self) -> anyhow::Result<()> {
        let app = Router::new()
            .route("/", get(root_handler))
            .route(
                "/api/n8n/trigger",
                post(trigger_n8n_workflow_handler),
            )
            .with_state(self.state);

        let addr = SocketAddr::from(([127, 0, 0, 1], 8080));
        println!("MCP Server listening on {}", addr);

        axum::Server::bind(&addr)
            .serve(app.into_make_service())
            .await?;

        Ok(())
    }
}

async fn root_handler() -> &'static str {
    "Skyscope Core MCP Server is running."
}

// The handler for our new N8N tool endpoint.
// It takes the application state and a JSON payload as input.
async fn trigger_n8n_workflow_handler(
    State(state): State<AppState>,
    Json(payload): Json<Value>,
) -> (StatusCode, Json<Value>) {
    // For this example, we'll extract a workflow_id from the payload.
    // A real implementation would have more robust validation.
    let workflow_id = payload
        .get("workflow_id")
        .and_then(Value::as_str)
        .unwrap_or("default_workflow");

    match state
        .n8n_tool
        .trigger_workflow(workflow_id, &payload)
        .await
    {
        Ok(response) => (
            StatusCode::OK,
            Json(json!({ "status": "success", "message": response })),
        ),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "status": "error", "message": e.to_string() })),
        ),
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let launcher = McpLauncher::new();
    launcher.run().await
}
