use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::Instant;

use rmcp::Json;
use rmcp::handler::server::wrapper::Parameters;
use rmcp::{
    ServerHandler,
    handler::server::tool::ToolRouter,
    model::{Implementation, ProtocolVersion, ServerCapabilities, ServerInfo},
    tool, tool_handler, tool_router,
    transport::{
        StreamableHttpService, streamable_http_server::session::local::LocalSessionManager,
    },
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use tokio::sync::{Mutex, Notify};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};
use uuid::Uuid;

/// Response for creating session
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct CreateSessionResponse {
    #[schemars(description = "The session id for partner LLMs to join this chat.")]
    session_id: String,
    #[schemars(description = "Your participant id which you'll use with every tool from now on.")]
    participant_id: String,
}

/// Request for joining session
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct JoinSessionRequest {
    #[schemars(description = "The session_id of chat to join.")]
    session_id: String,
}

/// Response for joining session
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct JoinSessionResponse {
    #[schemars(description = "Your participant id which you'll use with every tool from now on.")]
    participant_id: String,
}

/// When LLM wants to send a message in the session chat.
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct SendMessageRequest {
    #[schemars(description = "Your participant id.")]
    participant_id: String,
    #[schemars(description = "The message to send.")]
    message: String,
}

/// The response LLM receives on sending a message.
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct SendMessageResponse {
    #[schemars(description = "Whether message sent successfully!")]
    was_success: bool,
}

/// When LLM wants to wait, for others to message or something.
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct WaitRequest {
    #[schemars(description = "Your participant id.")]
    participant_id: String,
}

/// Consumes the inbox and sends the formatted message
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct WaitResponse {
    #[schemars(description = "Others' messages.")]
    other_response: Option<String>,
}

struct Participant {
    participant_id: String,         // The participant id for this participant
    session_id: String,             // The session_id for this participant
    inbox: Mutex<VecDeque<String>>, // Inbox of this participant
    notify: Notify,
}

struct Session {
    session_id: String,
    participants: Mutex<Vec<Arc<Participant>>>,
    max_participants: usize,
    last_accessed: Mutex<Instant>,
}

struct AppState {
    sessions: Mutex<HashMap<String, Arc<Session>>>, // session_id -> Session
    participants: Mutex<HashMap<String, Arc<Participant>>>, // participant_id -> (Session, idx)
}

#[derive(Clone)]
struct ChatServer {
    tool_router: ToolRouter<Self>,
    state: Arc<AppState>,
}

#[tool_router(router = tool_router)]
impl ChatServer {
    fn new() -> Self {
        Self {
            tool_router: Self::tool_router(),
            state: Arc::new(AppState {
                sessions: Mutex::new(HashMap::new()),
                participants: Mutex::new(HashMap::new()),
            }),
        }
    }

    #[tool(
        name = "create_session",
        description = "Creates a chat session. Returns a session_id to share with partner LLMs, and your participant_id to use in all future tool calls."
    )]
    async fn create_session(&self) -> Result<Json<CreateSessionResponse>, String> {
        let session_id = Uuid::new_v4().to_string();
        let participant_id = Uuid::new_v4().to_string();
        let participant = Arc::new(Participant {
            participant_id: participant_id.clone(),
            session_id: session_id.clone(),
            inbox: Mutex::new(VecDeque::new()),
            notify: Notify::new(),
        });

        let session = Arc::new(Session {
            session_id: session_id.clone(),
            participants: Mutex::new(vec![participant.clone()]),
            max_participants: 2,
            last_accessed: Mutex::new(Instant::now()),
        });

        // Spawn the reaper task
        let state_clone = self.state.clone();
        let session_id_clone = session_id.clone();
        tokio::spawn(async move {
            let mut sleep_for = std::time::Duration::from_secs(3660);
            loop {
                tokio::time::sleep(sleep_for).await;
                // Access elapsed time since the session was last accessesed
                let sessions = state_clone.sessions.lock().await;
                let Some(session) = sessions.get(&session_id_clone) else {
                    break;
                };
                let elapsed = session.last_accessed.lock().await.elapsed();
                let session_clone = session.clone();
                drop(sessions); // drop guard before mutating

                // If more than 1 hour has been elapsed
                if elapsed >= std::time::Duration::from_secs(3600) {
                    let participant_ids: Vec<String> = session_clone
                        .participants
                        .lock()
                        .await
                        .iter()
                        .map(|p| p.participant_id.clone())
                        .collect();
                    let mut participants_map = state_clone.participants.lock().await;
                    for id in participant_ids {
                        participants_map.remove(&id);
                    }
                    state_clone.sessions.lock().await.remove(&session_id_clone);
                    tracing::info!("Session {} expired, cleaned up", session_id_clone);
                    break;
                }
                sleep_for = std::time::Duration::from_secs(3605).saturating_sub(elapsed); // Sleep for exactly 5 seconds more than completing 1 hour
                if sleep_for.is_zero() {
                    sleep_for = std::time::Duration::from_secs(5);
                }
                // wake_up exactly when last_accessed + 1 hour hits
            }
        });

        self.state
            .sessions
            .lock()
            .await
            .insert(session_id.clone(), session);

        self.state
            .participants
            .lock()
            .await
            .insert(participant_id.clone(), participant);

        Ok(Json(CreateSessionResponse {
            participant_id,
            session_id,
        }))
    }

    #[tool(
        name = "join_session",
        description = "Join a chat session using session_id."
    )]
    async fn join_session(
        &self,
        Parameters(params): Parameters<JoinSessionRequest>,
    ) -> Result<Json<JoinSessionResponse>, String> {
        let session = {
            let sessions = self.state.sessions.lock().await;
            sessions
                .get(&params.session_id)
                .ok_or("Session not found!")?
                .clone()
        };

        let mut participants_vec = session.participants.lock().await;

        if participants_vec.len() >= session.max_participants {
            return Err("Maximum participants already in chat.".to_string());
        }

        let participant_id = Uuid::new_v4().to_string();

        let participant = Arc::new(Participant {
            participant_id: participant_id.clone(),
            session_id: session.session_id.clone(),
            inbox: Mutex::new(VecDeque::new()),
            notify: Notify::new(),
        });

        participants_vec.push(participant.clone());

        let mut participants_map = self.state.participants.lock().await;
        participants_map.insert(participant_id.clone(), participant);

        Ok(Json(JoinSessionResponse { participant_id }))
    }

    #[tool(
        name = "send_message",
        description = "Send a message to all other participant in the session."
    )]
    async fn send_message(
        &self,
        Parameters(params): Parameters<SendMessageRequest>,
    ) -> Result<Json<SendMessageResponse>, String> {
        let sender = self
            .state
            .participants
            .lock()
            .await
            .get(&params.participant_id)
            .ok_or("Not a valid participant_id")?
            .clone();

        let session = self
            .state
            .sessions
            .lock()
            .await
            .get(&sender.session_id)
            .ok_or("Session not found")?
            .clone();

        *session.last_accessed.lock().await = Instant::now();

        let participants_vec = session.participants.lock().await;

        for p in participants_vec.iter() {
            if !Arc::ptr_eq(p, &sender) {
                p.inbox.lock().await.push_back(params.message.clone());
                p.notify.notify_one();
            }
        }

        Ok(Json(SendMessageResponse { was_success: true }))
    }

    #[tool(
        name = "wait",
        description = "Wait for messages from other participants. Blocks until a message arrives or time out. If other_response is None or timed out error - call wait again. Don't call for many turns though. Let the user know that you've been waiting and aren't receiving any message."
    )]
    async fn wait(
        &self,
        Parameters(params): Parameters<WaitRequest>,
    ) -> Result<Json<WaitResponse>, String> {
        let participant = self
            .state
            .participants
            .lock()
            .await
            .get(&params.participant_id)
            .ok_or("Not a valid participant_id")?
            .clone();

        let session = self
            .state
            .sessions
            .lock()
            .await
            .get(&participant.session_id)
            .ok_or("Session not found")?
            .clone();

        *session.last_accessed.lock().await = Instant::now();

        let mut inbox = participant.inbox.lock().await;

        if !inbox.is_empty() {
            let messages: Vec<String> = inbox.drain(..).collect();
            return Ok(Json(WaitResponse {
                other_response: Some(messages.join("\n")),
            }));
        }
        drop(inbox);

        // Nothing in inbox, block until notified.
        let timeout = tokio::time::timeout(
            std::time::Duration::from_secs(30),
            participant.notify.notified(),
        )
        .await;

        if timeout.is_err() {
            return Ok(Json(WaitResponse {
                other_response: None,
            }));
        }

        *session.last_accessed.lock().await = Instant::now();

        let messages: Vec<String> = participant.inbox.lock().await.drain(..).collect();

        Ok(Json(WaitResponse {
            other_response: Some(messages.join("\n")),
        }))
    }
}

#[tool_handler(router = self.tool_router)]
impl ServerHandler for ChatServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo::new(ServerCapabilities::builder().enable_tools().build())
            .with_protocol_version(ProtocolVersion::V_2025_06_18)
            .with_instructions("A MCP server that allows you to chat with other LLMs.")
            .with_server_info(Implementation::new("chat-server", "0.1.0"))
    }
}

#[tokio::main]
async fn main() {
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info".to_string().into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    let server = ChatServer::new();
    for tool in server.tool_router.list_all() {
        eprintln!("\n{}: {}", tool.name, tool.description.unwrap_or_default());
        if let Some(output_schema) = &tool.output_schema {
            eprintln!(
                "  Output schema: {}",
                serde_json::to_string_pretty(output_schema).unwrap()
            )
        } else {
            eprintln!("  Output: Unstructured text");
        }
    }
    let service = StreamableHttpService::new(
        move || Ok(server.clone()),
        LocalSessionManager::default().into(),
        Default::default(),
    );

    let router = axum::Router::new().nest_service("/mcp", service);
    let tcp_listener = tokio::net::TcpListener::bind("0.0.0.0:80").await.unwrap();

    axum::serve(tcp_listener, router)
        .with_graceful_shutdown(async {
            tokio::signal::ctrl_c().await.unwrap();
        })
        .await
        .unwrap();
}
