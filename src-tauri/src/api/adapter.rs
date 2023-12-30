use axum::{extract::State, Json};

use crate::{AdapterList, ThreadRequest, ThreadState};

/// `/api/adapters`.
pub async fn adapters(State(ThreadState(sender)): State<ThreadState>) -> Json<Vec<String>> {
    let (list_sender, list_receiver) = flume::unbounded();
    let _ = sender.send(ThreadRequest::Adapter(list_sender));
    let AdapterList(list) = list_receiver.recv_async().await.unwrap_or_default();
    Json(list)
}
