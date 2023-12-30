use std::time::Duration;

use axum::{extract::State, Json};
use serde::Serialize;

use crate::{utils::request_info, ThreadState};

#[derive(Debug, Serialize)]
struct ModelChoice {
    object: String,
    id: String,
}

#[derive(Debug, Serialize)]
pub struct ModelResponse {
    data: Vec<ModelChoice>,
}

pub async fn models(State(ThreadState(sender)): State<ThreadState>) -> Json<ModelResponse> {
    let info = request_info(sender, Duration::from_secs(1)).await;
    let model_name = info
        .reload
        .model_path
        .file_stem()
        .map(|stem| stem.to_string_lossy())
        .unwrap_or_default();

    Json(ModelResponse {
        data: vec![ModelChoice {
            object: "models".into(),
            id: model_name.into(),
        }],
    })
}
