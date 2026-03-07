//! # NeuroLang Rust SDK
//!
//! Write neural networks in plain English from Rust.
//!
//! ```rust
//! use neurolang::NeuroLang;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let nl = NeuroLang::new("http://localhost:3000");
//!
//!     // One-liner
//!     let result = nl.run("Predict species from iris").await?;
//!     println!("Accuracy: {:.4}", result.accuracy.unwrap());
//!
//!     // Builder pattern
//!     let result = nl.predict("species")
//!         .from("iris")
//!         .features(&["petal_length", "petal_width"])
//!         .epochs(30)
//!         .train()
//!         .await?;
//!
//!     // Generate PyTorch code
//!     let code = nl.pytorch("Predict species from iris").await?;
//!     println!("{}", code);
//!
//!     Ok(())
//! }
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

pub struct NeuroLang {
    base_url: String,
    client: reqwest::Client,
}

#[derive(Debug, Deserialize)]
pub struct TrainResult {
    pub accuracy: Option<f64>,
    pub mse: Option<f64>,
    #[serde(rename = "finalLoss")]
    pub final_loss: f64,
    pub epochs: u32,
    pub code: String,
    pub predictions: Vec<HashMap<String, serde_json::Value>>,
}

#[derive(Debug, Deserialize)]
pub struct CompileResult {
    pub code: String,
    pub target: String,
    pub ir: serde_json::Value,
}

#[derive(Serialize)]
struct RunPayload {
    source: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    data: Option<Vec<HashMap<String, f64>>>,
}

#[derive(Serialize)]
struct CompilePayload {
    source: String,
    target: String,
}

impl NeuroLang {
    pub fn new(base_url: &str) -> Self {
        Self {
            base_url: base_url.to_string(),
            client: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(300))
                .build()
                .expect("Failed to create HTTP client"),
        }
    }

    pub async fn run(&self, source: &str) -> Result<TrainResult, Box<dyn std::error::Error>> {
        let payload = RunPayload {
            source: source.to_string(),
            data: None,
        };
        let resp = self.client
            .post(format!("{}/run", self.base_url))
            .json(&payload)
            .send()
            .await?;

        if !resp.status().is_success() {
            let text = resp.text().await?;
            return Err(format!("NeuroLang API error: {}", text).into());
        }

        Ok(resp.json().await?)
    }

    pub async fn compile(
        &self,
        source: &str,
        target: &str,
    ) -> Result<CompileResult, Box<dyn std::error::Error>> {
        let payload = CompilePayload {
            source: source.to_string(),
            target: target.to_string(),
        };
        let resp = self.client
            .post(format!("{}/compile", self.base_url))
            .json(&payload)
            .send()
            .await?;

        if !resp.status().is_success() {
            let text = resp.text().await?;
            return Err(format!("NeuroLang API error: {}", text).into());
        }

        Ok(resp.json().await?)
    }

    pub async fn pytorch(&self, source: &str) -> Result<String, Box<dyn std::error::Error>> {
        Ok(self.compile(source, "pytorch").await?.code)
    }

    pub async fn keras(&self, source: &str) -> Result<String, Box<dyn std::error::Error>> {
        Ok(self.compile(source, "keras").await?.code)
    }

    pub async fn jax(&self, source: &str) -> Result<String, Box<dyn std::error::Error>> {
        Ok(self.compile(source, "jax").await?.code)
    }

    /// Start building a model with the builder pattern.
    pub fn predict(&self, target: &str) -> ModelBuilder {
        ModelBuilder {
            client: self,
            target: target.to_string(),
            dataset: None,
            features: Vec::new(),
            epochs: None,
            batch_size: None,
            learning_rate: None,
            learn_mode: None,
            data: None,
        }
    }
}

pub struct ModelBuilder<'a> {
    client: &'a NeuroLang,
    target: String,
    dataset: Option<String>,
    features: Vec<String>,
    epochs: Option<u32>,
    batch_size: Option<u32>,
    learning_rate: Option<f64>,
    learn_mode: Option<String>,
    data: Option<Vec<HashMap<String, f64>>>,
}

impl<'a> ModelBuilder<'a> {
    pub fn from(mut self, dataset: &str) -> Self {
        self.dataset = Some(dataset.to_string());
        self
    }

    pub fn features(mut self, features: &[&str]) -> Self {
        self.features = features.iter().map(|s| s.to_string()).collect();
        self
    }

    pub fn epochs(mut self, n: u32) -> Self {
        self.epochs = Some(n);
        self
    }

    pub fn batch_size(mut self, n: u32) -> Self {
        self.batch_size = Some(n);
        self
    }

    pub fn learning_rate(mut self, lr: f64) -> Self {
        self.learning_rate = Some(lr);
        self
    }

    pub fn deep(mut self) -> Self {
        self.learn_mode = Some("deep".to_string());
        self
    }

    pub fn linear(mut self) -> Self {
        self.learn_mode = Some("linear".to_string());
        self
    }

    pub fn with_data(mut self, data: Vec<HashMap<String, f64>>) -> Self {
        self.data = Some(data);
        self
    }

    pub fn build_source(&self) -> String {
        let mut lines = vec![format!("predict {}", self.target)];
        if !self.features.is_empty() {
            lines.push(format!("inputs {}", self.features.join(" ")));
        }
        if let Some(ref ds) = self.dataset {
            lines.push(format!("dataset {}", ds));
        }
        if let Some(e) = self.epochs {
            lines.push(format!("epochs {}", e));
        }
        if let Some(bs) = self.batch_size {
            lines.push(format!("batch_size {}", bs));
        }
        if let Some(lr) = self.learning_rate {
            lines.push(format!("learning_rate {}", lr));
        }
        if let Some(ref mode) = self.learn_mode {
            lines.push(format!("learn {}", mode));
        }
        lines.join("\n")
    }

    pub async fn train(self) -> Result<TrainResult, Box<dyn std::error::Error>> {
        let payload = RunPayload {
            source: self.build_source(),
            data: self.data,
        };
        let resp = self.client.client
            .post(format!("{}/run", self.client.base_url))
            .json(&payload)
            .send()
            .await?;

        if !resp.status().is_success() {
            let text = resp.text().await?;
            return Err(format!("NeuroLang API error: {}", text).into());
        }

        Ok(resp.json().await?)
    }
}
