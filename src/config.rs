use serde::Deserialize;

#[derive(Debug, Deserialize, Clone)]
pub struct ModelConfig {
    pub d_model: usize,
    pub n_layer: usize,
    pub vocab_size: usize,
}

#[derive(Debug, Deserialize, Clone)]
pub struct TrainConfig {
    pub batch_size: usize,
    pub seq_len: usize,
    pub grad_accum_steps: usize,
    pub max_steps: usize,
    pub eval_every: usize,
    pub save_every: usize,
    pub log_every: usize,
    pub checkpoint_dir: String,
    pub clip_grad_norm: f32,
    pub lr: f32,
    pub min_lr: f32,
    pub warmup_steps: usize,
    pub weight_decay: f32,
}

#[derive(Debug, Deserialize, Clone)]
pub struct Config {
    pub model: ModelConfig,
    pub train: TrainConfig,
}
