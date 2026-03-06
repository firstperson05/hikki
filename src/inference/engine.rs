use crate::inference::sampler::Sampler;
use crate::model::lm::HikkiLM;
use crate::tensor::Tensor;
use crate::tokenizer::bpe::BpeTokenizer;

#[derive(Clone)]
pub struct InferenceConfig {
    pub max_new_tokens: usize,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    pub repetition_penalty: f32,
    pub repetition_window: usize,
    pub min_new_tokens: usize,
    pub use_beam_search: bool,
    pub beam_width: usize,
    pub max_sentences: usize, // NEW: stop after N sentences
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            max_new_tokens: 128,
            temperature: 0.5,        // Better for small models: less random
            top_p: 0.85,             // More focused sampling
            top_k: 20,               // More focused for undertrained models
            repetition_penalty: 1.5, // Stronger penalty for coherence
            repetition_window: 64,   // Larger context window
            min_new_tokens: 5,       // Prevent immediate EOS
            use_beam_search: false,
            beam_width: 4,
            max_sentences: 3, // Stop after 3 sentences
        }
    }
}

impl InferenceConfig {
    pub fn parse_config_string(&mut self, config_str: &str) -> Result<(), String> {
        for pair in config_str.split_whitespace() {
            if let Some((key, value)) = pair.split_once('=') {
                match key {
                    "temperature" => {
                        self.temperature = value.parse().map_err(|_| "Invalid temperature")?
                    }
                    "top_p" => self.top_p = value.parse().map_err(|_| "Invalid top_p")?,
                    "top_k" => self.top_k = value.parse().map_err(|_| "Invalid top_k")?,
                    "repetition_penalty" => {
                        self.repetition_penalty =
                            value.parse().map_err(|_| "Invalid repetition_penalty")?
                    }
                    "max_new_tokens" => {
                        self.max_new_tokens = value.parse().map_err(|_| "Invalid max_new_tokens")?
                    }
                    "max_sentences" => {
                        self.max_sentences = value.parse().map_err(|_| "Invalid max_sentences")?
                    }
                    _ => return Err(format!("Unknown config key: {}", key)),
                }
            }
        }
        Ok(())
    }

    pub fn print_current(&self) {
        println!("Current InferenceConfig:");
        println!("  temperature: {}", self.temperature);
        println!("  top_p: {}", self.top_p);
        println!("  top_k: {}", self.top_k);
        println!("  repetition_penalty: {}", self.repetition_penalty);
        println!("  max_new_tokens: {}", self.max_new_tokens);
        println!("  min_new_tokens: {}", self.min_new_tokens);
        println!("  max_sentences: {}", self.max_sentences);
    }
}

pub struct InferenceEngine {
    pub model: HikkiLM,
    pub tokenizer: BpeTokenizer,
    pub state: Vec<Tensor>,
    pub sampler: Sampler,
    pub recent_tokens: Vec<u32>,
    pub context_window: Vec<u32>, // NEW: sliding context for coherence
    pub max_context_size: usize,  // NEW: max tokens in context window
}

impl InferenceEngine {
    pub fn new(model: HikkiLM, tokenizer: BpeTokenizer, seed: u64) -> Self {
        let state = model.initial_state();
        Self {
            model,
            tokenizer,
            state,
            sampler: Sampler::new(seed),
            recent_tokens: Vec::new(),
            context_window: Vec::new(),
            max_context_size: 512, // Keep last 512 tokens for context
        }
    }

    pub fn reset(&mut self) {
        self.state = self.model.initial_state();
        self.recent_tokens.clear();
        self.context_window.clear();
    }

    /// Warm up the recurrent state with prompt tokens (no sampling)
    pub fn warm_up_state(&mut self, prompt_tokens: &[u32]) -> Result<(), String> {
        if prompt_tokens.len() > 10 {
            println!("Processing prompt... [{} tokens]", prompt_tokens.len());
        }

        for &token in prompt_tokens {
            let _logits = self.model.step(token, &mut self.state)?;
            self.update_context_window(token);
        }

        Ok(())
    }

    /// Update sliding context window for coherence
    fn update_context_window(&mut self, token: u32) {
        self.context_window.push(token);
        if self.context_window.len() > self.max_context_size {
            self.context_window.remove(0);
        }
    }

    /// Check if we've reached sentence boundary
    fn is_sentence_boundary(&self, token: u32) -> bool {
        let token_str = self.tokenizer.decode(&[token]);
        token_str.ends_with('.') || token_str.ends_with('!') || token_str.ends_with('?')
    }

    pub fn generate(&mut self, prompt: &str, config: &InferenceConfig) -> Result<String, String> {
        let mut result = String::new();
        let mut utf8_buffer = Vec::new();
        let tokens = self.tokenizer.encode(prompt);

        // 1. Warm up state with prompt (NEW: proper warmup)
        self.warm_up_state(&tokens)?;

        // Get last token for generation start
        let mut last_token = if tokens.is_empty() {
            0
        } else {
            *tokens.last().unwrap()
        };

        // 2. Generate new tokens with sentence awareness
        let mut tokens_generated = 0;
        let mut sentences_completed = 0;

        for _ in 0..config.max_new_tokens {
            let logits_tensor = self.model.step(last_token, &mut self.state)?;
            let mut logits = logits_tensor.data;

            // Apply repetition penalty
            Sampler::apply_repetition_penalty(
                &mut logits,
                &self.recent_tokens,
                config.repetition_penalty,
            );

            // Sample
            let next_token = if config.use_beam_search || config.temperature == 0.0 {
                self.sampler.greedy(&logits)
            } else if config.top_p < 1.0 {
                self.sampler
                    .top_p(&logits, config.top_p, config.temperature)
            } else if config.top_k < 50000 {
                self.sampler
                    .top_k(&logits, config.top_k, config.temperature)
            } else {
                self.sampler.greedy(&logits)
            };

            // Check for EOS but respect min_new_tokens
            if next_token == self.tokenizer.vocab.token_to_id("<EOS>") || next_token == 0 {
                if tokens_generated >= config.min_new_tokens {
                    break;
                } else {
                    continue; // Skip EOS and try again
                }
            }

            let piece = self.tokenizer.decode(&[next_token]);

            // Handle UTF-8 properly - accumulate bytes and only add valid UTF-8 sequences
            utf8_buffer.extend_from_slice(piece.as_bytes());

            // Try to decode what we have so far
            if let Ok(valid_str) = String::from_utf8(utf8_buffer.clone()) {
                result.push_str(&valid_str);
                utf8_buffer.clear();
            } else {
                // Keep the buffer for next iteration - maybe we need more bytes
                // But if buffer gets too large, clear it to prevent memory issues
                if utf8_buffer.len() > 16 {
                    utf8_buffer.clear();
                }
            }

            // NEW: Check sentence boundaries for stopping
            if self.is_sentence_boundary(next_token) {
                sentences_completed += 1;
                if sentences_completed >= config.max_sentences
                    && tokens_generated >= config.min_new_tokens
                {
                    break;
                }
            }

            last_token = next_token;
            self.update_history(next_token, config.repetition_window);
            self.update_context_window(next_token);
            tokens_generated += 1;
        }

        // Add any remaining valid UTF-8 bytes
        if let Ok(valid_str) = String::from_utf8(utf8_buffer) {
            result.push_str(&valid_str);
        }

        Ok(result)
    }

    pub fn generate_streaming<F>(
        &mut self,
        prompt: &str,
        config: &InferenceConfig,
        callback: F,
    ) -> Result<(), String>
    where
        F: Fn(&str),
    {
        let mut utf8_buffer = Vec::new();
        let tokens = self.tokenizer.encode(prompt);

        let mut last_token = 0;
        for &t in &tokens {
            self.model.step(t, &mut self.state)?;
            last_token = t;
            self.update_history(t, config.repetition_window);
        }

        let mut tokens_generated = 0;
        for _ in 0..config.max_new_tokens {
            let logits_tensor = self.model.step(last_token, &mut self.state)?;
            let mut logits = logits_tensor.data;

            Sampler::apply_repetition_penalty(
                &mut logits,
                &self.recent_tokens,
                config.repetition_penalty,
            );

            let next_token = if config.use_beam_search || config.temperature == 0.0 {
                self.sampler.greedy(&logits)
            } else if config.top_p < 1.0 {
                self.sampler
                    .top_p(&logits, config.top_p, config.temperature)
            } else if config.top_k < 50000 {
                self.sampler
                    .top_k(&logits, config.top_k, config.temperature)
            } else {
                self.sampler.greedy(&logits)
            };

            if next_token == self.tokenizer.vocab.token_to_id("<EOS>") || next_token == 0 {
                if tokens_generated >= config.min_new_tokens {
                    break;
                } else {
                    continue;
                }
            }

            let piece = self.tokenizer.decode(&[next_token]);

            // UTF-8 handling for streaming
            utf8_buffer.extend_from_slice(piece.as_bytes());

            if let Ok(valid_str) = String::from_utf8(utf8_buffer.clone()) {
                callback(&valid_str);
                utf8_buffer.clear();
            } else {
                if utf8_buffer.len() > 16 {
                    utf8_buffer.clear();
                }
            }

            last_token = next_token;
            self.update_history(next_token, config.repetition_window);
            tokens_generated += 1;
        }

        // Flush any remaining valid UTF-8
        if let Ok(valid_str) = String::from_utf8(utf8_buffer) {
            callback(&valid_str);
        }

        Ok(())
    }

    pub fn generate_beam_search(
        &mut self,
        prompt: &str,
        max_tokens: usize,
        _beam_width: usize,
    ) -> Result<String, String> {
        // For now, use greedy generation as beam search needs more complex state management
        // TODO: Implement proper beam search with state tracking per beam
        let mut config = InferenceConfig::default();
        config.temperature = 0.0; // Greedy
        config.max_new_tokens = max_tokens;
        self.generate(prompt, &config)
    }

    fn update_history(&mut self, token: u32, window: usize) {
        self.recent_tokens.push(token);
        if self.recent_tokens.len() > window {
            self.recent_tokens.remove(0);
        }
    }
}
