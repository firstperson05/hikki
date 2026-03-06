use crate::model::embedding::Embedding;
use crate::model::rwkv_block::channel_mix;
use crate::model::ssm::selective_scan;
use crate::tensor::autograd::{Node, NodeRef};
use crate::tensor::Tensor;
use rayon::prelude::*;
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

pub struct FastMindConfig {
    pub vocab_size: usize,
    pub n_layer: usize,
    pub n_embd: usize,
    pub seq_len: usize,
}

pub struct FastMindLM {
    pub config: FastMindConfig,
    pub embedding: Embedding,
    pub blocks: Vec<RWKVSSMBlock>,
    pub head: NodeRef, // Output projection weight
}

pub struct RWKVSSMBlock {
    pub ln1: (f32, f32), // Simplified LN params
    pub mixing: RWKVMixing,
    pub ssm: SSMParams,
}

pub struct RWKVMixing {
    pub time_mix_weight: Tensor,
    pub r_weight: NodeRef,
    pub k_weight: NodeRef,
    pub v_weight: NodeRef,
}

pub struct SSMParams {
    pub a_weight: Vec<f32>,
    pub b_weight: Vec<f32>,
}

impl FastMindLM {
    /// Estimate parameter count for model creation
    pub fn estimate_param_count(config: &FastMindConfig) -> usize {
        let embed_params = config.vocab_size * config.n_embd;
        let head_params = config.n_embd * config.vocab_size;

        // Each RWKV-SSM block has:
        // - time_mix_weight: n_embd
        // - r_weight: n_embd * n_embd
        // - k_weight: n_embd * n_embd
        // - v_weight: n_embd * n_embd
        // - a_weight: n_embd (SSM)
        // - b_weight: n_embd (SSM)
        let block_params = config.n_embd + 3 * (config.n_embd * config.n_embd) + 2 * config.n_embd;
        let all_blocks_params = config.n_layer * block_params;

        embed_params + head_params + all_blocks_params
    }

    pub fn new(config: FastMindConfig) -> Self {
        println!(
            "Creating model: {} layers, {} dim, vocab {} | {}M parameters",
            config.n_layer,
            config.n_embd,
            config.vocab_size,
            Self::estimate_param_count(&config) / 1_000_000
        );

        let embedding = Embedding::new(config.vocab_size, config.n_embd);
        let mut blocks = Vec::new();
        for _ in 0..config.n_layer {
            blocks.push(RWKVSSMBlock {
                ln1: (1.0, 0.0),
                mixing: RWKVMixing {
                    time_mix_weight: Tensor::new(vec![config.n_embd], vec![0.5; config.n_embd])
                        .unwrap(),
                    r_weight: Node::new_leaf(
                        Tensor::new(
                            vec![config.n_embd, config.n_embd],
                            vec![0.01; config.n_embd * config.n_embd],
                        )
                        .unwrap(),
                        true,
                    ),
                    k_weight: Node::new_leaf(
                        Tensor::new(
                            vec![config.n_embd, config.n_embd],
                            vec![0.01; config.n_embd * config.n_embd],
                        )
                        .unwrap(),
                        true,
                    ),
                    v_weight: Node::new_leaf(
                        Tensor::new(
                            vec![config.n_embd, config.n_embd],
                            vec![0.01; config.n_embd * config.n_embd],
                        )
                        .unwrap(),
                        true,
                    ),
                },
                ssm: SSMParams {
                    a_weight: vec![0.9; config.n_embd],
                    b_weight: vec![0.1; config.n_embd],
                },
            });
        }

        // Output head
        let head = Node::new_leaf(
            Tensor::new(
                vec![config.n_embd, config.vocab_size],
                vec![0.01; config.n_embd * config.vocab_size],
            )
            .unwrap(),
            true,
        );

        Self {
            config,
            embedding,
            blocks,
            head,
        }
    }

    pub fn forward(
        &self,
        input_ids: &[u32],
        last_state: Option<Vec<Tensor>>,
    ) -> Result<(NodeRef, Vec<Tensor>), String> {
        let mut x = self.embedding.forward(input_ids)?;
        let mut new_states = Vec::new();

        let seq_len = self.config.seq_len;
        let batch_size = input_ids.len() / seq_len;
        let dim = self.config.n_embd;

        for (layer_idx, block) in self.blocks.iter().enumerate() {
            // 1. LayerNorm
            let x_ln_val = {
                let x_lock = x.lock().unwrap();
                x_lock.value.layernorm(1e-5)?
            };

            // 2. Time Mixing (RWKV-style exponential moving average)
            let time_mixed_val = if let Some(ref prev_states) = last_state {
                let prev_state = &prev_states[layer_idx];
                let mut time_mixed = vec![0.0; x_ln_val.data.len()];

                time_mixed
                    .par_chunks_mut(seq_len * dim)
                    .enumerate()
                    .for_each(|(b, batch_out)| {
                        for t in 0..seq_len {
                            let token_start = t * dim;
                            let token_end = token_start + dim;

                            let current_token = &x_ln_val.data
                                [(b * seq_len + t) * dim..(b * seq_len + t) * dim + dim];
                            let prev_token = &prev_state.data[t * dim..(t + 1) * dim];

                            let out_slice = &mut batch_out[token_start..token_end];
                            for d in 0..dim {
                                let w = block.mixing.time_mix_weight.data[d];
                                out_slice[d] = w * current_token[d] + (1.0 - w) * prev_token[d];
                            }
                        }
                    });
                Tensor::new(x_ln_val.shape.clone(), time_mixed)?
            } else {
                // No previous state, use current tokens directly
                x_ln_val.clone()
            };

            // 3. Channel Mix (RWKV-style gated MLP)
            let channel_mixed_val = channel_mix(
                &time_mixed_val,
                &block.mixing.r_weight.lock().unwrap().value,
                &block.mixing.k_weight.lock().unwrap().value,
                &block.mixing.v_weight.lock().unwrap().value,
            )?;
            let channel_mixed_node = Node::new_leaf(channel_mixed_val, true);

            // 4. SSM Selective Scan (Mamba-style)
            let mut ssm_states = vec![0.0; batch_size * seq_len * dim];

            ssm_states
                .par_chunks_mut(seq_len * dim)
                .enumerate()
                .for_each(|(b, batch_out)| {
                    for c in 0..dim {
                        let mut a_seq = Vec::with_capacity(seq_len);
                        let mut b_seq = Vec::with_capacity(seq_len);

                        for t in 0..seq_len {
                            let idx = (b * seq_len + t) * dim + c;
                            a_seq.push(block.ssm.a_weight[c]);
                            b_seq.push(x_ln_val.data[idx]);
                        }

                        let h_init = last_state
                            .as_ref()
                            .and_then(|states| states.get(layer_idx))
                            .map(|state| state.data[c])
                            .unwrap_or(0.0);

                        let scanned = selective_scan(&a_seq, &b_seq, h_init);
                        for t in 0..seq_len {
                            batch_out[t * dim + c] = scanned[t];
                        }
                    }
                });
            let ssm_out_val = Tensor::new(x_ln_val.shape.clone(), ssm_states)?;
            let ssm_out_val_clone = ssm_out_val.clone();
            let ssm_out_node = Node::new_leaf(ssm_out_val, false); // SSM weights not trained for now

            // 5. Residual connections
            let x_new = Node::add(&x, &channel_mixed_node)?;
            let x_final = Node::add(&x_new, &ssm_out_node)?;
            x = x_final;

            new_states.push(ssm_out_val_clone);
        }

        // Output head
        let logits = Node::matmul(&x, &self.head)?;

        Ok((logits, new_states))
    }

    pub fn step(
        &self,
        token_id: u32,
        states: &mut [Tensor], // One per layer
    ) -> Result<Tensor, String> {
        // 1. Embedding
        let weight_data = {
            let weight_lock = self.embedding.weight.lock().unwrap();
            weight_lock.value.data.clone()
        };
        let start = token_id as usize * self.config.n_embd;
        let end = start + self.config.n_embd;
        let mut x_final = weight_data[start..end].to_vec();

        for (i, block) in self.blocks.iter().enumerate() {
            let dim = self.config.n_embd;
            // LayerNorm (simplified)
            let x_tensor = Tensor::new(vec![1, self.config.n_embd], x_final.clone())?;
            let x_ln = x_tensor.layernorm(1e-5)?;

            // Channel Mix (Recurrent)
            // r = sigmoid(x @ r_weight)
            let r_w = {
                let r_lock = block.mixing.r_weight.lock().unwrap();
                r_lock.value.clone()
            };
            let mut r = x_ln.matmul(&r_w)?;
            for val in r.data.iter_mut() {
                *val = 1.0 / (1.0 + (-*val).exp());
            }

            // k = relu(x @ k_weight)^2
            let k_w = {
                let k_lock = block.mixing.k_weight.lock().unwrap();
                k_lock.value.clone()
            };
            let k = x_ln.matmul(&k_w)?.relu();
            let mut k_sq = vec![0.0; k.data.len()];
            for j in 0..k.data.len() {
                k_sq[j] = k.data[j] * k.data[j];
            }
            let k_tensor = Tensor::new(k.shape, k_sq)?;

            // v = k @ v_weight
            let v_w = {
                let v_lock = block.mixing.v_weight.lock().unwrap();
                v_lock.value.clone()
            };
            let v = k_tensor.matmul(&v_w)?;

            // Mix result
            let mut mix_res = vec![0.0; dim];
            for j in 0..dim {
                mix_res[j] = r.data[j] * v.data[j];
            }

            // SSM Step
            let h_prev = &states[i].data;
            let mut h_next = vec![0.0; self.config.n_embd];
            for j in 0..self.config.n_embd {
                let a = block.ssm.a_weight[j];
                h_next[j] = a * h_prev[j] + x_ln.data[j];
            }
            states[i].data = h_next.clone();

            // Residual
            for j in 0..self.config.n_embd {
                x_final[j] += mix_res[j] + h_next[j];
            }
        }

        // Output head
        let head_weight = {
            let head_lock = self.head.lock().unwrap();
            head_lock.value.clone()
        };
        let x_tensor_final = Tensor::new(vec![1, self.config.n_embd], x_final)?;
        let logits = x_tensor_final.matmul(&head_weight)?;

        Ok(logits)
    }

    pub fn initial_state(&self) -> Vec<Tensor> {
        (0..self.config.n_layer)
            .map(|_| Tensor::zeros(vec![1, self.config.n_embd]))
            .collect()
    }

    pub fn load_checkpoint(&self, path: &Path) -> Result<(), String> {
        let mut file = BufReader::new(File::open(path).map_err(|e| e.to_string())?);
        let mut magic = [0u8; 4];
        file.read_exact(&mut magic).map_err(|e| e.to_string())?;
        if &magic != b"FMLM" {
            return Err("Invalid checkpoint format".to_string());
        }
        let mut u32_buf = [0u8; 4];
        let mut u64_buf = [0u8; 8];
        let mut f32_buf = [0u8; 4];
        file.read_exact(&mut u32_buf).map_err(|e| e.to_string())?;
        file.read_exact(&mut u64_buf).map_err(|e| e.to_string())?;
        file.read_exact(&mut u32_buf).map_err(|e| e.to_string())?;
        let num_params = u32::from_le_bytes(u32_buf);
        let mut param_nodes = Vec::new();
        param_nodes.push(&self.embedding.weight);
        for block in &self.blocks {
            param_nodes.push(&block.mixing.r_weight);
            param_nodes.push(&block.mixing.k_weight);
            param_nodes.push(&block.mixing.v_weight);
        }
        param_nodes.push(&self.head);
        if num_params as usize != param_nodes.len() {
            return Err("Checkpoint size mismatch".to_string());
        }
        for node in param_nodes {
            file.read_exact(&mut u32_buf).map_err(|e| e.to_string())?;
            let len = u32::from_le_bytes(u32_buf) as usize;
            let mut node_lock = node.lock().unwrap();
            for i in 0..len {
                file.read_exact(&mut f32_buf).map_err(|e| e.to_string())?;
                node_lock.value.data[i] = f32::from_le_bytes(f32_buf);
            }
        }
        Ok(())
    }
}
