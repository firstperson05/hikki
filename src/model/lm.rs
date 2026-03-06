use crate::model::embedding::Embedding;
use crate::model::ssm::selective_scan;
use crate::tensor::autograd::{Node, NodeRef};
use crate::tensor::Tensor;
use rayon::prelude::*;
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

#[cfg(feature = "cuda")]
use crate::cuda::CudaDevice;

pub struct HikkiConfig {
    pub vocab_size: usize,
    pub n_layer: usize,
    pub n_embd: usize,
    pub seq_len: usize,
}

pub struct BlockMixing {
    pub r_weight: NodeRef,
    pub k_weight: NodeRef,
    pub v_weight: NodeRef,
    pub time_mix_weight: Tensor,
}

pub struct BlockSSM {
    pub a_weight: Vec<f32>,
}

pub struct RWKVSSMBlock {
    pub mixing: BlockMixing,
    pub ssm: BlockSSM,
}

impl RWKVSSMBlock {
    pub fn new(dim: usize) -> Self {
        let r_weight = Node::new_leaf(Tensor::zeros(vec![dim, dim]), true);
        let k_weight = Node::new_leaf(Tensor::zeros(vec![dim, dim]), true);
        let v_weight = Node::new_leaf(Tensor::zeros(vec![dim, dim]), true);
        let time_mix_weight = Tensor::new(vec![dim], vec![0.5; dim]).unwrap();

        Self {
            mixing: BlockMixing {
                r_weight,
                k_weight,
                v_weight,
                time_mix_weight,
            },
            ssm: BlockSSM {
                a_weight: vec![0.9; dim],
            },
        }
    }
}

pub struct HikkiLM {
    pub config: HikkiConfig,
    pub embedding: Embedding,
    pub blocks: Vec<RWKVSSMBlock>,
    pub head: NodeRef,
}

impl HikkiLM {
    pub fn new(config: HikkiConfig) -> Self {
        let embedding = Embedding::new(config.vocab_size, config.n_embd);
        let blocks = (0..config.n_layer)
            .map(|_| RWKVSSMBlock::new(config.n_embd))
            .collect();
        let head = Node::new_leaf(
            Tensor::new(
                vec![config.n_embd, config.vocab_size],
                vec![0.0; config.n_embd * config.vocab_size],
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
        last_state: Option<&Vec<Tensor>>,
    ) -> Result<(NodeRef, Vec<Tensor>), String> {
        let mut x = self.embedding.forward(input_ids)?;
        let mut new_states = Vec::new();

        let batch_size = 1;
        let seq_len = input_ids.len();
        let dim = self.config.n_embd;

        for (layer_idx, block) in self.blocks.iter().enumerate() {
            let x_ln_val = x.lock().unwrap().value.clone();

            // 1. Time Mix (Mixing with previous token's output layer embedding)
            let time_mixed_val = if let Some(prev_states) = last_state {
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
                x_ln_val.clone()
            };

            // 2. Channel Mix (RWKV-style Gated MLP)
            let r_weight = block.mixing.r_weight.lock().unwrap().value.clone();
            let k_weight = block.mixing.k_weight.lock().unwrap().value.clone();
            let v_weight = block.mixing.v_weight.lock().unwrap().value.clone();

            let mut r = time_mixed_val.matmul(&r_weight)?;
            for v in r.data.iter_mut() {
                *v = 1.0 / (1.0 + (-*v).exp());
            }

            let mut k = time_mixed_val.matmul(&k_weight)?.relu();
            for v in k.data.iter_mut() {
                *v = (*v) * (*v);
            }

            let v = k.matmul(&v_weight)?;
            let mut mixed = vec![0.0; r.data.len()];
            for i in 0..mixed.len() {
                mixed[i] = r.data[i] * v.data[i];
            }
            let channel_mixed_val = Tensor::new(r.shape.clone(), mixed)?;
            let channel_mixed_node = Node::new_leaf(channel_mixed_val, true);

            // 3. SSM Selective Scan
            let mut ssm_states_data = vec![0.0; batch_size * seq_len * dim];
            ssm_states_data
                .par_chunks_mut(seq_len * dim)
                .enumerate()
                .for_each(|(b, batch_out)| {
                    for c in 0..dim {
                        let h_init = last_state
                            .as_ref()
                            .and_then(|st| st.get(layer_idx))
                            .map(|st| st.data[c])
                            .unwrap_or(0.0);

                        let mut a_seq = Vec::with_capacity(seq_len);
                        let mut b_seq = Vec::with_capacity(seq_len);
                        for t in 0..seq_len {
                            let idx = (b * seq_len + t) * dim + c;
                            a_seq.push(block.ssm.a_weight[c]);
                            b_seq.push(x_ln_val.data[idx]);
                        }

                        let scanned = selective_scan(&a_seq, &b_seq, h_init);
                        for t in 0..seq_len {
                            batch_out[t * dim + c] = scanned[t];
                        }
                    }
                });
            let ssm_out_val = Tensor::new(x_ln_val.shape.clone(), ssm_states_data)?;
            new_states.push(ssm_out_val.clone());
            let ssm_out_node = Node::new_leaf(ssm_out_val, false);

            let x_new = Node::add(&x, &channel_mixed_node)?;
            let x_final = Node::add(&x_new, &ssm_out_node)?;
            x = x_final;
        }

        let logits = Node::matmul(&x, &self.head)?;
        Ok((logits, new_states))
    }

    pub fn step(&self, token_id: u32, states: &mut [Tensor]) -> Result<Tensor, String> {
        let (logits_node, next_states) = self.forward(&[token_id], Some(&states.to_vec()))?;
        for (i, ns) in next_states.into_iter().enumerate() {
            states[i] = ns;
        }
        let result = logits_node.lock().unwrap().value.clone();
        Ok(result)
    }

    pub fn initial_state(&self) -> Vec<Tensor> {
        (0..self.config.n_layer)
            .map(|_| Tensor::zeros(vec![1, self.config.n_embd]))
            .collect()
    }

    pub fn estimate_param_count(config: &HikkiConfig) -> usize {
        let n_layer = config.n_layer;
        let d_model = config.n_embd;
        let vocab_size = config.vocab_size;

        let embedding = vocab_size * d_model;
        let mixing = n_layer * (d_model * d_model * 3 + d_model);
        let ssm = n_layer * d_model;
        let head = d_model * vocab_size;

        embedding + mixing + ssm + head
    }

    #[cfg(feature = "cuda")]
    pub fn to_cuda(&self, device: &CudaDevice) {
        {
            let mut w = self.embedding.weight.lock().unwrap();
            let _ = w.value.to_cuda(device);
        }
        for block in &self.blocks {
            {
                let mut w = block.mixing.r_weight.lock().unwrap();
                let _ = w.value.to_cuda(device);
            }
            {
                let mut w = block.mixing.k_weight.lock().unwrap();
                let _ = w.value.to_cuda(device);
            }
            {
                let mut w = block.mixing.v_weight.lock().unwrap();
                let _ = w.value.to_cuda(device);
            }
        }
        {
            let mut w = self.head.lock().unwrap();
            let _ = w.value.to_cuda(device);
        }
    }

    pub fn load_checkpoint(&self, path: &Path) -> Result<(), String> {
        let mut file = BufReader::new(File::open(path).map_err(|e| e.to_string())?);
        let mut magic = [0u8; 4];
        file.read_exact(&mut magic).map_err(|e| e.to_string())?;
        if &magic != b"FMLM" {
            return Err("Invalid format".to_string());
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
            if node_lock.value.data.len() != len {
                return Err("Parameter size mismatch".into());
            }
            for i in 0..len {
                file.read_exact(&mut f32_buf).map_err(|e| e.to_string())?;
                node_lock.value.data[i] = f32::from_le_bytes(f32_buf);
            }
        }
        Ok(())
    }
}
