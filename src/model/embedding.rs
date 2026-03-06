use crate::tensor::autograd::{Node, NodeRef};
use crate::tensor::Tensor;

pub struct Embedding {
    pub weight: NodeRef,
}

impl Embedding {
    pub fn new(vocab_size: usize, embed_dim: usize) -> Self {
        // Initialize weights (e.g., small random values)
        let mut data = vec![0.0; vocab_size * embed_dim];
        for x in data.iter_mut() {
            *x = (rand_f32() - 0.5) * 0.02;
        }
        let weight = Node::new_leaf(
            Tensor::new(vec![vocab_size, embed_dim], data).unwrap(),
            true,
        );
        Self { weight }
    }

    pub fn forward(&self, input_ids: &[u32]) -> Result<NodeRef, String> {
        let weight_lock = self.weight.lock().unwrap();
        let embed_dim = weight_lock.value.shape[1];
        let seq_len = input_ids.len();
        let weight_data = weight_lock.value.data.clone();
        drop(weight_lock);

        let mut out_data = Vec::with_capacity(seq_len * embed_dim);
        for &id in input_ids {
            let start = id as usize * embed_dim;
            let end = start + embed_dim;
            out_data.extend_from_slice(&weight_data[start..end]);
        }

        // Return a leaf node for now, as our autograd doesn't have an 'Embedding' op yet.
        // In a real impl, we'd need an Embedding op for backward pass.
        // For simplicity, we'll implement it as a slice/gather op if we had one.
        // Since we don't, we'll manually handle embedding gradients in the trainer or add an Op.
        Ok(Node::new_leaf(
            Tensor::new(vec![seq_len, embed_dim], out_data)?,
            true,
        ))
    }
}

fn rand_f32() -> f32 {
    static mut SEED: u32 = 42;
    unsafe {
        SEED = SEED.wrapping_mul(1664525).wrapping_add(1013904223);
        (SEED as f32) / (u32::MAX as f32)
    }
}
