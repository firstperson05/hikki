use crate::tensor::Tensor;

/// Sparse Cross Entropy Loss
/// targets: [batch_size, seq_len] (token IDs as f32)
/// logits: [batch_size, seq_len, vocab_size]
pub fn cross_entropy_loss(
    logits: &Tensor,
    targets: &Tensor,
    mask: &Tensor,
) -> Result<(f32, Tensor), String> {
    if logits.shape.len() < 2 || logits.shape.len() > 3 {
        return Err("Logits must be [B, T, V] or [N, V]".to_string());
    }

    let (batch_size, seq_len, vocab_size) = if logits.shape.len() == 3 {
        (logits.shape[0], logits.shape[1], logits.shape[2])
    } else {
        (1, logits.shape[0], logits.shape[1])
    };

    let mut total_loss = 0.0;
    let mut num_elements = 0.0;
    let mut grad_data = vec![0.0; logits.data.len()];

    for b in 0..batch_size {
        for t in 0..seq_len {
            let m = mask.data[b * seq_len + t];
            if m == 0.0 {
                continue;
            }

            let target_id = targets.data[b * seq_len + t] as usize;
            let offset = (b * seq_len + t) * vocab_size;
            let slice = &logits.data[offset..offset + vocab_size];

            // Numerical stability: subtract max
            let max_val = slice.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum_exp = 0.0;
            let mut probs = vec![0.0; vocab_size];

            // SIMD Optimization for Exp (Log-Sum-Exp trick)
            #[cfg(target_arch = "x86_64")]
            {
                if is_x86_feature_detected!("avx2") {
                    // Note: We'd typically use a math library for vectorized exp,
                    // but since we're "from scratch", we'll do a basic fallback or call
                    // a scalar exp in a loop which compiler might vectorize, or use intrinsics
                    // if we had a proper SLEEF-like impl. For now, we'll keep it correct
                    // but structured for future SIMD kernels.
                    for i in 0..vocab_size {
                        let p = (slice[i] - max_val).exp();
                        probs[i] = p;
                        sum_exp += p;
                    }
                } else {
                    for i in 0..vocab_size {
                        let p = (slice[i] - max_val).exp();
                        probs[i] = p;
                        sum_exp += p;
                    }
                }
            }
            #[cfg(not(target_arch = "x86_64"))]
            {
                for i in 0..vocab_size {
                    let p = (slice[i] - max_val).exp();
                    probs[i] = p;
                    sum_exp += p;
                }
            }

            for i in 0..vocab_size {
                probs[i] /= sum_exp;
            }

            let loss = -probs[target_id].ln();
            total_loss += loss;
            num_elements += 1.0;

            // Gradient: probs_i - 1(i == target)
            for i in 0..vocab_size {
                let g = if i == target_id {
                    probs[i] - 1.0
                } else {
                    probs[i]
                };
                grad_data[offset + i] = g;
            }
        }
    }

    let avg_loss = if num_elements > 0.0 {
        total_loss / num_elements
    } else {
        0.0
    };
    // Scale gradient by 1/N
    if num_elements > 0.0 {
        for g in grad_data.iter_mut() {
            *g /= num_elements;
        }
    }

    Ok((avg_loss, Tensor::new(logits.shape.clone(), grad_data)?))
}

pub fn perplexity(loss: f32) -> f32 {
    loss.exp()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Tensor;

    #[test]
    fn test_loss_uniform() {
        let vocab_size = 10;
        let logits = Tensor::new(vec![1, 1, vocab_size], vec![0.0; vocab_size]).unwrap();
        let targets = Tensor::new(vec![1, 1], vec![0.0]).unwrap();
        let mask = Tensor::new(vec![1, 1], vec![1.0]).unwrap();

        let (loss, _) = cross_entropy_loss(&logits, &targets, &mask).unwrap();
        let expected = (vocab_size as f32).ln();
        assert!((loss - expected).abs() < 1e-4);
    }

    #[test]
    fn test_perplexity() {
        let loss = 2.0f32.ln();
        assert_eq!(perplexity(loss), 2.0);
    }
}
