pub struct Sampler {
    pub seed: u64,
}

impl Sampler {
    pub fn new(seed: u64) -> Self {
        Self { seed }
    }

    /// Xorshift64 PRNG
    fn next_u64(&mut self) -> u64 {
        self.seed ^= self.seed << 13;
        self.seed ^= self.seed >> 7;
        self.seed ^= self.seed << 17;
        self.seed
    }

    fn next_f32(&mut self) -> f32 {
        (self.next_u64() as f32) / (u64::MAX as f32)
    }

    pub fn greedy(&self, logits: &[f32]) -> u32 {
        let cleaned_logits = self.clean_logits(logits);
        cleaned_logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i as u32)
            .unwrap_or(0)
    }

    /// Clean logits by handling NaN/Inf and extreme values
    fn clean_logits(&self, logits: &[f32]) -> Vec<f32> {
        let mut cleaned = Vec::with_capacity(logits.len());
        let mut has_issues = false;
        
        for &logit in logits {
            if logit.is_nan() || logit.is_infinite() {
                cleaned.push(0.0); // Replace NaN/Inf with 0
                has_issues = true;
            } else if logit > 100.0 {
                cleaned.push(100.0); // Clamp extreme positive values
                has_issues = true;
            } else if logit < -100.0 {
                cleaned.push(-100.0); // Clamp extreme negative values
                has_issues = true;
            } else {
                cleaned.push(logit);
            }
        }
        
        if has_issues {
            eprintln!("Warning: Cleaned problematic logits (NaN/Inf/extreme values)");
        }
        
        // If all logits are the same (common in untrained models), add small noise
        if cleaned.len() > 0 {
            let first = cleaned[0];
            let all_same = cleaned.iter().all(|&x| (x - first).abs() < 1e-6);
            if all_same {
                let len = cleaned.len() as f32;
                for (idx, val) in cleaned.iter_mut().enumerate() {
                    *val += (idx as f32 - len / 2.0) * 1e-3;
                }
            }
        }
        
        cleaned
    }

    pub fn apply_repetition_penalty(logits: &mut [f32], recent_tokens: &[u32], penalty: f32) {
        for &token in recent_tokens {
            let idx = token as usize;
            if idx < logits.len() {
                if logits[idx] > 0.0 {
                    logits[idx] /= penalty;
                } else {
                    logits[idx] *= penalty;
                }
            }
        }
    }

    pub fn top_k(&mut self, logits: &[f32], k: usize, temp: f32) -> u32 {
        let cleaned_logits = self.clean_logits(logits);
        let mut items: Vec<(usize, f32)> = cleaned_logits.iter().copied().enumerate().collect();

        // Apply temperature
        if temp != 1.0 {
            for item in &mut items {
                item.1 /= temp;
            }
        }

        // Sort by value descending
        items.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Keep only top k
        let k = k.min(items.len());
        items.truncate(k);

        self.softmax_sample(&items)
    }

    pub fn top_p(&mut self, logits: &[f32], p: f32, temp: f32) -> u32 {
        let cleaned_logits = self.clean_logits(logits);
        let mut items: Vec<(usize, f32)> = cleaned_logits.iter().copied().enumerate().collect();

        // Apply temperature
        if temp != 1.0 {
            for item in &mut items {
                item.1 /= temp;
            }
        }

        // Sort by value descending
        items.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Numerical stability: softmax first (conceptually) to get probabilities
        let max_val = items[0].1;
        let mut sum_exp = 0.0;
        for item in &mut items {
            item.1 = (item.1 - max_val).exp();
            sum_exp += item.1;
        }
        
        // Check for numerical issues
        if sum_exp == 0.0 || sum_exp.is_nan() || sum_exp.is_infinite() {
            eprintln!("Warning: Invalid softmax sum {}, falling back to uniform", sum_exp);
            // Fallback to uniform over top-k tokens
            let k = items.len().min(40);
            items.truncate(k);
            for (_i, item) in items.iter_mut().enumerate() {
                item.1 = 1.0 / k as f32;
            }
        } else {
            for item in &mut items {
                item.1 /= sum_exp;
            }
        }

        // Verify probabilities sum to 1.0 (within tolerance)
        let prob_sum: f32 = items.iter().map(|item| item.1).sum();
        if (prob_sum - 1.0).abs() > 1e-3 {
            eprintln!("Warning: Probabilities sum to {}, renormalizing", prob_sum);
            for item in &mut items {
                item.1 /= prob_sum;
            }
        }

        // Ensure all probabilities are non-negative
        for item in &mut items {
            if item.1 < 0.0 {
                item.1 = 0.0;
            }
        }

        // Cumulative sum for top-p
        let mut cumulative_prob = 0.0;
        let mut cutoff = items.len();
        for (idx, item) in items.iter().enumerate() {
            cumulative_prob += item.1;
            if cumulative_prob >= p {
                cutoff = idx + 1;
                break;
            }
        }
        items.truncate(cutoff);

        // Re-normalize for sampling
        let new_sum: f32 = items.iter().map(|item| item.1).sum();
        if new_sum > 0.0 {
            for item in &mut items {
                item.1 /= new_sum;
            }
        } else {
            // Fallback to uniform
            let len = items.len() as f32;
            for item in &mut items {
                item.1 = 1.0 / len;
            }
        }

        self.categorical_sample(&items)
    }

    fn softmax_sample(&mut self, items: &[(usize, f32)]) -> u32 {
        if items.is_empty() {
            return 0;
        }

        let mut probs: Vec<(usize, f32)> = items.to_vec();
        let max_val = probs
            .iter()
            .map(|item| item.1)
            .fold(f32::NEG_INFINITY, f32::max);
        let mut sum_exp = 0.0;
        for item in &mut probs {
            item.1 = (item.1 - max_val).exp();
            sum_exp += item.1;
        }
        
        if sum_exp == 0.0 {
            // Fallback to uniform
            let len = probs.len() as f32;
            for item in &mut probs {
                item.1 = 1.0 / len;
            }
        } else {
            for item in &mut probs {
                item.1 /= sum_exp;
            }
        }

        self.categorical_sample(&probs)
    }

    fn categorical_sample(&mut self, probs: &[(usize, f32)]) -> u32 {
        let r = self.next_f32();
        let mut cumulative = 0.0;
        for item in probs {
            cumulative += item.1;
            if r <= cumulative {
                return item.0 as u32;
            }
        }
        probs.last().map(|item| item.0 as u32).unwrap_or(0)
    }

    /// Beam search for more coherent generation from small models
    pub fn beam_search(&self, logits_fn: impl Fn(u32) -> Vec<f32>, width: usize, max_tokens: usize) -> Vec<u32> {
        let mut beams = vec![(vec![], 0.0)]; // (sequence, score)
        
        for _ in 0..max_tokens {
            let mut new_beams = Vec::new();
            
            for (seq, score) in beams {
                let next_token = if seq.is_empty() {
                    0 // Start token
                } else {
                    *seq.last().unwrap()
                };
                
                let logits = logits_fn(next_token);
                let cleaned_logits = self.clean_logits(&logits);
                
                // Get top candidates
                let mut candidates: Vec<(usize, f32)> = cleaned_logits.iter().copied().enumerate().collect();
                candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                candidates.truncate(width);
                
                for (token_id, logit) in candidates.iter().take(width) {
                    let mut new_seq = seq.clone();
                    new_seq.push(*token_id as u32);
                    let new_score = score + logit;
                    new_beams.push((new_seq, new_score));
                }
            }
            
            // Keep only top beams
            new_beams.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            new_beams.truncate(width);
            beams = new_beams;
        }
        
        // Return best beam
        beams.into_iter().next().map(|(seq, _)| seq).unwrap_or_default()
    }
}
