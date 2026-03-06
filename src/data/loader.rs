use crate::data::dataset::TextDataset;
use crate::tensor::Tensor;

pub struct Batch {
    pub inputs: Tensor,
    pub targets: Tensor,
    pub mask: Tensor,
}

pub struct DataLoader {
    pub dataset: TextDataset,
    pub batch_size: usize,
    pub seq_len: usize,
    pub shuffle: bool,
    indices: Vec<usize>,
    current: usize,
}

impl DataLoader {
    pub fn new(dataset: TextDataset, batch_size: usize, seq_len: usize, shuffle: bool) -> Self {
        let num_samples = dataset.num_samples(seq_len);
        let mut indices: Vec<usize> = (0..num_samples).collect();

        if shuffle && !indices.is_empty() {
            Self::fisher_yates_shuffle(&mut indices);
        }

        DataLoader {
            dataset,
            batch_size,
            seq_len,
            shuffle,
            indices,
            current: 0,
        }
    }

    fn fisher_yates_shuffle(indices: &mut [usize]) {
        // A simple weak PRNG to avoid external dependency for shuffle
        let mut seed = 123456789u32;
        let mut lcg = || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            seed
        };

        for i in (1..indices.len()).rev() {
            let j = (lcg() as usize) % (i + 1);
            indices.swap(i, j);
        }
    }

    pub fn reset(&mut self) {
        self.current = 0;
        if self.shuffle && !self.indices.is_empty() {
            Self::fisher_yates_shuffle(&mut self.indices);
        }
    }
}

impl Iterator for DataLoader {
    type Item = Batch;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current >= self.indices.len() {
            return None;
        }

        let end = std::cmp::min(self.current + self.batch_size, self.indices.len());
        let actual_batch_size = end - self.current;

        let mut inputs_data = Vec::with_capacity(actual_batch_size * self.seq_len);
        let mut targets_data = Vec::with_capacity(actual_batch_size * self.seq_len);
        let mut mask_data = Vec::with_capacity(actual_batch_size * self.seq_len);

        for i in self.current..end {
            let idx = self.indices[i];
            let (input_tokens, target_tokens) = self.dataset.get_sample(idx, self.seq_len);

            for &t in &input_tokens {
                inputs_data.push(t as f32);
                if t == 0 {
                    // PAD
                    mask_data.push(0.0);
                } else {
                    mask_data.push(1.0);
                }
            }

            for &t in &target_tokens {
                targets_data.push(t as f32);
            }
        }

        self.current = end;

        Some(Batch {
            inputs: Tensor::new(vec![actual_batch_size, self.seq_len], inputs_data).unwrap(),
            targets: Tensor::new(vec![actual_batch_size, self.seq_len], targets_data).unwrap(),
            mask: Tensor::new(vec![actual_batch_size, self.seq_len], mask_data).unwrap(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenizer::bpe::BpeTokenizer;
    use std::fs;
    use std::path::PathBuf;

    #[test]
    fn test_dataloader() {
        let corpus = "word1 word2 word3 word4 word5 word6 word7 word8 word9 word10";
        let bpe = BpeTokenizer::train(corpus, 300);
        let path = PathBuf::from("test_loader.txt");
        fs::write(&path, corpus).unwrap();

        let ds = TextDataset::from_file(&path, &bpe).unwrap();
        // Suppose sequence length is 3. We have ~10 tokens.
        // Sample 0: 0,1,2. Sample 1: 3,4,5. Sample 2: 6,7,8. Sample 3: 9 (pad).
        // Total 4 samples.
        let mut loader = DataLoader::new(ds, 2, 3, false);

        let batch1 = loader.next().unwrap();
        assert_eq!(batch1.inputs.shape, vec![2, 3]);
        assert_eq!(batch1.targets.shape, vec![2, 3]);
        assert_eq!(batch1.mask.shape, vec![2, 3]);

        let batch2 = loader.next().unwrap();
        assert_eq!(batch2.inputs.shape, vec![2, 3]);
        let batch3 = loader.next();
        assert!(batch3.is_none());

        fs::remove_file("test_loader.txt").unwrap();
        fs::remove_file("test_loader.txt.bin").unwrap();
    }
}
