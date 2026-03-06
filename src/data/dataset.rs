use memmap2::MmapOptions;
use rayon::prelude::*;
use std::fs::File;
use std::io::Write;
use std::path::Path;

use crate::tokenizer::bpe::BpeTokenizer;

pub struct TextDataset {
    mmap: memmap2::Mmap,
    len_tokens: usize,
}

impl TextDataset {
    /// Loads a dataset from a text file, caching the token IDs in a memory-mapped .bin file.
    pub fn from_file(path: &Path, tokenizer: &BpeTokenizer) -> std::io::Result<Self> {
        let mut bin_path = path.to_path_buf();
        if let Some(ext) = bin_path.extension() {
            let mut new_ext = ext.to_os_string();
            new_ext.push(".bin");
            bin_path.set_extension(new_ext);
        } else {
            bin_path.set_extension("bin");
        }

        if !bin_path.exists() {
            println!("[Dataset] Tokenizing corpus and building binary cache...");
            // Memory map original text and tokenize
            let file = File::open(path)?;
            let text_mmap = unsafe { MmapOptions::new().map(&file)? };
            let chunk_size = 1024 * 1024; // 1MB chunks
            let text_bytes = &text_mmap[..];

            // Find safe text boundaries (break on space or newline if possible)
            let mut boundaries = Vec::new();
            boundaries.push(0);
            let mut current = chunk_size;
            while current < text_bytes.len() {
                while current > 0 && text_bytes[current] != b' ' && text_bytes[current] != b'\n' {
                    current -= 1;
                }
                if current <= *boundaries.last().unwrap() {
                    // if no space found, just break at chunk_size
                    current = *boundaries.last().unwrap() + chunk_size;
                }
                boundaries.push(current);
                current += chunk_size;
            }
            boundaries.push(text_bytes.len());

            let mut chunks = Vec::new();
            for i in 0..boundaries.len() - 1 {
                let start = boundaries[i];
                let end = boundaries[i + 1];
                chunks.push(&text_bytes[start..end]);
            }

            let tokens_nested: Vec<Vec<u32>> = chunks
                .par_iter()
                .map(|&chunk| {
                    let s = String::from_utf8_lossy(chunk);
                    tokenizer.encode(&s)
                })
                .collect();

            let mut out_file = File::create(&bin_path)?;
            for t_vec in tokens_nested {
                for &t in &t_vec {
                    out_file.write_all(&t.to_le_bytes())?;
                }
            }
        }

        let bin_file = File::open(&bin_path)?;
        let mmap = unsafe { MmapOptions::new().map(&bin_file)? };
        let len_tokens = mmap.len() / 4;

        println!(
            "[Dataset] Loaded from cache: {} ({:.1}M tokens)",
            bin_path.display(),
            len_tokens as f64 / 1_000_000.0
        );

        Ok(TextDataset { mmap, len_tokens })
    }

    pub fn num_samples(&self, seq_len: usize) -> usize {
        if self.len_tokens == 0 || seq_len == 0 {
            return 0;
        }
        (self.len_tokens + seq_len - 1) / seq_len
    }

    /// Returns input and shifted target tokens for the given sample index.
    pub fn get_sample(&self, sample_idx: usize, seq_len: usize) -> (Vec<u32>, Vec<u32>) {
        let mut input = Vec::with_capacity(seq_len);
        let mut target = Vec::with_capacity(seq_len);

        let mut ptr = sample_idx * seq_len * 4;
        let mmap_len = self.mmap.len();

        for _ in 0..seq_len {
            if ptr + 8 <= mmap_len {
                let mut buf = [0u8; 4];
                buf.copy_from_slice(&self.mmap[ptr..ptr + 4]);
                input.push(u32::from_le_bytes(buf));

                buf.copy_from_slice(&self.mmap[ptr + 4..ptr + 8]);
                target.push(u32::from_le_bytes(buf));
            } else if ptr + 4 <= mmap_len {
                let mut buf = [0u8; 4];
                buf.copy_from_slice(&self.mmap[ptr..ptr + 4]);
                input.push(u32::from_le_bytes(buf));
                target.push(3); // EOS is 3
            } else {
                input.push(0); // PAD is 0
                target.push(0); // PAD
            }
            ptr += 4;
        }

        (input, target)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::PathBuf;

    #[test]
    fn test_dataset_padding() {
        let corpus = "Hello"; // very short
        let bpe = BpeTokenizer::train(corpus, 300);
        let path = PathBuf::from("test_pad.txt");
        fs::write(&path, corpus).unwrap();

        let ds = TextDataset::from_file(&path, &bpe).unwrap();
        assert_eq!(ds.num_samples(10), 1);
        let (inp, tgt) = ds.get_sample(0, 10);
        assert_eq!(inp.len(), 10);
        assert_eq!(tgt.len(), 10);

        let actual_len = bpe.encode(corpus).len();
        assert_eq!(tgt[actual_len - 1], 3); // EOS
        assert_eq!(inp[actual_len], 0); // PAD

        fs::remove_file("test_pad.txt").unwrap();
        fs::remove_file("test_pad.txt.bin").unwrap();
    }
}
