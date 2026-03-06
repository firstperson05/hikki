use crate::tokenizer::vocab::Vocab;
use rayon::prelude::*;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;
use std::time::Instant;

pub struct BpeTokenizer {
    pub vocab: Vocab,
    pub merges: Vec<((u32, u32), u32)>,
    pub special_tokens: HashMap<String, u32>,
}

// ── Priority queue entry for BPE merge selection ──────────────────────────────
#[derive(Eq, PartialEq)]
struct HeapEntry {
    count: u32,
    pair: (u32, u32),
}

impl Ord for HeapEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        // Max-heap by count, then deterministic tie-break: prefer smaller pair tuple
        match self.count.cmp(&other.count) {
            Ordering::Equal => other.pair.cmp(&self.pair), // smaller pair wins on tie
            other_ord => other_ord,
        }
    }
}

impl PartialOrd for HeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

// ── Helper: format bytes as human-readable ────────────────────────────────────
fn fmt_bytes(bytes: usize) -> String {
    if bytes >= 1_000_000_000 {
        format!("{:.1} GB", bytes as f64 / 1_000_000_000.0)
    } else if bytes >= 1_000_000 {
        format!("{:.0} MB", bytes as f64 / 1_000_000.0)
    } else if bytes >= 1_000 {
        format!("{:.0} KB", bytes as f64 / 1_000.0)
    } else {
        format!("{} B", bytes)
    }
}

fn progress_bar(frac: f64, width: usize) -> String {
    let filled = (frac * width as f64).round() as usize;
    let filled = filled.min(width);
    let empty = width - filled;
    format!("{}{}", "\u{2588}".repeat(filled), "\u{2591}".repeat(empty))
}

impl BpeTokenizer {
    /// Train tokenizer from scratch on a corpus — parallelized with rayon
    pub fn train(corpus: &str, vocab_size: usize) -> Self {
        let corpus_bytes = corpus.len();
        println!("[Tokenizer] Reading corpus...  {}", fmt_bytes(corpus_bytes));

        let mut vocab = Vocab::new();
        for i in 0..=255u8 {
            vocab.add_token(&format!("<0x{:02X}>", i));
        }

        // --- Step 1: Pre-tokenize into word frequencies ---
        print!("[Tokenizer] Counting word frequencies... ");
        std::io::stdout().flush().unwrap();

        let mut word_counts: HashMap<Vec<u32>, u32> = HashMap::with_capacity(100_000);
        let mut current_word = Vec::with_capacity(32);
        for b in corpus.bytes() {
            current_word.push(b as u32 + 4);
            if b == b' ' || b == b'\n' || b == b'\r' || b == b'\t' {
                *word_counts.entry(current_word.clone()).or_insert(0) += 1;
                current_word.clear();
            }
        }
        if !current_word.is_empty() {
            *word_counts.entry(current_word).or_insert(0) += 1;
        }
        println!("Done! ({} unique words)", word_counts.len());

        // --- Step 2: Initial pair counts ---
        print!("[Tokenizer] Initializing pair counts... ");
        std::io::stdout().flush().unwrap();
        let mut pair_counts: HashMap<(u32, u32), i64> = HashMap::with_capacity(200_000);
        for (word, &count) in &word_counts {
            for i in 0..word.len().saturating_sub(1) {
                let pair = (word[i], word[i + 1]);
                *pair_counts.entry(pair).or_insert(0) += count as i64;
            }
        }
        println!("Done!");

        // --- Step 3: Progressive Merge ---
        let mut merges = Vec::new();
        let mut next_id: u32 = 260;
        let num_merges = vocab_size.saturating_sub(260);

        // Initial Max-Heap
        let mut heap: BinaryHeap<HeapEntry> = pair_counts
            .iter()
            .map(|(&pair, &count)| HeapEntry {
                count: count as u32,
                pair,
            })
            .collect();

        let merge_start = Instant::now();
        let mut last_report = Instant::now();

        for merge_idx in 0..num_merges {
            // Find best pair (max count) and skip stale heap entries
            let mut best_pair = None;
            while let Some(entry) = heap.pop() {
                if let Some(&actual_count) = pair_counts.get(&entry.pair) {
                    if actual_count as u32 == entry.count {
                        best_pair = Some(entry.pair);
                        break;
                    } else if actual_count > 0 {
                        // Re-push updated count
                        heap.push(HeapEntry {
                            count: actual_count as u32,
                            pair: entry.pair,
                        });
                    }
                }
            }

            let pair = match best_pair {
                Some(p) => p,
                None => break,
            };

            let new_id = next_id;
            next_id += 1;
            merges.push((pair, new_id));
            vocab.add_token(&format!("<Token_{}>", new_id));

            // Update word counts and pair counts
            let mut new_word_counts = HashMap::with_capacity(word_counts.len());
            for (word, count) in word_counts {
                let count = count;
                let mut new_word = Vec::with_capacity(word.len());
                let mut i = 0;
                let mut changed = false;

                while i < word.len() {
                    if i < word.len() - 1 && word[i] == pair.0 && word[i + 1] == pair.1 {
                        // Before merge, remove old surrounding pairs
                        if i > 0 {
                            *pair_counts.entry((word[i - 1], word[i])).or_insert(0) -= count as i64;
                        }
                        if i < word.len() - 2 {
                            // Don't remove the pair we are currently merging
                            if (word[i + 1], word[i + 2]) != pair {
                                *pair_counts.entry((word[i + 1], word[i + 2])).or_insert(0) -=
                                    count as i64;
                            }
                        }

                        new_word.push(new_id);
                        i += 2;
                        changed = true;
                    } else {
                        new_word.push(word[i]);
                        i += 1;
                    }
                }

                if changed {
                    // Add new surrounding pairs
                    for j in 0..new_word.len().saturating_sub(1) {
                        let p = (new_word[j], new_word[j + 1]);
                        if p.0 == new_id || p.1 == new_id {
                            let c = pair_counts.entry(p).or_insert(0);
                            *c += count as i64;
                            heap.push(HeapEntry {
                                count: *c as u32,
                                pair: p,
                            });
                        }
                    }
                }
                *new_word_counts.entry(new_word).or_insert(0) += count;
            }
            word_counts = new_word_counts;
            pair_counts.remove(&pair);

            // Progress
            if last_report.elapsed().as_millis() > 500 {
                let frac = (merge_idx + 1) as f64 / num_merges as f64;
                let rate = (merge_idx + 1) as f64 / merge_start.elapsed().as_secs_f64();
                print!("\r[Tokenizer] Training BPE...    {}  {:3.0}% | {}/{} merges | {:.0} merges/s   ",
                    progress_bar(frac, 20), frac * 100.0, merge_idx + 1, num_merges, rate);
                std::io::stdout().flush().unwrap();
                last_report = Instant::now();
            }
        }
        println!();

        // ── Step 3: Save and report ───────────────────────────────────────────
        let mut special_tokens = HashMap::new();
        special_tokens.insert("<PAD>".to_string(), 0);
        special_tokens.insert("<UNK>".to_string(), 1);
        special_tokens.insert("<BOS>".to_string(), 2);
        special_tokens.insert("<EOS>".to_string(), 3);

        let tok = BpeTokenizer {
            vocab,
            merges,
            special_tokens,
        };

        // Show top-5 tokens by decoding merged token ids
        let mut top5 = Vec::new();
        for &(_, id) in tok.merges.iter().rev().take(5) {
            let decoded = tok.decode(&[id]);
            let display = decoded.replace('\n', "\\n").replace('\r', "\\r");
            top5.push(format!("\"{}\"", display));
        }
        top5.reverse();
        let top5_display = if top5.is_empty() {
            "[]".to_string()
        } else {
            format!("[{}]", top5.join(", "))
        };

        println!(
            "[Tokenizer] Done! {} tokens | top-5: {}",
            tok.vocab.len(),
            top5_display
        );

        tok
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        let merge_ranks: HashMap<(u32, u32), usize> = self
            .merges
            .iter()
            .enumerate()
            .map(|(i, &(pair, _))| (pair, i))
            .collect();
        let merge_to_id: HashMap<(u32, u32), u32> =
            self.merges.iter().map(|&(p, id)| (p, id)).collect();

        let mut final_ids = Vec::with_capacity(text.len() / 2);
        let mut current_word = Vec::with_capacity(64);

        for b in text.bytes() {
            current_word.push(b as u32 + 4); // basic shift
            if b == b' ' || b == b'\n' || b == b'\r' || b == b'\t' {
                Self::encode_chunk(&mut current_word, &merge_ranks, &merge_to_id);
                final_ids.extend_from_slice(&current_word);
                current_word.clear();
            }
        }
        if !current_word.is_empty() {
            Self::encode_chunk(&mut current_word, &merge_ranks, &merge_to_id);
            final_ids.extend_from_slice(&current_word);
        }

        final_ids
    }

    fn encode_chunk(
        ids: &mut Vec<u32>,
        merge_ranks: &HashMap<(u32, u32), usize>,
        merge_to_id: &HashMap<(u32, u32), u32>,
    ) {
        if ids.len() < 2 {
            return;
        }

        loop {
            let mut min_rank = usize::MAX;
            let mut best_pair = None;

            for i in 0..ids.len() - 1 {
                let pair = (ids[i], ids[i + 1]);
                if let Some(&rank) = merge_ranks.get(&pair) {
                    if rank < min_rank {
                        min_rank = rank;
                        best_pair = Some(pair);
                    }
                }
            }

            if let Some(pair_to_merge) = best_pair {
                let new_id = *merge_to_id.get(&pair_to_merge).unwrap();
                let mut new_ids = Vec::with_capacity(ids.len());
                let mut i = 0;
                while i < ids.len() {
                    if i < ids.len() - 1
                        && ids[i] == pair_to_merge.0
                        && ids[i + 1] == pair_to_merge.1
                    {
                        new_ids.push(new_id);
                        i += 2;
                    } else {
                        new_ids.push(ids[i]);
                        i += 1;
                    }
                }
                *ids = new_ids;
                if ids.len() < 2 {
                    break;
                }
            } else {
                break;
            }
        }
    }

    pub fn decode(&self, tokens: &[u32]) -> String {
        let mut id_to_pair = HashMap::with_capacity(self.merges.len());
        for &((p0, p1), id) in &self.merges {
            id_to_pair.insert(id, (p0, p1));
        }

        let mut bytes = Vec::new();
        for &id in tokens {
            self.expand_id(id, &mut bytes, &id_to_pair);
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }

    fn expand_id(&self, id: u32, bytes: &mut Vec<u8>, id_to_pair: &HashMap<u32, (u32, u32)>) {
        if id < 4 {
            let s = self.vocab.id_to_token(id);
            if s != "<UNK>" {
                bytes.extend_from_slice(s.as_bytes());
            }
        } else if id < 260 {
            bytes.push((id - 4) as u8);
        } else if let Some(&(p0, p1)) = id_to_pair.get(&id) {
            self.expand_id(p0, bytes, id_to_pair);
            self.expand_id(p1, bytes, id_to_pair);
        } else {
            let s = self.vocab.id_to_token(1);
            bytes.extend_from_slice(s.as_bytes());
        }
    }

    /// Save internal state to binary file with magic bytes "BPEV1"
    pub fn save(&self, path: &Path) -> std::io::Result<()> {
        let mut file = BufWriter::new(File::create(path)?);
        file.write_all(b"BPEV1")?;

        // Save merges
        file.write_all(&(self.merges.len() as u32).to_le_bytes())?;
        for &((p0, p1), id) in &self.merges {
            file.write_all(&p0.to_le_bytes())?;
            file.write_all(&p1.to_le_bytes())?;
            file.write_all(&id.to_le_bytes())?;
        }

        // Save vocab
        file.write_all(&(self.vocab.len() as u32).to_le_bytes())?;
        for (i, token) in self.vocab.id_to_token.iter().enumerate() {
            let bytes = token.as_bytes();
            file.write_all(&(bytes.len() as u16).to_le_bytes())?;
            file.write_all(bytes)?;
            file.write_all(&(i as u32).to_le_bytes())?;
        }

        // Save special tokens
        file.write_all(&(self.special_tokens.len() as u32).to_le_bytes())?;
        for (k, &v) in &self.special_tokens {
            let bytes = k.as_bytes();
            file.write_all(&(bytes.len() as u16).to_le_bytes())?;
            file.write_all(bytes)?;
            file.write_all(&v.to_le_bytes())?;
        }

        file.flush()?;

        let fname = path
            .file_name()
            .map(|f| f.to_string_lossy().to_string())
            .unwrap_or_else(|| path.display().to_string());
        println!("[Tokenizer] Saving vocab...    {}", fname);

        Ok(())
    }

    pub fn load(path: &Path) -> std::io::Result<Self> {
        let mut file = BufReader::new(File::open(path)?);
        let mut magic = [0u8; 5];
        file.read_exact(&mut magic)?;
        if &magic != b"BPEV1" {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Invalid magic bytes",
            ));
        }

        let mut u32_buf = [0u8; 4];
        let mut u16_buf = [0u8; 2];

        // Load merges
        file.read_exact(&mut u32_buf)?;
        let merges_len = u32::from_le_bytes(u32_buf);
        let mut merges = Vec::with_capacity(merges_len as usize);
        for _ in 0..merges_len {
            file.read_exact(&mut u32_buf)?;
            let p0 = u32::from_le_bytes(u32_buf);
            file.read_exact(&mut u32_buf)?;
            let p1 = u32::from_le_bytes(u32_buf);
            file.read_exact(&mut u32_buf)?;
            let id = u32::from_le_bytes(u32_buf);
            merges.push(((p0, p1), id));
        }

        // Load vocab
        file.read_exact(&mut u32_buf)?;
        let vocab_len = u32::from_le_bytes(u32_buf);
        let mut vocab = Vocab {
            token_to_id: HashMap::with_capacity(vocab_len as usize),
            id_to_token: vec![String::new(); vocab_len as usize],
        };
        for _ in 0..vocab_len {
            file.read_exact(&mut u16_buf)?;
            let len = u16::from_le_bytes(u16_buf);
            let mut str_buf = vec![0u8; len as usize];
            file.read_exact(&mut str_buf)?;
            file.read_exact(&mut u32_buf)?;
            let id = u32::from_le_bytes(u32_buf);

            let s = String::from_utf8(str_buf).unwrap_or_else(|_| "<UNK>".to_string());
            vocab.token_to_id.insert(s.clone(), id);
            if (id as usize) < vocab.id_to_token.len() {
                vocab.id_to_token[id as usize] = s;
            }
        }

        // Load special tokens
        file.read_exact(&mut u32_buf)?;
        let special_len = u32::from_le_bytes(u32_buf);
        let mut special_tokens = HashMap::with_capacity(special_len as usize);
        for _ in 0..special_len {
            file.read_exact(&mut u16_buf)?;
            let len = u16::from_le_bytes(u16_buf);
            let mut str_buf = vec![0u8; len as usize];
            file.read_exact(&mut str_buf)?;
            file.read_exact(&mut u32_buf)?;
            let id = u32::from_le_bytes(u32_buf);
            let s = String::from_utf8(str_buf).unwrap_or_else(|_| "<UNK>".to_string());
            special_tokens.insert(s, id);
        }

        Ok(BpeTokenizer {
            vocab,
            merges,
            special_tokens,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env::temp_dir;

    #[test]
    fn test_bpe_roundtrip() {
        let corpus = "Hello world! This is a test of the BPE tokenizer. Hello again.";
        let vocab_size = 300;
        let bpe = BpeTokenizer::train(corpus, vocab_size);
        let encoded = bpe.encode(corpus);
        let decoded = bpe.decode(&encoded);
        assert_eq!(corpus, decoded);
    }

    #[test]
    fn test_bpe_cyrillic() {
        let corpus = "Привет мир! Это проверка BPE.";
        let bpe = BpeTokenizer::train(corpus, 400);
        let encoded = bpe.encode(corpus);
        let decoded = bpe.decode(&encoded);
        assert_eq!(corpus, decoded);
    }

    #[test]
    fn test_bpe_fallback() {
        let corpus = "abc";
        let bpe = BpeTokenizer::train(corpus, 270);
        let encoded = bpe.encode("abcd");
        let decoded = bpe.decode(&encoded);
        assert_eq!("abcd", decoded);
    }

    #[test]
    fn test_bpe_binary() {
        let corpus = "test save load binary serialization";
        let bpe = BpeTokenizer::train(corpus, 280);
        let path = temp_dir().join("bpe.bin");
        bpe.save(&path).unwrap();

        let loaded = BpeTokenizer::load(&path).unwrap();
        assert_eq!(bpe.encode(corpus), loaded.encode(corpus));
    }
}
