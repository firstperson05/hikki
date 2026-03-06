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

        let mut vocab = Vocab::new(); // <PAD>, <UNK>, <BOS>, <EOS> (0..3)
        for i in 0..=255u8 {
            vocab.add_token(&format!("<0x{:02X}>", i));
        }

        // ── Step 1: Parallel corpus chunking ──────────────────────────────────
        let chunk_start = Instant::now();
        print!("[Tokenizer] Counting pairs...  ");
        std::io::stdout().flush().unwrap();

        // Split corpus into rayon-sized slices at whitespace boundaries
        let n_threads = rayon::current_num_threads();
        let slice_size = corpus_bytes / n_threads.max(1);

        // Find split points at whitespace boundaries
        let mut split_points = vec![0usize];
        for t in 1..n_threads {
            let approx = t * slice_size;
            // Walk forward to find next whitespace
            let mut pos = approx.min(corpus_bytes);
            let bytes_ref = corpus.as_bytes();
            while pos < corpus_bytes
                && bytes_ref[pos] != b' '
                && bytes_ref[pos] != b'\n'
                && bytes_ref[pos] != b'\r'
                && bytes_ref[pos] != b'\t'
            {
                pos += 1;
            }
            if pos < corpus_bytes {
                pos += 1; // include the whitespace in the previous chunk
            }
            if pos > *split_points.last().unwrap() && pos < corpus_bytes {
                split_points.push(pos);
            }
        }
        split_points.push(corpus_bytes);

        // Process each slice in parallel: build chunk frequency maps
        let chunks: HashMap<Vec<u32>, u32> = split_points
            .par_windows(2)
            .map(|w| {
                let slice = &corpus.as_bytes()[w[0]..w[1]];
                let mut local_chunks: HashMap<Vec<u32>, u32> = HashMap::with_capacity(100_000);
                let mut current = Vec::with_capacity(32);
                for &b in slice {
                    current.push(b as u32 + 4);
                    if b == b' ' || b == b'\n' || b == b'\r' || b == b'\t' {
                        *local_chunks.entry(current.clone()).or_insert(0) += 1;
                        current.clear();
                    }
                }
                if !current.is_empty() {
                    *local_chunks.entry(current).or_insert(0) += 1;
                }
                local_chunks
            })
            .reduce(
                || HashMap::with_capacity(100_000),
                |mut a, b| {
                    for (k, v) in b {
                        *a.entry(k).or_insert(0) += v;
                    }
                    a
                },
            );

        // Compute initial pair frequencies in parallel
        let mut pair_counts: HashMap<(u32, u32), u32> = chunks
            .par_iter()
            .fold(
                || HashMap::with_capacity(100_000),
                |mut acc, (chunk, &count)| {
                    for i in 0..chunk.len().saturating_sub(1) {
                        let pair = (chunk[i], chunk[i + 1]);
                        *acc.entry(pair).or_insert(0) += count;
                    }
                    acc
                },
            )
            .reduce(
                || HashMap::with_capacity(100_000),
                |mut a, b| {
                    for (k, v) in b {
                        *a.entry(k).or_insert(0) += v;
                    }
                    a
                },
            );

        let chunk_elapsed = chunk_start.elapsed().as_secs_f64();
        let speed = corpus_bytes as f64 / chunk_elapsed;
        println!(
            "{}  100% | {} | {}/s",
            progress_bar(1.0, 20),
            fmt_bytes(corpus_bytes),
            fmt_bytes(speed as usize)
        );

        // ── Step 2: BPE merge loop with BinaryHeap ────────────────────────────
        let mut merges = Vec::new();
        let mut next_id: u32 = 260;
        let num_merges = vocab_size.saturating_sub(260);
        let mut chunks_vec: Vec<(Vec<u32>, u32)> = chunks.into_iter().collect();
        let mut pair_to_chunks: HashMap<(u32, u32), HashSet<usize>> =
            HashMap::with_capacity(pair_counts.len());

        // Build the inverted index: which chunks contain which pairs
        for (chunk_idx, (chunk, _)) in chunks_vec.iter().enumerate() {
            for i in 0..chunk.len().saturating_sub(1) {
                let pair = (chunk[i], chunk[i + 1]);
                pair_to_chunks.entry(pair).or_default().insert(chunk_idx);
            }
        }

        // Build initial max-heap
        let mut heap: BinaryHeap<HeapEntry> = pair_counts
            .iter()
            .filter(|(_, &c)| c > 0)
            .map(|(&pair, &count)| HeapEntry { count, pair })
            .collect();

        let merge_start = Instant::now();
        let mut last_report = Instant::now();
        let mut _merges_since_report = 0usize;

        for merge_idx in 0..num_merges {
            // Pop stale entries until we find a valid one
            let best = loop {
                match heap.pop() {
                    None => break None,
                    Some(entry) => {
                        let actual = pair_counts.get(&entry.pair).copied().unwrap_or(0);
                        if actual == 0 {
                            continue; // stale
                        }
                        if actual != entry.count {
                            heap.push(HeapEntry {
                                count: actual,
                                pair: entry.pair,
                            });
                            continue;
                        }
                        break Some(entry);
                    }
                }
            };

            let best = match best {
                Some(b) => b,
                None => break,
            };

            let best_pair = best.pair;
            let new_id = next_id;
            next_id += 1;
            merges.push((best_pair, new_id));
            vocab.add_token(&format!("<Token_{}>", new_id));

            // Remove the merged pair everywhere
            pair_counts.remove(&best_pair);
            let target_chunks = pair_to_chunks.remove(&best_pair).unwrap_or_default();

            // Apply merge ONLY to the affected chunks
            for chunk_idx in target_chunks {
                let (chunk, count) = &chunks_vec[chunk_idx];
                let count = *count;

                // 1. Remove old pairs from global counts and inverted index
                for i in 0..chunk.len().saturating_sub(1) {
                    let p = (chunk[i], chunk[i + 1]);
                    if p != best_pair {
                        // best_pair is already removed from index
                        if let Some(c) = pair_counts.get_mut(&p) {
                            *c = c.saturating_sub(count);
                        }
                        if let Some(set) = pair_to_chunks.get_mut(&p) {
                            set.remove(&chunk_idx);
                        }
                    }
                }

                // 2. Apply merge
                let mut new_chunk = Vec::with_capacity(chunk.len());
                let mut i = 0;
                while i < chunk.len() {
                    if i < chunk.len() - 1 && chunk[i] == best_pair.0 && chunk[i + 1] == best_pair.1
                    {
                        new_chunk.push(new_id);
                        i += 2;
                    } else {
                        new_chunk.push(chunk[i]);
                        i += 1;
                    }
                }

                // 3. Add new pairs to global counts and inverted index
                for i in 0..new_chunk.len().saturating_sub(1) {
                    let p = (new_chunk[i], new_chunk[i + 1]);
                    let new_count = pair_counts.entry(p).or_insert(0);
                    *new_count += count;
                    heap.push(HeapEntry {
                        count: *new_count,
                        pair: p,
                    });
                    pair_to_chunks.entry(p).or_default().insert(chunk_idx);
                }

                // 4. Update the stored chunk
                chunks_vec[chunk_idx].0 = new_chunk;
            }
            _merges_since_report += 1;

            // Progress reporting ~ every 0.5s or every 100 merges
            let now = Instant::now();
            if now.duration_since(last_report).as_millis() > 500 || merge_idx == num_merges - 1 {
                let frac = (merge_idx + 1) as f64 / num_merges as f64;
                let elapsed_total = merge_start.elapsed().as_secs_f64();
                let rate = if elapsed_total > 0.0 {
                    (merge_idx + 1) as f64 / elapsed_total
                } else {
                    0.0
                };
                let remaining = if rate > 0.0 {
                    (num_merges - merge_idx - 1) as f64 / rate
                } else {
                    0.0
                };
                let eta_str = if remaining > 60.0 {
                    format!("{:.0}m {:02.0}s", remaining / 60.0, remaining % 60.0)
                } else {
                    format!("{:.0}s", remaining)
                };
                print!(
                    "\r[Tokenizer] Training BPE...    {}  {:3.0}% | {}/{} merges | {:.0} merges/s | eta {}   ",
                    progress_bar(frac, 20),
                    frac * 100.0,
                    merge_idx + 1,
                    num_merges,
                    rate,
                    eta_str
                );
                std::io::stdout().flush().unwrap();
                last_report = now;
                _merges_since_report = 0;
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
