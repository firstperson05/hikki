use hikki::{config, data, inference, model, tokenizer, training};

use clap::{Parser, Subcommand};
use std::fs;
use std::path::Path;

#[derive(Parser)]
#[command(author, version, about = "Hikki: Hybrid RWKV-SSM Architecture", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Train the model
    Train {
        #[arg(short, long)]
        config: String,
        #[arg(short, long)]
        data: String,
        #[arg(short, long)]
        val: Option<String>,
        #[arg(short, long)]
        tokenizer: Option<String>,
    },
    /// Evaluate the model
    Eval {
        #[arg(short, long)]
        checkpoint: String,
        #[arg(short, long)]
        data: String,
    },
    /// Chat with the model
    Chat {
        #[arg(short, long)]
        checkpoint: String,
    },
    /// Train the BPE tokenizer
    Tokenize {
        #[arg(short, long)]
        train_corpus: String,
        #[arg(short, long)]
        vocab_size: usize,
        #[arg(short, long)]
        output: String,
    },
    /// Run performance benchmarks
    Benchmark,
}

// ── Startup banner ────────────────────────────────────────────────────────────

fn format_number(n: usize) -> String {
    let s = n.to_string();
    let mut result = String::new();
    for (i, c) in s.chars().rev().enumerate() {
        if i > 0 && i % 3 == 0 {
            result.push(',');
        }
        result.push(c);
    }
    result.chars().rev().collect()
}

fn detect_avx2() -> &'static str {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return "enabled";
        }
        return "not available";
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        "N/A (non-x86)"
    }
}

fn print_startup_banner(vocab_size: usize, n_layer: usize, d_model: usize, param_count: usize) {
    let d_ffn = d_model * 4;
    let threads = num_cpus::get();

    println!();
    println!("Hikki v0.1.0");
    println!("Architecture : Hybrid RWKV-SSM");
    println!("Parameters   : {}", format_number(param_count));
    println!("Vocab size   : {}", format_number(vocab_size));
    println!(
        "Layers       : {}  |  d_model: {}  |  d_ffn: {}",
        n_layer, d_model, d_ffn
    );

    let device_str = if cfg!(feature = "cuda") {
        "CUDA (cuBLAS)"
    } else {
        "CPU (Rayon/AVX2)"
    };
    println!("Device       : {} | Threads: {}", device_str, threads);
    println!("AVX2         : {}", detect_avx2());
    println!();
}

fn main() -> Result<(), String> {
    let cli = Cli::parse();

    match &cli.command {
        Commands::Train {
            config,
            data,
            val,
            tokenizer: tokenizer_arg,
        } => {
            let config_str = fs::read_to_string(config).map_err(|e| e.to_string())?;
            let cfg: config::Config = toml::from_str(&config_str).map_err(|e| e.to_string())?;

            // 1. Load Tokenizer
            let default_tok = "tokenizer_16k.bpe".to_string();
            let tokenizer_path_str = tokenizer_arg.as_ref().unwrap_or(&default_tok);
            let tokenizer_path = Path::new(tokenizer_path_str);
            if !tokenizer_path.exists() {
                return Err(format!(
                    "{} not found. Run 'tokenize' first.",
                    tokenizer_path_str
                ));
            }
            let tokenizer =
                tokenizer::bpe::BpeTokenizer::load(tokenizer_path).map_err(|e| e.to_string())?;

            let actual_vocab_size = tokenizer.vocab.len();

            // 2. Init Model
            let model_config = model::lm::HikkiConfig {
                vocab_size: actual_vocab_size,
                n_layer: cfg.model.n_layer,
                n_embd: cfg.model.d_model,
                seq_len: cfg.train.seq_len,
            };
            let param_count = model::lm::HikkiLM::estimate_param_count(&model_config);

            // ── Startup Banner ──
            print_startup_banner(
                actual_vocab_size,
                cfg.model.n_layer,
                cfg.model.d_model,
                param_count,
            );

            let model = model::lm::HikkiLM::new(model_config);

            // 3. Init DataLoaders
            let train_ds = data::dataset::TextDataset::from_file(Path::new(data), &tokenizer)
                .map_err(|e| e.to_string())?;
            let train_loader = data::loader::DataLoader::new(
                train_ds,
                cfg.train.batch_size,
                cfg.train.seq_len,
                true,
            );

            let val_loader: Option<data::loader::DataLoader> = if let Some(val_path) = val {
                let val_ds = data::dataset::TextDataset::from_file(Path::new(val_path), &tokenizer)
                    .map_err(|e| e.to_string())?;
                Some(data::loader::DataLoader::new(
                    val_ds,
                    cfg.train.batch_size,
                    cfg.train.seq_len,
                    false,
                ))
            } else {
                None
            };

            // 4. Init Trainer
            let mut trainer =
                training::trainer::Trainer::new(model, train_loader, val_loader, &cfg.train);

            // 5. Run Training
            trainer.train(&cfg.train).map_err(|e| e.to_string())?;

            Ok(())
        }
        Commands::Eval { checkpoint, data } => {
            println!("Evaluating checkpoint: {} on data: {}", checkpoint, data);

            // 1. Load Tokenizer
            let tokenizer_path = Path::new("tokenizer_16k.bpe");
            if !tokenizer_path.exists() {
                return Err("tokenizer_16k.bpe not found. Run 'tokenize' first.".to_string());
            }
            let tokenizer =
                tokenizer::bpe::BpeTokenizer::load(tokenizer_path).map_err(|e| e.to_string())?;
            let vocab_size = tokenizer.vocab.len();

            // 2. Load model architecture
            let model_config = model::lm::HikkiConfig {
                vocab_size,
                n_layer: 4,
                n_embd: 128,
                seq_len: 128,
            };
            let param_count = model::lm::HikkiLM::estimate_param_count(&model_config);

            // ── Startup Banner ──
            print_startup_banner(vocab_size, 4, 128, param_count);

            let model = model::lm::HikkiLM::new(model_config);

            // 3. Load checkpoint weights
            model.load_checkpoint(Path::new(checkpoint))?;

            // 4. Load validation data
            let val_ds = data::dataset::TextDataset::from_file(Path::new(data), &tokenizer)
                .map_err(|e| e.to_string())?;
            let val_loader = data::loader::DataLoader::new(
                val_ds, 8,     // batch_size
                128,   // seq_len
                false, // no shuffle for evaluation
            );

            // 5. Run evaluation
            let mut trainer = training::trainer::Trainer::new(
                model,
                data::loader::DataLoader::new(
                    data::dataset::TextDataset::from_file(Path::new(data), &tokenizer)
                        .map_err(|e| e.to_string())?,
                    8,
                    128,
                    false,
                ),
                Some(val_loader),
                &config::TrainConfig {
                    batch_size: 8,
                    seq_len: 128,
                    grad_accum_steps: 1,
                    max_steps: 0,
                    eval_every: 0,
                    save_every: 0,
                    log_every: 0,
                    checkpoint_dir: "ckpts".to_string(),
                    clip_grad_norm: 1.0,
                    lr: 0.001,
                    min_lr: 0.0001,
                    warmup_steps: 0,
                    weight_decay: 0.01,
                },
            );

            let val_loss = trainer.eval(100)?; // Eval on 100 batches
            let val_ppl = val_loss.exp();

            println!("Evaluation Results:");
            println!("  Validation Loss: {:.4}", val_loss);
            println!("  Validation Perplexity: {:.2}", val_ppl);
            println!("  Batches evaluated: 100");

            Ok(())
        }
        Commands::Chat { checkpoint } => {
            // 1. Load Tokenizer (Required to get vocab_size)
            let tokenizer_path = Path::new("tokenizer_16k.bpe");
            if !tokenizer_path.exists() {
                return Err("tokenizer_16k.bpe not found.".to_string());
            }
            let tokenizer =
                tokenizer::bpe::BpeTokenizer::load(tokenizer_path).map_err(|e| e.to_string())?;
            let vocab_size = tokenizer.vocab.len();

            // 2. Init Model with architecture from tiny.toml
            let model_config = model::lm::HikkiConfig {
                vocab_size,
                n_layer: 4,
                n_embd: 128,
                seq_len: 128,
            };
            let param_count = model::lm::HikkiLM::estimate_param_count(&model_config);

            // ── Startup Banner ──
            print_startup_banner(vocab_size, 4, 128, param_count);

            let model = model::lm::HikkiLM::new(model_config);

            // 3. Load actual weights
            model.load_checkpoint(Path::new(checkpoint))?;

            // 4. Init Engine
            let mut engine = inference::engine::InferenceEngine::new(model, tokenizer, 42);
            let mut config = inference::engine::InferenceConfig::default();

            println!("Commands: /reset to clear state, /config key=val to change parameters.");
            println!("Use Ctrl+C to exit.");
            config.print_current();

            use std::io::{self, BufRead, Read, Write};
            let stdin = io::stdin();
            let mut input = String::new();

            // Check if input is piped by trying to read first line
            let mut first_line = String::new();
            match stdin.lock().read_line(&mut first_line) {
                Ok(0) => {
                    // No input, proceed with interactive mode
                }
                Ok(_) => {
                    // We have input, read the rest and treat as prompt
                    input.push_str(&first_line);
                    stdin.lock().read_to_string(&mut input).unwrap_or_default();
                    let prompt = input.trim();
                    if !prompt.is_empty() {
                        match engine.generate(prompt, &config) {
                            Ok(response) => {
                                print!("{}", response);
                                io::stdout().flush().unwrap();
                            }
                            Err(e) => {
                                eprintln!("Error: {}", e);
                            }
                        }
                        return Ok(());
                    }
                }
                Err(_) => {
                    // Error reading, proceed with interactive mode
                }
            }

            loop {
                print!("\n[user] > ");
                io::stdout().flush().unwrap();
                input.clear();
                if stdin.read_line(&mut input).is_err() || input.trim().is_empty() {
                    break;
                }

                let text = input.trim();
                if text == "/reset" {
                    engine.reset();
                    println!("State cleared.");
                    continue;
                }

                if text.starts_with("/config ") {
                    let config_str = &text[8..];
                    match config.parse_config_string(config_str) {
                        Ok(()) => {
                            println!("Config updated:");
                            config.print_current();
                        }
                        Err(e) => println!("Error: {}", e),
                    }
                    continue;
                }

                if text.starts_with("/config") {
                    config.print_current();
                    continue;
                }

                if text.starts_with("/prompt ") {
                    let prompt = &text[8..];
                    match engine.generate(prompt, &config) {
                        Ok(response) => {
                            print!("[bot] > {}", response);
                            io::stdout().flush().unwrap();
                            println!();
                        }
                        Err(e) => {
                            println!("Error: {}", e);
                        }
                    }
                    continue;
                }

                // Generate response
                match engine.generate(text, &config) {
                    Ok(response) => {
                        print!("[bot] > {}", response);
                        io::stdout().flush().unwrap();
                        println!();
                    }
                    Err(e) => {
                        println!("Error: {}", e);
                    }
                }
            }

            Ok(())
        }
        Commands::Tokenize {
            train_corpus,
            vocab_size,
            output,
        } => {
            println!(
                "Training BPE tokenizer on: {} with vocab size: {}",
                train_corpus, vocab_size
            );
            let corpus = fs::read_to_string(train_corpus).map_err(|e| e.to_string())?;
            let tokenizer = tokenizer::bpe::BpeTokenizer::train(&corpus, *vocab_size);
            tokenizer
                .save(Path::new(output))
                .map_err(|e| e.to_string())?;
            println!("Tokenizer saved to: {}", output);

            Ok(())
        }
        Commands::Benchmark => {
            println!("Running Hikki Benchmarks...");
            println!();

            // 1. Matmul
            let m = 256;
            let k = 256;
            let n = 256;
            let a = vec![1.0; m * k];
            let b = vec![1.0; k * n];
            let mut c = vec![0.0; m * n];

            let start = std::time::Instant::now();
            let iters = 1000;
            for _ in 0..iters {
                unsafe {
                    matrixmultiply::sgemm(
                        m,
                        k,
                        n,
                        1.0,
                        a.as_ptr(),
                        k as isize,
                        1,
                        b.as_ptr(),
                        n as isize,
                        1,
                        0.0,
                        c.as_mut_ptr(),
                        n as isize,
                        1,
                    );
                }
            }
            let elapsed = start.elapsed().as_secs_f64();
            let flops = (2.0 * m as f64 * k as f64 * n as f64) * iters as f64;
            let gflops = (flops / elapsed) / 1_000_000_000.0;
            println!("1. Matmul (256x256x256) x 1000:");
            println!("   Time   : {:.2} ms", elapsed * 1000.0);
            println!("   GFLOPS : {:.2}\n", gflops);

            // 2. Forward Pass
            println!("2. Forward Pass (Batch=8, Seq=128) x 100:");
            let m_cfg = model::lm::HikkiConfig {
                vocab_size: 4000,
                n_layer: 4,
                n_embd: 256,
                seq_len: 128,
            };
            let bench_model = model::lm::HikkiLM::new(m_cfg);
            let input_ids = vec![0; 8 * 128];
            let start = std::time::Instant::now();
            let fwd_iters = 100;
            for _ in 0..fwd_iters {
                let _ = bench_model.forward(&input_ids, None).unwrap();
            }
            let elapsed = start.elapsed().as_secs_f64();
            println!(
                "   Time/step  : {:.2} ms\n",
                (elapsed * 1000.0) / fwd_iters as f64
            );

            // 3. Tokenizer
            println!("3. Tokenizer (BPE 1MB text):");
            let text_1mb = "a ".repeat(500_000); // 1 MB string
            let bpe = tokenizer::bpe::BpeTokenizer::train(&text_1mb, 300);
            let start = std::time::Instant::now();
            let _ = bpe.encode(&text_1mb);
            let elapsed = start.elapsed().as_secs_f64();
            let mb_per_sec = 1.0 / elapsed.max(0.0001);
            println!("   Speed  : {:.2} MB/s\n", mb_per_sec);

            println!("Comparison: Typical modern CPU peak is 100+ GFLOPS.");
            println!("If GFLOPS < 10, check matrixmultiply CPU features.");
            Ok(())
        }
    }
}
