use crate::config::TrainConfig;
use crate::data::loader::DataLoader;
use crate::model::lm::HikkiLM;
use crate::tensor::autograd::Node;
use crate::training::loss::cross_entropy_loss;
use crate::training::optimizer::AdamW;
use crate::training::scheduler::CosineScheduler;
use std::fs;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::time::Instant;

pub struct Trainer {
    pub model: HikkiLM,
    pub optimizer: AdamW,
    pub scheduler: CosineScheduler,
    pub train_loader: DataLoader,
    pub val_loader: Option<DataLoader>,
}

// ── Display helpers ───────────────────────────────────────────────────────────

fn progress_bar(frac: f64, width: usize) -> String {
    let filled = (frac * width as f64).round() as usize;
    let filled = filled.min(width);
    let empty = width - filled;
    format!("{}{}", "\u{2588}".repeat(filled), "\u{2591}".repeat(empty))
}

fn fmt_duration(secs: f64) -> String {
    let h = (secs / 3600.0) as u64;
    let m = ((secs % 3600.0) / 60.0) as u64;
    let s = (secs % 60.0) as u64;
    format!("{}h {:02}m {:02}s", h, m, s)
}

fn fmt_eta(secs: f64) -> String {
    if secs > 3600.0 {
        let h = (secs / 3600.0) as u64;
        let m = ((secs % 3600.0) / 60.0) as u64;
        format!("{}h {:02}m", h, m)
    } else if secs > 60.0 {
        let m = (secs / 60.0) as u64;
        let s = (secs % 60.0) as u64;
        format!("{}m {:02}s", m, s)
    } else {
        format!("{:.0}s", secs)
    }
}

// ── Dashboard state ───────────────────────────────────────────────────────────

struct DashboardState {
    best_loss: f32,
    best_loss_step: usize,
    prev_loss: f32,
    last_val_loss: Option<f32>,
    last_val_step: usize,
    last_saved_ckpt: String,
    total_tokens_ever: usize,
}

impl DashboardState {
    fn new() -> Self {
        Self {
            best_loss: f32::MAX,
            best_loss_step: 0,
            prev_loss: f32::MAX,
            last_val_loss: None,
            last_val_step: 0,
            last_saved_ckpt: String::new(),
            total_tokens_ever: 0,
        }
    }
}

impl Trainer {
    pub fn new(
        model: HikkiLM,
        train_loader: DataLoader,
        val_loader: Option<DataLoader>,
        config: &TrainConfig,
    ) -> Self {
        let mut param_shapes = Vec::new();
        param_shapes.push({
            let embed_lock = model.embedding.weight.lock().unwrap();
            embed_lock.value.shape.clone()
        });
        for block in &model.blocks {
            param_shapes.push({
                let r_lock = block.mixing.r_weight.lock().unwrap();
                r_lock.value.shape.clone()
            });
            param_shapes.push({
                let k_lock = block.mixing.k_weight.lock().unwrap();
                k_lock.value.shape.clone()
            });
            param_shapes.push({
                let v_lock = block.mixing.v_weight.lock().unwrap();
                v_lock.value.shape.clone()
            });
        }
        param_shapes.push({
            let head_lock = model.head.lock().unwrap();
            head_lock.value.shape.clone()
        });

        let optimizer = AdamW::new(&param_shapes, config.lr, config.weight_decay);
        let scheduler = CosineScheduler {
            base_lr: config.lr,
            min_lr: config.min_lr,
            warmup_steps: config.warmup_steps,
            total_steps: config.max_steps,
        };

        Self {
            model,
            optimizer,
            scheduler,
            train_loader,
            val_loader,
        }
    }

    /// Print the rich training dashboard
    fn print_dashboard(
        &self,
        config: &TrainConfig,
        step: usize,
        loss: f32,
        _grad_norm: f32,
        tok_per_sec: f32,
        elapsed_total: f64,
        _state: &DashboardState,
    ) {
        let ppl = loss.exp();
        let lr = self.optimizer.lr;

        // ETA
        let steps_remaining = config.max_steps.saturating_sub(step);
        let secs_per_step = if step > 0 {
            elapsed_total / step as f64
        } else {
            0.0
        };
        let eta_secs = steps_remaining as f64 * secs_per_step;

        if step % 1 == 0 {
            let elapsed_str = fmt_duration(elapsed_total);
            let eta_str = fmt_eta(eta_secs);
            let lr_str = format!("{:.2e}", lr);

            // Single line for Colab/simple terminals
            print!("\r[Step {:>5}/{:<5}] Loss: {:.4} | Ppl: {:.2} | LR: {} | {:.1}k tok/s | ETA: {} | Elapsed: {}    ", 
                step, config.max_steps, loss, ppl, lr_str, tok_per_sec / 1000.0, eta_str, elapsed_str);
            std::io::stdout().flush().unwrap();
        }
    }

    pub fn train(&mut self, config: &TrainConfig) -> Result<(), String> {
        fs::create_dir_all(&config.checkpoint_dir).map_err(|e| e.to_string())?;

        let mut step = self.optimizer.step;
        let mut tokens_processed = 0usize;
        let mut last_log_time = Instant::now();
        let train_start = Instant::now();
        let mut dash_state = DashboardState::new();

        println!("Starting training from step {}...", step);

        while step < config.max_steps {
            let mut total_loss = 0.0;

            // 1. Accumulate gradients
            for _ in 0..config.grad_accum_steps {
                let batch = match self.train_loader.next() {
                    Some(b) => b,
                    None => {
                        self.train_loader.reset();
                        self.train_loader.next().unwrap()
                    }
                };

                // Forward
                let input_ids: Vec<u32> = batch.inputs.data.iter().map(|&x| x as u32).collect();
                let (logits_node, _) = self.model.forward(&input_ids, None)?;

                // Loss
                let (loss_val, mut loss_grad) = cross_entropy_loss(
                    &{
                        let logits_lock = logits_node.lock().unwrap();
                        logits_lock.value.clone()
                    },
                    &batch.targets,
                    &batch.mask,
                )?;

                // Scale loss and gradient by grad_accum_steps
                let scale = 1.0 / (config.grad_accum_steps as f32);
                total_loss += loss_val * scale;

                for g in loss_grad.data.iter_mut() {
                    *g *= scale;
                }

                // Backward
                {
                    let mut l_node = logits_node.lock().unwrap();
                    l_node.grad = Some(loss_grad);
                }
                Node::backward(logits_node);

                tokens_processed += batch.inputs.data.len();
                dash_state.total_tokens_ever += batch.inputs.data.len();
            }

            // 2. Gradient Clipping — compute norm before clipping
            let last_grad_norm = self.compute_grad_norm();
            self.clip_gradients(config.clip_grad_norm);

            // 3. Optimizer + Scheduler Step
            self.scheduler.step(&mut self.optimizer);

            // Execute optimizer step
            {
                {
                    let mut embed_lock = self.model.embedding.weight.lock().unwrap();
                    if let Some(g) = embed_lock.grad.take() {
                        self.optimizer.step_single(&mut embed_lock.value, &g);
                    }
                }
                {
                    let mut head_lock = self.model.head.lock().unwrap();
                    if let Some(g) = head_lock.grad.take() {
                        self.optimizer.step_single(&mut head_lock.value, &g);
                    }
                }
            }

            step += 1;

            // 4. Logging — rich dashboard
            if step % config.log_every == 0 {
                let elapsed_since_log = last_log_time.elapsed().as_secs_f32();
                let tok_per_sec = tokens_processed as f32 / elapsed_since_log.max(0.001);
                let elapsed_total = train_start.elapsed().as_secs_f64();

                // Update best loss
                if total_loss < dash_state.best_loss {
                    dash_state.best_loss = total_loss;
                    dash_state.best_loss_step = step;
                }

                self.print_dashboard(
                    config,
                    step,
                    total_loss,
                    last_grad_norm,
                    tok_per_sec,
                    elapsed_total,
                    &dash_state,
                );

                dash_state.prev_loss = total_loss;
                tokens_processed = 0;
                last_log_time = Instant::now();
            }

            // 5. Evaluation
            if step % config.eval_every == 0 && self.val_loader.is_some() {
                let num_eval_batches = 50;
                self.print_eval_header(step);
                let val_loss = self.eval_with_progress(num_eval_batches, step)?;
                let val_ppl = val_loss.exp();
                let train_ppl = total_loss.exp();
                let delta = val_loss - total_loss;

                let health = if delta < 0.2 {
                    "healthy gap, no overfitting"
                } else if delta < 0.5 {
                    "moderate gap"
                } else {
                    "WARNING: possible overfitting"
                };

                println!("Train loss : {:.4}  ppl {:.2}", total_loss, train_ppl);
                println!("Val   loss : {:.4}  ppl {:.2}", val_loss, val_ppl);
                println!("Delta      : {:+.4}  [{}]", delta, health);
                println!();

                dash_state.last_val_loss = Some(val_loss);
                dash_state.last_val_step = step;
            }

            // 6. Saving
            if step % config.save_every == 0 {
                let ckpt_name = format!("step_{:08}.ckpt", step);
                self.save_checkpoint(config, step)?;
                let ckpt_path = Path::new(&config.checkpoint_dir).join(&ckpt_name);
                let size = fs::metadata(&ckpt_path).map(|m| m.len()).unwrap_or(0);
                let size_str = if size >= 1_000_000 {
                    format!("{:.1} MB", size as f64 / 1_000_000.0)
                } else if size >= 1_000 {
                    format!("{:.1} KB", size as f64 / 1_000.0)
                } else {
                    format!("{} B", size)
                };
                println!("[Saved] {}  ({})", ckpt_name, size_str);
                dash_state.last_saved_ckpt = ckpt_name;
            }
        }

        println!(
            "\nTraining complete! {} steps, elapsed: {}",
            step,
            fmt_duration(train_start.elapsed().as_secs_f64())
        );

        // Final save as model_final.ckpt
        self.save_checkpoint_custom(config, "model_final.ckpt")?;
        println!("[Saved] model_final.ckpt");

        Ok(())
    }

    pub fn save_checkpoint_custom(&self, config: &TrainConfig, name: &str) -> Result<(), String> {
        let dir = Path::new(&config.checkpoint_dir);
        let ckpt_path = dir.join(name);
        let mut file = BufWriter::new(fs::File::create(&ckpt_path).map_err(|e| e.to_string())?);

        // Header
        file.write_all(b"FMLM").map_err(|e| e.to_string())?;
        file.write_all(&1u32.to_le_bytes())
            .map_err(|e| e.to_string())?; // version
        file.write_all(&0u64.to_le_bytes())
            .map_err(|e| e.to_string())?; // dummy step

        // Parameters
        let mut params = Vec::new();
        params.push({
            let embed_lock = self.model.embedding.weight.lock().unwrap();
            embed_lock.value.clone()
        });
        for block in &self.model.blocks {
            params.push({
                let r_lock = block.mixing.r_weight.lock().unwrap();
                r_lock.value.clone()
            });
            params.push({
                let k_lock = block.mixing.k_weight.lock().unwrap();
                k_lock.value.clone()
            });
            params.push({
                let v_lock = block.mixing.v_weight.lock().unwrap();
                v_lock.value.clone()
            });
        }
        params.push({
            let head_lock = self.model.head.lock().unwrap();
            head_lock.value.clone()
        });

        file.write_all(&(params.len() as u32).to_le_bytes())
            .map_err(|e| e.to_string())?;
        for p in params {
            file.write_all(&(p.data.len() as u32).to_le_bytes())
                .map_err(|e| e.to_string())?;
            for &val in &p.data {
                file.write_all(&val.to_le_bytes())
                    .map_err(|e| e.to_string())?;
            }
        }
        Ok(())
    }

    fn compute_grad_norm(&self) -> f32 {
        let mut total_norm_sq: f64 = 0.0;

        let params = self.collect_params();
        for p_ref in &params {
            let p_lock = p_ref.lock().unwrap();
            if let Some(grad) = &p_lock.grad {
                for &g in &grad.data {
                    total_norm_sq += (g as f64) * (g as f64);
                }
            }
        }

        (total_norm_sq as f32).sqrt()
    }

    fn collect_params(&self) -> Vec<crate::tensor::autograd::NodeRef> {
        let mut params = Vec::new();
        params.push(self.model.embedding.weight.clone());
        for block in &self.model.blocks {
            params.push(block.mixing.r_weight.clone());
            params.push(block.mixing.k_weight.clone());
            params.push(block.mixing.v_weight.clone());
        }
        params.push(self.model.head.clone());
        params
    }

    fn clip_gradients(&self, max_norm: f32) {
        let mut total_norm_sq = 0.0;

        let params = self.collect_params();

        for p_ref in &params {
            let p_lock = p_ref.lock().unwrap();
            if let Some(grad) = &p_lock.grad {
                for &g in &grad.data {
                    total_norm_sq += g * g;
                }
            }
        }

        let total_norm = (total_norm_sq as f32).sqrt();
        if total_norm > max_norm {
            let scale = max_norm / total_norm;
            for p_ref in &params {
                let mut p = p_ref.lock().unwrap();
                if let Some(grad) = &mut p.grad {
                    for g in &mut grad.data {
                        *g *= scale;
                    }
                }
            }
        }
    }

    fn print_eval_header(&self, step: usize) {
        print!("[Eval @ step {}] ", step);
        std::io::stdout().flush().unwrap();
    }

    pub fn eval_with_progress(&mut self, num_batches: usize, step: usize) -> Result<f32, String> {
        let loader = self.val_loader.as_mut().ok_or("No validation loader")?;
        let mut total_loss = 0.0;
        let mut count = 0;

        loader.reset();
        for i in 0..num_batches {
            if let Some(batch) = loader.next() {
                let input_ids: Vec<u32> = batch.inputs.data.iter().map(|&x| x as u32).collect();
                let (logits_node, _) = self.model.forward(&input_ids, None)?;
                let (loss_val, _) = cross_entropy_loss(
                    &{
                        let logits_lock = logits_node.lock().unwrap();
                        logits_lock.value.clone()
                    },
                    &batch.targets,
                    &batch.mask,
                )?;
                total_loss += loss_val;
                count += 1;

                // Progress
                let frac = (i + 1) as f64 / num_batches as f64;
                print!(
                    "\r[Eval @ step {}] {} {}/{} batches",
                    step,
                    progress_bar(frac, 20),
                    i + 1,
                    num_batches
                );
                std::io::stdout().flush().unwrap();
            } else {
                break;
            }
        }
        println!();

        Ok(if count > 0 {
            total_loss / count as f32
        } else {
            0.0
        })
    }

    pub fn eval(&mut self, num_batches: usize) -> Result<f32, String> {
        let loader = self.val_loader.as_mut().ok_or("No validation loader")?;
        let mut total_loss = 0.0;
        let mut count = 0;

        loader.reset();
        for _ in 0..num_batches {
            if let Some(batch) = loader.next() {
                let input_ids: Vec<u32> = batch.inputs.data.iter().map(|&x| x as u32).collect();
                let (logits_node, _) = self.model.forward(&input_ids, None)?;
                let (loss_val, _) = cross_entropy_loss(
                    &{
                        let logits_lock = logits_node.lock().unwrap();
                        logits_lock.value.clone()
                    },
                    &batch.targets,
                    &batch.mask,
                )?;
                total_loss += loss_val;
                count += 1;
            } else {
                break;
            }
        }

        Ok(if count > 0 {
            total_loss / count as f32
        } else {
            0.0
        })
    }

    pub fn save_checkpoint(&self, config: &TrainConfig, step: usize) -> Result<(), String> {
        let dir = Path::new(&config.checkpoint_dir);
        let ckpt_path = dir.join(format!("step_{:08}.ckpt", step));
        let mut file = BufWriter::new(fs::File::create(&ckpt_path).map_err(|e| e.to_string())?);

        // Header
        file.write_all(b"FMLM").map_err(|e| e.to_string())?;
        file.write_all(&1u32.to_le_bytes())
            .map_err(|e| e.to_string())?; // version
        file.write_all(&(step as u64).to_le_bytes())
            .map_err(|e| e.to_string())?;

        // Parameters
        let mut params = Vec::new();
        params.push({
            let embed_lock = self.model.embedding.weight.lock().unwrap();
            embed_lock.value.clone()
        });
        for block in &self.model.blocks {
            params.push({
                let r_lock = block.mixing.r_weight.lock().unwrap();
                r_lock.value.clone()
            });
            params.push({
                let k_lock = block.mixing.k_weight.lock().unwrap();
                k_lock.value.clone()
            });
            params.push({
                let v_lock = block.mixing.v_weight.lock().unwrap();
                v_lock.value.clone()
            });
        }
        params.push({
            let head_lock = self.model.head.lock().unwrap();
            head_lock.value.clone()
        });

        file.write_all(&(params.len() as u32).to_le_bytes())
            .map_err(|e| e.to_string())?;
        for p in params {
            file.write_all(&(p.data.len() as u32).to_le_bytes())
                .map_err(|e| e.to_string())?;
            for &val in &p.data {
                file.write_all(&val.to_le_bytes())
                    .map_err(|e| e.to_string())?;
            }
        }

        // Optimizer state
        let opt_path = ckpt_path.with_extension("opt");
        self.optimizer
            .save_state(&opt_path)
            .map_err(|e| e.to_string())?;

        // Rotate checkpoints
        self.rotate_checkpoints(&config.checkpoint_dir)
            .map_err(|e| e.to_string())?;

        Ok(())
    }

    fn rotate_checkpoints(&self, dir: &str) -> std::io::Result<()> {
        let mut ckpts: Vec<_> = fs::read_dir(dir)?
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().map_or(false, |ext| ext == "ckpt"))
            .collect();

        ckpts.sort_by_key(|e| e.path());

        if ckpts.len() > 10 {
            for i in 0..ckpts.len() - 10 {
                let path = ckpts[i].path();
                let name = path
                    .file_name()
                    .map(|f| f.to_string_lossy().to_string())
                    .unwrap_or_default();
                let _ = fs::remove_file(&path);
                let _ = fs::remove_file(path.with_extension("opt"));
                println!("[Saved] [deleted {}]", name);
            }
        }
        Ok(())
    }
}
