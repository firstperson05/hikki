use std::f32::consts::PI;

pub struct CosineScheduler {
    pub base_lr: f32,
    pub min_lr: f32,
    pub warmup_steps: usize,
    pub total_steps: usize,
}

impl CosineScheduler {
    pub fn get_lr(&self, step: usize) -> f32 {
        if step < self.warmup_steps {
            return self.base_lr * (step as f32 / self.warmup_steps as f32);
        }
        if step > self.total_steps {
            return self.min_lr;
        }

        let decay_ratio =
            (step - self.warmup_steps) as f32 / (self.total_steps - self.warmup_steps) as f32;
        let coeff = 0.5 * (1.0 + (PI * decay_ratio).cos());
        self.min_lr + coeff * (self.base_lr - self.min_lr)
    }

    pub fn step(&self, optimizer: &mut crate::training::optimizer::AdamW) {
        let lr = self.get_lr(optimizer.step);
        optimizer.lr = lr;
    }
}
