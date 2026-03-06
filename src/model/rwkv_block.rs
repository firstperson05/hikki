use crate::tensor::Tensor;

/// Time mixing via exponential moving average (RWKV-style).
/// Linearly interpolates between the current token `x` and the previous token `last_x`
/// based on learnable parameter `mix`.
pub fn time_mix(x: &Tensor, last_x: &Tensor, mix: &Tensor) -> Result<Tensor, String> {
    if x.shape != last_x.shape || x.shape != mix.shape {
        return Err("Shape mismatch in time_mix".to_string());
    }

    let mut out = vec![0.0; x.data.len()];
    for i in 0..x.data.len() {
        // x_mixed = x * mix_weight + last_x * (1 - mix_weight)
        let m = mix.data[i];
        out[i] = x.data[i] * m + last_x.data[i] * (1.0 - m);
    }

    Tensor::new(x.shape.clone(), out)
}

/// Channel mixing via Gated MLP.
/// A standard RWKV channel mix performs essentially:
/// r = sigmoid(x * r_weight)
/// k = relu(x * k_weight)^2
/// return r * (k * v_weight)
///
/// We simulate this using the existing tensor ops (matmul, relu).
pub fn channel_mix(
    x: &Tensor,
    r_weight: &Tensor,
    k_weight: &Tensor,
    v_weight: &Tensor,
) -> Result<Tensor, String> {
    // Receptance gate (r)
    let mut r = x.matmul(r_weight)?;
    // Apply sigmoid to r inline
    for val in r.data.iter_mut() {
        *val = 1.0 / (1.0 + (-*val).exp());
    }

    // Key (k)
    let mut k = x.matmul(k_weight)?.relu();
    // Square relu output
    for val in k.data.iter_mut() {
        *val = (*val) * (*val);
    }

    // Value (v)
    let v = k.matmul(v_weight)?;

    // Element-wise multiply r and v
    if r.shape != v.shape {
        return Err("Shape mismatch between r and v in channel_mix".to_string());
    }

    let mut out = vec![0.0; r.data.len()];
    for i in 0..out.len() {
        out[i] = r.data[i] * v.data[i];
    }

    Tensor::new(r.shape.clone(), out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_time_mix() {
        let x = Tensor::new(vec![2], vec![1.0, 2.0]).unwrap();
        let last_x = Tensor::new(vec![2], vec![0.0, 4.0]).unwrap();
        let mix = Tensor::new(vec![2], vec![1.0, 0.5]).unwrap();

        let out = time_mix(&x, &last_x, &mix).unwrap();
        // i=0 => 1.0*1.0 + 0.0*0.0 = 1.0
        // i=1 => 2.0*0.5 + 4.0*0.5 = 3.0
        assert_eq!(out.data, vec![1.0, 3.0]);
    }
}
