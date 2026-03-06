use rayon::prelude::*;

/// Mamba-style selective scan with parallel prefix sum.
/// Evaluates the recurrence: h_t = a_t * h_{t-1} + b_t
/// This allows pure recurrent inference while maintaining parallel training capabilities.
///
/// In a State Space Model (SSM), a_t depends on the state transition matrix
/// after discretization, and b_t depends on the input and input projection.
pub fn selective_scan(a: &[f32], b: &[f32], h_init: f32) -> Vec<f32> {
    let n = a.len();
    if n == 0 {
        return vec![];
    }

    // We demonstrate the associative property used for parallel prefix scan here.
    // Given elements (a_i, b_i), the composition is:
    // (a_i, b_i) ∘ (a_j, b_j) = (a_j * a_i, a_j * b_i + b_j)
    // This could be done with rayon's sequence scanning tools, but for simplicity
    // and correct compilation without extra cyclic dependencies, we implement
    // the chunked parallel reduction manually or fallback to a fast sequential loop.

    let chunk_size = 1024;

    if n <= chunk_size {
        // Sequential fallback for small inputs (O(1) inference mode)
        let mut h = vec![0.0; n];
        h[0] = a[0] * h_init + b[0];
        for i in 1..n {
            h[i] = a[i] * h[i - 1] + b[i];
        }
        return h;
    }

    // Parallel Prefix Sum (Scan) for large inputs (e.g. parallel training mode)
    // Step 1: Compute block reductions
    let num_chunks = (n + chunk_size - 1) / chunk_size;
    let block_reductions: Vec<(f32, f32)> = (0..num_chunks)
        .into_par_iter()
        .map(|idx| {
            let start = idx * chunk_size;
            let end = std::cmp::min(start + chunk_size, n);
            let mut a_cum = 1.0;
            let mut b_cum = 0.0;
            for i in start..end {
                b_cum = a[i] * b_cum + b[i];
                a_cum = a[i] * a_cum;
            }
            (a_cum, b_cum)
        })
        .collect();

    // Step 2: Exclusive scan on the block reductions
    let mut block_scans = vec![(1.0, 0.0); num_chunks];
    let mut current_a = 1.0;
    let mut current_b = h_init;
    for i in 0..num_chunks {
        block_scans[i] = (current_a, current_b);
        let (ba, bb) = block_reductions[i];
        current_b = ba * current_b + bb;
        current_a = ba * current_a;
    }

    // Step 3: Compute final values in parallel per block
    let mut h = vec![0.0; n];
    h.par_chunks_mut(chunk_size)
        .enumerate()
        .for_each(|(idx, chunk)| {
            let start = idx * chunk_size;
            let mut _a_cum = block_scans[idx].0; // Not perfectly needed for final h, but if we track
            let mut current_h = block_scans[idx].1;

            for i in 0..chunk.len() {
                current_h = a[start + i] * current_h + b[start + i];
                chunk[i] = current_h;
            }
        });

    h
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_selective_scan_basic() {
        let a = vec![0.5, 0.5, 0.5, 0.5];
        let b = vec![1.0, 1.0, 1.0, 1.0];
        let h_init = 0.0;
        let h = selective_scan(&a, &b, h_init);
        // h0 = 0.5*0 + 1.0 = 1.0
        // h1 = 0.5*1.0 + 1.0 = 1.5
        // h2 = 0.5*1.5 + 1.0 = 1.75
        // h3 = 0.5*1.75 + 1.0 = 1.875
        assert_eq!(h, vec![1.0, 1.5, 1.75, 1.875]);
    }

    #[test]
    fn test_selective_scan_parallel() {
        let a = vec![0.5; 2000];
        let b = vec![1.0; 2000];

        // Compute sequentially
        let mut expected = vec![0.0; 2000];
        expected[0] = 0.5 * 0.0 + 1.0;
        for i in 1..2000 {
            expected[i] = a[i] * expected[i - 1] + b[i];
        }

        // Compute via parallel function
        let actual = selective_scan(&a, &b, 0.0);

        for i in 0..2000 {
            assert!(
                (expected[i] - actual[i]).abs() < 1e-3,
                "Mismatch at index {}: exp {} got {}",
                i,
                expected[i],
                actual[i]
            );
        }
    }
}
