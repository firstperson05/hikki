use criterion::{black_box, criterion_group, criterion_main, Criterion};
use fastmind_lm::inference::engine::{InferenceConfig, InferenceEngine};
use fastmind_lm::model::lm::{FastMindConfig, FastMindLM};
use fastmind_lm::tensor::Tensor;
use fastmind_lm::tokenizer::bpe::BpeTokenizer;

fn bench_inference(c: &mut Criterion) {
    let config = FastMindConfig {
        vocab_size: 32000,
        n_layer: 12,
        n_embd: 768,
        seq_len: 1024,
    };
    let model = FastMindLM::new(config);
    // Mock tokenizer
    let tokenizer = BpeTokenizer::train("hello world", 100);
    let mut engine = InferenceEngine::new(model, tokenizer, 42);
    let _inf_config = InferenceConfig::default();

    c.bench_function("single_token_gen", |b| {
        b.iter(|| {
            // We use a dummy prompt to update state then generate 1 token
            let _ = engine.model.step(black_box(1), &mut engine.state);
        })
    });
}

fn bench_matmul(c: &mut Criterion) {
    let size = 1024;
    let a = Tensor::new(vec![size, size], vec![0.1; size * size]).unwrap();
    let b = Tensor::new(vec![size, size], vec![0.2; size * size]).unwrap();

    c.bench_function("matmul_1024x1024", |bench| {
        bench.iter(|| {
            let _ = black_box(&a).matmul(black_box(&b)).unwrap();
        })
    });
}

criterion_group!(benches, bench_inference, bench_matmul);
criterion_main!(benches);
