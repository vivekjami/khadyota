use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use khadyota::distance::*;

fn bench_cosine_distance(c: &mut Criterion) {
    let dims = vec![128, 256, 512, 768, 1024];
    
    for dim in dims {
        let a: Vec<f32> = (0..dim).map(|i| (i as f32).sin()).collect();
        let b: Vec<f32> = (0..dim).map(|i| (i as f32).cos()).collect();
        
        let mut group = c.benchmark_group(format!("cosine_distance_{}", dim));
        
        group.bench_function("scalar", |bench| {
            bench.iter(|| {
                scalar::cosine_distance_scalar(black_box(&a), black_box(&b))
            })
        });
        
        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("avx2") && dim % 8 == 0 {
            group.bench_function("avx2", |bench| {
                bench.iter(|| unsafe {
                    simd::cosine_distance_avx2(black_box(&a), black_box(&b))
                })
            });
        }
        
        group.bench_function("auto_dispatch", |bench| {
            bench.iter(|| {
                cosine_distance(black_box(&a), black_box(&b))
            })
        });
        
        group.finish();
    }
}

fn bench_euclidean_distance(c: &mut Criterion) {
    let dims = vec![128, 512, 1024];
    
    for dim in dims {
        let a: Vec<f32> = (0..dim).map(|i| i as f32 / dim as f32).collect();
        let b: Vec<f32> = (0..dim).map(|i| 1.0 - i as f32 / dim as f32).collect();
        
        let mut group = c.benchmark_group(format!("euclidean_{}", dim));
        
        group.bench_function("scalar", |bench| {
            bench.iter(|| {
                scalar::euclidean_distance_scalar(black_box(&a), black_box(&b))
            })
        });
        
        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("avx2") && dim % 8 == 0 {
            group.bench_function("avx2", |bench| {
                bench.iter(|| unsafe {
                    simd::euclidean_distance_avx2(black_box(&a), black_box(&b))
                })
            });
        }
        
        group.finish();
    }
}

criterion_group!(benches, bench_cosine_distance, bench_euclidean_distance);
criterion_main!(benches);